"""切分方式对比评测：同一批 PDF 用三种切分方式各建一个 Milvus 库，跑同一套标注问题。

对比的三种切分方式：

    semantic        语义切分，按相邻句组的语义差异分布选断点（生产默认）
    recursive_char  递归字符切分，优先在段落/换行/中文标点处断开，固定长度加重叠
    element         Unstructured 元素切分，按标题层级聚合，利用文档结构

每种方式有自己独立的 Milvus 库和片段缓存，互不干扰：

    tests/results/chunk_compare/<策略>/milvus.db       该策略的向量库
    tests/results/chunk_compare/<策略>/chunks.pkl      该策略的片段（BM25 也从这里读）
    tests/results/chunk_compare/<策略>/manifest.json   该策略的文件哈希清单
    tests/results/chunk_compare/<策略>/chunks.json     该策略的片段（可读格式）
    tests/results/chunk_compare/<策略>/stats.json      chunk 数量与长度分布
    tests/results/chunk_compare/<策略>/metrics.json    检索指标
    tests/results/chunk_compare/summary.json           三者的横向对比

表格片段与切分方式无关，作为常数基线加入每种策略，可用 --no-tables 关闭。

图片描述不参与对比，因此本脚本建出的库比生产索引少 figure 片段（当前语料为 129 条，
来自 6 份报告类 PDF 的图表）。提取图片描述需要视觉模型、明显拖慢评测，而它对三种
切分方式是同一份常数，不影响横向比较，所以脚本不提供该开关。

三种策略都走生产的检索链路（向量 + BM25 + RRF，可选 rerank），差异只来自切分方式。
semantic 与 recursive_char 共用同一份解析结果，只换切分器；element 直接对 PDF 元素
聚合，不经过按页合并正文的步骤。

用法：
    python -m tests.evaluation.run_chunk_compare --max-files 5     # 建议先小集合试跑
    python -m tests.evaluation.run_chunk_compare                   # 全量 44 个 PDF
    python -m tests.evaluation.run_chunk_compare --strategies semantic element
    python -m tests.evaluation.run_chunk_compare --no-rerank       # 只跑纯向量召回
"""

import argparse
import json
import pickle
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.bootstrap import create_runtime
from src.chunker import Chunker, meaningful_len
from src.config import load_settings
from src.ingestion import file_hash
from src.schemas import ParseResult
from tests.evaluation.run_retrieval_eval import (
    COMPARE_NOTE,
    METRIC_NOTES,
    RELEVANCE_NOTE,
    _print_result,
)
from tests.evaluation.run_retrieval_eval import (
    run as run_retrieval,
)

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
DEFAULT_OUT_DIR = HERE.parent / "results" / "chunk_compare"
STRATEGIES = ("semantic", "recursive_char", "element")
HEADLINE = ("recall@5", "hit@5", "mrr", "ndcg@5")

# unstructured 元素类别的保留集合，与生产 PDFParser 的 fast 通道一致。
# 不含 Table，表格统一由表格基线提供，避免与元素切分的结果重复计入。
ELEMENT_CATEGORIES = ("Title", "NarrativeText", "Text", "ListItem")

# 递归字符切分的分隔符优先级，靠前的先尝试；空串表示最后按字符硬切。
RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


class NoVision:
    """占位视觉组件：available 恒为 False，让 PDF 解析器跳过图片描述。

    图片描述对三种切分方式是同一份常数，提取又需要视觉模型，
    因此不参与横向对比；库会比生产索引少 figure 片段。
    """

    available = False


def build_pdf_parser(settings):
    """创建只做正文和表格提取的 PDF 解析器，不经过按后缀分发的注册表。

    注册表只暴露统一的 parse，无法单独取到 PDF 解析器；这里按生产的构造方式
    直接创建，并用 NoVision 关掉图片描述。
    """
    from src.parsers.pdf import PDFParser

    return PDFParser(NoVision(), settings.vision)


def list_pdfs(doc_dir: Path, max_files: int | None) -> list[Path]:
    """按文件名排序列出文档目录第一层的 PDF，可用 max_files 只取前若干个。"""
    paths = sorted(p for p in doc_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    return paths[:max_files] if max_files else paths


def parse_texts(parser, path: Path, cache: dict) -> list:
    """解析正文并在策略之间复用，解析是整条链路里最慢的一步。"""
    if path.name not in cache:
        cache[path.name] = parser.parse_text(path)
    return cache[path.name]


def table_baseline(parser, paths: list[Path], cache: dict) -> list:
    """抽取表格片段作为三种策略共同的基线，与切分方式无关，只取一次。"""
    tables = []
    for path in paths:
        if path.name not in cache:
            try:
                cache[path.name] = parser.table_extractor(path)
            except Exception as exc:  # 单个文件表格失败不影响其他文件
                print(f"[warn] 表格抽取失败 {path.name}: {exc}")
                cache[path.name] = []
        tables.extend(cache[path.name])
    return tables


def filter_noise(chunks: list, min_chars: int) -> list:
    """丢弃有效字符数不足的片段，判定规则与生产 Chunker.filter 一致。"""
    return [c for c in chunks if meaningful_len(c.page_content) >= min_chars]


def make_recursive_factory(chunk_size: int, chunk_overlap: int):
    """生成忽略语义参数的切分器工厂，让 Chunker 用同一套流程跑递归字符切分。

    Chunker 创建切分器时还会传入 breakpoint_threshold_type 等语义参数，
    这里统一忽略，只把递归切分自己的长度和重叠参数固定进去。
    """

    def factory(embedder, **semantic_options):
        """返回按固定长度和重叠递归断开的切分器，不使用嵌入模型。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=RECURSIVE_SEPARATORS,
        )

    return factory


def split_elements(path: Path) -> list:
    """用 unstructured 的 fast 通道解析元素，抽不到文字时退回 OCR 通道。"""
    from unstructured.partition.pdf import partition_pdf

    for strategy in ("fast", "ocr_only"):
        elements = partition_pdf(
            filename=str(path),
            strategy=strategy,
            languages=["chi_sim"],
        )
        if elements:
            return elements
    return []


def element_chunks(path: Path, max_characters: int, new_after: int, combine_under: int) -> list:
    """按标题层级聚合元素切分，跨页 section 取第一页作为页码。

    保留类别与生产 fast 通道一致，因此表格仍由统一的基线提供，不在这里重复。
    """
    from langchain_core.documents import Document
    from unstructured.chunking.title import chunk_by_title

    kept = [element for element in split_elements(path) if element.category in ELEMENT_CATEGORIES]
    if not kept:
        return []
    sections = chunk_by_title(
        kept,
        max_characters=max_characters,
        new_after_n_chars=new_after,
        combine_text_under_n_chars=combine_under,
    )
    documents = []
    for section in sections:
        text = (section.text or "").strip()
        if not text:
            continue
        page = getattr(section.metadata, "page_number", None)
        # 跨页 section 的页码是列表，取第一页代表该片段所在位置。
        if isinstance(page, list):
            page = page[0] if page else None
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page or 0, "type": "text"},
            )
        )
    return documents


def build_chunks(strategy: str, parser, paths: list[Path], chunker, options: dict):
    """按指定策略切出各文件的文本片段，返回 (片段, 失败原因)。

    单文件解析或切分失败时跳过并记录，不中断其余文件。
    """
    chunks, failed = [], {}
    for path in paths:
        try:
            if strategy == "element":
                text_chunks = element_chunks(
                    path,
                    options["element_max_characters"],
                    options["element_new_after"],
                    options["element_combine_under"],
                )
                text_chunks = filter_noise(text_chunks, options["min_chars"])
            else:
                # 只把正文交给 Chunker，表格由基线统一提供，避免重复计入。
                parsed = ParseResult(texts=parse_texts(parser, path, options["parse_cache"]))
                text_chunks = chunker.split(parsed)
            if not text_chunks:
                raise ValueError("没有切出可用片段")
            chunks.extend(text_chunks)
        except Exception as exc:
            failed[path.name] = str(exc)
            print(f"[warn] 切分失败 {path.name}: {exc}")
    return chunks, failed


def chunk_stats(chunks: list) -> dict:
    """统计片段数量、长度分布和来源文件分布。"""
    lengths = [len(c.page_content) for c in chunks]
    sources = Counter(c.metadata.get("source", "") for c in chunks)
    return {
        "count": len(chunks),
        "total_chars": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "min_chars": min(lengths) if lengths else 0,
        "sources": dict(sources.most_common()),
    }


def strategy_settings(settings, strategy_dir: Path):
    """把索引和向量库路径改到该策略自己的目录，其余配置保持不变。"""
    return replace(
        settings,
        chunks_path=strategy_dir / "chunks.pkl",
        manifest_path=strategy_dir / "manifest.json",
        milvus=replace(
            settings.milvus,
            connection_args={"uri": str(strategy_dir / "milvus.db")},
        ),
    )


def metrics_report(result: dict) -> dict:
    """给指标结果加上中文说明，结构与 run_retrieval_eval 的报告保持一致。"""
    return {
        "指标说明": {
            "recall@k": METRIC_NOTES["recall@k"],
            "precision@k": METRIC_NOTES["precision@k"],
            "hit@k": METRIC_NOTES["hit@k"],
            "mrr": METRIC_NOTES["mrr"],
            "ndcg@k": METRIC_NOTES["ndcg@k"],
            "vector_only vs vector_rerank": COMPARE_NOTE,
            "相关片段判定方式": RELEVANCE_NOTE,
        },
        **result,
    }


def write_artifacts(strategy_dir: Path, chunks: list, stats: dict, result: dict) -> None:
    """把片段、统计和指标写入该策略自己的目录。"""
    strategy_dir.mkdir(parents=True, exist_ok=True)
    (strategy_dir / "chunks.json").write_text(
        json.dumps(
            [
                {
                    "source": chunk.metadata.get("source", ""),
                    "page": chunk.metadata.get("page", ""),
                    "type": chunk.metadata.get("type", "text"),
                    "content": chunk.page_content,
                }
                for chunk in chunks
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (strategy_dir / "chunks.pkl").open("wb") as stream:
        pickle.dump(chunks, stream)
    (strategy_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (strategy_dir / "metrics.json").write_text(
        json.dumps(metrics_report(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def evaluate_strategy(strategy: str, settings, paths: list[Path], questions: list, options: dict):
    """为单一策略建库、跑指标并落盘，返回 (指标结果, chunk 统计)。

    每个策略使用独立的 Runtime，跑完释放模型引用，避免三套模型同时占用显存。
    切分器和向量库共用同一个 Runtime 的嵌入组件，与生产结构一致。
    """
    strategy_dir = Path(options["out_dir"]) / strategy
    runtime = create_runtime(strategy_settings(settings, strategy_dir))
    try:
        chunker = None
        if strategy == "semantic":
            chunker = Chunker(runtime.embedder, settings.splitting)
        elif strategy == "recursive_char":
            chunker = Chunker(
                runtime.embedder,
                settings.splitting,
                splitter_factory=make_recursive_factory(
                    options["recursive_chunk_size"], options["recursive_chunk_overlap"]
                ),
            )

        chunks, failed = build_chunks(strategy, options["parser"], paths, chunker, options)
        if options["with_tables"]:
            chunks.extend(
                filter_noise(
                    table_baseline(options["parser"], paths, options["table_cache"]),
                    options["min_chars"],
                )
            )
        if not chunks:
            raise RuntimeError(f"{strategy} 没有切出任何片段，无法建库")
        stats = chunk_stats(chunks)

        # 向量库和片段缓存都指向该策略目录，BM25 因此也只读本策略的片段。
        runtime.vector_store.replace_all(chunks)
        runtime.chunk_store.publish(chunks, {path.name: file_hash(path) for path in paths})

        result = {"vector_only": run_retrieval(questions, runtime, with_rerank=False)}
        if options["with_rerank"]:
            result["vector_rerank"] = run_retrieval(questions, runtime, with_rerank=True)
        result["failed"] = failed
        write_artifacts(strategy_dir, chunks, stats, result)
        return result, stats
    finally:
        runtime.release_models()


def headline_line(strategy: str, stats: dict, result: dict, modes: list) -> str:
    """拼出汇总表的一行：片段数量、平均长度，以及各模式的头部指标。"""
    line = f"{strategy:<16} chunks={stats['count']:>5} avg_len={stats['avg_chars']:>7.1f}"
    for mode in modes:
        average = (result.get(mode) or {}).get("avg") or {}
        if not average:
            line += f"  [{mode}] (无样本)"
            continue
        values = " ".join(f"{key}={average.get(key, 0):.3f}" for key in HEADLINE)
        line += f"  [{mode}] {values}"
    return line


def main():
    """解析参数，逐策略建库并跑检索指标，最后写出横向对比。"""
    parser = argparse.ArgumentParser(description="切分方式对比评测")
    parser.add_argument(
        "--strategies",
        nargs="*",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="要对比的切分方式（默认全部）",
    )
    parser.add_argument("--max-files", type=int, default=None, help="只取前 N 个 PDF，便于试跑")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-rerank", action="store_true", help="只跑纯向量召回")
    parser.add_argument("--no-tables", action="store_true", help="不加入表格基线片段")
    parser.add_argument("--recursive-chunk-size", type=int, default=800)
    parser.add_argument("--recursive-chunk-overlap", type=int, default=100)
    parser.add_argument("--element-max-characters", type=int, default=800)
    parser.add_argument("--element-new-after", type=int, default=700)
    parser.add_argument("--element-combine-under", type=int, default=200)
    args = parser.parse_args()

    settings = load_settings(args.config, args.data_root)
    questions = [
        item
        for item in json.loads(Path(args.questions).read_text(encoding="utf-8"))
        if item.get("question")
    ]
    paths = list_pdfs(settings.doc_dir, args.max_files)
    print(f"加载 {len(questions)} 条测试问题，对比 {len(paths)} 个 PDF")
    print(f"切分方式：{', '.join(args.strategies)}")
    if args.no_tables:
        print("已关闭表格基线，只对比文本片段")

    # 解析和表格结果在策略之间复用；element 走自己的元素解析，不用这两个缓存。
    options = {
        "out_dir": args.out_dir,
        "parser": build_pdf_parser(settings),
        "with_rerank": not args.no_rerank,
        "with_tables": not args.no_tables,
        "min_chars": settings.splitting.min_chars,
        "parse_cache": {},
        "table_cache": {},
        "recursive_chunk_size": args.recursive_chunk_size,
        "recursive_chunk_overlap": args.recursive_chunk_overlap,
        "element_max_characters": args.element_max_characters,
        "element_new_after": args.element_new_after,
        "element_combine_under": args.element_combine_under,
    }

    results, stats = {}, {}
    for strategy in args.strategies:
        print(f"\n{'=' * 72}\n策略：{strategy}\n{'=' * 72}")
        result, strategy_stats = evaluate_strategy(strategy, settings, paths, questions, options)
        results[strategy], stats[strategy] = result, strategy_stats
        if result["failed"]:
            print(f"失败文件 {len(result['failed'])} 个：{list(result['failed'])[:3]}")
        if "vector_only" in result:
            _print_result(f"{strategy} · 纯向量召回", result["vector_only"])
        if "vector_rerank" in result:
            _print_result(f"{strategy} · 向量召回 + rerank", result["vector_rerank"])

    modes = ["vector_only"] + ([] if args.no_rerank else ["vector_rerank"])
    print("\n" + "=" * 100)
    print("切分方式对比汇总（指标均值，0~1 越大越好）")
    print("=" * 100)
    for strategy in args.strategies:
        print(headline_line(strategy, stats[strategy], results[strategy], modes))

    summary = {
        "strategies": args.strategies,
        "max_files": args.max_files,
        "questions": args.questions,
        "chunk_stats": stats,
        "results": results,
        "参数": {
            "recursive_chunk_size": args.recursive_chunk_size,
            "recursive_chunk_overlap": args.recursive_chunk_overlap,
            "element_max_characters": args.element_max_characters,
            "element_new_after": args.element_new_after,
            "element_combine_under": args.element_combine_under,
            "表格基线": not args.no_tables,
            "重排对比": not args.no_rerank,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n产物已保存到 {out_dir}")


if __name__ == "__main__":
    main()
