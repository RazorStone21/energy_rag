"""受控实验：只改「片段重叠」这一个变量，其余全部保持不变。

做法是复用已入库的片段（生产索引的 chunks.pkl），对同一批正文片段生成
「不重叠」和「按比例重叠」两个版本，各自建库后用同一批题对比检索指标。
这样切分器、片段大小、题目、模型、检索参数全都相同，差异只可能来自重叠。

只有正文片段会加重叠：表格和图片本来就是完整单元、不参与句子切分，
给它们加重叠只会制造重复内容，不是这次要检验的变量。

除了常规检索指标，还统计 top-k 内部的重复度——用来检验「重叠片段
会互相挤占上下文位置」这个判断是否成立：context_top_k 只有 5 个位置，
如果其中两条高度相似，就等于白白少了一条真正不同的证据。

用法：
    python -m tests.evaluation.run_overlap_eval                    # 0% 与 20%
    python -m tests.evaluation.run_overlap_eval --ratios 0 0.1 0.3
    python -m tests.evaluation.run_overlap_eval --source-chunks data/chunks.pkl

报告写入 tests/results/overlap_eval_report.json。
"""

import argparse
import json
import re
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from src.bootstrap import create_runtime
from src.config import load_settings

from .run_retrieval_eval import (
    METRIC_NOTES,
    _average,
    _jaccard,
    _print_result,
    metrics_for_query,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "chunks.pkl"
DEFAULT_QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE.parent / "results"

# 报告里附上的说明，避免只看数字不知道在比什么。
EXPERIMENT_NOTE = (
    "对照组与实验组使用同一批片段、同一批题目和同一套检索参数，"
    "唯一区别是正文片段是否带上与上一块的重叠。表格和图片片段两组完全相同。"
)
DUPLICATION_NOTE = (
    "top-k 最大重复度：取前 k 条结果中任意两条之间最高的 3-gram Jaccard 相似度。"
    "数值越高说明返回的片段越像，真正不同的证据越少——context_top_k 只有 5 个位置，"
    "重复的片段会直接挤掉本可以放进去的其他内容。"
)


def _with_content(chunk, text):
    """复制片段并替换正文；兼容 LangChain Document 与其他实现。"""
    copy_method = getattr(chunk, "model_copy", None)
    if copy_method is not None:
        return copy_method(update={"page_content": text})
    return type(chunk)(page_content=text, metadata=dict(chunk.metadata))


def overlap_tail(text, char_count, sentence_regex):
    """从上一块末尾截出用于重叠的一段文字，尽量从一个句子开头开始。

    不按句边界对齐就会把半句话接在下一块前面，让模型读到断头的内容。
    """
    if char_count <= 0 or not text:
        return ""
    tail = text[-char_count:]
    # sentence_regex 是后顾断言，匹配到的是句子结束符之后的零宽位置。
    marks = list(re.finditer(sentence_regex, tail))
    if marks and 0 < marks[0].end() < len(tail):
        tail = tail[marks[0].end() :]
    return tail


def add_overlap(chunks, ratio, sentence_regex):
    """给正文片段加上与同一来源上一块的重叠。

    重叠取自上一块的原文而不是已经加过重叠的结果，否则重叠会一层层累积。
    表格、图片以及每个来源的第一块都不加。
    """
    if ratio <= 0:
        return list(chunks)
    result = []
    previous_text = {}
    for chunk in chunks:
        metadata = chunk.metadata
        source = metadata.get("source")
        if metadata.get("type", "text") != "text":
            result.append(chunk)
            continue
        text = chunk.page_content
        previous = previous_text.get(source)
        if previous:
            tail = overlap_tail(previous, int(len(text) * ratio), sentence_regex)
            if tail:
                text = tail + text
        previous_text[source] = chunk.page_content
        result.append(_with_content(chunk, text))
    return result


def top_k_duplication(documents, k):
    """返回前 k 条里最相似的一对的 3-gram Jaccard 相似度；不足两条时为 0。"""
    texts = [document.page_content for document in documents[:k]]
    if len(texts) < 2:
        return 0.0
    return max(_jaccard(a, b) for a, b in combinations(texts, 2))


# 判定重叠时要求的最短匹配长度：中文里一两字的偶然重复很常见，
# 不过滤掉会把基线也算出几个百分点，看不出真实差异。
_MIN_SHARED_CHARS = 5


def overlap_ratio(chunks):
    """估算片段之间的实际重叠：相邻同源正文片段的最长公共前后缀占当前块的比例。"""
    ratios = []
    previous = {}
    for chunk in chunks:
        if chunk.metadata.get("type", "text") != "text":
            continue
        source = chunk.metadata.get("source")
        text = chunk.page_content
        prior = previous.get(source)
        previous[source] = text
        if not prior or not text:
            continue
        # 只有前缀确实是上一块的结尾时才算重叠，普通相同的开头不算。
        shared = 0
        for size in range(min(len(prior), len(text)), _MIN_SHARED_CHARS - 1, -1):
            if prior.endswith(text[:size]):
                shared = size
                break
        if shared:
            ratios.append(shared / len(text))
    return sum(ratios) / len(ratios) if ratios else 0.0


def run(questions, runtime, with_rerank, duplication_k):
    """逐题检索并计算指标，同时统计 context_top_k 内部的重复度。"""
    per_query = []
    duplicated = []
    for question in questions:
        relevant = question.get("relevant_chunks") or []
        if not relevant:
            continue
        hits = runtime.pipeline.retrieve(
            question["question"], with_rerank=with_rerank, hybrid=False
        )
        documents = [hit.document for hit in hits]
        metrics = metrics_for_query(documents, relevant)
        metrics["question"] = question["question"]
        per_query.append(metrics)
        duplicated.append(top_k_duplication(documents, duplication_k))
    if not per_query:
        return {"avg": {}, "per_query": [], "n": 0}
    numeric = [
        {key: value for key, value in m.items() if isinstance(value, (int, float))}
        for m in per_query
    ]
    return {
        "avg": _average(numeric),
        "per_query": per_query,
        "n": len(per_query),
        "平均 top-k 重复度": round(sum(duplicated) / len(duplicated), 4),
    }


def evaluate(ratio, source_chunks, settings, questions, duplication_k, out_dir):
    """为某个重叠比例建库并评测，返回指标、分块统计和实际重叠率。"""
    chunks = add_overlap(source_chunks, ratio, settings.splitting.sentence_regex)
    variant_dir = Path(out_dir) / f"overlap_{int(ratio * 100)}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    runtime = create_runtime(
        replace(
            settings,
            chunks_path=variant_dir / "chunks.pkl",
            manifest_path=variant_dir / "manifest.json",
            milvus=replace(
                settings.milvus,
                connection_args={"uri": str(variant_dir / "milvus.db")},
            ),
        )
    )
    try:
        runtime.vector_store.replace_all(chunks)
        # 发布片段缓存：检索前的就绪检查会读它，BM25 也依赖同一份片段。
        runtime.chunk_store.publish(
            chunks, {name: "eval" for name in {c.metadata.get("source", "") for c in chunks}}
        )
        result = {"vector_only": run(questions, runtime, False, duplication_k)}
        result["vector_rerank"] = run(questions, runtime, True, duplication_k)
        stats = {
            "count": len(chunks),
            "avg_chars": round(sum(len(c.page_content) for c in chunks) / len(chunks), 1),
            "实际重叠率": round(overlap_ratio(chunks), 4),
        }
        (variant_dir / "metrics.json").write_text(
            json.dumps({"参数": {"ratio": ratio}, **stats, **result}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result, stats
    finally:
        runtime.release_models()


def main():
    """按给定的重叠比例逐个建库评测，最后打印横向对比。"""
    parser = argparse.ArgumentParser(description="片段重叠的受控实验")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.2])
    parser.add_argument("--source-chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--duplication-k", type=int, default=5, help="统计重复度时取前几条")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "overlap_compare"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "overlap_eval_report.json"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    import pickle

    source_chunks = pickle.loads(Path(args.source_chunks).read_bytes())
    body = [c for c in source_chunks if c.metadata.get("type", "text") == "text"]
    print(f"来源片段 {len(source_chunks)} 条，其中正文 {len(body)} 条（只有正文参与重叠实验）")

    settings = load_settings(args.config, args.data_root)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"加载 {len(questions)} 条测试问题\n")

    report, stats_by_ratio = {}, {}
    for ratio in args.ratios:
        result, stats = evaluate(
            ratio,
            source_chunks,
            settings,
            questions,
            args.duplication_k,
            args.out_dir,
        )
        key = f"重叠 {int(ratio * 100)}%"
        report[key] = {**stats, **result}
        stats_by_ratio[ratio] = stats
        print(
            f"=== {key}（片段 {stats['count']} 条，平均 {stats['avg_chars']} 字，"
            f"实测重叠率 {stats['实际重叠率'] * 100:.1f}%）"
        )
        for mode, label in (("vector_only", "纯向量"), ("vector_rerank", "向量+重排")):
            _print_result(f"{label} · {key}", result[mode])
            print(
                f"  平均 top-{args.duplication_k} 重复度 : {result[mode]['平均 top-k 重复度']:.4f}"
            )
        print()

    suffix = f"（取前 {args.duplication_k} 条统计）"
    report = {
        "指标说明": METRIC_NOTES,
        "实验设计": EXPERIMENT_NOTE,
        f"top-k 重复度{suffix}": DUPLICATION_NOTE,
        "对照结果": report,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("受控对比（唯一变量：正文片段是否重叠）")
    print("=" * 72)
    print(
        f"{'重叠比例':<12}{'片段数':>7}{'平均字数':>9}  {'模式':<10}{'MRR':>8}{'Recall@5':>10}{'重复度':>9}"
    )
    for ratio in args.ratios:
        stats = stats_by_ratio[ratio]
        for mode, label in (("vector_only", "纯向量"), ("vector_rerank", "向量+重排")):
            average = report[f"重叠 {int(ratio * 100)}%"][mode]
            print(
                f"{int(ratio * 100)}%{'':<9}{stats['count']:>7}{stats['avg_chars']:>9}"
                f"  {label:<10}{average['avg']['mrr']:>8.4f}{average['avg']['recall@5']:>10.4f}"
                f"{average['平均 top-k 重复度']:>9.4f}"
            )
    print(f"\n报告已保存到 {args.out}")


if __name__ == "__main__":
    main()
