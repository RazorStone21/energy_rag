"""运行本地检索评测，比较向量搜索与向量搜索加重排，不调用答案生成模型。

指标：Recall@k / Precision@k / Hit@k / MRR / nDCG@k（k = 1/3/5/10）。
同时对比「纯向量召回」vs「向量召回 + rerank」，量化重排序收益。

用法：
    python -m tests.evaluation.run_retrieval_eval              # 使用 tests/evaluation/questions.json
    python -m tests.evaluation.run_retrieval_eval --questions xxx.json
    python -m tests.evaluation.run_retrieval_eval --no-rerank  # 只跑纯向量召回

报告写入 tests/results/。
"""

import argparse
import json
import math
from pathlib import Path

from src.bootstrap import create_runtime
from src.config import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE.parent / "results"
KS = (1, 3, 5, 10)

# ---- 指标中文说明（写入报告 JSON 的「指标说明」字段，并在控制台末尾打印）----
METRIC_NOTES = {
    "recall@k": "召回率@k：前 k 条检索结果中命中的「正确片段」数 ÷ 该题标注的正确片段总数。回答「该找的关键内容找回来了多少」，数值越低说明漏掉的相关内容越多。",
    "precision@k": "精确率@k：前 k 条结果中正确片段数 ÷ k（返回总数）。回答「返回的结果里有多少是真正相关的」，数值越低说明混入的无关噪声越多。",
    "hit@k": "命中率@k：前 k 条结果里只要出现 ≥1 条正确片段，该题即记 1（否则记 0），再对所有题目取平均。回答「有多少比例的题目至少找到了一条相关内容」。",
    "mrr": "MRR（平均倒数排名）：每题取第一条正确片段所在位置 r，该题得 1/r（没找到记 0），再对全部题目平均。回答「正确答案排得有多靠前」，满分 1.0 表示每题第一名就是正确片段。",
    "ndcg@k": "nDCG@k（归一化折损累计增益）：带排序位置权重的命中指标——排得越靠前权重越高（第 i 位权重 1/log2(i+2)），再除以理想排序的最大值做归一化。比 Recall/Hit 更严格，能区分「相关片段排第 1 还是排第 10」。",
}
COMPARE_NOTE = (
    "vector_only = 纯向量召回；vector_rerank = 向量召回 + bge-reranker 重排序。"
    "同一套题目各跑一遍做对比，用于量化重排序（rerank）带来的召回与排序收益。"
)
RELEVANCE_NOTE = (
    "「正确片段」的判定：检索到的 chunk 与标注的 relevant_chunks 在忽略空格和换行后存在"
    "子串包含关系，或 3-gram Jaccard 相似度 ≥ 0.6，即视为命中。上述所有指标取值都在 0~1 之间，越大越好。"
)


def _strip_layout(s: str) -> str:
    """去掉空格和换行；PDF 抽取出的换行位置随排版变化，不携带语义。"""
    return s.replace("\n", "").replace(" ", "")


def _ngrams(s: str, n: int = 3) -> set:
    """按每次移动一个字符提取连续 n 个字符，返回去重后的集合。"""
    s = _strip_layout(s)
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: str, b: str, n: int = 3) -> float:
    """用共有字符片段数除以全部不同字符片段数，粗略比较两段文字的重合程度。"""
    A, B = _ngrams(a, n), _ngrams(b, n)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def is_relevant(retrieved_text: str, relevant_texts: list[str], threshold: float = 0.6) -> bool:
    """用文字包含关系或字符片段相似度判断是否命中标注，并不判断语义是否一致。

    包含关系也在去掉空格和换行后比较：同一句话在标注里和在 PDF 片段里的断行
    位置往往不同，逐字比较会把明明命中的结果判成未命中。Jaccard 一直按这套
    归一化计算，两处口径必须一致。
    """
    text = _strip_layout(retrieved_text)
    for rt in relevant_texts:
        stripped = _strip_layout(rt)
        if text in stripped or stripped in text:
            return True
        if _jaccard(retrieved_text, rt) >= threshold:
            return True
    return False


def metrics_for_query(retrieved_docs, relevant_texts: list[str]) -> dict:
    """根据按排名排列的片段计算单题指标；precision 的分母是实际返回数量。

    当前按检索结果逐条计命中，多个结果可能对应同一条标注，未做一一匹配。
    recall 和 nDCG 被限制在 1 以内，结果只能作为近似对比。
    """
    rel = [is_relevant(d.page_content, relevant_texts) for d in retrieved_docs]
    n_rel = len(relevant_texts)
    out = {}

    for k in KS:
        topk = rel[:k]
        hits = sum(topk)
        out[f"recall@{k}"] = min(1.0, hits / n_rel) if n_rel else 0.0
        out[f"precision@{k}"] = hits / len(topk) if topk else 0.0
        out[f"hit@{k}"] = 1.0 if hits > 0 else 0.0

        # 命中的片段排得越靠前，贡献越大；再用理想排序得分进行归一化。
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(topk))
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
        out[f"ndcg@{k}"] = min(1.0, dcg / ideal) if ideal else 0.0

    mrr = 0.0
    for i, r in enumerate(rel, start=1):
        if r:
            mrr = 1.0 / i
            break
    out["mrr"] = mrr
    return out


def _average(metric_list: list[dict]) -> dict:
    """对非空评测结果取平均，跳过 question 字段；其余字段应都是数值指标。"""
    keys = [k for k in metric_list[0].keys() if k != "question"]
    return {k: round(sum(m[k] for m in metric_list) / len(metric_list), 4) for k in keys}


def run(questions: list[dict], runtime, with_rerank: bool = True) -> dict:
    """逐题运行纯向量检索及可选重排，跳过没有片段标注的问题并汇总指标。"""
    per_query = []
    for q in questions:
        query = q["question"]
        relevant = q.get("relevant_chunks") or []
        if not relevant:
            print(f"[skip] 缺少 relevant_chunks 标注：{query}")
            continue
        # 关闭 BM25，保证两组只比较是否启用重排；重排结果数量受 context_top_k 限制。
        hits = runtime.pipeline.retrieve(query, with_rerank=with_rerank, hybrid=False)
        docs = [hit.document for hit in hits]
        m = metrics_for_query(docs, relevant)
        m["question"] = query
        per_query.append(m)

    if not per_query:
        return {"avg": {}, "per_query": [], "n": 0}

    return {"avg": _average(per_query), "per_query": per_query, "n": len(per_query)}


def _print_result(title: str, result: dict):
    """在控制台展示一组检索评测的汇总指标，空样本时明确提示。"""
    print(f"\n### {title}  (n={result['n']})")
    avg = result["avg"]
    if not avg:
        print("  无有效样本")
        return
    print(
        f"  Recall@1/3/5/10 : {avg['recall@1']:.4f} / {avg['recall@3']:.4f} / {avg['recall@5']:.4f} / {avg['recall@10']:.4f}"
    )
    print(
        f"  Precision@1/3/5 : {avg['precision@1']:.4f} / {avg['precision@3']:.4f} / {avg['precision@5']:.4f}"
    )
    print(
        f"  Hit@1/3/5/10    : {avg['hit@1']:.4f} / {avg['hit@3']:.4f} / {avg['hit@5']:.4f} / {avg['hit@10']:.4f}"
    )
    print(f"  MRR             : {avg['mrr']:.4f}")
    print(
        f"  nDCG@1/3/5/10   : {avg['ndcg@1']:.4f} / {avg['ndcg@3']:.4f} / {avg['ndcg@5']:.4f} / {avg['ndcg@10']:.4f}"
    )


def _print_notes():
    """在控制台末尾打印每个指标的中文含义，便于阅读报告。"""
    print("\n" + "=" * 60)
    print("指标说明（各指标取值均为 0~1，越大越好）")
    print("=" * 60)
    for k, v in METRIC_NOTES.items():
        print(f"  · {k}\n      {v}")
    print(f"  · 对比\n      {COMPARE_NOTE}")
    print(f"  · 判定\n      {RELEVANCE_NOTE}")


def main():
    """加载标注问题，对比纯向量与重排检索，并写出指标报告。"""
    parser = argparse.ArgumentParser(description="检索指标评测")
    parser.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--no-rerank", action="store_true", help="只跑纯向量召回")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "retrieval_eval_report.json"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    runtime = create_runtime(load_settings(args.config, args.data_root))

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"加载 {len(questions)} 条测试问题")

    report = {}

    if args.no_rerank:
        report["vector_only"] = run(questions, runtime, with_rerank=False)
        _print_result("纯向量召回 (vector only)", report["vector_only"])
    else:
        report["vector_only"] = run(questions, runtime, with_rerank=False)
        report["vector_rerank"] = run(questions, runtime, with_rerank=True)
        _print_result("纯向量召回 (vector only)", report["vector_only"])
        _print_result("向量召回 + rerank", report["vector_rerank"])

    # 把中文说明一并写入报告文件，并把「指标说明」放到最前面
    report = {
        "指标说明": {
            "recall@k": METRIC_NOTES["recall@k"],
            "precision@k": METRIC_NOTES["precision@k"],
            "hit@k": METRIC_NOTES["hit@k"],
            "mrr": METRIC_NOTES["mrr"],
            "ndcg@k": METRIC_NOTES["ndcg@k"],
            "vector_only vs vector_rerank": COMPARE_NOTE,
            "相关片段判定方式": RELEVANCE_NOTE,
        },
        **report,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存到 {args.out}")
    _print_notes()


if __name__ == "__main__":
    main()
