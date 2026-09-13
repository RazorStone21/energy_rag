"""评测多轮追问的查询改写效果，量化「结合历史还原指代」带来的检索收益。

对照三档，指标与 run_retrieval_eval 完全一致（Recall@k / Precision@k / Hit@k / MRR / nDCG@k）：

    raw       直接用追问原文检索，代表当前的现状基线；
    resolved  用题库里标注的「还原句」检索，代表改写做到完美时的上限；
    rewritten 用真实改写组件的输出检索，需要 --with-rewriter 才跑。

raw 与 resolved 之间的差距就是改写能吃到的收益空间；rewritten 离 resolved 有多近，
则说明改写组件本身的质量。

用法：
    python -m tests.evaluation.run_rewrite_eval                  # 跑 raw 与 resolved
    python -m tests.evaluation.run_rewrite_eval --with-rewriter   # 额外跑真实改写
    python -m tests.evaluation.run_rewrite_eval --questions xxx.json

报告写入 tests/results/rewrite_eval_report.json。
"""

import argparse
import json
from pathlib import Path

from src.bootstrap import create_runtime
from src.config import load_settings

from .run_retrieval_eval import METRIC_NOTES, _average, _print_result, metrics_for_query

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "followup_questions.json"
RESULTS_DIR = HERE.parent / "results"

MODE_NOTES = {
    "raw": "追问原文：直接拿用户的话去检索，不结合历史。代表当前的现状。",
    "resolved": "还原句：题库人工标注的、不依赖上下文的独立问句。代表改写做到完美时的上限。",
    "rewritten": "改写输出：真实改写组件结合历史生成的问题，用来衡量组件本身的质量。",
}


def run(questions: list[dict], runtime, mode: str, with_rerank: bool, rewriter=None) -> dict:
    """逐题按指定模式取检索词并计算指标，跳过缺少标注的问题。"""
    per_query = []
    for q in questions:
        relevant = q.get("relevant_chunks") or []
        if not relevant:
            print(f"[skip] 缺少 relevant_chunks 标注：{q['question']}")
            continue

        history = [(t["role"], t["text"]) for t in q.get("history", [])]
        if mode == "raw":
            search_query = q["question"]
        elif mode == "resolved":
            search_query = q["resolved"]
        elif mode == "rewritten":
            if rewriter is None:
                raise ValueError("rewritten 模式需要传入 rewriter")
            search_query = rewriter.rewrite(q["question"], history=history)
        else:
            raise ValueError(f"未知模式: {mode}")

        # 关闭 BM25，与 run_retrieval_eval 保持一致，两组只比较是否启用重排。
        hits = runtime.pipeline.retrieve(search_query, with_rerank=with_rerank, hybrid=False)
        docs = [hit.document for hit in hits]
        m = metrics_for_query(docs, relevant)
        m["question"] = q["question"]
        m["search_query"] = search_query
        per_query.append(m)

    if not per_query:
        return {"avg": {}, "per_query": [], "n": 0}
    # per_query 里还带着 question 和 search_query 两个字符串字段，
    # 求平均前先只保留数值型指标，避免把它们也算进去。
    numeric = [{k: v for k, v in m.items() if isinstance(v, (int, float))} for m in per_query]
    return {"avg": _average(numeric), "per_query": per_query, "n": len(per_query)}


def _print_queries(result: dict, limit: int = 3):
    """打印前若干题的检索词，便于人工核对改写是否真的还原了指代。"""
    rows = result.get("per_query") or []
    if not rows:
        return
    print(f"  检索词示例（前 {min(limit, len(rows))} 题）：")
    for m in rows[:limit]:
        print(f"    「{m['question']}」")
        print(f"      → 实际检索：{m['search_query']}")


def main():
    """加载追问题库，按模式对比检索指标并写出报告。"""
    parser = argparse.ArgumentParser(description="多轮追问的查询改写评测")
    parser.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--with-rewriter", action="store_true", help="额外评测真实改写组件")
    parser.add_argument("--no-rerank", action="store_true", help="只跑纯向量召回")
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "rewrite_eval_report.json"))
    parser.add_argument("--show", type=int, default=3, help="每档打印几条检索词示例")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config, args.data_root)
    runtime = create_runtime(settings)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"加载 {len(questions)} 条多轮追问")

    modes = ["raw", "resolved"]
    if args.with_rewriter:
        modes.append("rewritten")
    # 直接用 Runtime 按 config.toml 组装好的改写器，评测的就是线上真正生效的那一个。
    rewriter = runtime.rewriter if args.with_rewriter else None

    variants = (("vector_only", False),) if args.no_rerank else (
        ("vector_only", False),
        ("vector_rerank", True),
    )  # fmt: skip

    report = {}
    for variant, with_rerank in variants:
        report[variant] = {}
        for mode in modes:
            result = run(questions, runtime, mode, with_rerank, rewriter)
            report[variant][mode] = result
            _print_result(f"{variant} · {MODE_NOTES[mode].split('：')[0]}", result)
            if args.show:
                _print_queries(result, args.show)

    report = {
        "指标说明": {**METRIC_NOTES, "模式说明": MODE_NOTES},
        **report,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存到 {args.out}")

    # 三种模式的横向对比，直接看改写吃掉了多少差距。
    if "vector_rerank" in report and report["vector_rerank"].get("raw", {}).get("avg"):
        rr = report["vector_rerank"]
        print("\n" + "=" * 60)
        print("改写收益（vector_rerank，MRR / Recall@5）")
        print("=" * 60)
        for mode in modes:
            avg = rr[mode]["avg"]
            print(f"  {mode:10s} MRR={avg['mrr']:.4f}  Recall@5={avg['recall@5']:.4f}")


if __name__ == "__main__":
    main()
