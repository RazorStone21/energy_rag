"""生成阶段评测：Faithfulness / Answer Relevancy / Context Precision / Context Recall。

默认用 RAGAS，评分使用配置中的文本模型和嵌入模型，与生成答案共用模型实例。
若 RAGAS 未安装或运行失败，改用字符层面的相似度指标，不能当作 RAGAS 得分。

生成 121 题耗时长，因此支持按题型分层抽样（--limit）、只跑指定题型（--only-type）、
断点续跑（--resume）以及每题生成后立即落盘：中途中断只损失当前这一题。

用法：
    python -m tests.evaluation.run_rag_eval                    # 全部题目
    python -m tests.evaluation.run_rag_eval --limit 30         # 分层抽 30 题
    python -m tests.evaluation.run_rag_eval --only-type table numeric
    python -m tests.evaluation.run_rag_eval --limit 30 --resume   # 接着上次的结果跑

报告写入 tests/results/。
"""

import argparse
import json
import os
import random
from pathlib import Path

# 在模型加载前设置分配策略，缓解显存碎片化导致的内存不足
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


from src.bootstrap import create_runtime
from src.config import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE.parent / "results"


def select_questions(questions: list[dict], limit, only_types, seed: int) -> list[dict]:
    """按题型筛选并分层抽样，返回本轮要跑的题目。

    分层而不是整体随机，是因为 table、numeric 这类题量少又是当前短板：
    纯随机抽样很容易让某一类整类缺席，指标就失去参考价值。
    同一个 seed 抽出的题目固定，换种子等于换一套样本。
    """
    selected = [q for q in questions if not only_types or q.get("type") in only_types]
    if limit is None or limit >= len(selected):
        return selected
    groups: dict[str, list[dict]] = {}
    for question in selected:
        groups.setdefault(question.get("type", ""), []).append(question)
    generator = random.Random(seed)
    for group in groups.values():
        generator.shuffle(group)
    picked: list[dict] = []
    # 逐轮从每个题型各取一题，题量少的类型先被取空，剩下的位置再分给题量多的类型。
    while len(picked) < limit:
        progressed = False
        for key in sorted(groups):
            if not groups[key]:
                continue
            picked.append(groups[key].pop())
            progressed = True
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def collect_predictions(questions: list[dict], runtime, on_sample=None) -> list[dict]:
    """逐题生成答案，记录问题、实际使用的片段、答案和参考答案，供后续评分。

    每题完成后调用 on_sample，让调用方把已完成的结果落盘：整轮跑完可能要几小时，
    中途中断时已生成的答案不该丢掉。
    """
    samples = []
    for q in questions:
        query = q["question"]
        result = runtime.pipeline.ask(query)
        contexts = [hit.document.page_content for hit in result.evidence]
        answer = result.answer
        samples.append(
            {
                "question": query,
                "type": q.get("type", ""),
                "contexts": contexts,
                "answer": answer.strip(),
                "reference": q.get("reference_answer", ""),
            }
        )
        print(f"[gen] 完成：{query[:30]}…")
        if on_sample is not None:
            on_sample(samples)
    return samples


def write_report(path: Path, report: dict) -> None:
    """把评测报告写到磁盘；未完成时也会调用，保存已经生成的部分。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_completed_samples(path: Path, questions: list[dict]) -> list[dict]:
    """读取上次未跑完的生成结果，按题目顺序复用，供 --resume 跳过已完成的题。

    只认报告里带 answer 的条目；题目仍在本轮范围内才复用。
    """
    if not path.exists():
        return []
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    completed = {item.get("question"): item for item in previous.get("samples", [])}
    return [completed[q["question"]] for q in questions if q["question"] in completed]


def run_ragas(samples: list[dict], runtime) -> dict:
    """RAGAS 评测（需要 ragas 已安装）。

    max_workers=1 让评分任务依次执行，减少多个生成请求同时占用显存。
    每次评分允许等待 300 秒，以适应本地模型速度；仍可能因资源不足或依赖问题失败。
    """
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm = LangchainLLMWrapper(runtime.generator.load())
    embedder = LangchainEmbeddingsWrapper(runtime.embedder.load())

    ds = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                retrieved_contexts=s["contexts"],
                response=s["answer"],
                reference=s["reference"] or None,
            )
            for s in samples
        ]
    )
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embedder,
        run_config=RunConfig(max_workers=1, timeout=300, max_retries=2),
    )
    return result


# RAGAS 不可用时使用的字符相似度指标。
def _rouge_l(ref: str, hyp: str) -> float:
    """按字符计算 ROUGE-L F1：ref 是参考答案，hyp 是生成答案，用最长公共子序列比较。"""
    if not ref or not hyp:
        return 0.0
    m, n = len(ref), len(hyp)
    # dp[i][j] 表示两段文本各取前 i、j 个字符时，最长公共子序列的长度。
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / len(hyp) if hyp else 0.0
    rec = lcs / len(ref) if ref else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def run_lexical(samples: list[dict]) -> dict:
    """汇总答案与参考文本的字符相似度，作为 RAGAS 无法运行时的粗略对照。

    bleu_1 是历史字段名，实际计算去重字符重合率，并非标准 BLEU-1。
    faithfulness_heuristic 只比较答案与上下文的字符，不验证答案事实是否正确。
    """
    rouge, bleu, faithful = [], [], []

    def _bleu1(ref: str, hyp: str) -> float:
        """计算答案与参考答案的去重字符重合率，属于词法近似而非标准 BLEU。"""
        if not ref or not hyp:
            return 0.0
        ref_tok, hyp_tok = set(ref), set(hyp)
        return len(ref_tok & hyp_tok) / len(hyp_tok) if hyp_tok else 0.0

    for s in samples:
        ref = s["reference"]
        hyp = s["answer"]
        rouge.append(_rouge_l(ref, hyp))
        bleu.append(_bleu1(ref, hyp))
        # 只统计答案的不同字符中有多少也出现在上下文里，不判断句意或事实。
        ctx = "".join(s["contexts"])
        if hyp and ctx:
            faithful.append(len(set(hyp) & set(ctx)) / len(set(hyp)))
        else:
            faithful.append(0.0)

    def avg(x):
        """计算列表平均值并保留四位小数，空列表返回零。"""
        return round(sum(x) / len(x), 4) if x else 0.0

    return {
        "avg": {
            "rouge_l": avg(rouge),
            "bleu_1": avg(bleu),
            "faithfulness_heuristic": avg(faithful),
        },
        "note": "RAGAS 未运行，使用词法近似指标（仅供参考，不代表真实 RAGAS 指标）",
    }


def main():
    """运行生成评测并保存报告，RAGAS 不可用时记录词法近似结果。"""
    parser = argparse.ArgumentParser(description="生成阶段评测")
    parser.add_argument("--questions", type=str, default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out", type=str, default=str(RESULTS_DIR / "rag_eval_report.json"))
    parser.add_argument("--limit", type=int, default=None, help="最多评测多少题，按题型分层抽取")
    parser.add_argument(
        "--only-type",
        nargs="+",
        default=None,
        help="只评测这些题型，例如 --only-type table numeric",
    )
    parser.add_argument("--seed", type=int, default=17, help="分层抽样的随机种子")
    parser.add_argument("--resume", action="store_true", help="复用报告中已生成的答案")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in questions if q.get("question")]
    questions = select_questions(questions, args.limit, args.only_type, args.seed)
    out_path = Path(args.out)
    print(f"本轮评测 {len(questions)} 题（题库过滤与抽样后），开始生成回答…")

    samples = load_completed_samples(out_path, questions) if args.resume else []
    done = {sample["question"] for sample in samples}
    if done:
        print(f"[resume] 复用已生成的 {len(done)} 题，跳过生成。")
    pending = [q for q in questions if q["question"] not in done]

    # 生成期间随时可能中断，因此每题完成后就把已有结果写进报告文件。
    def checkpoint(current):
        """把已完成的问题与答案写入报告文件，评测指标等全部生成后再补。"""
        write_report(
            out_path,
            {
                "n": len(current),
                "planned": len(questions),
                "seed": args.seed,
                "only_type": args.only_type,
                "samples": current,
            },
        )

    if pending:
        runtime = create_runtime(load_settings(args.config, args.data_root))
        samples += collect_predictions(pending, runtime, on_sample=checkpoint)
    else:
        # 全部题目都已生成过，只需要一个运行环境来做评分。
        runtime = create_runtime(load_settings(args.config, args.data_root))

    # 答案已生成，不再需要重排模型；先清理它以减少随后评分的显存占用。
    runtime.release_reranker()

    report = {
        "n": len(samples),
        "planned": len(questions),
        "seed": args.seed,
        "only_type": args.only_type,
        "samples": samples,
    }
    try:
        result = run_ragas(samples, runtime)
        report["ragas"] = result
        print("\n=== RAGAS 指标 ===")
        print(result)
    except Exception as e:
        print(f"\n[RAGAS 运行失败，回退词法指标] {e}")
        report["lexical"] = run_lexical(samples)
        print("\n=== 词法近似指标 ===")
        print(json.dumps(report["lexical"], ensure_ascii=False, indent=2))

    write_report(out_path, report)
    print(f"\n报告已保存到 {args.out}")


if __name__ == "__main__":
    main()
