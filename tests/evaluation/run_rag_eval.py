"""生成阶段评测：Faithfulness / Answer Relevancy / Context Precision / Context Recall。

默认用 RAGAS，评分使用配置中的文本模型和嵌入模型，与生成答案共用模型实例。
若 RAGAS 未安装或运行失败，改用字符层面的相似度指标，不能当作 RAGAS 得分。

用法：
    python -m tests.evaluation.run_rag_eval

报告写入 tests/results/。
"""

import argparse
import json
import os
from pathlib import Path

# 在模型加载前设置分配策略，缓解显存碎片化导致的内存不足
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


from src.bootstrap import create_runtime
from src.config import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE.parent / "results"


def collect_predictions(questions: list[dict], runtime) -> list[dict]:
    """逐题生成答案，记录问题、实际使用的片段、答案和参考答案，供后续评分。"""
    samples = []
    for q in questions:
        query = q["question"]
        result = runtime.pipeline.ask(query)
        contexts = [hit.document.page_content for hit in result.evidence]
        answer = result.answer
        samples.append(
            {
                "question": query,
                "contexts": contexts,
                "answer": answer.strip(),
                "reference": q.get("reference_answer", ""),
            }
        )
        print(f"[gen] 完成：{query[:30]}…")
    return samples


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
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    runtime = create_runtime(load_settings(args.config, args.data_root))

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in questions if q.get("question")]
    print(f"加载 {len(questions)} 条测试问题，开始生成回答…")

    samples = collect_predictions(questions, runtime)

    # 答案已生成，不再需要重排模型；先清理它以减少随后评分的显存占用。
    runtime.release_reranker()

    report = {"n": len(samples)}
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n报告已保存到 {args.out}")


if __name__ == "__main__":
    main()
