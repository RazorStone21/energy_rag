"""生成阶段评测：Faithfulness / Answer Relevancy / Context Precision / Context Recall。

默认用 RAGAS，评分使用 [judge] 配置的判分模型（OpenAI 兼容 API）和本地嵌入模型。
判分模型必须与生成模型分开：同一个模型给自己生成的答案打分时指标会失真——实测
Qwen3-8B 自评时 81% 的题 faithfulness 恰好得 1.0，失去区分度。
API Key 从 [judge] api_key_env 指向的环境变量读取，不写进配置文件。
评测会把检索到的文档片段发给判分服务，涉密语料不要走外部 API。
若 RAGAS 未安装或运行失败，改用字符层面的相似度指标，不能当作 RAGAS 得分。

生成 121 题耗时长，因此支持按题型分层抽样（--limit）、只跑指定题型（--only-type）、
断点续跑（--resume）以及每题生成后立即落盘：中途中断只损失当前这一题。

评分同样按批落盘（每批 SCORE_BATCH 题）：--resume 既复用已生成的答案，也复用已评分的题，
评分阶段中断后接着评即可，不必从头重评。评分并发数由 --workers 控制，
默认值及「本地判分模型相反」的实测结论见 DEFAULT_WORKERS 的说明。

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
from math import isnan
from pathlib import Path

# 在模型加载前设置分配策略，缓解显存碎片化导致的内存不足
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# 关闭 RAGAS 的使用统计上报。上报会在评分过程中发起 HTTP 请求，在连不通外网的环境里
# 每次都要等到连接超时：实测 2 题评分从 85 秒涨到 438 秒（慢 5 倍），指标结果不受影响。
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")


from src.bootstrap import create_runtime
from src.config import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = HERE / "questions.json"
RESULTS_DIR = HERE.parent / "results"

# 评分的 4 个指标，顺序与 RAGAS 返回的逐题分数键一致。
RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_LABELS = {
    "faithfulness": "Faithfulness（忠实度）",
    "answer_relevancy": "Answer Relevancy（答案相关性）",
    "context_precision": "Context Precision（上下文精确率）",
    "context_recall": "Context Recall（上下文召回率）",
}
# 评分并发数。判分走外部 API 时并发是正收益（瓶颈是网络往返），默认 4。
# 注意与本地判分模型的历史结论相反：那时在 RTX 4090 + Qwen3-8B 4bit 上实测
# 并发是负收益——同时 4 个请求时单次调用从约 3 秒涨到 8~16 秒，瓶颈是 RAGAS
# 与 LangChain 的 Python 侧开销（GIL 下不并行）而不是算力。若把判分换回本地
# 模型，应把这个值调回 1。
DEFAULT_WORKERS = 4
# 每批评分多少题就落盘一次：批越大，每批的固定开销摊得越薄；批越小，中断损失越少。
SCORE_BATCH = 8
# 单个评分操作允许等待的秒数，比原先的 300 秒放宽，避免排队时误触发重试。
SCORE_TIMEOUT = 600


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


def read_report(path: Path) -> dict:
    """读取上次的报告文件；文件不存在或内容损坏时按空报告处理，不抛异常。"""
    if not path.exists():
        return {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return previous if isinstance(previous, dict) else {}


def load_completed_samples(report: dict, questions: list[dict]) -> list[dict]:
    """读取上次未跑完的生成结果，按题目顺序复用，供 --resume 跳过已完成的题。

    只认报告里带 answer 的条目；题目仍在本轮范围内才复用。答案与判分模型无关，
    换判分模型后仍然可以复用。
    """
    completed = {item.get("question"): item for item in report.get("samples", [])}
    return [completed[q["question"]] for q in questions if q["question"] in completed]


def load_completed_scores(report: dict, questions: list[dict], judge: str) -> list[dict]:
    """读取上次已完成的逐题评分，按题目顺序复用，供 --resume 跳过已评分的题。

    评分与答案分开复用：生成中断时答案先落盘，评分中断时评分先落盘，互不影响。
    判分模型换过时一律不复用：同一个答案被不同模型评出的分数不可比，混进同一份
    报告会让均值失去意义——报告里的 judge 字段就是用来判断这一点的。
    """
    if str(report.get("judge", "")) != str(judge):
        return []
    done = {row.get("question"): row for row in report.get("scores", [])}
    return [done[q["question"]] for q in questions if q["question"] in done]


def build_judge_llm(settings):
    """按 [judge] 配置创建 RAGAS 判分模型，走 OpenAI 兼容接口。

    API Key 从配置指定的环境变量读取，不写进配置文件也不落进报告。
    缺这个变量时直接报错：与其让每一题都失败一次，不如在开跑前就说清楚。
    """
    import os

    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get(settings.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"环境变量 {settings.api_key_env} 未设置，无法调用判分模型 "
            f"{settings.model}（{settings.base_url}）"
        )
    return LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.model,
            base_url=settings.base_url,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=3,
            timeout=SCORE_TIMEOUT,
        ),
        # answer_relevancy 默认要 n=3 条候选（strictness）。多数 OpenAI 兼容服务
        # 只支持 n=1，DeepSeek 会直接返回 400 "Invalid n value"，该题就变成 null。
        # bypass_n 改为发 3 次独立的单条请求，指标语义不变。
        bypass_n=True,
    )


def run_ragas(samples: list[dict], runtime, workers: int = DEFAULT_WORKERS):
    """用 RAGAS 给一批样本评分，返回带逐题分数的结果对象（scores 与输入同序）。

    workers 控制并发评分任务数。判分走外部 API 时并发是正收益：瓶颈是网络往返
    而不是本地算力（换成本地模型判分时实测并发是负收益，见 DEFAULT_WORKERS）。
    """
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm = build_judge_llm(runtime.settings.judge)
    # 嵌入模型仍用本地的 bge-m3：它与生成模型不同，不构成自评，而且不用为
    # answer_relevancy 单独买一套嵌入 API。
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
    return evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embedder,
        run_config=RunConfig(max_workers=workers, timeout=SCORE_TIMEOUT, max_retries=2),
    )


def sample_scores(samples: list[dict], result) -> list[dict]:
    """把 RAGAS 结果整理成逐题分数，供写盘和 --resume 复用。

    指标失败时 RAGAS 记 NaN，这里转成 None；numpy 标量必须转成内置 float，
    否则 json.dumps 会靠 default=str 把它写成字符串。
    """
    rows = []
    for sample, scores in zip(samples, result.scores):
        row = {"question": sample["question"], "type": sample.get("type", "")}
        for name in RAGAS_METRIC_NAMES:
            value = scores.get(name)
            row[name] = None if value is None or isnan(float(value)) else round(float(value), 4)
        rows.append(row)
    return rows


def average_scores(rows: list[dict], workers: int) -> dict:
    """对逐题分数按指标求均值，跳过失败的 None，并记录参与统计的题数。"""
    metrics = {}
    for name in RAGAS_METRIC_NAMES:
        values = [row[name] for row in rows if row.get(name) is not None]
        metrics[name] = round(sum(values) / len(values), 4) if values else None
    return {
        "n": len(rows),
        "workers": workers,
        "metrics": metrics,
        "note": "逐题分数见 scores 字段；某题指标失败时该题为 null，均值只统计成功的题。",
    }


def print_scores(rows: list[dict], workers: int) -> None:
    """在控制台打印各指标均值；指标名用中文解释，便于直接读结论。"""
    summary = average_scores(rows, workers)
    print("\n=== RAGAS 指标 ===")
    for name in RAGAS_METRIC_NAMES:
        value = summary["metrics"][name]
        print(f"  {METRIC_LABELS[name]}: {'—' if value is None else f'{value:.4f}'}")
    print(f"  （{summary['n']} 题，评分并发 {workers}）")


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
    parser.add_argument("--resume", action="store_true", help="复用报告中已生成的答案与评分")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"RAGAS 评分并发数（默认 {DEFAULT_WORKERS}）：判分走外部 API 时并发更快；换回本地判分模型应设为 1",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit 必须为正整数")
    if args.workers <= 0:
        parser.error("--workers 必须为正整数")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [q for q in questions if q.get("question")]
    questions = select_questions(questions, args.limit, args.only_type, args.seed)
    out_path = Path(args.out)
    settings = load_settings(args.config, args.data_root)
    print(f"判分模型：{settings.judge.model}（{settings.judge.base_url}）")
    print(f"本轮评测 {len(questions)} 题（题库过滤与抽样后），开始生成回答…")

    previous = read_report(out_path)
    samples = load_completed_samples(previous, questions) if args.resume else []
    score_rows = load_completed_scores(previous, questions, settings.judge.model) if args.resume else []
    if args.resume and previous.get("scores") and not score_rows:
        print(f"[resume] 已有评分出自 {previous.get('judge', '未知判分模型')}，与当前判分模型不同，不复用。")
    done = {sample["question"] for sample in samples}
    if done:
        print(f"[resume] 复用已生成的 {len(done)} 题，跳过生成。")
    if score_rows:
        print(f"[resume] 复用已评分的 {len(score_rows)} 题，跳过评分。")
    pending = [q for q in questions if q["question"] not in done]

    # 生成和评分都需要运行环境；模型是延迟加载的，提前创建不占显存。
    runtime = create_runtime(settings)

    def build_report(current, scored, error=None):
        """组装报告；生成与评分都可能中途中断，因此两者都要能单独写盘。

        报告里记录判分模型：换判分模型后分数不可直接横向比较，必须能看出
        某份报告是哪一次、由哪个模型评出来的。
        """
        report = {
            "n": len(current),
            "planned": len(questions),
            "seed": args.seed,
            "only_type": args.only_type,
            "judge": runtime.settings.judge.model,
            "samples": current,
            "scores": scored,
        }
        if scored:
            report["ragas"] = average_scores(scored, args.workers)
        if error is not None:
            report["ragas_error"] = error
        return report

    # 生成期间随时可能中断，因此每题完成后就把已有结果写进报告文件。
    def checkpoint(current):
        """把已完成的问题与答案写入报告文件，评分结果随后补进同一份报告。"""
        write_report(out_path, build_report(current, score_rows))

    if pending:
        samples += collect_predictions(pending, runtime, on_sample=checkpoint)

    # 答案已生成，不再需要重排模型；先清理它以减少随后评分的显存占用。
    runtime.release_reranker()

    # 评分是整轮最慢的一段，因此按批评分并随时落盘：中断只损失当前这一批，
    # --resume 会跳过已评分的题接着评。批不小于并发数，避免并发流空转。
    scored = {row["question"] for row in score_rows}
    to_score = [sample for sample in samples if sample["question"] not in scored]
    batch_size = max(SCORE_BATCH, args.workers)
    if to_score:
        print(f"[score] 待评分 {len(to_score)} 题，并发 {args.workers}，每批 {batch_size} 题。")
    try:
        for start in range(0, len(to_score), batch_size):
            batch = to_score[start : start + batch_size]
            score_rows += sample_scores(batch, run_ragas(batch, runtime, args.workers))
            write_report(out_path, build_report(samples, score_rows))
            print(f"[score] 已完成 {len(score_rows)}/{len(samples)} 题")
    except Exception as e:
        # 评分整轮重跑代价很高，所以失败时保留已完成的题，交给 --resume 接着评。
        print(f"\n[RAGAS 运行失败] {e}")
        report = build_report(samples, score_rows, error=str(e))
        if score_rows:
            print(f"已保留中断前完成的 {len(score_rows)} 题评分，加 --resume 可接着评。")
        else:
            report["lexical"] = run_lexical(samples)
            print("\n=== 词法近似指标 ===")
            print(json.dumps(report["lexical"], ensure_ascii=False, indent=2))
        write_report(out_path, report)
        print(f"\n报告已保存到 {args.out}")
        return

    if score_rows:
        print_scores(score_rows, args.workers)
    write_report(out_path, build_report(samples, score_rows))
    print(f"\n报告已保存到 {args.out}")


if __name__ == "__main__":
    main()
