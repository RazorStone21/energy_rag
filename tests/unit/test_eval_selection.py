"""生成评测的抽样与续跑回归测试；只验证纯函数，不加载模型也不连数据库。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evaluation.run_rag_eval import (
    average_scores,
    load_completed_samples,
    load_completed_scores,
    read_report,
    sample_scores,
    select_questions,
)

QUESTIONS = [
    {"question": f"{kind}-{index}", "type": kind}
    for kind, count in (("table", 13), ("numeric", 25), ("factual", 18), ("figure", 15))
    for index in range(count)
]

# 报告里记录的判分模型名，用于验证换模型后不复用旧评分。
JUDGE = "deepseek-v4-pro"


def test_limit_keeps_every_type_present():
    """验证分层抽样：题量少的题型不会被随机抽没。"""
    picked = select_questions(QUESTIONS, 12, None, seed=17)
    assert len(picked) == 12
    counts = Counter(q["type"] for q in picked)
    # 4 个题型各 12/4 = 3 题，最小的 table（13 题）也不会缺席。
    assert set(counts) == {"table", "numeric", "factual", "figure"}
    assert set(counts.values()) == {3}


def test_sampling_is_deterministic_and_seed_dependent():
    """验证同一 seed 抽出的题目固定，换 seed 会换样本。"""
    first = select_questions(QUESTIONS, 10, None, seed=17)
    again = select_questions(QUESTIONS, 10, None, seed=17)
    other = select_questions(QUESTIONS, 10, None, seed=18)
    assert [q["question"] for q in first] == [q["question"] for q in again]
    assert [q["question"] for q in first] != [q["question"] for q in other]


def test_only_type_filters_before_sampling():
    """验证题型过滤先于抽样，且样本不超出所选题型。"""
    picked = select_questions(QUESTIONS, 6, ["table", "numeric"], seed=17)
    assert len(picked) == 6
    assert {q["type"] for q in picked} == {"table", "numeric"}


def test_limit_above_question_count_returns_everything():
    """验证上限大于题库时不抽样，直接返回全部题目。"""
    picked = select_questions(QUESTIONS, 999, None, seed=17)
    assert [q["question"] for q in picked] == [q["question"] for q in QUESTIONS]
    assert select_questions(QUESTIONS, None, None, seed=17) == picked


def test_resume_reuses_only_questions_in_this_round(tmp_path):
    """验证断点续跑只复用本轮范围内的已完成题目，并按顺序对齐。"""
    report = tmp_path / "rag_eval_report.json"
    report.write_text(
        json.dumps(
            {
                "samples": [
                    {"question": "table-0", "answer": "已生成"},
                    {"question": "不在本轮范围内", "answer": "已生成"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    questions = [{"question": "table-1"}, {"question": "table-0"}]
    completed = load_completed_samples(read_report(report), questions)
    assert [sample["question"] for sample in completed] == ["table-0"]
    # 文件不存在或内容损坏时按没有历史处理，不抛异常。
    assert load_completed_samples(read_report(tmp_path / "missing.json"), questions) == []
    broken = tmp_path / "broken.json"
    broken.write_text("{不是 JSON", encoding="utf-8")
    assert load_completed_samples(read_report(broken), questions) == []


def test_resume_reuses_completed_scores_only_in_this_round(tmp_path):
    """验证评分续跑只复用本轮范围内的题目，文件缺失或损坏时按没有历史处理。"""
    report = tmp_path / "rag_eval_report.json"
    report.write_text(
        json.dumps(
            {
                "judge": JUDGE,
                "scores": [
                    {"question": "table-0", "faithfulness": 0.8},
                    {"question": "不在本轮范围内", "faithfulness": 0.9},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    questions = [{"question": "table-1"}, {"question": "table-0"}]
    completed = load_completed_scores(read_report(report), questions, JUDGE)
    assert [row["question"] for row in completed] == ["table-0"]
    assert load_completed_scores(read_report(tmp_path / "missing.json"), questions, JUDGE) == []
    broken = tmp_path / "broken.json"
    broken.write_text("{不是 JSON", encoding="utf-8")
    assert load_completed_scores(read_report(broken), questions, JUDGE) == []


def test_resume_drops_scores_from_a_different_judge():
    """验证换过判分模型后不复用旧评分。

    同一个答案被不同判分模型评出的分数不可比，混进同一份报告的均值没有意义。
    换判分模型之前生成的报告没有 judge 字段，同样不能复用。
    """
    scores = [{"question": "table-0", "faithfulness": 0.8}]
    questions = [{"question": "table-0"}]
    assert load_completed_scores({"judge": JUDGE, "scores": scores}, questions, JUDGE)
    assert load_completed_scores({"judge": JUDGE, "scores": scores}, questions, "别的判分模型") == []
    assert load_completed_scores({"scores": scores}, questions, JUDGE) == []


def test_sample_scores_turns_failed_metrics_into_none():
    """验证失败指标记 None、数值转成内置 float，逐题分数才能直接写成 JSON。"""
    numpy = pytest.importorskip("numpy")
    samples = [{"question": "table-0", "type": "table"}]
    result = SimpleNamespace(
        scores=[
            {
                "faithfulness": numpy.float64(0.81234),
                "answer_relevancy": float("nan"),
                "context_precision": 1.0,
                "context_recall": numpy.float64(0.5),
            }
        ]
    )
    rows = sample_scores(samples, result)
    assert rows == [
        {
            "question": "table-0",
            "type": "table",
            "faithfulness": 0.8123,
            "answer_relevancy": None,
            "context_precision": 1.0,
            "context_recall": 0.5,
        }
    ]
    # 不借助 default=str 也能写盘，说明没有残留 numpy 标量。
    json.dumps(rows, ensure_ascii=False)


def test_average_scores_skips_failed_metrics():
    """验证均值只统计成功的题，整列失败或缺失时该指标记 None，不抛异常。"""
    rows = [
        {"question": "a", "faithfulness": 1.0, "answer_relevancy": None},
        {"question": "b", "faithfulness": 0.0, "answer_relevancy": None},
    ]
    summary = average_scores(rows, workers=4)
    assert summary["n"] == 2
    assert summary["workers"] == 4
    assert summary["metrics"]["faithfulness"] == 0.5
    assert summary["metrics"]["answer_relevancy"] is None
    # 整列缺失（例如旧报告只有部分指标）同样按失败处理。
    assert summary["metrics"]["context_recall"] is None


def test_question_bank_keeps_the_documented_type_mix():
    """验证题库仍是 README 描述的题型构成，抽样测试的假设成立才有效。"""
    bank = json.loads(
        (Path(__file__).resolve().parents[1] / "evaluation" / "questions.json").read_text(
            encoding="utf-8"
        )
    )
    counts = Counter(q["type"] for q in bank)
    assert counts["table"] > 0 and counts["numeric"] > 0
    assert sum(1 for q in bank if not q.get("relevant_chunks")) == counts["negative"]
