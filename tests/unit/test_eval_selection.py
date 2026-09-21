"""生成评测题目筛选的回归测试；只验证抽样逻辑，不加载模型也不连数据库。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from tests.evaluation.run_rag_eval import load_completed_samples, select_questions

QUESTIONS = [
    {"question": f"{kind}-{index}", "type": kind}
    for kind, count in (("table", 13), ("numeric", 25), ("factual", 18), ("figure", 15))
    for index in range(count)
]


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
    completed = load_completed_samples(report, questions)
    assert [sample["question"] for sample in completed] == ["table-0"]
    # 文件不存在或内容损坏时按没有历史处理，不抛异常。
    assert load_completed_samples(tmp_path / "missing.json", questions) == []
    broken = tmp_path / "broken.json"
    broken.write_text("{不是 JSON", encoding="utf-8")
    assert load_completed_samples(broken, questions) == []


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
