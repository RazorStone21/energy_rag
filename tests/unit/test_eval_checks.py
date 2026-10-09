"""生成评测里两条不依赖模型的常驻指标与索引指纹的回归测试；不加载模型、不连数据库。"""

from __future__ import annotations

from types import SimpleNamespace

from tests.evaluation.run_rag_eval import (
    index_fingerprint,
    load_completed_samples,
    numeric_support,
    refusal_stats,
)
from tests.evaluation.run_retrieval_eval import annotation_alignment


def fake_chunk(content, page=1, block=1):
    """造一条片段替身，只需 page_content 与 metadata。"""
    return SimpleNamespace(
        page_content=content,
        metadata={"source": "报告.pdf", "page": page, "block_index": block},
    )


def fake_runtime(chunks):
    """造一个只提供 chunk_store.load_chunks() 的运行环境替身。"""
    return SimpleNamespace(
        chunk_store=SimpleNamespace(load_chunks=lambda: list(chunks)),
    )


def test_index_fingerprint_tracks_content_and_position():
    """同一份片段集合指纹稳定；正文、页码或块序号变化都要反映出来。"""
    chunks = [fake_chunk("第一段", page=1), fake_chunk("第二段", page=2)]
    base = index_fingerprint(fake_runtime(chunks))
    assert base == index_fingerprint(fake_runtime(chunks))
    assert base["chunks"] == 2 and len(base["fingerprint"]) == 16

    # 正文变了（换了切分方式或解析器）
    changed_text = index_fingerprint(fake_runtime([fake_chunk("第一段改", page=1), chunks[1]]))
    assert changed_text["fingerprint"] != base["fingerprint"]
    # 位置变了（同一段文字落到不同页码）
    changed_page = index_fingerprint(fake_runtime([fake_chunk("第一段", page=9), chunks[1]]))
    assert changed_page["fingerprint"] != base["fingerprint"]


def test_samples_are_reused_only_for_the_same_index():
    """续跑复用答案要求索引一致：换了索引（重解析、重切分）后旧答案不再可用。"""
    questions = [{"question": "问题一", "type": "factual"}]
    report = {
        "index": {"fingerprint": "aaaa"},
        "samples": [{"question": "问题一", "answer": "旧答案"}],
    }
    assert load_completed_samples(report, questions, "aaaa")
    assert load_completed_samples(report, questions, "bbbb") == []
    # 不传指纹时保持旧行为（测试与人工调用默认可复用）。
    assert load_completed_samples(report, questions)


def test_refusal_stats_counts_only_negative_questions():
    """拒答率只统计 negative 题，其余题型拒答不算数。"""
    samples = [
        {"type": "negative", "answer": "根据提供的文档无法回答该问题"},
        {"type": "negative", "answer": "2025 年底前基本实现全覆盖"},
        {"type": "factual", "answer": "根据提供的文档无法回答该问题"},
    ]
    stats = refusal_stats(samples)
    assert stats["n"] == 2 and stats["refused"] == 1 and stats["rate"] == 0.5
    assert refusal_stats([{"type": "factual", "answer": "答案"}])["rate"] is None


def test_numeric_support_flags_numbers_without_evidence():
    """答案里的数字在片段里找不到依据时计入无据数字，并保留可复核的清单。"""
    samples = [
        {
            "question": "装机多少？",
            "answer": "累计装机 2.8 亿千瓦，同比增长 84.3%，新增 13593 万千瓦。",
            "contexts": ["全球新型储能累计装机规模达 2.8 亿千瓦。", "同比增长 84.3%。"],
        },
        {"question": "没有数字的题", "answer": "文档未提及。", "contexts": ["正文"]},
    ]
    stats = numeric_support(samples)
    # 只有第一条计入：2.8 与 84.3 有依据，13593 无依据。
    assert stats["n"] == 1 and stats["numbers"] == 3 and stats["unsupported"] == 1
    assert stats["unsupported_rate"] == round(1 / 3, 4)
    assert stats["questions_with_unsupported"][0]["numbers"] == ["13593"]


def test_numeric_support_ignores_chinese_numerals():
    """中文数字不在覆盖范围内，这是已知下界：报告里要注明它不是真值。"""
    samples = [{"question": "几倍？", "answer": "增长超四十倍。", "contexts": ["增长超 40 倍"]}]
    stats = numeric_support(samples)
    assert stats["n"] == 0 and stats["unsupported_rate"] is None


def test_annotation_alignment_counts_substring_hits():
    """对齐诊断只看标注能否在索引里找到：它是报告里 Recall 可比性的下界。"""
    chunks = [fake_chunk("将绿证合作列为政府交流重点议题，加快绿证国际互认进程。")]
    questions = [
        {"question": "有对齐的题", "relevant_chunks": ["加快绿证国际互认进程"]},
        {"question": "没对齐的题", "relevant_chunks": ["这段文字来自上一次切分的索引"]},
    ]
    report = annotation_alignment(questions, chunks)
    assert report["annotations"] == 2 and report["aligned"] == 1 and report["rate"] == 0.5
    assert report["not_aligned"][0]["question"] == "没对齐的题"
    # 忽略空格与换行的差异与命中判定同口径。
    spaced = [fake_chunk("加快 绿证 国际互认 进程")]
    assert annotation_alignment([questions[0]], spaced)["aligned"] == 1
