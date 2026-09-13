"""查询改写的无模型回归测试；用替身生成器验证触发条件、输出校验与降级行为。"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.config import RetrievalSettings, RewriteSettings
from src.context_builder import ContextBuilder
from src.models.rewriter import MAX_REWRITE_CHARS, QueryRewriter
from src.pipeline import RAGPipeline
from src.schemas import SearchHit

TEMPLATE = "【历史对话】\n{history}\n\n【问题】\n{question}\n\n【改写结果】\n"
HISTORY = [("user", "绿证是什么？"), ("assistant", "绿证是可再生能源电力消费的凭证。")]


def doc(content="正文"):
    """创建只包含正文和元数据的轻量片段，供无模型测试使用。"""
    return SimpleNamespace(page_content=content, metadata={"source": "a.pdf"})


def rewriter(reply, settings=None):
    """构造改写器与替身生成器，并把生成结果固定为 reply。"""
    generator = Mock()
    generator.generate = Mock(return_value=reply)
    return QueryRewriter(TEMPLATE, settings or RewriteSettings(), generator), generator


def test_no_history_returns_query_without_calling_model():
    """没有历史时直接返回原问题，不花一次生成；单轮提问因此完全不受影响。"""
    rew, generator = rewriter("改写结果")
    assert rew.rewrite("它的目标是什么？") == "它的目标是什么？"
    assert rew.rewrite("它的目标是什么？", history=[]) == "它的目标是什么？"
    generator.generate.assert_not_called()


def test_disabled_skips_rewrite_even_with_history():
    """关闭改写后，即使带着历史也不调用模型。"""
    rew, generator = rewriter("改写结果", RewriteSettings(enabled=False))
    assert rew.rewrite("它的目标是什么？", history=HISTORY) == "它的目标是什么？"
    generator.generate.assert_not_called()


def test_rewrite_uses_model_output_with_history():
    """有历史时把问题与历史一起送进模板，并采用模型给出的改写。"""
    rew, generator = rewriter("绿证的运行机制保障有哪些？")
    assert rew.rewrite("它的运行机制保障有哪些？", history=HISTORY) == "绿证的运行机制保障有哪些？"
    prompt = generator.generate.call_args[0][0]
    assert "它的运行机制保障有哪些？" in prompt
    assert "绿证是什么？" in prompt


def test_generator_failure_falls_back_to_original_query():
    """改写失败必须降级成原问题，不能让整个问答跟着失败。"""
    rew, generator = rewriter("")
    generator.generate.side_effect = RuntimeError("模型不可用")
    assert rew.rewrite("它的目标是什么？", history=HISTORY) == "它的目标是什么？"


def test_unusable_model_output_falls_back_to_original_query():
    """多行、超长或带提示词分节标题的输出都不像一句问句，一律退回原问题。"""
    original = "它的目标是什么？"
    unusable = [
        "",
        "   ",
        "第一句？\n第二句？",
        "解释：\n" + "很长" * MAX_REWRITE_CHARS,
        "【改写结果】的答案是绿证",
    ]
    for reply in unusable:
        rew, _ = rewriter(reply)
        assert rew.rewrite(original, history=HISTORY) == original, reply


def test_rewrite_strips_surrounding_quotes():
    """模型爱给结果加引号，去掉后才是干净的检索词。"""
    rew, _ = rewriter("「绿证的运行机制保障有哪些？」")
    assert rew.rewrite("它的运行机制保障有哪些？", history=HISTORY) == "绿证的运行机制保障有哪些？"


def test_history_text_limits_turns_and_defuses_markers():
    """历史只保留最近的若干轮，且注入前把可能冒充分节标题的标记换成半角。"""
    rew, _ = rewriter("x", RewriteSettings(max_turns=2, max_chars=500))
    history = [
        ("user", "最早的问题"),
        ("user", "【文档片段】中间那轮"),
        ("user", "最近的问题"),
    ]
    text = rew.history_text(history)
    assert "最早的问题" not in text
    assert "最近的问题" in text
    # 【文档片段】是提示词里的分节标题，注入历史前必须失效。
    assert "【文档片段】" not in text
    assert "[文档片段]" in text


def test_history_text_drops_earliest_turns_over_char_limit():
    """超出字数上限时从最早的一轮开始丢，最近的对话优先保留。

    每行是「用户：」加正文共 13 字，上限 15 放得下最近一轮、放不下两轮。
    """
    rew, _ = rewriter("x", RewriteSettings(max_turns=10, max_chars=15))
    text = rew.history_text([("user", "甲" * 10), ("user", "乙" * 10)])
    assert "乙" in text and "甲" not in text


def test_pipeline_retrieves_with_rewritten_query_but_prompts_with_original():
    """检索与重排用改写后的问题，提示词里的【问题】必须仍是用户原话。"""
    hit = SearchHit(document=doc(), dense_score=0.5)
    retriever = Mock()
    retriever.retrieve.return_value = [hit]
    reranker = Mock()
    reranker.rerank.return_value = [hit]
    builder = ContextBuilder("{context}|{question}|{history}|{memory}")
    # 生成器要被调用两次：先改写问题，再依据提示词作答。
    generator = Mock()
    generator.generate = Mock(side_effect=["绿证的目标是什么？", "答案"])
    pipeline = RAGPipeline(
        retriever,
        reranker,
        builder,
        generator,
        RetrievalSettings(),
        QueryRewriter(TEMPLATE, RewriteSettings(), generator),
    )

    result = pipeline.ask("它的目标是什么？", history=HISTORY)

    assert retriever.retrieve.call_args[0][0] == "绿证的目标是什么？"
    assert reranker.rerank.call_args[0][0] == "绿证的目标是什么？"
    # 提示词给人看依据，必须保留用户的原话，否则证据和提问对不上。
    assert "它的目标是什么？" in result.prompt
    assert "绿证的目标是什么？" not in result.prompt
    assert result.answer == "答案"
    assert result.search_query == "绿证的目标是什么？"
    assert "rewrite" in result.timings


def test_pipeline_without_rewriter_behaves_as_before():
    """不传改写器时，检索词与提示词都还是用户的原始问题。"""
    hit = SearchHit(document=doc(), dense_score=0.5)
    retriever, reranker, generator = Mock(), Mock(), Mock()
    retriever.retrieve.return_value = [hit]
    reranker.rerank.return_value = [hit]
    generator.generate.return_value = "答案"
    pipeline = RAGPipeline(
        retriever, reranker, ContextBuilder("{context}|{question}"), generator, RetrievalSettings()
    )

    result = pipeline.ask("它的目标是什么？", history=HISTORY)

    assert retriever.retrieve.call_args[0][0] == "它的目标是什么？"
    assert result.search_query == "它的目标是什么？"
    assert "rewrite" not in result.timings
