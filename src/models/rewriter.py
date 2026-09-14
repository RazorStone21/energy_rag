"""结合历史对话把依赖上下文的追问改写成可以独立检索的问句。

只在有历史对话时触发：没有历史就无从还原指代，单轮提问因此完全不受影响，
既不增加一次生成，检索词也和改动前逐字相同。
"""

from __future__ import annotations

import logging

from ..context_builder import ROLE_LABELS, defuse
from ..schemas import valid_query

logger = logging.getLogger(__name__)

# 改写结果应该只有一句问句；超过这个长度说明模型没有按要求只输出问句。
MAX_REWRITE_CHARS = 200


class QueryRewriter:
    def __init__(self, template, settings, generator=None):
        """保存改写模板、历史上限和用于生成问句的文本模型，不在这里加载模型。

        generator 复用问答流程已经加载的同一个模型实例，不额外占显存。
        """
        self.template = template
        self.settings = settings
        self.generator = generator

    def history_text(self, history):
        """把最近若干轮对话整理成改写提示词里的历史段落，没有可用内容时返回空字符串。

        这里有意不复用 context_builder.history_block：那里会额外加上【历史对话】标题，
        而本模块的模板自带该标题，复用会让标题出现两次。
        """
        if self.settings.max_turns <= 0:
            return ""
        selected = list(history or [])[-self.settings.max_turns :]
        lines = []
        for role, text in selected:
            label = ROLE_LABELS.get(role, ROLE_LABELS["user"])
            lines.append(f"{label}：{defuse(str(text).strip())}")
        # 从最早的一轮开始丢，最近的对话对还原指代最重要。
        while lines and sum(len(line) for line in lines) > self.settings.max_chars:
            lines.pop(0)
        return "\n".join(lines)

    def accept(self, candidate, query):
        """检查改写输出是否可用，不合格时退回原问题。

        模型偶尔会连同提示词的分节标题一起吐出来，或返回一段解释而不是问句；
        这类输出直接拿去检索比用原问题更糟，因此宁可放弃改写。
        """
        text = str(candidate or "").strip().strip("「」『』\"'“”")
        if not text or "\n" in text or len(text) > MAX_REWRITE_CHARS:
            return query
        # 提示词里的分节标题一律用【】括起，正文问句不会出现它；
        # 命中就说明模型把模板里的标题带出来了，不是干净的改写结果。
        if "【" in text or "】" in text:
            return query
        return text

    def rewrite(self, query, history=None):
        """结合历史把追问改写成独立问句；不需要改写或改写失败时原样返回。

        改写是尽力而为的增强：生成失败不该让整个问答失败，退回原问题仍然能作答。
        """
        query = valid_query(query)
        if not self.settings.enabled or self.generator is None:
            return query
        history_block = self.history_text(history)
        if not history_block:
            return query
        prompt = self.template.format(history=history_block, question=defuse(query))
        try:
            candidate = self.generator.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - 改写失败必须降级，不能让问答整体失败
            logger.warning("Query rewrite failed, using original query: %s", exc)
            return query
        return self.accept(candidate, query)
