"""组装问答模板，同时记录实际进入上下文的证据。"""

from __future__ import annotations

from .config import ConversationSettings, MemorySettings
from .schemas import ContextBundle, SearchHit, valid_query

ELEMENT_TYPE_LABELS = {"table": "表格", "figure": "图表", "text": "正文"}
HISTORY_HEADING = "【历史对话】"
MEMORY_HEADING = "【长期记忆】"
ROLE_LABELS = {"user": "用户", "assistant": "助手"}

# 注入文本里可能冒充提示词分节标题的标记。旧回答若被恶意文档片段影响过，
# 可以在下一轮伪造一个【文档片段】段落；注入前统一换成半角方括号即可失效。
PROMPT_MARKERS = ("【长期记忆】", "【历史对话】", "【文档片段】", "【问题】", "【回答】")
_HALF_WIDTH_BRACKETS = str.maketrans("【】", "[]")


def defuse(text: str) -> str:
    """把注入文本里可能冒充分节标题的标记换成半角方括号，避免伪造段落。

    只在拼提示词时处理，落盘和界面仍然显示原话。
    """
    for marker in PROMPT_MARKERS:
        text = text.replace(marker, marker.translate(_HALF_WIDTH_BRACKETS))
    return text


def memory_section(text, max_chars: int) -> str:
    """把长期记忆整理成提示词段落，超过上限时从开头截取并注明。

    手写文件的阅读顺序是从上到下，截取头部能让文件顶部成为稳定的优先前缀：
    在文件末尾追加内容不会改变模型看到的部分，用户不必数字数就能预知结果。
    没有内容时返回空字符串，模板里因此不会多出空行。
    """
    body = defuse(str(text or "").strip())
    if not body:
        return ""
    if len(body) > max_chars:
        body = f"{body[:max_chars]}\n（长期记忆超过 {max_chars} 字上限，后续内容未注入）"
    return f"{MEMORY_HEADING}\n{body}\n\n"


def source_label(metadata: dict) -> str:
    """按元数据里实际存在的字段拼出人类可读的来源说明。

    提示词与 Web 接口共用这一份实现，界面显示的位置就和模型看到的完全一致。
    这里有意不合并 scripts/export_chunks.py 的 _location_label：导出结果不含文件名，
    连接方式也不同，合并会改动其中一方的输出。
    """
    source = metadata.get("source", "?")
    page = metadata.get("page")
    # TXT 没有页码，PDF 的 0 表示页码未知，都只显示来源文件名。
    if page not in (None, "", 0, "0", "?"):
        source = f"{source} 第{page}页"
    # 图表和表格的题注是它们最重要的身份信息：同一页可能有好几张图，
    # 只给页码无法说明这条片段对应哪一张。
    caption = metadata.get("caption")
    if caption:
        source = f"{source} | {caption}"
    heading_path = metadata.get("heading_path")
    if heading_path:
        source = f"{source} | 标题: {heading_path}"
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    if line_start is not None and line_end is not None:
        # 语义切分后的片段继承原文块范围，因此这里不声称是片段的精确行号。
        source = f"{source} | 所属原文块: 第{line_start}—{line_end}行"
    paragraph_start = metadata.get("paragraph_start")
    paragraph_end = metadata.get("paragraph_end")
    if paragraph_start is not None and paragraph_end is not None:
        source = f"{source} | 所属原文段落: 第{paragraph_start}—{paragraph_end}段"
    table_index = metadata.get("table_index")
    if table_index is not None:
        source = f"{source} | 第{table_index}个表格"
    if metadata.get("sheet_name"):
        source = f"{source} | 工作表: {metadata['sheet_name']}"
        source += f" | 第{metadata['row_start']}—{metadata['row_end']}行"
        source += f" | {metadata['column_start']}—{metadata['column_end']}列"
        if metadata.get("header_row_start") is not None:
            header_start = metadata["header_row_start"]
            header_end = metadata["header_row_end"]
            source += f" | 表头: 第{header_start}—{header_end}行"
    return source


def history_block(turns, max_turns: int, max_chars: int) -> str:
    """把最近若干轮问答整理成提示词里的历史段落，超出限制的部分从最早处丢弃。

    turns 中每项是 (role, text)，role 为 user 或 assistant；没有可用内容时返回空字符串，
    调用方因此不需要为「没有历史」单独准备模板。
    """
    selected = list(turns or [])[-max_turns:] if max_turns > 0 else []
    lines = []
    for role, text in selected:
        label = ROLE_LABELS.get(role, ROLE_LABELS["user"])
        # 历史文本来自用户和模型，与长期记忆同属外部输入，一样要防伪造分节标题。
        lines.append(f"{label}：{defuse(str(text).strip())}")
    while lines and sum(len(line) for line in lines) > max_chars:
        lines.pop(0)
    if not lines:
        return ""
    return f"{HISTORY_HEADING}\n" + "\n".join(lines) + "\n\n"


class ContextBuilder:
    def __init__(
        self,
        template: str,
        conversation: ConversationSettings | None = None,
        memory_settings: MemorySettings | None = None,
    ):
        """保存提示词模板、历史长度限制和记忆注入上限，不加载模型。"""
        self.template = template
        self.conversation = conversation or ConversationSettings()
        self.memory_settings = memory_settings or MemorySettings()

    def build(self, query: str, hits: list[SearchHit], history=None, memory=None) -> ContextBundle:
        """给片段加上来源说明，再与长期记忆、历史对话和问题一起填入提示词模板。

        按传入顺序使用全部 hits，不在这里重新排序或截断；同时返回实际使用的片段。
        history 只用于理解当前问题的指代，memory 是长期记忆全文，两者为空时
        提示词与不带它们时完全一致。
        """
        query = valid_query(query)
        evidence = list(hits)
        blocks = []
        for index, hit in enumerate(evidence, start=1):
            document = hit.document
            label = source_label(document.metadata)
            type_label = ELEMENT_TYPE_LABELS.get(document.metadata.get("type"), "正文")
            block = f"[片段 {index} | {type_label} | 来源: {label}]\n{document.page_content}"
            blocks.append(block)
        history_text = history_block(
            history,
            self.conversation.max_turns,
            self.conversation.max_chars,
        )
        prompt = self.template.format(
            context="\n\n".join(blocks),
            question=query,
            history=history_text,
            memory=memory_section(memory, self.memory_settings.max_chars),
        )
        return ContextBundle(prompt=prompt, evidence=evidence)
