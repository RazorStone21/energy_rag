"""正文语义切分与噪声过滤，不负责数据库或文件写入。"""

from __future__ import annotations

import re

from .config import SplitSettings
from .headings import heading_level
from .schemas import DocumentLike, ParseResult

MEANINGFUL_CHAR_PATTERN = re.compile(r"[一-鿿A-Za-z0-9]")

# 解析器标为这些类型的片段不参与句子切分，整体保留。
WHOLE_UNIT_KINDS = ("code", "list")


def meaningful_len(text: str) -> int:
    """统计中文、英文字母和数字的数量；标点与空白不计入有效字符数。"""
    return len(MEANINGFUL_CHAR_PATTERN.findall(text))


def _with_content(chunk, text):
    """复制片段并替换正文；兼容 LangChain Document 与其他实现。"""
    copy_method = getattr(chunk, "model_copy", None)
    if copy_method is not None:
        return copy_method(update={"page_content": text})
    return type(chunk)(page_content=text, metadata=dict(chunk.metadata))


def _page_key(chunk):
    """片段的来源标识，用于判断两条片段是否真的连续。"""
    metadata = chunk.metadata
    return metadata.get("source"), metadata.get("page")


def _split_trailing_heading(text, sentence_regex):
    """末尾一句若是编号标题就拆出来，返回 (标题, 剩余正文)。

    只剩一句时不动：搬走它会把这一块搬空。标题本身也要求是有效标题，
    实测末尾短句里绝大多数是 PDF 提取留下的碎片，那些不能当标题搬。
    """
    marks = list(re.finditer(sentence_regex, text))
    if len(marks) < 2:
        return "", text
    cut = marks[-2].end()
    tail = text[cut:].strip()
    if heading_level(tail) is None:
        return "", text
    return tail, text[:cut].rstrip()


def restore_heading_boundaries(chunks, sentence_regex, min_chars):
    """把被切在块尾的章节标题交给下一块。

    语义断点可能正好落在「（三）推进构网型技术应用。」和它的正文之间，
    标题孤零零留在上一块末尾、正文在下一块开头，检索时问题里的关键词
    和答案就分到了两条片段上。标题属于它下面的内容，因此挪到下一块开头。

    只搬编号标题：实测末尾的短句（≤20 字）里只有约一成是编号标题，
    其余大多是被排版拆散的数字和页码残留，搬来搬去只会添乱。
    跨文件或跨页不搬——那两处的内容本来就不连续。
    """
    if len(chunks) < 2:
        return list(chunks)
    texts = [chunk.page_content for chunk in chunks]
    # carried[i] 是从第 i 块末尾挪出来、要接到第 i+1 块开头的标题。
    carried = [""] * len(chunks)
    for index in range(len(chunks) - 1):
        if _page_key(chunks[index]) != _page_key(chunks[index + 1]):
            continue
        heading, rest = _split_trailing_heading(texts[index], sentence_regex)
        # 搬走标题后剩下的部分若会被长度门槛过滤掉，就等于把内容搬丢了，宁可不搬。
        if not heading or meaningful_len(rest) < min_chars:
            continue
        carried[index] = heading
        texts[index] = rest
    return [
        _with_content(chunk, (carried[index - 1] if index else "") + texts[index])
        for index, chunk in enumerate(chunks)
    ]


def _load_splitter_factory():
    """尝试从不同模块导入 SemanticChunker；都不可用时保留导入错误。"""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        try:
            from langchain_text_splitters import SemanticChunker
        except ImportError:
            from langchain.text_splitter import SemanticChunker
    return SemanticChunker


class Chunker:
    """负责正文切分和统一过滤，表格及图片描述保持完整。"""

    def __init__(self, embedder, settings: SplitSettings, splitter_factory=None):
        """保存嵌入组件和切分参数；也可传入自定义切分器创建函数，用到时才创建。"""
        self.embedder = embedder
        self.settings = settings
        self._factory = splitter_factory
        self._splitter = None

    def _get_splitter(self):
        """首次使用时创建切分器，后续文件复用同一个实例。"""
        if self._splitter is None:
            factory = self._factory
            if factory is None:
                factory = _load_splitter_factory()
            self._splitter = factory(
                self.embedder,
                breakpoint_threshold_type=self.settings.threshold_type,
                breakpoint_threshold_amount=self.settings.threshold_amount,
                buffer_size=self.settings.buffer_size,
                sentence_split_regex=self.settings.sentence_regex,
            )
        return self._splitter

    def split_texts(self, docs: list[DocumentLike]) -> list[DocumentLike]:
        """按语义断点拆分正文，继承来源和页码；空输入直接返回空列表。

        拆分后修一次标题边界：断点可能正好落在小标题和它的正文之间，
        那样标题会孤零零留在上一块末尾，见 restore_heading_boundaries。
        """
        if not docs:
            return []
        chunks = self._get_splitter().split_documents(docs)
        return restore_heading_boundaries(
            chunks,
            self.settings.sentence_regex,
            self.settings.min_chars,
        )

    def filter(self, chunks: list[DocumentLike]) -> list[DocumentLike]:
        """保留有效字符数不少于 min_chars 的片段；只检查长度，不判断内容是否相关。"""
        return [
            chunk
            for chunk in chunks
            if meaningful_len(chunk.page_content) >= self.settings.min_chars
        ]

    def split(self, parsed: ParseResult) -> list[DocumentLike]:
        """切分普通正文，完整保留代码块、列表、表格和图片描述，再过滤噪声。

        长度门槛只用于判断「这次切分是否切出了没用的残片」，因此只作用在
        按句子切出来的正文上；表格、图片、代码块和列表本就是完整单元，
        再短也自成一条信息，按同一门槛砍掉会直接丢内容。
        """
        prose, whole_units = [], []
        for document in parsed.texts:
            # 代码块和列表都不能按句子切分：代码切开会打断结构，
            # 列表切开会把并列的几条条目分到不同片段里。
            if document.metadata.get("block_kind") in WHOLE_UNIT_KINDS:
                whole_units.append(document)
            else:
                prose.append(document)
        text_chunks = self.filter(self.split_texts(prose))
        return text_chunks + whole_units + parsed.tables + parsed.figures
