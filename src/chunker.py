"""正文语义切分与噪声过滤，不负责数据库或文件写入。"""

from __future__ import annotations

import re

from .config import SplitSettings
from .schemas import DocumentLike, ParseResult

MEANINGFUL_CHAR_PATTERN = re.compile(r"[一-鿿A-Za-z0-9]")


def meaningful_len(text: str) -> int:
    """统计中文、英文字母和数字的数量；标点与空白不计入有效字符数。"""
    return len(MEANINGFUL_CHAR_PATTERN.findall(text))


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
        """按语义断点拆分正文，继承来源和页码；空输入直接返回空列表。"""
        if not docs:
            return []
        return self._get_splitter().split_documents(docs)

    def filter(self, chunks: list[DocumentLike]) -> list[DocumentLike]:
        """保留有效字符数不少于 min_chars 的片段；只检查长度，不判断内容是否相关。"""
        return [
            chunk
            for chunk in chunks
            if meaningful_len(chunk.page_content) >= self.settings.min_chars
        ]

    def split(self, parsed: ParseResult) -> list[DocumentLike]:
        """切分普通正文，完整保留独立代码块、表格和图片描述，再过滤噪声。

        长度门槛只用于判断「这次切分是否切出了没用的残片」，因此只作用在
        按句子切出来的正文上；表格、图片和代码块本就是完整单元，
        再短也自成一条信息，按同一门槛砍掉会直接丢内容。
        """
        prose, code_blocks = [], []
        for document in parsed.texts:
            # Markdown 已单独提取的代码块不能再按句子切分，否则会打断代码结构。
            if document.metadata.get("block_kind") == "code":
                code_blocks.append(document)
            else:
                prose.append(document)
        text_chunks = self.filter(self.split_texts(prose))
        return text_chunks + code_blocks + parsed.tables + parsed.figures
