"""按标题组织 Markdown 正文，单独保留顶层表格、列表和代码块，记录原文位置。

使用语法解析器识别块边界，再读取对应原文，避免改写列表、链接和代码缩进。
顶层列表单独成块：列表项是并列的一组条目，按句子切分会把其中几条和其余条拆开，
检索到的片段只覆盖一部分条目。列表或引用内部的表格与代码保留在所属正文中；
图片链接不触发图片读取。
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from ..schemas import ParseResult
from .text import read_utf8_text

# 会被单独切成一个块的顶层元素。列表也在这里：列表项是一组并列条目，
# 交给按句子切分的分块器会把其中几条和其余条拆开。
BLOCK_TOKENS = (
    "heading_open",
    "table_open",
    "fence",
    "code_block",
    "bullet_list_open",
    "ordered_list_open",
)
BLOCK_KINDS = {
    "table_open": "table",
    "fence": "code",
    "code_block": "code",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
}
LIST_TOKENS = ("bullet_list_open", "ordered_list_open")


def _list_intro_start(lines, block_start, limit):
    """返回列表块应当从哪一行开始，把紧邻的引出语并进来。

    列表前面通常有一句引出语（「……遵循以下原则：」）。它属于这个列表，
    但单独成块往往太短，会被切分器的长度门槛当成残片过滤掉，
    整句引出语就此消失。这里只吃掉紧邻的那一段，遇到空行或标题就停。
    """
    index = block_start
    # 先跳过列表与引出语之间的空行。
    while index > limit and not lines[index - 1].strip():
        index -= 1
    end = index
    # 再往前吃掉紧邻的那一段非空行。
    while index > limit and lines[index - 1].strip():
        index -= 1
    if index == end:
        return block_start
    # 不要把标题并进来：标题自带一个块，并进列表会让它从章节结构里消失。
    if lines[index].lstrip().startswith("#"):
        return block_start
    return index


class MarkdownParser:
    """读取 UTF-8 Markdown，返回带标题路径与原文行号的文档元素。"""

    def __init__(self, document_factory=None):
        """保存文档创建函数，Markdown 依赖在首次解析时加载。"""
        self.document_factory = document_factory
        self._parser = None

    def _load_parser(self):
        """创建并复用 Markdown 语法解析器，启用竖线表格规则。"""
        if self._parser is None:
            from markdown_it import MarkdownIt

            self._parser = MarkdownIt("commonmark").enable("table")
        return self._parser

    def _make_document(self, lines, start, end, path, headings, block_kind):
        """取出非空原文块，记录所属标题与行号；不修改代码缩进和 Markdown 标记。"""
        # 去掉块两端的空行，但保留非空行的全部缩进和末尾空格。
        while start < end and not lines[start].strip():
            start += 1
        while end > start and not lines[end - 1].strip():
            end -= 1
        if start == end:
            return None
        factory = self.document_factory
        if factory is None:
            from langchain_core.documents import Document

            factory = Document
        return factory(
            page_content="".join(lines[start:end]),
            metadata={
                "source": path.name,
                "type": "table" if block_kind == "table" else "text",
                "heading_path": " > ".join(title for _, title in headings),
                "line_start": start + 1,
                "line_end": end,
                "block_kind": block_kind,
            },
        )

    def _parse_content(self, content: str, path: Path) -> ParseResult:
        """根据顶层标题、表格、列表和代码块划分原文，每个块只放入结果一次。"""
        tokens = self._load_parser().parse(content)
        lines = StringIO(content).readlines()
        result = ParseResult()
        headings = []
        start = 0
        for index, token in enumerate(tokens):
            if token.level != 0 or token.map is None:
                continue
            if token.type not in BLOCK_TOKENS:
                continue
            block_start, block_end = token.map
            if token.type in LIST_TOKENS:
                # 列表前紧邻的引出语属于这个列表，见 _list_intro_start。
                block_start = _list_intro_start(lines, block_start, start)
            document = self._make_document(lines, start, block_start, path, headings, "section")
            if document is not None:
                result.texts.append(document)

            if token.type == "heading_open":
                level = int(token.tag[1:])
                # 同级或更高层标题出现时，结束原来的子章节；代码中的 # 不会被识别为标题。
                while headings and headings[-1][0] >= level:
                    headings.pop()
                title = tokens[index + 1].content.strip()
                headings.append((level, title))
                start = block_start
                continue

            block_kind = BLOCK_KINDS[token.type]
            document = self._make_document(
                lines, block_start, block_end, path, headings, block_kind
            )
            if document is not None:
                if block_kind == "table":
                    result.tables.append(document)
                else:
                    # 列表和代码块都作为完整单元放进正文，由切分器整体保留。
                    result.texts.append(document)
            start = block_end

        document = self._make_document(lines, start, len(lines), path, headings, "section")
        if document is not None:
            result.texts.append(document)
        return result

    def parse(self, path: str | Path, on_status=None) -> ParseResult:
        """读取 Markdown 并提取元素；文件、依赖或解析错误会阻止该文件替换旧索引。

        on_status 是解析器协议要求的进度回调；Markdown 没有需要分步上报的环节，
        这里接受但不使用，这样注册表可以用同一种方式调用所有解析器。
        """
        path = Path(path)
        try:
            return self._parse_content(read_utf8_text(path), path)
        except Exception as exc:
            return ParseResult(errors=[f"markdown: {exc}"])
