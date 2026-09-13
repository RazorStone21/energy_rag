"""按标题组织 Markdown 正文，单独保留顶层表格和代码块，记录原文位置。

使用语法解析器识别块边界，再读取对应原文，避免改写列表、链接和代码缩进。
列表或引用内部的表格与代码保留在所属正文中；图片链接不触发图片读取。
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from ..schemas import ParseResult
from .text import read_utf8_text


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
        """根据顶层标题、表格和代码块划分原文，每个块只放入结果一次。"""
        tokens = self._load_parser().parse(content)
        lines = StringIO(content).readlines()
        result = ParseResult()
        headings = []
        start = 0
        for index, token in enumerate(tokens):
            if token.level != 0 or token.map is None:
                continue
            if token.type not in ("heading_open", "table_open", "fence", "code_block"):
                continue
            block_start, block_end = token.map
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

            block_kind = "table" if token.type == "table_open" else "code"
            document = self._make_document(
                lines, block_start, block_end, path, headings, block_kind
            )
            if document is not None:
                if block_kind == "table":
                    result.tables.append(document)
                else:
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
