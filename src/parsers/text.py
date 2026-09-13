"""读取 UTF-8 编码的 TXT 正文，保留段落，具体切分交给 Chunker。"""

from __future__ import annotations

from pathlib import Path

from ..schemas import ParseResult


def read_utf8_text(path: Path) -> str:
    """读取 UTF-8 正文，兼容 BOM；编码错误、空文件和空字符都明确报错。"""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"{path.name} 不是有效的 UTF-8 文本，请转换编码后重试") from exc
    if not content.strip():
        raise ValueError(f"{path.name} 没有可用正文")
    if "\x00" in content:
        raise ValueError(f"{path.name} 含空字符，请确认文件是 UTF-8 纯文本")
    return content


class TextParser:
    """把一份 TXT 转为正文文档，不生成表格、图片描述或虚构页码。"""

    def __init__(self, document_factory=None):
        """允许替换文档创建函数；默认在实际解析时导入 LangChain Document。"""
        self.document_factory = document_factory

    def parse(self, path: str | Path, on_status=None) -> ParseResult:
        """读取正文并记录文件名；读取失败、编码错误和空文件都作为解析失败返回。

        on_status 是解析器协议要求的进度回调；纯文本没有需要分步上报的环节，
        这里接受但不使用，这样注册表可以用同一种方式调用所有解析器。
        """
        path = Path(path)
        try:
            content = read_utf8_text(path)
        except (ValueError, OSError) as exc:
            return ParseResult(errors=[f"text: {exc}"])

        factory = self.document_factory
        if factory is None:
            from langchain_core.documents import Document

            factory = Document
        # 保留原始段落和缩进；整份正文没有可靠页码，后续片段仅继承文件名。
        document = factory(
            page_content=content,
            metadata={"source": path.name, "type": "text"},
        )
        return ParseResult(texts=[document])
