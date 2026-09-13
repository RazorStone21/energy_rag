"""根据文件后缀选择解析器，让入库流程通过同一个 parse 方法读取不同格式。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..interfaces import Parser
from ..schemas import ParseResult


class ParserRegistry:
    """保存已实现的解析器，同时提供文件发现时使用的后缀集合。"""

    def __init__(self, parsers: Mapping[str, Parser]):
        """复制后缀与解析器的对应关系，拒绝空配置和重复的大小写变体。"""
        if not parsers:
            raise ValueError("至少需要注册一种文件格式")
        self._parsers = {}
        for suffix, parser in parsers.items():
            if not suffix.startswith(".") or len(suffix) == 1:
                raise ValueError(f"文件后缀必须以点开头，例如 .pdf：{suffix!r}")
            suffix = suffix.lower()
            if suffix in self._parsers:
                raise ValueError(f"文件后缀重复注册：{suffix}")
            self._parsers[suffix] = parser

    @property
    def supported_suffixes(self) -> frozenset[str]:
        """返回已注册的文件后缀，供入库流程筛选目录中的文件。"""
        return frozenset(self._parsers)

    def parse(self, path: str | Path, on_status=None) -> ParseResult:
        """按后缀调用对应解析器；未支持的格式明确报错，不返回空结果。

        on_status 原样转发给解析器，用于显示解析进度；不传时按原来的方式调用，
        这样只实现了 parse(path) 的自定义解析器仍然可以直接注册。
        """
        path = Path(path)
        parser = self._parsers.get(path.suffix.lower())
        if parser is None:
            supported = ", ".join(sorted(self.supported_suffixes))
            raise ValueError(f"暂不支持文件 {path.name}，当前支持：{supported}")
        if on_status is None:
            return parser.parse(path)
        return parser.parse(path, on_status=on_status)
