"""生成 Markdown 表格时共用的单元格转义。

Excel 解析与 PDF（MinerU）解析都要把二维表写成 Markdown 管道表格，两处必须用
同一套转义规则，否则同一份内容在不同来源里会出现不同的转义口径。
"""

from __future__ import annotations

from html import escape


def escape_cell(value) -> str:
    """转义竖线、反斜线和换行，防止单元格内容改变 Markdown 表格结构。"""
    return (
        escape(str(value))
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\n", "<br>")
    )
