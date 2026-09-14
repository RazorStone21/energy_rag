"""识别中文文档里的章节标题编号。

PDF 解析时用它跟踪每个片段属于哪一节，正文切分时用它避免把标题和它的正文
切到两条片段里——两处必须是同一套规则，否则「什么算标题」会出现两种口径，
因此集中在这里。
"""

from __future__ import annotations

import re

# 中文报告的章节标题有固定的编号形式，正文极少这样开头；配合长度上限即可区分。
HEADING_PATTERNS = (
    (1, re.compile(r"^第[一二三四五六七八九十百]+[章节篇]")),
    (1, re.compile(r"^[一二三四五六七八九十]+、")),
    (2, re.compile(r"^（[一二三四五六七八九十]+）")),
    (3, re.compile(r"^\d+(?:\.\d+)*[、．]\s*\S")),
    (3, re.compile(r"^\d+(?:\.\d+)*\.\s+\S")),
)
# 超过这个长度的「标题」实际上是被误套了标题样式或误判的正文。
MAX_HEADING_CHARS = 40


def heading_level(text: str):
    """文本是章节标题时返回它的编号层级，否则返回 None。

    长度上限用来排除以编号开头、实际是正文的句子。
    """
    text = (text or "").strip()
    if not text or len(text) > MAX_HEADING_CHARS:
        return None
    for level, pattern in HEADING_PATTERNS:
        if pattern.match(text):
            return level
    return None
