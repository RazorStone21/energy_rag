"""把 MinerU 的内容项转换成入库用的正文、表格与图表片段。

MinerU 的 `content_list` 是它的对外稳定接口：每一项都带 `type`、`page_idx`（0 起）
与 `bbox`（0-1 归一化），正文项给 `text` 与可选的 `text_level`，表格给 `table_body`
（HTML），图与图表给 `img_path`（内联 data URL）与各自的题注字段。

转换的三条规则：

1. 页码 +1：项目内部统一 1 起，0 在下游被当成"页码未知"而不显示。
2. 表格转成 Markdown 再入库：HTML 标签会被 jieba 切进倒排并污染 IDF，前端也会
   把裸标签当文本显示；转 Markdown 后与 Excel 片段同一形态。
3. 图表交给项目自己的视觉模型描述：MinerU 只负责版面与裁剪，描述口径与既有
   评测保持一致（分批送模型，整批失败退回逐张）。
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser

from ..headings import heading_level, push_heading
from ..schemas import ParseResult
from .mineru_engine import FIGURE_TYPES, SKIPPED_TYPES, decode_image
from .tables import escape_cell

logger = logging.getLogger(__name__)

# 中文排版常在标题的字之间插空白（实测有「专 栏 1 ……」）；匹配前先去掉汉字间的空白。
_CJK_GAP = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")

# 内容项里可能出现的题注字段，按表格/图表/图片各自的命名取值。
_CAPTION_KEYS = ("table_caption", "chart_caption", "image_caption")


def normalize_caption(text) -> str:
    """清理题注：去掉汉字间的排版空白与 Markdown 强调标记。

    汉字与数字之间的空格保留，避免把编号粘起来；MinerU 会把续表的题注写成
    `**续表**`，引用标签里不该出现这些星号。
    """
    cleaned = _CJK_GAP.sub("", str(text or "")).strip()
    return cleaned.strip("*").strip()


def caption_of(item: dict) -> str:
    """取出内容项的题注，列表形式用空格连接；没有题注时返回空串。"""
    for key in _CAPTION_KEYS:
        value = item.get(key)
        if not value:
            continue
        if isinstance(value, (list, tuple)):
            joined = " ".join(str(part) for part in value if part)
        else:
            joined = str(value)
        caption = normalize_caption(joined)
        if caption:
            return caption
    return ""


class _TableCollector(HTMLParser):
    """把 `<table>` 收集成二维文本行。

    合并单元格按占位补齐：`colspan` 在右侧补空单元格，`rowspan` 在后续行里占住
    同一列。旧实现把合并单元格整体拍平，列会向左塌陷、不同行的单元格错位；
    占位补齐后每行的列数一致，读取时至少不会串列。
    """

    def __init__(self):
        """初始化行列缓冲。"""
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._parts: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1
        # 生效中的跨行占位：列号 -> 还需要在后续多少行里补一个空单元格。
        self._spans: dict[int, int] = {}

    def _fill_spans(self):
        """在当前行补齐恰好落在插入位置上的跨行占位。

        只在占位列正好等于当前行长度时消耗：开始时若已越过该列，说明本行对应位置
        已经放过真实单元格（跨行登记属于"后续行"，不能在同一行里被消耗）。
        """
        while True:
            pending = [column for column, left in self._spans.items() if left > 0]
            if not pending or min(pending) != len(self._row):
                return
            column = min(pending)
            self._spans[column] -= 1
            if self._spans[column] <= 0:
                del self._spans[column]
            self._row.append("")

    def handle_starttag(self, tag, attrs):
        """识别行与单元格的开始标签，并记下跨行跨列属性。"""
        if tag == "tr":
            self._row = []
            return
        if tag in ("td", "th"):
            self._parts = []
            settings = dict(attrs)
            self._colspan = _positive_int(settings.get("colspan"))
            self._rowspan = _positive_int(settings.get("rowspan"))
        elif tag in ("br", "p") and self._parts is not None:
            # 单元格内的换行与段落按空格处理，稍后统一压空白。
            self._parts.append(" ")

    def handle_endtag(self, tag):
        """单元格结束时按 colspan 落位，并按 rowspan 登记后续行的占位。"""
        if tag not in ("td", "th") or self._row is None or self._parts is None:
            if tag == "tr" and self._row is not None:
                self.rows.append(self._row)
                self._row = None
            return
        text = re.sub(r"\s+", " ", "".join(self._parts)).strip()
        self._fill_spans()
        column = len(self._row)
        self._row.append(text)
        for _ in range(self._colspan - 1):
            self._row.append("")
        if self._rowspan > 1:
            # 从下一行起，在同一个列号上补空单元格。
            self._spans[column] = self._rowspan - 1
        self._parts = None

    def handle_data(self, data):
        """累积单元格内的文本。"""
        if self._parts is not None:
            self._parts.append(data)


def _positive_int(value) -> int:
    """把 HTML 属性解析成正整数；缺失或非法时按 1 处理。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return parsed if parsed > 0 else 1


def parse_table_html(html: str) -> list[list[str]]:
    """解析 MinerU 的 HTML 表格，返回补齐列宽后的二维文本。"""
    collector = _TableCollector()
    collector.feed(str(html or ""))
    collector.close()
    return collector.rows


def html_table_to_markdown(
    html: str, caption: str, *, rows_per_chunk: int, min_rows: int, min_columns: int
) -> list[tuple[str, int, int]]:
    """把 HTML 表格转成 Markdown 分片，返回 [(正文, 起始行, 结束行)]。

    少于 min_rows 行或 min_columns 列的"表格"是被误判的正文（通知抬头、章节标题），
    直接丢弃——与旧实现 `_table_to_markdown` 的判据保持一致。
    行号是表内数据行的序号（1 起，不含表头），既让引用能定位到具体几行，
    也让去重身份能区分同一张表的多个切片。
    """
    rows = [row for row in parse_table_html(html) if any(cell.strip() for cell in row)]
    if len(rows) < min_rows:
        return []
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    used = [index for index in range(width) if any(row[index].strip() for row in rows)]
    if len(used) < min_columns:
        return []
    trimmed = [[row[index] for index in used] for row in rows]

    header, data_rows = trimmed[0], trimmed[1:]
    if not data_rows:
        # 只有表头没有数据行，说明是被误判的短块，不是表格。
        return []
    header_line = "| " + " | ".join(escape_cell(cell) for cell in header) + " |"
    separator = "| " + " | ".join(["---"] * len(header)) + " |"

    chunks = []
    for start in range(0, len(data_rows), rows_per_chunk):
        window = data_rows[start : start + rows_per_chunk]
        row_start, row_end = start + 1, start + len(window)
        label = f"表格：{caption}（第 {row_start}—{row_end} 行）" if caption else (
            f"表格（第 {row_start}—{row_end} 行）"
        )
        lines = [label, header_line, separator]
        lines.extend(
            "| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in window
        )
        chunks.append(("\n".join(lines), row_start, row_end))
    return chunks


def describe_figures(vision, items, batch_size, on_status=None) -> list[str]:
    """分批描述图片，返回与 items 等长的描述列表。

    items 每项是 (image, prompt)。整批失败时退回逐张重试：一张图出问题
    （例如显存不足）不该让同一批里其他图的描述一起丢掉。
    """
    descriptions = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        if on_status is not None:
            on_status(f"描述第 {start + 1}—{start + len(chunk)} 张图表")
        try:
            descriptions.extend(
                vision.describe_batch(
                    [image for image, _ in chunk],
                    [prompt for _, prompt in chunk],
                )
            )
        except Exception as exc:  # noqa: BLE001 - 整批失败要降级，不能放弃这一批图片
            logger.warning("批量描述失败，退回逐张重试：%s", exc)
            for image, prompt in chunk:
                descriptions.append(vision.describe(image, prompt))
    return descriptions


def _document_factory(factory):
    """取得文档构造器，默认用 LangChain 的 Document。"""
    if factory is not None:
        return factory
    from langchain_core.documents import Document

    return Document


def _base_metadata(item: dict, source: str, index_on_page: int, heading: str) -> dict:
    """拼出正文类片段共用的元数据：来源、页码、页内序号、位置与所属章节。"""
    metadata = {
        "source": source,
        "page": int(item.get("page_idx") or 0) + 1,
        "type": "text",
        "block_kind": "section",
        "block_index": index_on_page,
        "parser": "mineru",
    }
    bbox = item.get("bbox")
    if bbox:
        metadata["bbox"] = [round(float(value), 4) for value in bbox]
    if heading:
        metadata["heading_path"] = heading
    return metadata


def blocks_to_parse_result(
    items,
    source: str,
    vision,
    vision_settings,
    mineru_settings,
    document_factory=None,
    on_status=None,
) -> ParseResult:
    """把内容项分成正文、表格与图表三类，图表交给视觉模型描述。"""
    factory = _document_factory(document_factory)
    result = ParseResult()
    stack: list[str] = []
    page_indexes: dict[int, int] = {}
    pending: list[tuple] = []

    for item in items:
        kind = str(item.get("type") or "")
        if kind in SKIPPED_TYPES:
            continue
        page_no = int(item.get("page_idx") or 0) + 1
        page_indexes[page_no] = page_indexes.get(page_no, 0) + 1
        index_on_page = page_indexes[page_no]

        if kind == "table":
            caption = caption_of(item)
            chunks = html_table_to_markdown(
                item.get("table_body") or "",
                caption,
                rows_per_chunk=mineru_settings.rows_per_chunk,
                min_rows=mineru_settings.min_table_rows,
                min_columns=mineru_settings.min_table_columns,
            )
            for position, (markdown, row_start, row_end) in enumerate(chunks, start=1):
                metadata = _base_metadata(item, source, index_on_page, _heading_of(stack))
                metadata.update(
                    type="table",
                    block_kind="table",
                    table_index=position,
                    row_start=row_start,
                    row_end=row_end,
                    caption=caption or None,
                )
                result.tables.append(factory(page_content=markdown, metadata=metadata))
            continue

        if kind in FIGURE_TYPES:
            image = decode_image(item.get("img_path"))
            if image is None:
                continue
            bbox = item.get("bbox") or [0, 0, 1, 1]
            width_ratio = abs(float(bbox[2]) - float(bbox[0]))
            height_ratio = abs(float(bbox[3]) - float(bbox[1]))
            if (
                width_ratio < mineru_settings.min_figure_width_ratio
                or height_ratio < mineru_settings.min_figure_height_ratio
            ):
                # 页码旁的装饰性小图标、版式分隔线：描述它们只会挤占检索候选位。
                continue
            caption = caption_of(item)
            prompt = _figure_prompt(vision_settings.prompt, caption)
            pending.append((item, image, caption, prompt, index_on_page))
            continue

        text = str(item.get("text") or "").strip()
        if not text:
            continue
        # 标题识别用项目自己的编号规则，不采信 MinerU 的 text_level：实测它会把加粗的
        # 长句也标成 paragraph_title，导致整句正文进入 heading_path 与引用标签。
        level = heading_level(text)
        if level:
            push_heading(stack, text, level)
        metadata = _base_metadata(item, source, index_on_page, _heading_of(stack))
        if kind in ("code", "list"):
            metadata["block_kind"] = kind
        result.texts.append(factory(page_content=text, metadata=metadata))

    if pending and vision is not None and getattr(vision, "available", False):
        descriptions = describe_figures(
            vision,
            [(image, prompt) for _, image, _, prompt, _ in pending],
            vision_settings.batch_size,
            on_status=on_status,
        )
        for (item, _image, caption, _prompt, index_on_page), description in zip(
            pending, descriptions
        ):
            content = f"{caption}\n{description}".strip() if caption else description.strip()
            metadata = _base_metadata(item, source, index_on_page, _heading_of(stack))
            metadata.update(type="figure", block_kind="figure", caption=caption or None)
            result.figures.append(factory(page_content=content, metadata=metadata))
    return result


def _heading_of(stack: list[str]) -> str:
    """把章节栈拼成 heading_path；空槽不参与拼接。"""
    return " > ".join(part for part in stack if part)


def _figure_prompt(prompt: str, caption: str) -> str:
    """题注存在时把它放在描述要求前面，让模型知道这张图的标题。"""
    if not caption:
        return prompt
    return f"这张图的标题是「{caption}」。\n{prompt}"
