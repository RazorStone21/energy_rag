"""从 PDF 提取表格与图表：表格转 Markdown，图表渲染成图片后交给视觉模型描述。

表格和图片分别读取 PDF，返回 LangChain Document 列表，type 为 table 或 figure。
片段正文保存 Markdown 表格或图片描述，不保存图片向量。

图表有两条来源：
- extract_figures 处理 PDF 里的内嵌位图；
- extract_vector_charts 处理用矢量线条绘制、且没有内嵌位图的图表，
  这类图的数据标签往往没有可用的字符映射（提取出来是乱码），
  只能把区域渲染成图片让视觉模型直接看。

第三方库在对应提取函数中导入；提取失败由 PDFParser 记录并交给入库流程判断。
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

# 屏蔽 pdfminer 解析字体描述符的噪声警告（pdfplumber 表格抽取会触发大量此类日志）
logging.getLogger("pdfminer").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# 题注以「图1」「表 2」这类编号开头；用它把题注和正文里的普通「图」字区分开。
_CAPTION_START = re.compile(r"^\s*(图|表|专栏|附表|附件)\s*\d+")
# 中文排版常在标题的字之间插空白（实测有「专 栏 1 ……」），匹配前先去掉汉字间的空白。
_CJK_GAP = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def normalize_caption(text: str) -> str:
    """去掉汉字之间被排版插入的空白；汉字与数字之间的空格保留，避免把编号粘起来。"""
    return _CJK_GAP.sub("", text).strip()


def is_caption(text: str) -> bool:
    """判断一行是否为「图N」「表N」「专栏N」这类题注。"""
    return bool(_CAPTION_START.match(normalize_caption(text)))
# 同一视觉行的判定容差：纵坐标相差小于该值即视为同一行，用于拼接被拆开的题注。
_SAME_LINE_TOLERANCE = 3.0

_MIN_TABLE_ROWS = 2
_MIN_TABLE_COLUMNS = 2
# 包围盒占页面宽高都超过这个比例时，判定为「整页框线」而不是表格。
_PAGE_SPAN_RATIO = 0.85
# 判定某一行是否为正文；图表内部的标签不满足这里的条件，详见 _looks_like_prose。
_PROSE_MIN_CHARS = 15
_PROSE_READABLE_RATIO = 0.6
_READABLE_CHAR = re.compile(r"[一-鿿A-Za-z0-9%．。，、；：（）()\-—.,]")


# ---------------- 通用：题注定位 ----------------


def _line_blocks(page, tolerance: float = _SAME_LINE_TOLERANCE) -> list[tuple]:
    """把页面文本块按视觉行合并，返回 (top, bottom, x0, x1, text) 列表。

    同一个题注常被拆成多个文本块（如「图3 全球生物燃料乙醇产量占比」+「情况」），
    不合并就会只拿到半句，因此这里按纵坐标重叠把同行的块拼回一行。

    表格抽取拿到的是 pdfplumber 的页面对象，图表抽取拿到的是 PyMuPDF 的，
    两者取文本的接口不同，这里统一成同一种结构再处理。
    """
    if hasattr(page, "extract_text_lines"):  # pdfplumber
        raw = [
            (line["top"], line["bottom"], line["x0"], line["x1"], line.get("text") or "")
            for line in page.extract_text_lines()
        ]
    else:  # PyMuPDF
        raw = [(b[1], b[3], b[0], b[2], b[4]) for b in page.get_text("blocks")]

    rows = []
    for y0, y1, x0, x1, text in raw:
        text = " ".join(str(text).split()).strip()
        if text:
            rows.append([y0, y1, x0, x1, text])
    rows.sort(key=lambda r: (r[0], r[2]))
    merged = []
    for row in rows:
        if merged and row[0] - merged[-1][0] <= tolerance:
            last = merged[-1]
            last[1] = max(last[1], row[1])
            last[2] = min(last[2], row[2])
            last[3] = max(last[3], row[3])
            last[4] = f"{last[4]} {row[4]}"
        else:
            merged.append(row)
    return [tuple(r) for r in merged]


def _find_caption(page, bbox, direction: str = "above", max_dist: float = 30.0, max_len: int = 90):
    """在表格或图片附近找题注。

    direction 为 above/below/both：中文图表题注通常写在图下方、表格题注写在表格上方，
    两种位置都要能找。

    只认「图N」「表N」这类编号开头的行。正文里带「图」字的句子（「如下图所示」）
    和图表区域内的说明文字都很多，放宽到「含关键词即可」会把它们当题注，
    而错误的题注比没有题注更糟——它会让模型把片段对到错误的图上。
    """
    x0, top, x1, bottom = bbox
    candidates = []
    for y0, y1, bx0, bx1, text in _line_blocks(page):
        if len(text) > max_len or not is_caption(text):
            continue
        if bx1 < x0 - 5 or bx0 > x1 + 5:
            continue
        above = top - max_dist <= y0 <= top
        below = bottom <= y0 <= bottom + max_dist
        if (direction in ("above", "both") and above) or (direction in ("below", "both") and below):
            candidates.append((abs(y0 - (top if above else bottom)), normalize_caption(text)))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][1]


def _looks_like_prose(text: str) -> bool:
    """判断一行文字是正文，还是图表内部的标签。

    只看长度会误判：图表的坐标轴和图例文字在字体缺少字符映射时，
    提取出来是一长串乱码，长度和正文相当。正文则是成句的可读中文，
    因此除了长度还要看可读字符的占比。
    """
    if len(text) < _PROSE_MIN_CHARS:
        return False
    readable = sum(1 for char in text if _READABLE_CHAR.match(char))
    return readable / len(text) >= _PROSE_READABLE_RATIO


def caption_prompt(prompt: str, caption: str) -> str:
    """把题注拼进图片描述要求，让视觉模型知道这张图在讲什么。

    没有题注时原样返回：模型只能靠图内文字判断，描述会不可靠，
    但比编一个不存在的标题要好。
    """
    if not caption:
        return prompt
    return f"这张图的标题是「{caption}」。\n{prompt}"


# ---------------- 表格 ----------------


def _table_to_markdown(table) -> str:
    """把表格的行列数据转换成 Markdown，去掉全空行并将第一行作为表头。

    少于两行或少于两列的「表格」其实是正文被误判，返回空字符串让调用方跳过：
    PDF 里的通知抬头、章节标题常被 pdfplumber 当成单列表格，
    存进索引只会污染检索，不如不要。
    """
    if not table:
        return ""
    rows = []
    for row in table:
        # 单元格里常有硬换行（表头折行、多行说明），直接拼进 Markdown 会把一行拆成两行，
        # 整个表格的列结构就散了，所以这里把单元格内的空白统一压成单个空格。
        cleaned_row = [" ".join((cell or "").split()) for cell in row]
        rows.append(cleaned_row)
    rows = [r for r in rows if any(r)]  # 去全空行
    if len(rows) < _MIN_TABLE_ROWS:
        return ""
    # 各行可能有不同数量的单元格，先补齐列数；这里不会还原原 PDF 的合并单元格。
    ncols = max(len(r) for r in rows)
    if ncols < _MIN_TABLE_COLUMNS:
        return ""
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _table_bboxes(page, settings=None):
    """返回该页找到的表格及其包围盒；抽取失败时返回空列表。"""
    try:
        tables = page.find_tables(settings) if settings else page.find_tables()
    except Exception:
        return []
    found = []
    for table in tables:
        try:
            bbox = tuple(table.bbox)
        except Exception:
            continue
        found.append((bbox, table))
    return found


def _overlaps(bbox, others, tolerance: float = 8.0) -> bool:
    """判断包围盒是否与已有表格显著重叠，用于合并两种抽取策略的结果。

    同一个表在有框线和无框线两种策略下都会被找到，位置几乎一致；
    留一点容差，避免边线判定差异导致同一个表被收两次。
    """
    ax0, ay0, ax1, ay1 = bbox
    for bx0, by0, bx1, by1 in others:
        if ax0 < bx1 - tolerance and bx0 < ax1 - tolerance and ay0 < by1 - tolerance and by0 < ay1 - tolerance:
            return True
    return False


def _spans_page(page, bbox) -> bool:
    """判断「表格」是否几乎盖住整页。

    页面四周的框线会被 pdfplumber 当成一张大表，把页眉、页脚和正文一起裹进来；
    这类结果的包围盒接近整页，与真正的表格有明显区别。
    """
    x0, top, x1, bottom = bbox
    return (x1 - x0) >= page.width * _PAGE_SPAN_RATIO and (
        bottom - top
    ) >= page.height * _PAGE_SPAN_RATIO


def _swallows_caption(page, bbox) -> bool:
    """判断表格区域是否把一条题注框了进去。

    文本对齐策略很贪心，会把相邻的图和表并成一块；并过头的直接特征是
    题注落在了表格内部——正常表格不会包含题注行。这类结果的边界已经错了。
    """
    _x0, top, _x1, bottom = bbox
    for y0, y1, _bx0, _bx1, text in _line_blocks(page):
        if top < y0 and y1 < bottom and is_caption(text):
            return True
    return False


def extract_tables(pdf_path: Path) -> list[Document]:
    """用 pdfplumber 抽取表格，转成 Markdown 表格字符串。

    先用默认的框线策略，再用文本对齐策略补无框线表格：政务文档里大量表格
    只有列对齐、没有框线，只跑框线策略会整张漏掉。两种策略的结果按位置去重。
    """
    import pdfplumber
    from langchain_core.documents import Document

    docs: list[Document] = []
    text_settings = {"vertical_strategy": "text", "horizontal_strategy": "text"}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            boxes: list[tuple] = []
            for settings in (None, text_settings):
                for bbox, table in _table_bboxes(page, settings):
                    if (
                        _overlaps(bbox, boxes)
                        or _spans_page(page, bbox)
                        or _swallows_caption(page, bbox)
                    ):
                        continue
                    md = _table_to_markdown(table.extract())
                    if not md:
                        continue
                    boxes.append(bbox)
                    # 表格题注通常在上方，个别排在下方，两处都找。
                    caption = _find_caption(page, bbox, "both")
                    content = f"{caption}\n{md}" if caption else md
                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_no,
                                "type": "table",
                                "caption": caption or None,
                            },
                        )
                    )
    return docs


# ---------------- 图表 ----------------


def _figure_document(Document, pdf_path, page_no, bbox, caption, description):
    """把一次图表描述组装成片段，题注同时写进元数据和正文开头。"""
    content = f"{caption}\n{description}" if caption else description
    return Document(
        page_content=content,
        metadata={
            "source": pdf_path.name,
            "page": page_no,
            "type": "figure",
            "bbox": [round(v, 1) for v in bbox],
            "caption": caption or None,
        },
    )


def _collect_descriptions(vision, items, batch_size, on_status=None):
    """分批描述收集到的图片，返回与 items 等长的描述列表。

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


def _render_region(page, bbox, settings):
    """把页面的一块区域渲染成图片并载入内存，供随后批量描述。"""
    import pymupdf
    from PIL import Image

    pix = page.get_pixmap(clip=pymupdf.Rect(*bbox), dpi=settings.dpi)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    # 渲染结果先读进内存再交给模型：稍后批量描述时要一次性持有整批图片，
    # 不能像逐张描述那样依赖 with 语句在离开代码块时关闭。
    image.load()
    return image


def extract_figures(pdf_path, vision, settings, on_status=None):
    """截取 PDF 内嵌图片所在区域，生成文字描述并记录页码与位置。

    先把整份文档的图片收集齐再分批描述：视觉模型支持一次输入多张图，
    批量走一次前向能把单张耗时降到五分之一左右。
    只处理 get_image_info 找到的内嵌位图；矢量绘制的图表由 extract_vector_charts 处理。
    """
    import pymupdf
    from langchain_core.documents import Document

    pending = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_no in range(len(pdf)):
            page = pdf[page_no]
            for info in page.get_image_info():
                x0, y0, x1, y1 = info["bbox"]
                # 这里按 PDF 页面坐标筛掉小图片（页眉横幅、图标）；dpi 只影响随后生成图片的清晰度。
                if x1 - x0 < settings.min_width or y1 - y0 < settings.min_height:
                    continue
                caption = _find_caption(page, (x0, y0, x1, y1), "below")
                image = _render_region(page, (x0, y0, x1, y1), settings)
                pending.append(
                    (page_no + 1, (x0, y0, x1, y1), caption, image, caption_prompt(settings.prompt, caption))
                )  # fmt: skip

    descriptions = _collect_descriptions(
        vision,
        [(image, prompt) for _, _, _, image, prompt in pending],
        settings.batch_size,
        on_status,
    )
    return [
        _figure_document(Document, pdf_path, page_no, bbox, caption, description)
        for (page_no, bbox, caption, _image, _prompt), description in zip(pending, descriptions)
    ]


def _chart_region(page, caption) -> tuple | None:
    """由题注位置推断它上方那块图表区域的包围盒。

    中文报告的图是「正文段落 → 图表 → 题注」的顺序，因此图表区域
    就是上一段正文的底边到题注顶边之间。找不到正文时从页面上沿开始。
    """
    cap_top, _cap_bottom, cap_x0, cap_x1, _text = caption
    top = None
    for _y0, y1, bx0, bx1, text in _line_blocks(page):
        # 题注本身及其下方的内容不算作图表区域。
        if y1 > cap_top:
            continue
        if bx1 < cap_x0 - 20 or bx0 > cap_x1 + 20:
            continue
        # 坐标轴和图例的文字就排在图表内部，不排除它们，
        # 区域上界会被压到图表内部，整块图表反而被切掉。
        if not _looks_like_prose(text):
            continue
        top = y1 if top is None else max(top, y1)
    page_box = page.rect
    return (
        min(cap_x0, page_box.x0 + 30),
        (top + 2) if top is not None else page_box.y0 + 30,
        max(cap_x1, page_box.x1 - 30),
        cap_top - 2,
    )


def extract_vector_charts(pdf_path, vision, settings, on_status=None):
    """渲染矢量绘制的图表并生成描述；这类图没有内嵌位图，extract_figures 找不到。

    图表的数据标签常常没有可用的字符映射，文本提取只能得到乱码，
    因此这里把题注上方的区域整块渲染成图片，让视觉模型直接读。
    """
    import pymupdf
    from langchain_core.documents import Document

    pending = []
    with pymupdf.open(str(pdf_path)) as pdf:
        for page_no in range(len(pdf)):
            page = pdf[page_no]
            embedded = [
                tuple(i["bbox"])
                for i in page.get_image_info()
                if i["bbox"][2] - i["bbox"][0] >= settings.min_width
                and i["bbox"][3] - i["bbox"][1] >= settings.min_height
            ]
            for caption in _line_blocks(page):
                title = normalize_caption(caption[4])
                match = _CAPTION_START.match(title)
                # 只处理图表题注；表格题注由 extract_tables 负责。
                if not match or match.group(1) != "图":
                    continue
                region = _chart_region(page, caption)
                if region is None:
                    continue
                x0, y0, x1, y1 = region
                if x1 - x0 < settings.min_width or y1 - y0 < settings.min_height:
                    continue
                # 内嵌位图已经覆盖的区域交给 extract_figures，避免同一个图描述两次。
                if _overlaps((x0, y0, x1, y1), embedded):
                    continue
                image = _render_region(page, (x0, y0, x1, y1), settings)
                pending.append(
                    (page_no + 1, (x0, y0, x1, y1), title, image, caption_prompt(settings.prompt, title))
                )  # fmt: skip

    descriptions = _collect_descriptions(
        vision,
        [(image, prompt) for _, _, _, image, prompt in pending],
        settings.batch_size,
        on_status,
    )
    return [
        _figure_document(Document, pdf_path, page_no, bbox, title, description)
        for (page_no, bbox, title, _image, _prompt), description in zip(pending, descriptions)
    ]
