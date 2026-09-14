"""PDF 正文、OCR 与多模态解析，并明确记录各阶段失败信息。"""

from __future__ import annotations

import logging
import re

from ..headings import heading_level
from ..schemas import ParseResult
from . import pdf_elements

logger = logging.getLogger(__name__)
# Header/Footer 是每页重复的页眉页脚，保留只会让每个片段都带上同样的噪声，因此排除。
# UncategorizedText 曾一并排除，但中文报告里它装着真实正文段落和章节标题，需要保留，
# 只过滤其中提取失败的乱码块，见 clean_element_text。
FAST_CATEGORIES = ("Title", "NarrativeText", "Text", "ListItem", "UncategorizedText")

# 字体缺少字符映射时，unstructured 会把每个字输出成 (cid:1234) 这样的占位符。
_CID_TOKEN = re.compile(r"\(cid:\d+\)")
_MAX_CID_RATIO = 0.5

# 页码页脚（如「— 12 —」）常被归到 UncategorizedText，躲过了 Header/Footer 分类的排除，
# 又总落在页面末尾，跨页合并后会卡进句子中间。只认两侧带装饰符的独立页码：
# 光秃秃的数字行可能是年份、章节编号或表格里的取值，误删的代价比留噪声大得多。
_PAGE_NUMBER_FOOTER = re.compile(r"^[—–\-]{1,3}\s*\d{1,4}\s*[—–\-]{1,3}$")


def clean_element_text(text: str) -> str:
    """清掉提取失败留下的 (cid:1234) 占位符；整块基本都是占位符时返回空字符串。

    图表坐标轴一类文字常常没有可用的字符映射，提取出来整块都是占位符，
    既不能检索也没有内容；但同一个分类里也混着真实正文，所以只丢乱码块。
    """
    text = (text or "").strip()
    if not text:
        return ""
    placeholder_chars = sum(len(token) for token in _CID_TOKEN.findall(text))
    if placeholder_chars / len(text) > _MAX_CID_RATIO:
        return ""
    return _CID_TOKEN.sub("", text).strip()


def _push_heading(stack: list[str], text: str, level: int) -> None:
    """把标题放到编号层级对应的位置上，并丢掉同级和更深层的旧标题。

    栈比层级浅时先补空位，让标题落在自己的层级上。少了这一步，
    文档开头就出现的「（一）」会落到第 0 层冒充顶层，后面同级的「（六）」
    清不掉它，路径里就会串起两个互不相干的同级标题。
    """
    del stack[level - 1 :]
    while len(stack) < level - 1:
        stack.append("")
    stack.append(text)


def _page_entry(value) -> tuple:
    """把提取结果统一成 (文本列表, 标题路径)，兼容只返回文本列表的自定义提取函数。"""
    if isinstance(value, tuple):
        return value
    return value, ""


# allowed_categories 是允许保留的元素类别集合，None 表示不按类别过滤。
def extract_page_texts_with_unstructured(path, strategy, allowed_categories):
    """用指定的 unstructured 解析方式提取内容，按页返回文本和该页所属章节。

    标题路径在这里跟踪：它跨页延续，只有按元素顺序扫描全文档才能得到，
    因此不能等到按页组装片段时再补。
    """
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=str(path),
        strategy=strategy,
        languages=["chi_sim"],
    )
    # 每页保存 {"texts": [...], "heading_path": "..."}；页码缺失的放在 0，表示页码未知。
    page_texts = {}
    # 章节栈按编号层级存放，遇到同级或更高级的标题时丢弃它下面的层级。
    stack: list[str] = []
    for element in elements:
        if allowed_categories is not None and element.category not in allowed_categories:
            continue
        text = clean_element_text(element.text)
        if not text:
            continue
        # 页码不进正文，也不参与标题判断：它既不是内容也不是章节。
        if _PAGE_NUMBER_FOOTER.match(text):
            continue
        level = heading_level(text)
        if level is not None:
            _push_heading(stack, text, level)
        page_number = getattr(element.metadata, "page_number", None) or 0
        entry = page_texts.setdefault(page_number, {"texts": [], "heading_path": ""})
        # 记录本页第一条内容出现时的章节：一页可能跨两节，
        # 但引用时只需要说明这段内容属于哪一节，取开头的那节最贴近阅读顺序。
        if not entry["texts"]:
            entry["heading_path"] = " > ".join(title for title in stack if title)
        entry["texts"].append(text)
    return {page: (value["texts"], value["heading_path"]) for page, value in page_texts.items()}


def extract_page_texts_with_pypdf(path):
    """用 pypdf 读取 PDF 中已有的文字，在 fast 解析失败或没有文字时尝试。

    这条路径拿不到元素结构，因此没有章节信息。
    """
    from pypdf import PdfReader

    page_texts = {}
    with path.open("rb") as stream:
        for page_number, page in enumerate(PdfReader(stream).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                page_texts[page_number] = ([text], "")
    return page_texts


class PDFParser:
    def __init__(
        self,
        vision,
        vision_settings,
        unstructured_text_extractor=None,
        fallback_text_extractor=None,
        document_factory=None,
        table_extractor=None,
        figure_extractor=None,
        chart_extractor=None,
    ):
        """保存视觉模型及配置，并允许单独替换正文、表格和图片的提取函数。"""
        self.vision = vision
        self.vision_settings = vision_settings
        self.unstructured_text_extractor = (
            unstructured_text_extractor or extract_page_texts_with_unstructured
        )
        self.fallback_text_extractor = fallback_text_extractor or extract_page_texts_with_pypdf
        self.document_factory = document_factory
        self.table_extractor = table_extractor or pdf_elements.extract_tables
        self.figure_extractor = figure_extractor or pdf_elements.extract_figures
        self.chart_extractor = chart_extractor or pdf_elements.extract_vector_charts

    def extract_page_texts(self, path):
        """按 fast、pypdf、OCR 顺序尝试正文提取，返回首个非空页面集合。

        任一方式取到文字就停止，不再补查该 PDF 的其他页面。
        都没有文字时，有异常则汇总抛出，否则返回空字典。
        """
        failures = []
        strategies = (
            ("fast", lambda: self.unstructured_text_extractor(path, "fast", FAST_CATEGORIES)),
            ("pypdf", lambda: self.fallback_text_extractor(path)),
            ("ocr_only", lambda: self.unstructured_text_extractor(path, "ocr_only", None)),
        )
        for name, extract in strategies:
            try:
                page_texts = extract()
                if page_texts:
                    return page_texts
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        if failures:
            raise RuntimeError("; ".join(failures))
        return {}

    def parse_text(self, path):
        """把页面文本转换为文档片段，保留来源文件名、页码、正文类型和所属章节。"""
        page_texts = self.extract_page_texts(path)
        factory = self.document_factory
        if factory is None:
            from langchain_core.documents import Document

            factory = Document
        documents = []
        for page_number, value in sorted(page_texts.items()):
            texts, heading = _page_entry(value)
            content = "\n".join(texts).strip()
            if not content:
                continue
            metadata = {"source": path.name, "page": page_number, "type": "text"}
            if heading:
                metadata["heading_path"] = heading
            documents.append(factory(page_content=content, metadata=metadata))
        return documents

    def parse_tables_and_figures(self, path, on_status=None):
        """独立提取表格及图片描述，将每个阶段的异常记录到结果中。

        图表有两条来源：内嵌位图，以及用矢量线条绘制、没有位图的图表；
        两者互补，不会描述同一张图（后者会跳过已被前者覆盖的区域）。
        on_status 是可选函数，接收一句描述当前动作的文字，由提取函数按批上报。
        """
        result = ParseResult()
        if on_status is not None:
            on_status("提取表格")
        try:
            result.tables = self.table_extractor(path)
        except Exception as exc:
            result.errors.append(f"tables: {exc}")
        # 没有模型配置文件时跳过图片描述；有配置但加载或解析失败时会记录错误。
        if self.vision.available:
            try:
                # 传入视觉对象本身而不是单张描述函数：提取函数要先把图片收集齐，
                # 再分批交给 describe_batch，批量能把单张耗时降到五分之一左右。
                result.figures = self.figure_extractor(
                    path, self.vision, self.vision_settings, on_status=on_status
                )
                result.figures += self.chart_extractor(
                    path, self.vision, self.vision_settings, on_status=on_status
                )
            except Exception as exc:
                result.errors.append(f"figures: {exc}")
        return result

    def parse(self, path, on_status=None):
        """把正文、表格、图片描述和错误放入 ParseResult，供入库流程检查和使用。

        on_status 是可选函数，接收一句描述当前动作的文字，用于显示解析进度。
        """
        result = ParseResult()
        try:
            if on_status is not None:
                on_status("提取正文")
            result.texts = self.parse_text(path)
        except Exception as exc:
            result.errors.append(f"text: {exc}")
        tables_and_figures = self.parse_tables_and_figures(path, on_status=on_status)
        result.tables = tables_and_figures.tables
        result.figures = tables_and_figures.figures
        result.errors.extend(tables_and_figures.errors)
        return result
