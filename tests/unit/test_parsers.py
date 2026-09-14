"""验证格式选择、TXT 内容读取和来源展示，不加载模型或 PDF 依赖。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.bootstrap import Runtime
from src.config import load_settings
from src.context_builder import ContextBuilder
from src.ingestion import list_documents
from src.parsers import pdf_elements
from src.parsers.pdf import PDFParser
from src.parsers.registry import ParserRegistry
from src.parsers.text import TextParser
from src.schemas import SearchHit


def test_registry_routes_by_case_insensitive_suffix():
    """不同后缀交给不同解析器，大小写不影响选择，未知格式不触发解析。"""
    pdf, text = Mock(), Mock()
    registry = ParserRegistry({".PDF": pdf, ".txt": text})
    assert registry.parse("report.PdF") is pdf.parse.return_value
    assert registry.parse(Path("notes.TXT")) is text.parse.return_value
    pdf.parse.assert_called_once_with(Path("report.PdF"))
    text.parse.assert_called_once_with(Path("notes.TXT"))
    with pytest.raises(ValueError, match=".docx"):
        registry.parse("report.docx")
    assert registry.supported_suffixes == {".pdf", ".txt"}
    assert pdf.parse.call_count == text.parse.call_count == 1


@pytest.mark.parametrize("suffixes", [[], ["txt"], ["."], [".txt", ".TXT"]])
def test_registry_rejects_invalid_configuration(suffixes):
    """拒绝空注册表、没有点的后缀和重复的大小写变体。"""
    with pytest.raises(ValueError):
        ParserRegistry(dict.fromkeys(suffixes, Mock()))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_text_preserves_chinese_paragraphs_and_indentation(tmp_path, encoding):
    """普通 UTF-8 和带 BOM 的正文都保留中文、段落及缩进，不生成页码。"""
    path = tmp_path / "能源说明.TXT"
    content = "能源政策说明\n\n    保留缩进和段落。\n"
    path.write_bytes(content.replace("\n", "\r\n").encode(encoding))
    parsed = TextParser(document_factory=SimpleNamespace).parse(path)
    assert not parsed.errors
    assert parsed.tables == parsed.figures == []
    assert len(parsed.texts) == 1
    assert parsed.texts[0].page_content == content
    assert parsed.texts[0].metadata == {"source": path.name, "type": "text"}


@pytest.mark.parametrize(
    "content",
    [b"", b" \r\n\t", b"\xff", "中文".encode("gb18030"), b"t\x00e\x00x\x00t"],
)
def test_text_rejects_empty_invalid_encoding_and_binary_content(tmp_path, content):
    """空文件、错误编码和含空字符的内容明确失败，不生成可入库文档。"""
    path = tmp_path / "bad.txt"
    path.write_bytes(content)
    factory = Mock()
    parsed = TextParser(document_factory=factory).parse(path)
    assert parsed.errors and not parsed.texts
    factory.assert_not_called()


def test_text_reports_missing_file(tmp_path):
    """文件不可读取时通过解析错误返回原因。"""
    parsed = TextParser().parse(tmp_path / "missing.txt")
    assert parsed.errors and not parsed.texts


def test_document_discovery_filters_files_without_recursing(tmp_path):
    """发现 PDF 和 TXT 的大小写后缀，忽略未支持格式、同名后缀目录和子目录文件。"""
    for name in ("a.PDF", "b.TXT", "c.txt", "d.docx"):
        (tmp_path / name).write_text("content", encoding="utf-8")
    nested = tmp_path / "folder.pdf"
    nested.mkdir()
    (nested / "hidden.txt").write_text("content", encoding="utf-8")
    paths = list_documents(tmp_path, {".pdf", ".txt"})
    assert [path.name for path in paths] == ["a.PDF", "b.TXT", "c.txt"]
    assert list_documents(tmp_path, {".pdf", ".txt"}, max_files=1) == paths[:1]
    with pytest.raises(ValueError):
        list_documents(tmp_path, {".txt"}, max_files=0)
    with pytest.raises(ValueError):
        list_documents(tmp_path / "missing", {".txt"})


def test_runtime_uses_the_registered_formats_for_ingestion():
    """正式 Runtime 接通五种已实现的格式，文件发现与解析使用相同后缀。"""
    settings = load_settings(Path(__file__).resolve().parents[2] / "config.toml")
    runtime = Runtime(settings)
    assert isinstance(runtime.parser, ParserRegistry)
    assert runtime.ingestion.parser is runtime.parser
    assert runtime.ingestion.supported_suffixes == {".pdf", ".txt", ".md", ".docx", ".xlsx"}


def test_pdf_uses_migrated_element_extractors():
    """确认 PDF 的默认表格和图片函数已指向迁移后的辅助模块。"""
    parser = PDFParser(SimpleNamespace(available=False), None)
    assert parser.table_extractor is pdf_elements.extract_tables
    assert parser.figure_extractor is pdf_elements.extract_figures
    assert pdf_elements._table_to_markdown([["项目", "数值"], ["能源", "10"]]) == (
        "| 项目 | 数值 |\n| --- | --- |\n| 能源 | 10 |"
    )


def test_context_shows_pdf_pages_and_txt_filename(tmp_path):
    """混合证据保留 PDF 页码，TXT 和页码未知的 PDF 不显示占位页码。"""
    path = tmp_path / "说明.txt"
    path.write_text("能源政策说明", encoding="utf-8")
    parsed = TextParser(document_factory=SimpleNamespace).parse(path)
    pdf = SimpleNamespace(
        page_content="PDF 正文",
        metadata={"source": "报告.pdf", "page": 3, "type": "text"},
    )
    unknown_page = SimpleNamespace(
        page_content="页码未知",
        metadata={"source": "扫描.pdf", "page": 0, "type": "text"},
    )
    hits = [SearchHit(pdf), SearchHit(parsed.texts[0]), SearchHit(unknown_page)]
    bundle = ContextBuilder("{context}\n{question}").build("有哪些政策？", hits)
    assert "来源: 报告.pdf 第3页" in bundle.prompt
    assert "来源: 说明.txt]" in bundle.prompt
    assert "来源: 扫描.pdf]" in bundle.prompt
    assert "第?页" not in bundle.prompt and "第0页" not in bundle.prompt
    assert bundle.evidence == hits


# ---------------- 图表与表格标题 ----------------


def fake_page(lines):
    """按 pdfplumber 的 extract_text_lines 形状造一个页面替身。

    每项是 (top, bottom, x0, x1, text)，坐标含义与真实页面一致。
    """
    return SimpleNamespace(
        extract_text_lines=lambda: [
            {"top": top, "bottom": bottom, "x0": x0, "x1": x1, "text": text}
            for top, bottom, x0, x1, text in lines
        ]
    )


def test_find_caption_reads_below_for_figures_and_above_for_tables():
    """图表题注在下方、表格题注在上方，两个方向都要能取到。"""
    page = fake_page(
        [
            (80, 92, 100, 400, "表1 各类能源装机规模"),
            (100, 260, 100, 400, "正文内容区域"),
            (280, 420, 100, 400, "绘图内容区域"),
            (430, 442, 120, 380, "图7 2024年各省绿证出售量情况"),
        ]
    )
    assert pdf_elements._find_caption(page, (100, 100, 400, 260), "above") == "表1 各类能源装机规模"
    assert pdf_elements._find_caption(page, (100, 280, 400, 420), "below") == (
        "图7 2024年各省绿证出售量情况"
    )
    # 方向反了就找不到：上方查不到图题，下方查不到表题。
    assert pdf_elements._find_caption(page, (100, 280, 400, 420), "above") == ""
    assert pdf_elements._find_caption(page, (100, 100, 400, 260), "below") == ""


def test_find_caption_requires_numbered_prefix():
    """正文里带「图」字的叙述句不是题注，必须有「图N」这类编号才算。"""
    page = fake_page(
        [
            (434, 446, 120, 380, "如下图所示，各省出售量差异较大"),
            (452, 464, 120, 380, "图8 2024年各省绿证购买量情况"),
        ]
    )
    assert pdf_elements._find_caption(page, (100, 300, 400, 430), "below") == (
        "图8 2024年各省绿证购买量情况"
    )


def test_line_blocks_joins_fragments_split_across_blocks():
    """同一个题注被拆成多个文本块时要拼回一行，否则只会拿到半句。"""
    page = fake_page(
        [
            (430, 442, 120, 300, "图3 全球生物燃料乙醇产量占比"),
            (431, 443, 305, 380, "情况"),
        ]
    )
    blocks = pdf_elements._line_blocks(page)
    assert len(blocks) == 1
    assert "全球生物燃料乙醇产量占比" in blocks[0][4] and "情况" in blocks[0][4]


def test_table_to_markdown_rejects_degenerate_tables():
    """单列或单行的「表格」其实是正文被误判，返回空字符串让调用方跳过。"""
    assert pdf_elements._table_to_markdown([["通知抬头", "国务院办公厅"]]) == ""
    assert pdf_elements._table_to_markdown([["一、发展形势"], ["二、重点任务"]]) == ""


def test_caption_prompt_prepends_caption_only_when_present():
    """题注拼进描述要求，模型才知道这张图在讲什么；没有题注时保持原样。"""
    prompt = "请描述这张图。"
    assert pdf_elements.caption_prompt(prompt, "") == prompt
    with_caption = pdf_elements.caption_prompt(prompt, "图7 各省绿证出售量情况")
    assert with_caption.startswith("这张图的标题是「图7 各省绿证出售量情况」")
    assert with_caption.endswith(prompt)


def test_looks_like_prose_rejects_garbled_chart_labels():
    """图表标签有两种形态：很短，或字体缺字符映射时提取出的长串乱码。"""
    assert pdf_elements._looks_like_prose("截至2025 年底，全球新型储能累计装机规模约2.8 亿千瓦")
    assert not pdf_elements._looks_like_prose("ԍ᎖ቇඡϲᑟ")
    assert not pdf_elements._looks_like_prose("ᜉ఻᜻വὅʺӢၧ὆ࣱ کϲᑟ௑᫂ ὅ࠵௑὆")
    assert not pdf_elements._looks_like_prose("0.4% A。")


def test_context_label_shows_caption_for_figures_and_tables():
    """来源说明带上题注：同一页可能有好几张图，只给页码说不清是哪一张。"""
    figure = SimpleNamespace(
        page_content="描述",
        metadata={
            "source": "报告.pdf",
            "page": 24,
            "type": "figure",
            "caption": "图7 2024年各省绿证出售量情况",
        },
    )
    bundle = ContextBuilder("{context}\n{question}").build("问题", [SearchHit(figure)])
    assert "来源: 报告.pdf 第24页 | 图7 2024年各省绿证出售量情况" in bundle.prompt


# ---------------- 正文清洗与章节路径 ----------------


def test_clean_element_text_drops_pure_placeholder_blocks():
    """整块都是 (cid:1234) 的块没有可检索内容，去掉；只带少量占位符的正文保留。"""
    from src.parsers.pdf import clean_element_text

    assert clean_element_text("(cid:1293)(cid:5014)(cid:4679)(cid:3489)") == ""
    assert clean_element_text("") == ""
    kept = clean_element_text("全球储能装机(cid:1234)规模持续增长，同比增长超过一半")
    assert "全球储能装机" in kept and "(cid:" not in kept


def test_page_number_pattern_leaves_bare_numbers_alone():
    """只认两侧带装饰符的页码；光秃秃的数字行是年份或编号，误删代价更大。"""
    from src.parsers.pdf import _PAGE_NUMBER_FOOTER

    assert _PAGE_NUMBER_FOOTER.match("— 12 —")
    assert _PAGE_NUMBER_FOOTER.match("- 3 -")
    assert not _PAGE_NUMBER_FOOTER.match("2024")
    assert not _PAGE_NUMBER_FOOTER.match("3.")
    assert not _PAGE_NUMBER_FOOTER.match("— 推动储能建设")


def test_page_number_footer_is_dropped_before_chunking(monkeypatch):
    """页码页脚要在提取阶段丢掉：跨页合并后会卡进句子中间，污染检索文本。"""
    updf = pytest.importorskip("unstructured.partition.pdf")
    from src.parsers.pdf import extract_page_texts_with_unstructured

    def element(category, text, page):
        return SimpleNamespace(
            category=category, text=text, metadata=SimpleNamespace(page_number=page)
        )

    monkeypatch.setattr(
        updf,
        "partition_pdf",
        lambda **kwargs: [
            element("NarrativeText", "优化加强电网主网架。", 1),
            element("UncategorizedText", "— 1 —", 1),
            element("NarrativeText", "开展电力系统设计工作。", 2),
        ],
    )

    page_texts = extract_page_texts_with_unstructured(Path("x.pdf"), "fast", None)

    assert page_texts[1][0] == ["优化加强电网主网架。"]
    assert page_texts[2][0] == ["开展电力系统设计工作。"]


def test_heading_path_detects_chinese_section_numbering():
    """中文报告的章节标题有固定编号形式；正文句子即使以编号开头也不该误判。

    同一套规则同时供 PDF 解析器跟踪章节、供切分器判断能不能把标题挪到下一块，
    因此放在共享模块里，这里一并核对。
    """
    from src.headings import heading_level

    for text in ("三、绿证市场活力持续增强", "（一）交易规模实现翻两番", "第一章 总体要求", "2. 装机规模超预期增长"):
        assert heading_level(text) is not None, text
    for text in (
        "0.4% A。",
        "截至2025 年底，全国新型储能累计装机规模13593 万千瓦，同比增长84.3%，储能时长呈逐年上升趋势。",
        "2025 年，全球新型储能新增装机规模约1.1 亿千瓦",
    ):
        assert heading_level(text) is None, text


def test_heading_stack_never_chains_same_level_headings():
    """同级标题不能互相套娃，二级标题也不该在缺少一级标题时冒充顶层。"""
    from src.parsers.pdf import _push_heading

    def titles(stack):
        return [title for title in stack if title]

    stack = []
    # 文档开头直接出现二级标题：它应当独占一层，而不是落到第 0 层。
    _push_heading(stack, "（一）世界新型储能发展", 2)
    assert titles(stack) == ["（一）世界新型储能发展"]
    # 再来一个同级标题，前一个必须被替换，不能串成父子。
    _push_heading(stack, "（六）新型储能产业集群扩能提质", 2)
    assert titles(stack) == ["（六）新型储能产业集群扩能提质"]
    # 出现一级标题后，二级标题挂在它下面。
    _push_heading(stack, "三、绿证市场活力持续增强", 1)
    _push_heading(stack, "（一）交易规模实现翻两番", 2)
    assert titles(stack) == ["三、绿证市场活力持续增强", "（一）交易规模实现翻两番"]
    # 同级二级标题替换旧的二级标题，一级标题保留。
    _push_heading(stack, "（二）参与主体数量显著增加", 2)
    assert titles(stack) == ["三、绿证市场活力持续增强", "（二）参与主体数量显著增加"]


def test_table_markdown_collapses_newlines_inside_cells():
    """单元格里的硬换行要压成空格，否则一行被拆成两行，整个表格的列结构会散掉。"""
    md = pdf_elements._table_to_markdown(
        [
            ["2025年度中国电力市场发展报告\n· 2 ·", ""],
            ["发文机关：", "国家能源局"],
        ]
    )
    lines = md.split("\n")
    assert lines[0] == "| 2025年度中国电力市场发展报告 · 2 · |  |"
    # 每一行的管道符数量必须一致，否则 Markdown 表格渲染不出来。
    assert len({line.count("|") for line in lines}) == 1


def test_spans_page_rejects_page_framing_lines():
    """页面四周的框线会被当成一张大表，把页眉页脚一起裹进来，必须拒掉。"""
    page = SimpleNamespace(width=541, height=754)
    # 覆盖整页的框线：宽高都接近页面尺寸。
    assert pdf_elements._spans_page(page, (0, 0, 764, 754))
    # 正常表格：只占页面的一部分。
    assert not pdf_elements._spans_page(page, (82, 454, 465, 684))
    # 横跨页面宽度但很矮的条带（如页眉）不算整页。
    assert not pdf_elements._spans_page(page, (0, 40, 541, 60))


# ---------------- 批量图片描述 ----------------


class FakeVision:
    """记录批量与逐张调用，用来验证分批策略和降级路径。"""

    def __init__(self, fail_batches=False):
        self.fail_batches = fail_batches
        self.batches = []
        self.singles = []

    def describe_batch(self, images, prompts):
        if self.fail_batches:
            raise RuntimeError("显存不足")
        self.batches.append((list(images), list(prompts)))
        return [f"批量描述{prompt}" for prompt in prompts]

    def describe(self, image, prompt):
        self.singles.append(prompt)
        return f"单张描述{prompt}"


def test_collect_descriptions_splits_into_batches_and_keeps_order():
    """按 batch_size 切批，每批只调一次批量接口，返回顺序与输入一致。"""
    vision = FakeVision()
    items = [(f"图{i}", f"提示{i}") for i in range(5)]

    out = pdf_elements._collect_descriptions(vision, items, batch_size=2)

    assert out == [f"批量描述提示{i}" for i in range(5)]
    assert [len(images) for images, _ in vision.batches] == [2, 2, 1]
    assert vision.singles == []
    # 图片必须和提示词一一对应地传下去，错位会让描述张冠李戴。
    for images, prompts in vision.batches:
        assert [p.removeprefix("提示") for p in prompts] == [
            i.removeprefix("图") for i in images
        ]


def test_collect_descriptions_falls_back_to_one_by_one(caplog):
    """整批失败时退回逐张：一张图出问题不该让同批其他图也丢了描述。"""
    vision = FakeVision(fail_batches=True)
    items = [(f"图{i}", f"提示{i}") for i in range(3)]

    out = pdf_elements._collect_descriptions(vision, items, batch_size=2)

    assert out == [f"单张描述提示{i}" for i in range(3)]
    assert vision.singles == ["提示0", "提示1", "提示2"]
    assert "退回逐张重试" in caplog.text


def test_collect_descriptions_reports_batch_progress():
    """进度按批上报，让长时间的文件级解析仍能看到推进。"""
    vision = FakeVision()
    notes = []
    pdf_elements._collect_descriptions(
        vision, [(f"图{i}", f"提示{i}") for i in range(5)], batch_size=2, on_status=notes.append
    )
    assert notes == ["描述第 1—2 张图表", "描述第 3—4 张图表", "描述第 5—5 张图表"]


def test_collect_descriptions_with_empty_input_makes_no_calls():
    """没有图片时不调用模型，也不产生进度。"""
    vision = FakeVision()
    assert pdf_elements._collect_descriptions(vision, [], batch_size=4) == []
    assert vision.batches == [] and vision.singles == []


def test_vision_settings_rejects_non_positive_batch_size(tmp_path):
    """批量大小必须是正整数；0 或负数会让分批循环永远不前进。"""
    from src.config import VisionSettings

    def build(batch_size):
        return VisionSettings(
            path=tmp_path,
            max_new_tokens=256,
            prompt="描述",
            min_width=150,
            min_height=70,
            dpi=150,
            batch_size=batch_size,
        )

    assert build(1).batch_size == 1
    for invalid in (0, -3, True):
        with pytest.raises(ValueError):
            build(invalid)


def test_every_registered_parser_accepts_the_progress_callback():
    """注册表无差别转发 on_status，每个解析器都必须接受这个参数。

    这个契约只在解析具体格式时才被触发：漏改一个解析器，对应格式的文件就会在入库时
    整体失败——真实事故是新增 .docx 文档后，Word 解析器没跟上签名，该文件直接报
    unexpected keyword argument，而其他格式照常工作。这里按注册表逐个核对签名。
    """
    import inspect

    parser = Runtime(load_settings(Path(__file__).resolve().parents[2] / "config.toml")).parser
    assert parser.supported_suffixes == {".pdf", ".txt", ".md", ".docx", ".xlsx"}

    for suffix, implementation in parser._parsers.items():  # noqa: SLF001 - 要覆盖真正注册的实现
        parameters = inspect.signature(implementation.parse).parameters
        assert "on_status" in parameters, f"{suffix} 的解析器没有接受 on_status"
        assert parameters["on_status"].default is None, f"{suffix} 的 on_status 必须有默认值"
