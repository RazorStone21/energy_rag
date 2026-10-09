"""解析器注册、文本格式与来源标签的回归测试；PDF 侧见 test_mineru_parser.py。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.bootstrap import Runtime
from src.config import load_settings
from src.context_builder import ContextBuilder
from src.ingestion import list_documents
from src.parsers.registry import ParserRegistry
from src.parsers.text import TextParser
from src.schemas import SearchHit


def test_registry_routes_by_case_insensitive_suffix(tmp_path):
    """后缀大小写不敏感；未注册的格式明确报错，不返回空结果。"""
    path = tmp_path / "report.PDF"
    path.write_text("x", encoding="utf-8")
    calls = []
    parser = ParserRegistry({".pdf": SimpleNamespace(parse=lambda p: calls.append(p) or "ok")})
    assert parser.parse(path) == "ok"
    assert calls == [path]
    with pytest.raises(ValueError, match="暂不支持"):
        parser.parse(tmp_path / "report.docx")


@pytest.mark.parametrize("suffixes", [[], ["txt"], ["."], [".txt", ".TXT"]])
def test_registry_rejects_invalid_configuration(suffixes):
    """空注册表、缺少点、只有点、大小写重复的后缀都必须在构造时拒绝。"""
    with pytest.raises(ValueError):
        ParserRegistry({suffix: SimpleNamespace(parse=lambda p: None) for suffix in suffixes})


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_text_preserves_chinese_paragraphs_and_indentation(tmp_path, encoding):
    """TXT 整份文件作为一个片段，保留空行与行内缩进；BOM 不进入正文。"""
    path = tmp_path / "说明.txt"
    path.write_text("第一段\n  缩进内容\n\n第二段", encoding=encoding)
    parsed = TextParser(document_factory=SimpleNamespace).parse(path)
    assert len(parsed.texts) == 1
    assert parsed.texts[0].page_content == "第一段\n  缩进内容\n\n第二段"
    assert parsed.texts[0].metadata == {"source": "说明.txt", "type": "text"}


@pytest.mark.parametrize(
    "content",
    ["", "   \n\n  ", "﻿"],
)
def test_text_rejects_empty_invalid_encoding_and_binary_content(tmp_path, content):
    """空文件与只有空白的内容不产出片段，进而在入库时报错而不是写入空索引。"""
    path = tmp_path / "空.txt"
    path.write_text(content, encoding="utf-8")
    factory = SimpleNamespace
    parsed = TextParser(document_factory=factory).parse(path)
    assert parsed.texts == []


def test_text_reports_missing_file(tmp_path):
    """文件不存在时记入 errors，由入库流程判该文件失败。"""
    parsed = TextParser(document_factory=SimpleNamespace).parse(tmp_path / "缺失.txt")
    assert parsed.texts == []
    assert parsed.errors


def test_document_discovery_filters_files_without_recursing(tmp_path):
    """只扫第一层目录，且按后缀过滤；子目录里的文档不会被发现。"""
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "note.md").write_text("x", encoding="utf-8")
    nested = tmp_path / "子目录"
    nested.mkdir()
    (nested / "c.txt").write_text("x", encoding="utf-8")
    found = {path.name for path in list_documents(tmp_path, {".txt", ".pdf"})}
    assert found == {"a.txt", "b.pdf"}
    with pytest.raises(ValueError, match="does not exist"):
        list_documents(tmp_path / "missing", {".txt"})


def test_runtime_uses_the_registered_formats_for_ingestion():
    """正式 Runtime 接通五种已实现的格式，文件发现与解析使用相同后缀。"""
    settings = load_settings(Path(__file__).resolve().parents[2] / "config.toml")
    runtime = Runtime(settings)
    assert isinstance(runtime.parser, ParserRegistry)
    assert runtime.ingestion.parser is runtime.parser
    assert runtime.ingestion.supported_suffixes == {".pdf", ".txt", ".md", ".docx", ".xlsx"}


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


def test_heading_path_detects_chinese_section_numbering():
    """中文报告的章节标题有固定编号形式；正文句子即使以编号开头也不该误判。

    同一套规则同时供 PDF 解析器跟踪章节、供切分器判断能不能把标题挪到下一块，
    因此放在共享模块里，这里一并核对。
    """
    from src.headings import heading_level

    for text in (
        "三、绿证市场活力持续增强",
        "（一）交易规模实现翻两番",
        "第一章 总体要求",
        "2. 装机规模超预期增长",
    ):
        assert heading_level(text) is not None, text
    for text in (
        "0.4% A。",
        "截至2025 年底，全国新型储能累计装机规模13593 万千瓦，同比增长84.3%，储能时长呈逐年上升趋势。",
        "2025 年，全球新型储能新增装机规模约1.1 亿千瓦",
    ):
        assert heading_level(text) is None, text


def test_heading_stack_never_chains_same_level_headings():
    """同级标题不能互相套娃，二级标题也不该在缺少一级标题时冒充顶层。"""
    from src.headings import push_heading

    def titles(stack):
        """取出标题栈里非空的部分，便于直接比较。"""
        return [title for title in stack if title]

    stack = []
    # 文档开头直接出现二级标题：它应当独占一层，而不是落到第 0 层。
    push_heading(stack, "（一）世界新型储能发展", 2)
    assert titles(stack) == ["（一）世界新型储能发展"]
    # 再来一个同级标题，前一个必须被替换，不能串成父子。
    push_heading(stack, "（六）新型储能产业集群扩能提质", 2)
    assert titles(stack) == ["（六）新型储能产业集群扩能提质"]
    # 出现一级标题后，二级标题挂在它下面。
    push_heading(stack, "三、绿证市场活力持续增强", 1)
    push_heading(stack, "（一）交易规模实现翻两番", 2)
    assert titles(stack) == ["三、绿证市场活力持续增强", "（一）交易规模实现翻两番"]
    # 同级二级标题替换旧的二级标题，一级标题保留。
    push_heading(stack, "（二）参与主体数量显著增加", 2)
    assert titles(stack) == ["三、绿证市场活力持续增强", "（二）参与主体数量显著增加"]


def test_vision_settings_rejects_non_positive_batch_size(tmp_path):
    """批量大小必须是正整数；0 或负数会让分批循环永远不前进。"""
    from src.config import VisionSettings

    def build(batch_size):
        """用给定的批量大小构造视觉配置。"""
        return VisionSettings(
            path=tmp_path,
            max_new_tokens=256,
            prompt="描述",
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
