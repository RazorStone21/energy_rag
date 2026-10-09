"""MinerU 解析路径的回归测试：不加载模型，全部用内容项替身与假视觉模型。

MinerU 的 `content_list` 是它对外承诺的稳定接口，这里按它真实的字段形状构造
替身（`type`/`text`/`text_level`/`table_body`/`img_path`/各 `*_caption`/`bbox`/`page_idx`），
这样即使 MinerU 内部结构变了，只要这个接口不变，测试就仍然说明问题。
"""

from __future__ import annotations

import base64
import io
import re
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from src.config import MineruSettings, VisionSettings
from src.parsers.mineru_engine import (
    MineruEngine,
    MineruParseError,
    MineruUnavailable,
    decode_image,
)
from src.parsers.mineru_pdf import (
    blocks_to_parse_result,
    caption_of,
    describe_figures,
    html_table_to_markdown,
    normalize_caption,
    parse_table_html,
)
from src.parsers.pdf import PDFParser


def _image_data_url(size=(120, 80), color="white") -> str:
    """造一张内联图片，形状与 MinerU 的 img_path 一致（data URL + base64）。"""
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def text_item(text, page_idx=0, *, level=None, bbox=(0.1, 0.1, 0.9, 0.2)):
    """构造一个正文项；level 表示它是标题（MinerU 的 text_level）。"""
    item = {"type": "text", "text": text, "bbox": list(bbox), "page_idx": page_idx}
    if level is not None:
        item["text_level"] = level
    return item


def table_item(html, page_idx=0, *, caption=None, bbox=(0.1, 0.3, 0.9, 0.6)):
    """构造一个表格项；题注按 MinerU 的列表形式给出。"""
    item = {"type": "table", "table_body": html, "bbox": list(bbox), "page_idx": page_idx}
    if caption:
        item["table_caption"] = [caption]
    return item


def figure_item(kind="chart", page_idx=0, *, caption=None, bbox=(0.1, 0.3, 0.9, 0.8)):
    """构造一个图表项（chart 或 image），带内联图片。"""
    item = {
        "type": kind,
        "img_path": _image_data_url(),
        "bbox": list(bbox),
        "page_idx": page_idx,
    }
    if caption:
        item["chart_caption" if kind == "chart" else "image_caption"] = [caption]
    return item


class FakeVision:
    """记录批量与逐张调用，用来验证分批策略和降级路径。"""

    available = True

    def __init__(self, fail_batches=False):
        """保存是否让批量接口失败，并准备记录调用。"""
        self.fail_batches = fail_batches
        self.batches = []
        self.singles = []

    def describe_batch(self, images, prompts):
        """记录一次批量描述调用，按开关决定是否抛错。"""
        if self.fail_batches:
            raise RuntimeError("显存不足")
        self.batches.append((list(images), list(prompts)))
        return [f"批量描述{prompt}" for prompt in prompts]

    def describe(self, image, prompt):
        """记录一次单张描述调用。"""
        self.singles.append(prompt)
        return f"单张描述{prompt}"


def vision_settings(tmp_path, batch_size=4):
    """构造视觉配置；图片尺寸与渲染参数已随旧实现移除，只剩模型与批量。"""
    return VisionSettings(
        path=tmp_path,
        max_new_tokens=256,
        prompt="请描述这张图。",
        batch_size=batch_size,
    )


# ---------------- 表格：HTML → Markdown ----------------


def test_table_html_rejects_degenerate_tables():
    """单列或单行的「表格」是被误判的正文，转出来是空列表。"""
    settings = dict(rows_per_chunk=50, min_rows=2, min_columns=2)
    assert html_table_to_markdown("<table><tr><td>通知抬头</td><td>国务院</td></tr></table>", "", **settings) == []
    assert html_table_to_markdown(
        "<table><tr><td>一、发展形势</td></tr><tr><td>二、重点任务</td></tr></table>", "", **settings
    ) == []


def test_table_markdown_escapes_cells_and_keeps_column_counts():
    """竖线、反斜线与硬换行必须转义，否则单元格内容会撑破表格结构。"""
    html = (
        "<table>"
        "<tr><td>项目</td><td>说明</td></tr>"
        "<tr><td>a|b</td><td>第一行<br/>第二行</td></tr>"
        "</table>"
    )
    chunks = html_table_to_markdown(html, "", rows_per_chunk=50, min_rows=2, min_columns=2)
    assert len(chunks) == 1
    markdown, row_start, row_end = chunks[0]
    assert (row_start, row_end) == (1, 1)
    lines = markdown.split("\n")
    # 首行是片段自己的说明文字（不含竖线），表格部分每一行的分隔竖线数量必须一致，
    # Markdown 才能渲染成表格；单元格里的竖线是转义过的，不算分隔符。
    separators = [len(re.findall(r"(?<!\\)\|", line)) for line in lines[1:]]
    assert len(set(separators)) == 1
    assert lines[0].startswith("表格（第 1—1 行）")
    assert r"a\|b" in markdown
    assert "第一行 第二行" in markdown


def test_table_markdown_splits_rows_and_repeats_header():
    """超长表格按行切片，每片重复表头并把行号写进首行说明。"""
    rows = "".join(f"<tr><td>{index}</td><td>值{index}</td></tr>" for index in range(1, 6))
    html = f"<table><tr><td>序号</td><td>数值</td></tr>{rows}</table>"
    chunks = html_table_to_markdown(
        html, "表1 示例", rows_per_chunk=2, min_rows=2, min_columns=2
    )
    assert [(start, end) for _, start, end in chunks] == [(1, 2), (3, 4), (5, 5)]
    for markdown, _, _ in chunks:
        lines = markdown.split("\n")
        assert lines[0].startswith("表格：表1 示例（第")
        # 表头与分隔行在每片都要出现，否则后面的片段读不出列含义。
        assert lines[1] == "| 序号 | 数值 |" and lines[2] == "| --- | --- |"


def test_table_html_expands_merged_cells():
    """跨行跨列按占位补齐：不然行的列数会塌陷，读取时列会串位。"""
    html = (
        "<table>"
        "<tr><td rowspan='2'>在运</td><td>1</td><td>中国</td></tr>"
        "<tr><td>2</td><td>美国</td></tr>"
        "</table>"
    )
    rows = parse_table_html(html)
    assert rows == [["在运", "1", "中国"], ["", "2", "美国"]]


def test_table_caption_is_normalized():
    """题注里汉字之间的排版空白要压掉，编号与汉字之间的空格保留。"""
    assert normalize_caption("专 栏 1 绿证发展情况") == "专栏 1 绿证发展情况"
    # 列表形式的题注按空格连接后同样过一遍归一化，因此"指数 续表"里的空格会消失。
    assert caption_of({"table_caption": ["表 2-1 综合指数", "续表"]}) == "表 2-1 综合指数续表"
    assert caption_of({"chart_caption": []}) == ""
    assert caption_of({}) == ""


# ---------------- 内容项 → 片段 ----------------


def test_blocks_skip_repeated_page_blocks(tmp_path):
    """页眉、页脚、页码与目录项每页重复或只是导航，不能进索引。"""
    items = [
        {"type": "header", "text": "中国能源发展报告", "page_idx": 0, "bbox": [0, 0, 1, 0.05]},
        {"type": "page_number", "text": "12", "page_idx": 0, "bbox": [0.9, 0.9, 0.95, 0.95]},
        {"type": "index", "text": "目录 第一章 ……", "page_idx": 0, "bbox": [0.1, 0.1, 0.9, 0.5]},
        text_item("正文内容", page_idx=0),
        {"type": "page_footnote", "text": "注：数据来源见附表", "page_idx": 0, "bbox": [0, 0.9, 1, 0.95]},
    ]
    result = blocks_to_parse_result(
        items, "报告.pdf", None, vision_settings(tmp_path), MineruSettings()
    )
    assert [document.page_content for document in result.texts] == ["正文内容"]
    assert result.tables == [] and result.figures == []


def test_blocks_ignore_mineru_title_without_numbering(tmp_path):
    """MinerU 会把加粗长句标成标题，这种块不能进 heading_path。

    实测它把「截至2025年底，我国新型储能装机比"十三五"末增长超40倍」当成
    paragraph_title，若照单全收，引用标签里会出现整句正文冒充章节名。
    标题识别只认项目自己的编号规则，与切分器共用同一口径。
    """
    items = [
        text_item("一、总体要求", page_idx=0, level=2),
        text_item("截至2025年底，我国新型储能装机比“十三五”末增长超40倍", page_idx=0, level=2),
        text_item("新型储能装机规模五年增长超过四十倍。", page_idx=0),
    ]
    result = blocks_to_parse_result(
        items, "储能.pdf", None, vision_settings(tmp_path), MineruSettings()
    )
    assert result.texts[0].metadata["heading_path"] == "一、总体要求"
    # 被误标的句子只算正文：它仍属于"一、总体要求"这一节，但不会替换章节名。
    assert result.texts[1].metadata["heading_path"] == "一、总体要求"
    assert result.texts[2].metadata["heading_path"] == "一、总体要求"


def test_blocks_add_page_index_heading_and_block_index(tmp_path):
    """页码从 1 起、页内序号递增、编号标题进入 heading_path。"""
    items = [
        text_item("加快构建新型电力系统行动方案", page_idx=0, level=1, bbox=(0.2, 0.1, 0.8, 0.15)),
        text_item("一、总体要求", page_idx=0, level=2, bbox=(0.2, 0.2, 0.5, 0.25)),
        text_item("以习近平新时代中国特色社会主义思想为指导。", page_idx=0, bbox=(0.1, 0.3, 0.9, 0.5)),
        text_item("二、电力系统稳定保障行动", page_idx=1, level=2, bbox=(0.2, 0.1, 0.6, 0.15)),
        text_item("优化加强电网主网架。", page_idx=1, bbox=(0.1, 0.2, 0.9, 0.4)),
    ]
    result = blocks_to_parse_result(
        items, "方案.pdf", None, vision_settings(tmp_path), MineruSettings()
    )
    first = result.texts[0].metadata
    # 文档标题（text_level=1）不进 heading_path，它属于文件名。
    assert "heading_path" not in first and first["page"] == 1 and first["block_index"] == 1
    body = result.texts[2].metadata
    assert body["page"] == 1 and body["block_index"] == 3
    assert body["heading_path"] == "一、总体要求"
    assert body["block_kind"] == "section" and body["parser"] == "mineru"
    last = result.texts[4].metadata
    assert last["page"] == 2 and last["heading_path"] == "二、电力系统稳定保障行动"


def test_blocks_build_tables_with_caption_and_page(tmp_path):
    """表格片段带题注、页内序号与行号范围，正文是转好的 Markdown。"""
    html = (
        "<table><tr><td>国家</td><td>数量</td></tr>"
        "<tr><td>中国</td><td>59</td></tr><tr><td>美国</td><td>94</td></tr></table>"
    )
    result = blocks_to_parse_result(
        [table_item(html, page_idx=14, caption="表 2-1 世界主要核电国家机组情况")],
        "核电.pdf",
        None,
        vision_settings(tmp_path),
        MineruSettings(),
    )
    assert len(result.tables) == 1
    metadata = result.tables[0].metadata
    assert metadata["type"] == "table" and metadata["page"] == 15
    assert metadata["caption"] == "表 2-1 世界主要核电国家机组情况"
    assert (metadata["row_start"], metadata["row_end"]) == (1, 2)
    assert metadata["block_kind"] == "table" and metadata["block_index"] == 1
    assert result.tables[0].page_content.startswith("表格：表 2-1")


def test_blocks_describe_figures_in_batches(tmp_path):
    """图表交给视觉模型描述，题注拼进提示词，描述写进正文。"""
    vision = FakeVision()
    items = [
        figure_item("chart", page_idx=13, caption="图1 全球新型储能累计装机规模"),
        figure_item("image", page_idx=15),
    ]
    result = blocks_to_parse_result(
        items, "储能.pdf", vision, vision_settings(tmp_path), MineruSettings()
    )
    assert len(result.figures) == 2
    first = result.figures[0]
    assert first.metadata["type"] == "figure" and first.metadata["page"] == 14
    assert first.metadata["caption"] == "图1 全球新型储能累计装机规模"
    assert first.page_content.startswith("图1 全球新型储能累计装机规模\n")
    # 题注存在时进入提示词；没有题注的那张按原样提问。
    prompts = vision.batches[0][1]
    assert prompts[0].startswith("这张图的标题是「图1 全球新型储能累计装机规模」")
    assert prompts[1] == "请描述这张图。"


def test_blocks_skip_decorative_small_figures(tmp_path):
    """页边的小图标与分隔线不描述：描述它们只会挤占检索候选位。"""
    vision = FakeVision()
    items = [figure_item("image", page_idx=0, bbox=(0.01, 0.01, 0.05, 0.03))]
    result = blocks_to_parse_result(
        items, "报告.pdf", vision, vision_settings(tmp_path), MineruSettings()
    )
    assert result.figures == []
    assert vision.batches == []


def test_blocks_without_vision_model_keep_text_and_tables(tmp_path):
    """视觉模型不可用时仍产出正文与表格，只是没有图片描述。"""
    items = [text_item("正文", page_idx=0), figure_item("chart", page_idx=0)]
    result = blocks_to_parse_result(
        items, "报告.pdf", None, vision_settings(tmp_path), MineruSettings()
    )
    assert len(result.texts) == 1 and result.figures == []


# ---------------- 批量图片描述 ----------------


def test_describe_figures_splits_into_batches_and_keeps_order():
    """按 batch_size 切批，每批只调一次批量接口，返回顺序与输入一致。"""
    vision = FakeVision()
    items = [(f"图{i}", f"提示{i}") for i in range(5)]
    out = describe_figures(vision, items, batch_size=2)
    assert out == [f"批量描述提示{i}" for i in range(5)]
    assert [len(images) for images, _ in vision.batches] == [2, 2, 1]
    assert vision.singles == []


def test_describe_figures_falls_back_to_one_by_one(caplog):
    """整批失败时退回逐张：一张图出问题不该让同批其他图也丢了描述。"""
    vision = FakeVision(fail_batches=True)
    out = describe_figures(vision, [(f"图{i}", f"提示{i}") for i in range(3)], batch_size=2)
    assert out == [f"单张描述提示{i}" for i in range(3)]
    assert "退回逐张重试" in caplog.text


def test_describe_figures_reports_progress():
    """进度按批上报，让长时间的文件级解析仍能看到推进。"""
    notes = []
    describe_figures(
        FakeVision(), [(f"图{i}", f"提示{i}") for i in range(5)], batch_size=2, on_status=notes.append
    )
    assert notes == ["描述第 1—2 张图表", "描述第 3—4 张图表", "描述第 5—5 张图表"]


# ---------------- PDFParser：错误语义 ----------------


class FakeEngine:
    """MinerU 引擎替身：只提供 PDFParser 用到的三个接口。"""

    def __init__(self, items=None, error=None, failures=0):
        """按 (内容项, 抛出的异常, 连续失败数) 构造替身。"""
        self.items = items if items is not None else []
        self.error = error
        self.consecutive_failures = failures

    def parse_items(self, path):
        """返回预设内容项，或抛出预设异常。"""
        if self.error is not None:
            raise self.error
        return self.items

    def release(self):
        """记录释放调用（本组测试不校验，仅为接口完整）。"""


def test_pdf_parser_reports_single_file_failure_as_error(tmp_path):
    """单份文件失败写进 errors：入库流程据此保留该文件的旧索引。"""
    parser = PDFParser(
        None,
        vision_settings(tmp_path),
        FakeEngine(error=MineruParseError("a.pdf: 版面解析失败")),
        MineruSettings(),
    )
    result = parser.parse(tmp_path / "a.pdf")
    assert result.texts == [] and result.tables == [] and result.figures == []
    assert result.errors == ["mineru: a.pdf: 版面解析失败"]


def test_pdf_parser_stops_when_environment_keeps_failing(tmp_path):
    """连续失败达到上限时抛 MineruUnavailable：环境问题不能让整轮构建逐份失败。"""
    parser = PDFParser(
        None,
        vision_settings(tmp_path),
        FakeEngine(items=[text_item("正文")], failures=3),
        MineruSettings(max_consecutive_failures=3),
    )
    with pytest.raises(MineruUnavailable, match="连续 3 份文件解析失败"):
        parser.parse(tmp_path / "a.pdf")


def test_pdf_parser_reports_empty_document(tmp_path):
    """解析结果为空说明提取没成功，按错误上报而不是写入空索引。"""
    parser = PDFParser(None, vision_settings(tmp_path), FakeEngine(items=[]), MineruSettings())
    result = parser.parse(tmp_path / "a.pdf")
    assert result.errors == ["mineru: 没有解析出任何版面块"]


def test_pdf_parser_can_be_disabled(tmp_path):
    """配置关闭 PDF 解析时明确报错，而不是静默产出空索引。"""
    parser = PDFParser(
        None, vision_settings(tmp_path), FakeEngine(items=[text_item("正文")]), MineruSettings(enabled=False)
    )
    result = parser.parse(tmp_path / "a.pdf")
    assert result.errors and "[mineru].enabled" in result.errors[0]


def test_pdf_parser_fingerprint_tracks_tier(tmp_path):
    """指纹随档位变化：换档位产出的版面块不同，必须触发全量重建。"""
    parser = PDFParser(None, vision_settings(tmp_path), FakeEngine(), MineruSettings(tier="basic"))
    assert parser.fingerprint == "mineru-basic"


# ---------------- MineruEngine：缓存与错误包装 ----------------


def _stub_mineru(monkeypatch, items, calls):
    """把 mineru 与 mineru.render 换成替身，并跳过模型就绪检查。"""
    fake = SimpleNamespace(
        parse=lambda path, **kwargs: calls.append((path, kwargs)) or SimpleNamespace(middle_json=object())
    )
    render = SimpleNamespace(render_content_list=lambda middle: items)
    monkeypatch.setitem(sys.modules, "mineru", fake)
    monkeypatch.setitem(sys.modules, "mineru.render", render)
    monkeypatch.setattr(MineruEngine, "preflight", lambda self: None)


def test_engine_reuses_result_until_the_file_changes(tmp_path, monkeypatch):
    """同一文件只解析一次；文件被改写（mtime/大小变化）后重新解析。"""
    calls = []
    _stub_mineru(monkeypatch, [text_item("正文")], calls)
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")

    engine = MineruEngine(MineruSettings())
    first = engine.parse_items(path)
    second = engine.parse_items(path)
    assert first == second == [text_item("正文")]
    assert len(calls) == 1
    assert calls[0][1]["tier"] == "basic" and calls[0][1]["page_range"] == "all"

    path.write_bytes(b"%PDF-1.4 longer")
    engine.parse_items(path)
    assert len(calls) == 2


def test_engine_wraps_failure_and_counts_it(tmp_path, monkeypatch):
    """解析异常包装成 MineruParseError 并累计连续失败数。"""

    def boom(path, **kwargs):
        """模拟 MinerU 抛错。"""
        raise RuntimeError("模型缺失")

    monkeypatch.setitem(sys.modules, "mineru", SimpleNamespace(parse=boom))
    monkeypatch.setitem(sys.modules, "mineru.render", SimpleNamespace(render_content_list=lambda m: []))
    monkeypatch.setattr(MineruEngine, "preflight", lambda self: None)

    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-1.4")
    engine = MineruEngine(MineruSettings())
    with pytest.raises(MineruParseError, match="模型缺失"):
        engine.parse_items(path)
    assert engine.consecutive_failures == 1


def test_decode_image_returns_none_for_unusable_input():
    """图片字段不是 data URL 或无法解码时返回 None，由调用方跳过这一张。"""
    assert decode_image("images/page_1_chart_2.jpg") is None
    assert decode_image("") is None
    assert decode_image("data:image/png;base64,不是base64") is None
    assert decode_image(_image_data_url((10, 10))) is not None


def test_caption_strips_markdown_emphasis():
    """MinerU 会把续表题注写成 **续表**，引用标签里不该出现星号。"""
    assert normalize_caption("**续表**") == "续表"
    assert caption_of({"table_caption": ["**续表**"]}) == "续表"
    # 正常题注不受影响，编号与汉字之间的空格保留。
    assert normalize_caption("表 2-1 综合指数") == "表 2-1 综合指数"
