"""使用真实 Markdown 语法解析器验证内容结构，通过替身隔离模型和数据库。"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.export_chunks import _export_record, export, to_markdown
from src.chunker import Chunker
from src.config import load_settings
from src.context_builder import ContextBuilder
from src.parsers.markdown import MarkdownParser
from src.retrieval.hybrid import rrf_fuse
from src.schemas import SearchHit
from src.storage.milvus import MilvusStore


def parse_markdown(tmp_path, content):
    """写入带 BOM 的 Markdown 样本，返回实际语法解析结果。"""
    path = tmp_path / "能源报告.MD"
    path.write_text(content, encoding="utf-8-sig")
    result = MarkdownParser(document_factory=SimpleNamespace).parse(path)
    assert not result.errors, result.errors
    return result


def test_markdown_preserves_structure_locations_and_code(tmp_path):
    """标题、列表、代码和表格按原文提取，代码中的伪标题或伪表格不改变章节。"""
    content = (
        "文档说明\n\n# 能源报告\n\n## 电力\n\n- 火电\n  - 煤电\n\n"
        "```python\n# 不是章节\nprint('a|b')\n| A | B |\n| --- | --- |\n```\n\n"
        "| 类型 | 数值 |\n| --- | --- |\n| 发电量 | 10 |\n\n"
        "## 燃气\n\n![统计图](missing.png)\n\n> 保留引用\n"
    )
    parsed = parse_markdown(tmp_path, content)
    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.page_content == "| 类型 | 数值 |\n| --- | --- |\n| 发电量 | 10 |\n"
    assert table.metadata["heading_path"] == "能源报告 > 电力"
    code = next(d for d in parsed.texts if d.metadata["block_kind"] == "code")
    assert "# 不是章节\nprint('a|b')" in code.page_content
    assert code.page_content.startswith("```python\n")
    assert code.page_content.endswith("```\n")
    assert code.metadata["heading_path"] == "能源报告 > 电力"
    assert any("- 火电\n  - 煤电" in d.page_content for d in parsed.texts)
    assert parsed.texts[-1].metadata["heading_path"] == "能源报告 > 燃气"
    assert "![统计图](missing.png)" in parsed.texts[-1].page_content
    assert not parsed.figures

    # 每个非空原文行恰好属于一个元素，验证提取没有遗漏或重复。
    lines = content.splitlines(keepends=True)
    covered = []
    for document in parsed.texts + parsed.tables:
        start = document.metadata["line_start"] - 1
        end = document.metadata["line_end"]
        assert document.page_content == "".join(lines[start:end])
        assert "page" not in document.metadata
        covered.extend(range(start, end))
    assert len(covered) == len(set(covered))
    assert all(index in covered for index, line in enumerate(lines) if line.strip())


def test_setext_headings_nested_blocks_and_unicode_line_separator(tmp_path):
    """支持下划线式标题，引用内标题不改变章节，Unicode 分隔符不影响原文定位。"""
    content = (
        "总标题\n======\n\n> # 引用标题\n> | a | b |\n> | --- | --- |\n"
        "> | 1 | 2 |\n\n正文\u2028仍在同一行\n\n子标题\n------\n\n末尾正文\n"
    )
    parsed = parse_markdown(tmp_path, content)
    assert len(parsed.texts) == 2 and not parsed.tables
    assert parsed.texts[0].metadata["heading_path"] == "总标题"
    assert "> | 1 | 2 |" in parsed.texts[0].page_content
    assert parsed.texts[-1].metadata["heading_path"] == "总标题 > 子标题"
    assert parsed.texts[-1].metadata["line_start"] == 11


@pytest.mark.parametrize(
    "content", ["普通段落\n\n没有标题", "```\n未闭合的代码", "    print('缩进代码')\n"]
)
def test_markdown_without_headings_or_closed_fence_keeps_content(tmp_path, content):
    """无标题、未闭合围栏和缩进代码都按 Markdown 规则保留内容。"""
    parsed = parse_markdown(tmp_path, content)
    assert len(parsed.texts) == 1
    assert parsed.texts[0].page_content == content


@pytest.mark.parametrize("content", [b"", b" \n", b"\xff", b"binary\x00text"])
def test_markdown_invalid_input_is_reported(tmp_path, content):
    """空白、无效编码和空字符不会被作为正常 Markdown 入库。"""
    path = tmp_path / "bad.md"
    path.write_bytes(content)
    result = MarkdownParser(document_factory=SimpleNamespace).parse(path)
    assert result.errors and not result.texts and not result.tables


def test_markdown_dependency_failure_is_explicit(tmp_path, monkeypatch):
    """依赖不可用时明确报告解析失败，不回退成丢失结构的普通文本。"""
    path = tmp_path / "report.md"
    path.write_text("# 能源报告", encoding="utf-8")
    parser = MarkdownParser(document_factory=SimpleNamespace)
    monkeypatch.setattr(parser, "_load_parser", Mock(side_effect=ImportError("missing parser")))
    assert parser.parse(path).errors == ["markdown: missing parser"]


def test_chunker_preserves_code_tables_and_section_metadata(tmp_path):
    """语义切分只收到普通正文，代码和表格整体保留，章节信息继续用于引用。"""
    parsed = parse_markdown(
        tmp_path,
        "# 能源报告\n政策正文。\n\n```python\nprint('energy')\n```\n\n"
        "| 项目 | 数值 |\n| --- | --- |\n| 能源 | 10 |\n",
    )
    settings = load_settings(Path(__file__).resolve().parents[2] / "config.toml")
    splitter = Mock()
    splitter.split_documents.side_effect = lambda docs: docs
    chunker = Chunker(Mock(), settings.splitting, splitter_factory=lambda *a, **k: splitter)
    chunks = chunker.split(parsed)
    code = next(d for d in parsed.texts if d.metadata["block_kind"] == "code")
    assert code in chunks and parsed.tables[0] in chunks
    assert all(
        d.metadata["block_kind"] == "section" for d in splitter.split_documents.call_args.args[0]
    )
    context = ContextBuilder("{context}\n{question}").build(
        "能源情况？", [SearchHit(d) for d in chunks]
    )
    assert "标题: 能源报告" in context.prompt
    assert "所属原文块: 第" in context.prompt
    assert "第?页" not in context.prompt


def test_rrf_does_not_merge_equal_content_at_different_markdown_locations(tmp_path):
    """相同标题和正文出现在不同位置时分别保留，同一位置的跨路召回合并。"""
    parsed = parse_markdown(tmp_path, "# 概况\n能源情况相同\n\n# 概况\n能源情况相同\n")
    first, second = [SearchHit(d) for d in parsed.texts]
    assert first.document.page_content == second.document.page_content
    hits = rrf_fuse([first, second], [first, second], 10, 60)
    assert len(hits) == 2
    assert hits[0].rrf_score == pytest.approx(2 / 61)


def test_milvus_dynamic_metadata_and_legacy_collection_check(tmp_path):
    """新集合启用动态字段，旧固定字段集合在增量写入前明确要求重建。"""
    settings = load_settings(Path(__file__).resolve().parents[2] / "config.toml")
    settings = replace(settings.milvus, connection_args={"uri": str(tmp_path / "index.db")})
    backend, factory, embedder = Mock(), Mock(), Mock()
    # 动态字段开关只能从底层 MilvusClient 拿到：langchain_milvus 的 col.schema
    # 只暴露 fields，没有这个属性。
    backend.client.describe_collection.return_value = {"enable_dynamic_field": False}
    factory.return_value = backend
    store = MilvusStore(embedder, settings, factory=factory)
    with pytest.raises(RuntimeError, match="python main.py build"):
        store.validate_update()
    backend.delete.assert_not_called()
    assert factory.call_args.kwargs["enable_dynamic_field"] is True
    backend.client.describe_collection.return_value = {"enable_dynamic_field": True}
    store.validate_update()
    # 服务端没有回答这个字段时不能凭猜测拦截，否则会把正常的增量更新挡在门外。
    backend.client.describe_collection.return_value = {}
    store.validate_update()
    store.replace_all([SimpleNamespace(page_content="正文", metadata={"source": "report.md"})])
    assert factory.from_documents.call_args.kwargs["enable_dynamic_field"] is True


def test_markdown_location_survives_json_csv_and_preview_export(tmp_path):
    """JSON、CSV 和 Markdown 预览保留章节与行号，混合 PDF 记录允许这些字段缺失。"""
    import csv
    import json

    parsed = parse_markdown(tmp_path, "# 能源报告\n政策正文。\n")
    record = _export_record(parsed.texts[0])
    records = [record, {"source": "report.pdf", "page": 3, "type": "text", "content": "PDF"}]
    json_path, csv_path = tmp_path / "chunks.json", tmp_path / "chunks.csv"
    export(records, "json", json_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["heading_path"] == "能源报告"
    export(records, "csv", csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["line_start"] == "1"
    assert rows[1]["heading_path"] == ""
    preview = to_markdown(records)
    assert "标题: 能源报告" in preview and "所属原文块: 第1—2行" in preview
    assert "第3页" in preview


def test_top_level_list_stays_whole_with_its_intro(tmp_path):
    """顶层列表整体成块，前面的引出语一起带走。

    列表项是一组并列条目，按句子切分会把其中几条和其余条拆开；
    而引出语（「……遵循以下原则：」）单独成块又太短，会被长度门槛当残片过滤掉，
    整句凭空消失。两者都要求引出语跟着列表走。
    """
    content = (
        "## 总则\n\n**第三条** 数据分类分级遵循以下原则：\n\n"
        "- **依据明确。** 以数据的特性和用途作为分类的主要依据。\n"
        "- **边界清晰。** 所有数据均有确定的类别、等级。\n"
        "- **就高从严。** 涉及多个方面的安全风险时按最高级别确定。\n\n"
        "## 附则\n\n末尾正文。\n"
    )
    parsed = parse_markdown(tmp_path, content)
    lists = [d for d in parsed.texts if d.metadata["block_kind"] == "list"]
    assert len(lists) == 1
    block = lists[0]
    assert block.page_content.startswith("**第三条** 数据分类分级遵循以下原则：")
    for item in ("依据明确", "边界清晰", "就高从严"):
        assert item in block.page_content
    assert block.metadata["heading_path"] == "总则"


def test_list_intro_never_swallows_a_heading(tmp_path):
    """列表紧跟标题时不能把标题并进列表，否则标题会从章节结构里消失。"""
    content = "## 识别规则\n\n- 第一条规则；\n- 第二条规则。\n"
    parsed = parse_markdown(tmp_path, content)
    lists = [d for d in parsed.texts if d.metadata["block_kind"] == "list"]
    assert len(lists) == 1
    assert not lists[0].page_content.startswith("#")
    assert "第一条规则" in lists[0].page_content
    assert lists[0].metadata["heading_path"] == "识别规则"


def test_ordered_list_is_also_a_whole_block(tmp_path):
    """有序列表与无序列表同样处理。"""
    content = "## 流程\n\n按以下顺序执行：\n\n1. 先核对数据来源；\n2. 再判定数据等级。\n"
    parsed = parse_markdown(tmp_path, content)
    lists = [d for d in parsed.texts if d.metadata["block_kind"] == "list"]
    assert len(lists) == 1
    assert lists[0].page_content.startswith("按以下顺序执行：")
    assert "1. 先核对数据来源；" in lists[0].page_content
