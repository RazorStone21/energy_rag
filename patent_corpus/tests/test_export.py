"""抽取、构建与输出测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from patent_corpus.export import (
    PatentRecord,
    build_corpus,
    iter_records,
    write_jsonl,
    write_text_corpus,
)
from patent_corpus.loader import load_table


@pytest.fixture
def mixed_export(tmp_path: Path) -> Path:
    """构造一份混合了相关与无关专利的导出文件。"""
    path = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "公开(公告)号": [
                "CN110123456A",  # 电力，应收录
                "CN110123457B",  # 风电，应收录
                "CN110123458C",  # 半导体无关键词，应筛除
                "CN110123459D",  # 食品，应筛除
                "CN110123460E",  # 光伏，应收录
                "CN110123461F",  # 无IPC但有关键词，应收录
                "CN110123462G",  # 无IPC无关键词，应筛除
                "CN110123456A",  # 与第一行重复，应去重
            ],
            "专利名称": [
                "一种配电网故障定位方法",
                "一种风机叶片检测装置",
                "一种半导体刻蚀工艺",
                "一种食品加工方法",
                "一种光伏组件清洗装置",
                "一种微电网能量管理方法",
                "一种包装盒结构",
                "一种配电网故障定位方法",
            ],
            "摘要": [
                "本发明公开了一种配电网故障定位方法，属于电力系统技术领域。",
                "本发明公开了一种风机叶片检测装置，用于风力发电机组。",
                "本发明公开了一种半导体刻蚀工艺。",
                "本发明公开了一种食品加工方法。",
                "本发明公开了一种光伏组件清洗装置。",
                "本发明公开了一种微电网能量管理方法。",
                "本发明公开了一种包装盒结构。",
                "本发明公开了一种配电网故障定位方法，属于电力系统技术领域。",
            ],
            "权利要求书": [
                "1.一种配电网故障定位方法，其特征在于包括步骤A。",
                "1.一种风机叶片检测装置，其特征在于包括探头。",
                "1.一种半导体刻蚀工艺，其特征在于包括步骤B。",
                "1.一种食品加工方法，其特征在于包括步骤C。",
                "1.一种光伏组件清洗装置，其特征在于包括刷头。",
                "1.一种微电网能量管理方法，其特征在于包括调度。",
                "1.一种包装盒结构，其特征在于包括盒体。",
                "1.一种配电网故障定位方法，其特征在于包括步骤A。",
            ],
            "IPC分类号": [
                "H02J13/00", "F03D17/00", "H01L21/02", "A23L1/00",
                "H02S40/10", "", "", "H02J13/00",
            ],
        }
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def test_抽取记录字段(mixed_export: Path):
    """逐行抽取应保留原始字段。"""
    table = load_table(mixed_export)
    records = list(iter_records(table))

    assert len(records) == 8
    assert records[0].pub_number == "CN110123456A"
    assert records[0].title == "一种配电网故障定位方法"
    assert "配电网" in records[0].abstract
    assert records[0].claims.startswith("1.")


def test_构建语料按领域筛选(mixed_export: Path):
    """应只保留能源电力领域的记录。"""
    records, stats = build_corpus([load_table(mixed_export)])

    assert stats.rows_seen == 8
    # 5 条相关（含被去重的那条计为重复）
    titles = [r.title for r in records]
    assert "一种配电网故障定位方法" in titles
    assert "一种风机叶片检测装置" in titles
    assert "一种光伏组件清洗装置" in titles
    assert "一种微电网能量管理方法" in titles
    assert "一种半导体刻蚀工艺" not in titles
    assert "一种食品加工方法" not in titles
    assert "一种包装盒结构" not in titles


def test_构建语料去重(mixed_export: Path):
    """同一件专利用不同批次导出时会产生重复，应去重。"""
    _, stats = build_corpus([load_table(mixed_export)])

    # 8 行里：4 条唯一且属领域（配电网、风机、光伏、微电网），
    # 3 条筛除（半导体、食品、包装盒），1 条重复
    assert stats.reasons.get("重复") == 1
    assert stats.rows_included == 4


def test_记录带判定依据(mixed_export: Path):
    """每条记录应保留领域判定的依据，便于事后核查。"""
    records, _ = build_corpus([load_table(mixed_export)])
    by_title = {r.title: r for r in records}

    assert "H02J" in by_title["一种配电网故障定位方法"].domain_reason
    assert "F03D" in by_title["一种风机叶片检测装置"].domain_reason
    assert "关键词" in by_title["一种微电网能量管理方法"].domain_reason


def test_文档ID优先用公开号():
    """文档 ID 应稳定，优先取公开号。"""
    record = PatentRecord(pub_number="CN110123456A", title="标题")
    assert record.doc_id == "CN110123456A"

    fallback = PatentRecord(app_number="CN201910123456", title="标题")
    assert fallback.doc_id == "CN201910123456"


def test_文档ID剔除非法字符():
    """公开号里若含路径字符要剔除，避免写文件时出事。"""
    record = PatentRecord(pub_number="CN/110:123*A")
    assert "/" not in record.doc_id
    assert "*" not in record.doc_id


def test_渲染语料包含各段():
    """渲染出的文本应含标题、摘要、权利要求书三段。"""
    record = PatentRecord(
        pub_number="CN1A",
        title="一种配电网方法",
        abstract="本发明公开了一种配电网方法。",
        claims="1.一种配电网方法，其特征在于。",
        ipc="H02J13/00",
        applicant="某某电力公司",
    )
    text = record.render_text()

    assert "【发明名称】" in text
    assert "【摘要】" in text
    assert "【权利要求书】" in text
    assert "一种配电网方法" in text
    assert "IPC：H02J13/00" in text


def test_渲染可排除说明书():
    """--no-description 时应不含说明书段。"""
    record = PatentRecord(title="标题", abstract="摘要", description="说明书正文")
    text = record.render_text(include_description=False)

    assert "【说明书】" not in text
    assert "说明书正文" not in text


def test_空记录不渲染():
    """没有任何内容的记录不应产出空文件。"""
    assert PatentRecord().render_text() == ""


def test_写出JSONL(mixed_export: Path, tmp_path: Path):
    """JSONL 应可逐行解析。"""
    records, _ = build_corpus([load_table(mixed_export)])
    out = tmp_path / "out.jsonl"
    count = write_jsonl(records, out)

    assert count == len(records)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == count

    first = json.loads(lines[0])
    assert first["title"]
    assert "domain_reason" in first


def test_写出文本语料(mixed_export: Path, tmp_path: Path):
    """每件一个 txt，并统计字节数。"""
    records, _ = build_corpus([load_table(mixed_export)])
    out_dir = tmp_path / "corpus"
    stats = write_text_corpus(records, out_dir, single_file=tmp_path / "all.txt")

    assert stats.written == len(records)
    assert stats.bytes_written > 0
    assert len(list(out_dir.glob("*.txt"))) == len(records)
    assert (tmp_path / "all.txt").exists()


def test_统计不被覆盖(mixed_export: Path, tmp_path: Path):
    """写出语料时，build 阶段的筛选计数不能被写出的统计覆盖掉。"""
    records, stats = build_corpus([load_table(mixed_export)])
    seen_before = stats.rows_seen

    write_text_corpus(records, tmp_path / "c", stats=stats)

    assert stats.rows_seen == seen_before
    assert stats.written == len(records)
    # 筛选组的计数不应被写出环节重复累加
    assert stats.with_claims <= stats.rows_included


def test_字节统计用于估算到GB的距离(mixed_export: Path, tmp_path: Path):
    """统计摘要应给出到 1GB 的件数估算。"""
    records, _ = build_corpus([load_table(mixed_export)])
    stats = write_text_corpus(records, tmp_path / "c", single_file=None)

    summary = stats.summary()
    assert "总字节数" in summary
    assert "1GB" in summary
