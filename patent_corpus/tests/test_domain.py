"""领域筛选与文本清洗测试。

筛选是决定语料质量的关键：漏筛会把无关专利混进来，过筛会丢掉目标数据。
这里对每类 IPC 的判定都做覆盖。
"""

from __future__ import annotations

import pytest

from patent_corpus.domain import (
    classify_domain,
    clean_claims,
    clean_text,
    has_domain_keyword,
    normalize_for_claims_split,
    parse_ipc_codes,
)

# ---------------- IPC 解析 ----------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("H02J13/00", ["H02J"]),
        ("H02J 3/38", ["H02J"]),
        ("H02J3/38;H02M7/48", ["H02J", "H02M"]),
        ("H02J3/38, F03D9/00", ["H02J", "F03D"]),
        ("", []),
        ("无", []),
    ],
)
def test_解析IPC分类号(raw: str, expected: list[str]):
    """IPC 文本形态多样，都应正确抽出小类号。"""
    assert parse_ipc_codes(raw) == expected


def test_解析IPC去重():
    """同一个分类号重复出现只保留一次。"""
    assert parse_ipc_codes("H02J3/38;H02J13/00") == ["H02J"]


# ---------------- 领域判定 ----------------


def test_严格分类直接收录():
    """H02（发电变电配电）命中即收录，无需关键词。"""
    hit, reason = classify_domain("H02J13/00", "一种配电网监测方法")

    assert hit is True
    assert "H02J" in reason


def test_风电分类直接收录():
    """F03D（风力发电机）是严格分类。"""
    hit, reason = classify_domain("F03D17/00", "一种风机叶片检测装置")

    assert hit is True
    assert "F03D" in reason


def test_储能电池分类直接收录():
    """H01M（电池）是严格分类。"""
    hit, _ = classify_domain("H01M10/613", "一种电池热管理系统")

    assert hit is True


def test_核能分类直接收录():
    """G21（核工程）是严格分类。"""
    hit, _ = classify_domain("G21C3/00", "一种反应堆燃料组件")

    assert hit is True


def test_宽泛分类无关键词时筛除():
    """H01L（半导体）太宽，无领域关键词时不应收录。

    这条很关键：如果不加限制，所有芯片专利都会被当成光伏收进来。
    """
    hit, reason = classify_domain("H01L21/02", "一种半导体刻蚀工艺")

    assert hit is False
    assert "宽泛分类" in reason


def test_宽泛分类有领域关键词时收录():
    """H01L 配合「光伏」关键词则应收录。"""
    hit, reason = classify_domain("H01L31/18", "一种光伏电池片制备方法")

    assert hit is True
    assert "关键词" in reason


def test_无关分类筛除():
    """食品类专利应被筛除。"""
    hit, _ = classify_domain("A23L1/00", "一种食品加工方法")

    assert hit is False


def test_无IPC时退回关键词判定():
    """没有 IPC 字段时用关键词兜底，并如实说明依据。"""
    hit, reason = classify_domain("", "一种微电网能量管理方法")

    assert hit is True
    assert "无 IPC" in reason


def test_无IPC且无关键词时筛除():
    """既无 IPC 又无关键词，无法判定为相关领域。"""
    hit, _ = classify_domain("", "一种包装盒结构")

    assert hit is False


def test_关键词在摘要里也能命中():
    """关键词判定应覆盖摘要和权利要求文本。"""
    hit, _ = classify_domain("", "一种控制方法", "应用于光伏逆变器的控制方法")

    assert hit is True


def test_关键词检测返回命中的词():
    """返回具体命中的关键词，便于解释判定依据。"""
    assert has_domain_keyword("一种风力发电机组") == "风电" or has_domain_keyword(
        "一种风力发电机组"
    ) in ("风电", "风力", "发电")
    assert has_domain_keyword("一种食品加工方法") is None


# ---------------- 文本清洗 ----------------


def test_清洗HTML实体():
    """导出里常有 &amp; 这类实体，要还原。"""
    assert clean_text("电压&amp;电流") == "电压&电流"


def test_清洗缺失值():
    """pandas 读进来的 NaN 会变成 'nan' 字符串，必须清成空串。"""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("nan") == ""
    assert clean_text("") == ""


def test_清洗控制字符与多余空白():
    """控制字符和连续空白要清掉。"""
    text = clean_text("一种\x00方法    用于\r\n\r\n\r\n\r\n配电网")

    assert "\x00" not in text
    assert "    " not in text
    assert "\n\n\n" not in text


def test_清洗保留段落结构():
    """段落之间的单个空行要保留，不能全压成一行。"""
    text = clean_text("第一段\n\n第二段")

    assert "\n\n" in text


def test_权利要求切分保留权项():
    """权利要求应按权项编号切分，编号是重要的层级信息。"""
    raw = "1.一种方法，其特征在于包括步骤A。\n2.根据权利要求1所述的方法，其特征在于步骤B。\n3.根据权利要求1所述的方法，其特征在于步骤C。"
    items = normalize_for_claims_split(clean_claims(raw))

    assert len(items) == 3
    assert items[0].startswith("1.")
    assert items[2].startswith("3.")


def test_权利要求清洗不做段落合并():
    """权利要求书的换行结构不能被压平。"""
    raw = "1.一种方法，其特征在于：\n包括步骤A；\n包括步骤B。"
    cleaned = clean_claims(raw)

    assert cleaned.count("\n") >= 2
