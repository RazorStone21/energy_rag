"""列名识别测试。

这是整个管道最脆的一环：列识别错了会把权利要求书当摘要用，
产出的语料直接报废。所以这里对各种表头写法做覆盖。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from patent_corpus.loader import (
    LoadError,
    collect_files,
    detect_columns,
    load_table,
    normalize_header,
)


@pytest.fixture
def baili_export(tmp_path: Path) -> Path:
    """模拟佰腾风格的导出文件。"""
    path = tmp_path / "baiten.xlsx"
    pd.DataFrame(
        {
            "公开(公告)号": ["CN110123456A", "CN110123457B"],
            "申请号": ["CN201910123456", "CN201910123457"],
            "专利名称": ["一种配电网故障定位方法", "一种风机叶片检测装置"],
            "摘要": ["本发明公开了一种配电网故障定位方法…", "本实用新型公开了一种风机叶片检测装置…"],
            "权利要求书": ["1.一种配电网故障定位方法，其特征在于…", "1.一种风机叶片检测装置，其特征在于…"],
            "IPC分类号": ["H02J13/00", "F03D17/00"],
            "申请人": ["某某电力公司", "某某风电公司"],
            "发明人": ["张三", "李四"],
            "申请日": ["2019-01-01", "2019-01-02"],
        }
    ).to_excel(path, index=False)
    return path


@pytest.fixture
def zhuanlizhixing_export(tmp_path: Path) -> Path:
    """模拟专利之星风格的导出文件（表头写法不同）。"""
    path = tmp_path / "zhuanlizhixing.csv"
    pd.DataFrame(
        {
            "名称": ["一种储能电池热管理系统"],
            "摘要(中文)": ["本发明涉及一种储能电池热管理系统…"],
            "主权项": ["1.一种储能电池热管理系统，包括…"],
            "主分类号": ["H01M10/613"],
            "申请人名称": ["某某新能源公司"],
            "申请日": ["2020-05-05"],
        }
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def test_归一化表头去掉标点与空白():
    """归一化应抹平括号、空白、大小写的差异。"""
    assert normalize_header("公开(公告)号") == normalize_header("公开（公告）号")
    assert normalize_header("IPC 分类号") == "ipc分类号"


def test_识别佰腾风格表头(baili_export: Path):
    """佰腾的表头应全部识别出来。"""
    table = load_table(baili_export)

    assert table.row_count == 2
    assert table.columns["title"] == "专利名称"
    assert table.columns["abstract"] == "摘要"
    assert table.columns["claims"] == "权利要求书"
    assert table.columns["pub_number"] == "公开(公告)号"
    assert table.columns["ipc"] == "IPC分类号"
    assert table.columns["applicant"] == "申请人"


def test_识别专利之星风格表头(zhuanlizhixing_export: Path):
    """专利之星的表头写法完全不同，也要能识别。"""
    table = load_table(zhuanlizhixing_export)

    assert table.columns["title"] == "名称"
    assert table.columns["abstract"] == "摘要(中文)"
    assert table.columns["claims"] == "主权项"
    assert table.columns["ipc"] == "主分类号"


def test_缺少标题和摘要时抛错(tmp_path: Path):
    """没有标题也没有摘要的表没有利用价值，应明确报错。"""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"发明人": ["张三"], "法律状态": ["授权"]}).to_csv(path, index=False)

    with pytest.raises(LoadError) as excinfo:
        load_table(path)
    assert "标题" in str(excinfo.value) or "摘要" in str(excinfo.value)


def test_损坏的xls抛出LoadError(tmp_path: Path):
    """损坏的 .xls 应抛 LoadError，而不是让底层异常冒到调用方。"""
    path = tmp_path / "broken.xls"
    path.write_bytes(b"\xd0\xcf\x11\xe0" + b"\x00" * 60)

    with pytest.raises(LoadError):
        load_table(path)


def test_主分类号与完整分类号分开识别(tmp_path: Path):
    """真实的专利之星导出同时有两列：主分类号 和 分类号。

    这两列必须分开映射——领域判定只用主分类号，完整分类号仅作参考。
    合并成一列会误收「NFC 锁」这类带副分类 H02 的无关专利。
    """
    path = tmp_path / "zlsx.csv"
    pd.DataFrame(
        {
            "标题": ["一种光伏光热一体化集热组件"],
            "主分类号": ["H02S40/44"],
            "分类号": ["H02S40/44;F24S10/30;F24S10/40;H02S40/34"],
            "摘要": ["本申请提供了一种光伏光热一体化集热组件…"],
        }
    ).to_csv(path, index=False, encoding="utf-8")

    table = load_table(path)

    assert table.columns["ipc"] == "主分类号"
    assert table.columns["ipc_all"] == "分类号"


def test_全角括号表头也能识别(tmp_path: Path):
    """专利之星用的是全角括号「公开（公告）号」，佰腾用半角，都要能认。"""
    path = tmp_path / "fullwidth.csv"
    pd.DataFrame(
        {
            "标题": ["一种配电网方法"],
            "摘要": ["摘要内容"],
            "公开（公告）号": ["CN110123456A"],
            "公开（公告）日": ["20260922"],
        }
    ).to_csv(path, index=False, encoding="utf-8")

    table = load_table(path)

    assert table.columns["pub_number"] == "公开（公告）号"
    assert table.columns["pub_date"] == "公开（公告）日"


def test_文件不存在(tmp_path: Path):
    """不存在的文件应抛 LoadError。"""
    with pytest.raises(LoadError):
        load_table(tmp_path / "没有这个文件.xlsx")


def test_目录收集按后缀过滤(tmp_path: Path):
    """目录扫描应只收受支持的后缀，且结果有序。"""
    (tmp_path / "sub").mkdir()
    for name in ("a.csv", "b.xlsx", "c.xlsm", "d.json", "e.txt"):
        (tmp_path / "sub" / name).write_text("", encoding="utf-8")

    files = collect_files(tmp_path)
    names = [f.name for f in files]

    assert names == ["a.csv", "b.xlsx", "c.xlsm", "e.txt"]
    assert "d.json" not in names


def test_列名识别不误配(tmp_path: Path):
    """「摘要附图」这类相近表头不应该被当成摘要列。"""
    mapping, _ = detect_columns(["发明名称", "摘要附图", "摘要"])
    assert mapping["abstract"] == "摘要"
