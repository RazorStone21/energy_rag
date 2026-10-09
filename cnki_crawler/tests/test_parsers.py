"""解析器的离线单元测试，全部基于实测保存的真实页面。

这些 fixtures 是 2026-09-22 从 wap.cnki.net 抓下来的原始响应。知网改版后
测试会失败——这正是它的价值：第一时间告诉你选择器失效了。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cnki_crawler.parsers import (
    ParseError,
    parse_article_id,
    parse_article_page,
    parse_issue_page,
    parse_journal_page,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    """读取 fixture 文件内容。"""
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------- 文章详情页 ----------------


def test_详情页_基本字段():
    """详情页应解析出标题、刊名、年卷期。"""
    record = parse_article_page(load("ja.html"), article_id="DYLC202608000F")

    assert record.title == "人文经济学与县域经济高质量发展:理论逻辑与转化机制"
    assert record.journal == "东岳论丛"
    assert record.year == 2026
    assert record.issue == 8
    assert record.article_id == "DYLC202608000F"


def test_详情页_作者与机构去重():
    """同一机构在作者行和机构字段各出现一次，必须去重。"""
    record = parse_article_page(load("ja.html"), article_id="DYLC202608000F")

    assert record.authors == ["赵跃先", "王逸然"]
    assert record.affiliations == ["山西师范大学马克思主义学院"]


def test_详情页_关键词与领域已清洗():
    """关键词和领域的尾部分号必须被去掉。"""
    record = parse_article_page(load("ja.html"), article_id="DYLC202608000F")

    assert record.keywords == ["人文经济学", "县域经济", "高质量发展", "人文价值转化", "群众路线"]
    assert record.subjects == ["经济理论及经济思想史", "经济体制改革"]
    assert all(not k.endswith("；") for k in record.keywords)


def test_详情页_摘要标记为截断():
    """wap 页面的摘要是服务端截断的，必须如实标记。

    这是本项目最重要的诚实性约束：截断的摘要不能被当成完整摘要。
    """
    record = parse_article_page(load("ja.html"), article_id="DYLC202608000F")

    assert record.abstract_truncated is True
    assert record.abstract
    assert not record.abstract.endswith("...")
    # 截断后的正文应当明显短于正常摘要
    assert len(record.abstract) < 200


def test_详情页_统计与附件字段():
    """被引、下载、PDF 大小、期刊代码都要解析出来。"""
    record = parse_article_page(load("ja.html"), article_id="DYLC202608000F")

    assert record.pdf_size_kb == 1610
    assert record.pykm == "DYLC"
    assert record.backend == "wap"
    assert record.url.endswith("DYLC202608000F.html")


def test_详情页_结构不符时抛错():
    """拿到的不是详情页时应抛 ParseError，而不是返回空壳记录。"""
    with pytest.raises(ParseError):
        parse_article_page("<html><body><p>无关页面</p></body></html>")


# ---------------- 期次页 ----------------


def test_期次页_解析出全部文章():
    """期次页应解析出该期全部 20 篇文章及其 ID。"""
    info = parse_issue_page(load("list.html"), pykm="DLXT")

    assert info.journal == "电力系统自动化"
    assert info.year == 2026
    assert info.issue == 13
    assert len(info.articles) == 20

    first = info.articles[0]
    assert first.article_id == "DLXT202613001"
    assert first.title == "特约主编寄语"


def test_期次页_文章ID连续():
    """期次内的文章序号应当连续，可用于校验解析完整性。"""
    info = parse_issue_page(load("list.html"), pykm="DLXT")

    ids = [a.article_id for a in info.articles]
    assert ids[0] == "DLXT202613001"
    assert ids[-1] == "DLXT202613020"
    assert len(set(ids)) == len(ids), "文章 ID 不应重复"


def test_期次页_栏目分组():
    """目录按栏目分组，栏目名应挂到对应的文章上。"""
    info = parse_issue_page(load("list.html"), pykm="DLXT")

    assert info.articles[0].section == "新型电力系统大模型关键技术及应用"


# ---------------- 期刊页 ----------------


def test_期刊页_元数据():
    """期刊页应解析出刊名与学科代码。"""
    meta = parse_journal_page(load("dlxt.html"), pykm="DLXT")

    assert meta.name == "电力系统自动化"
    assert meta.subject_codes == "C042;I140"


def test_期刊页_网络首发条目字段分离():
    """网络首发的标题不能把作者和日期粘进来。"""
    meta = parse_journal_page(load("dlxt.html"), pykm="DLXT")

    assert len(meta.online_first) == 6
    first = meta.online_first[0]
    assert first.article_id == "DLXT20260922001"
    assert first.title == "基于时序基础模型知识迁移与多元特征融合的负荷预测方法"
    assert first.authors == ["何蕾", "耿闯", "包铁", "肖望", "侯嘉璐"]
    assert first.date == "2026-09-22"
    assert "何蕾" not in first.title


# ---------------- 工具函数 ----------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("//wap.cnki.net/touch/web/Journal/Article/DLXT202613007.html", "DLXT202613007"),
        ("https://wap.cnki.net/touch/web/Journal/Article/DYLC2026092000F.html", "DYLC2026092000F"),
        ("https://example.com/nothing", ""),
        ("", ""),
    ],
)
def test_提取文章ID(url: str, expected: str):
    """文章 ID 提取应覆盖各种链接形态。"""
    assert parse_article_id(url) == expected
