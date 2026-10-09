"""正文抽取与链接抽取测试。

fixtures 是 2026-09-22 从真实站点抓下来的原始响应。站点改版后测试会失败——
这正是它存在的意义。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_docs.extract import (
    ExtractError,
    clean_text,
    extract_links,
    extract_main_text,
    find_data_proxy,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    """读取 fixture 文件内容。"""
    return (FIXTURES / name).read_text(encoding="utf-8", errors="replace")


# ---------------- 正文抽取 ----------------


def test_抽取法规正文():
    """国家能源局法规页应抽出《能源法》全文。"""
    doc = extract_main_text(load("nea_law.html"))

    assert doc.title == "中华人民共和国能源法"
    assert doc.chars > 5000
    assert "第一章" in doc.text
    assert "第八十条" in doc.text


def test_标题剥离站点后缀():
    """标题里的「---国家能源局」后缀要被剥掉。"""
    doc = extract_main_text(load("nea_law.html"))

    assert "国家能源局" not in doc.title
    assert doc.title == "中华人民共和国能源法"


def test_正文不含导航与页脚():
    """抽出的正文不能混入导航、页脚、版权信息。"""
    doc = extract_main_text(load("nea_law.html"))

    for noise in ("版权所有", "京ICP备", "网站地图", "Toggle navigation"):
        assert noise not in doc.text


def test_记录命中方式():
    """应记录正文是靠哪个选择器抽到的，便于发现站点改版。"""
    doc = extract_main_text(load("nea_law.html"))

    assert doc.selector  # 非空表示命中具名选择器而非密度兜底


def test_抽取省级站正文():
    """山东省能源局的文章页也应能抽出正文。"""
    doc = extract_main_text(load("sd_article.html"), min_chars=100)

    assert doc.chars > 100
    assert "山东" in doc.title or "调研" in doc.title or doc.title


def test_非正文页抛错():
    """内容过少的页面应抛 ExtractError 而不是返回垃圾。"""
    with pytest.raises(ExtractError):
        extract_main_text("<html><body><p>短</p></body></html>", min_chars=200)


# ---------------- 文本清洗 ----------------


def test_清洗噪声行():
    """发布时间、责任编辑这类噪声行要被去掉。"""
    text = clean_text("发布时间：2026-09-22\n正文第一段\n责任编辑：张三\n正文第二段")

    assert "发布时间" not in text
    assert "责任编辑" not in text
    assert "正文第一段" in text
    assert "正文第二段" in text


def test_清洗HTML实体():
    """HTML 实体要还原。"""
    assert clean_text("电压&amp;电流") == "电压&电流"


def test_清洗保留段落结构():
    """段落间空行要保留。"""
    assert "\n" in clean_text("第一段\n\n\n\n第二段")


# ---------------- 链接抽取 ----------------


def test_抽取静态链接():
    """国家能源局法规列表页是标准 <a href> 结构。"""
    links = extract_links(
        load("nea_law_list.html"),
        r"nea\.gov\.cn/\d{4}-\d{2}/\d{2}/c_\d+\.htm",
        "https://www.nea.gov.cn/nyflfg/",
    )

    assert len(links) >= 20
    titles = [t for t, _ in links]
    assert any("能源法" in t for t in titles)


def test_抽取CDATA数据岛链接():
    """大汉 CMS 把列表数据塞在 CDATA 里，标准解析抓不到。

    这类页面的 <a> 在 <record><![CDATA[...]]></record> 内部，会被
    BeautifulSoup 当作纯文本，必须单独再解析一遍 CDATA 段。
    """
    links = extract_links(
        load("sd_notice_list.html"),
        r"/art/\d{4}/\d{1,2}/\d{1,2}/art_\d+_\d+\.html",
        "http://nyj.shandong.gov.cn/col/col59960/index.html",
    )

    assert len(links) >= 30, "CDATA 里的链接没被抽出来"
    titles = [t for t, _ in links]
    assert any("通知" in t or "公告" in t for t in titles)


def test_链接去重():
    """同一链接重复出现只保留一次。"""
    links = extract_links(
        load("sd_notice_list.html"),
        r"/art/\d{4}/\d{1,2}/\d{1,2}/art_\d+_\d+\.html",
        "http://nyj.shandong.gov.cn/",
    )

    urls = [u for _, u in links]
    assert len(urls) == len(set(urls))


def test_链接正则不匹配时返回空():
    """正则写错时应返回空列表而不是乱抓。"""
    links = extract_links(load("nea_law.html"), r"/never/matches/\d+", "")

    assert links == []


# ---------------- 分页接口识别 ----------------


def test_识别大汉分页接口():
    """应能从页面脚本里解析出完整的分页接口模板。"""
    proxy = find_data_proxy(load("sd_notice_list.html"))

    assert proxy is not None
    assert "{page}" in proxy
    assert "columnid=59960" in proxy
    # 关键参数必须从 JS 对象里解析出来，光抄页面 URL 是不完整的
    assert "webid=355" in proxy
    assert "unitid=745545" in proxy


def test_无分页接口时返回None():
    """普通静态页不应误报有分页接口。"""
    assert find_data_proxy(load("nea_law_list.html")) is None
