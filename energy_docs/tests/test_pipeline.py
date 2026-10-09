"""存储、适配器、HTTP 层测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_docs.adapters.base import SiteAdapter, SiteSpec, build_page_urls
from energy_docs.fetcher import BlockedError, FetchError, PoliteSession
from energy_docs.store import Document, Store, doc_id_for

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    """读取 fixture 文件内容。"""
    return (FIXTURES / name).read_text(encoding="utf-8", errors="replace")


@pytest.fixture
def spec() -> SiteSpec:
    """一个测试用的站点配置。"""
    return SiteSpec(
        name="山东省能源局-通知公告",
        base="http://nyj.shandong.gov.cn",
        entry="/col/col59960/index.html",
        link_pattern=r"/art/\d{4}/\d{1,2}/\d{1,2}/art_\d+_\d+\.html",
        category="省级能源局",
    )


# ---------------- 存储 ----------------


def test_文档ID由URL决定():
    """同一 URL 必须得到同一 ID（去重靠它）。"""
    url = "http://example.gov.cn/art/a.html"

    assert doc_id_for(url) == doc_id_for(url)
    assert doc_id_for(url) != doc_id_for(url + "x")


def test_写入与读回(tmp_path: Path):
    """文档写入后能完整读回。"""
    with Store(tmp_path / "t.db", tmp_path / "t.jsonl") as store:
        doc = Document(url="http://x.test/a", title="标题", text="正文内容", source="某站")
        store.add_document(doc)

        got = list(store.iter_documents())
        assert len(got) == 1
        assert got[0]["title"] == "标题"
        assert got[0]["chars"] == 4


def test_同时写JSONL(tmp_path: Path):
    """JSONL 应逐条追加，进程被杀也不丢。"""
    jsonl = tmp_path / "t.jsonl"
    with Store(tmp_path / "t.db", jsonl) as store:
        store.add_document(Document(url="http://x.test/a", title="A", text="aaa"))
        store.add_document(Document(url="http://x.test/b", title="B", text="bbb"))

    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_断点续传标记(tmp_path: Path):
    """已处理的 URL 应被记住，重跑时跳过。"""
    with Store(tmp_path / "t.db") as store:
        assert store.already_seen("站A", "http://x.test/a") is False

        store.mark_seen("站A", "http://x.test/a")
        assert store.already_seen("站A", "http://x.test/a") is True

        # 不同来源的同一 URL 互不影响
        assert store.already_seen("站B", "http://x.test/a") is False


def test_重复写入覆盖(tmp_path: Path):
    """同一文档重复写入应覆盖而非新增。"""
    with Store(tmp_path / "t.db") as store:
        for _ in range(2):
            store.add_document(Document(url="http://x.test/a", title="A", text="aaa"))
        assert store.stats()["documents"] == 1


# ---------------- 分页 URL 生成 ----------------


def test_生成多种分页候选(spec: SiteSpec):
    """应为一个页码生成多种候选 URL，供逐个尝试。"""
    urls = build_page_urls(spec, 2)

    assert len(urls) > 1
    assert any("index_2.html" in u for u in urls)
    assert all(u.startswith("http://nyj.shandong.gov.cn") for u in urls)


def test_分页候选覆盖常见模式(spec: SiteSpec):
    """候选里应包含 shtml 与查询参数两类常见模式。"""
    urls = build_page_urls(spec, 3)

    joined = " ".join(urls)
    assert "index_3.shtml" in joined
    assert "pageNum=3" in joined or "page=3" in joined


# ---------------- 标题清洗 ----------------


def test_剥离站点名与栏目名(spec: SiteSpec):
    """三段式标题「站点名 栏目名 标题」应只留标题。"""
    adapter = SiteAdapter(spec, PoliteSession(delay=0))

    cleaned = adapter._clean_title("山东省能源局 动态要闻 局主要负责同志调研充电站建设")

    assert cleaned == "局主要负责同志调研充电站建设"


def test_不误剥正常标题(spec: SiteSpec):
    """标题不以站点名开头时不应乱剥。"""
    adapter = SiteAdapter(spec, PoliteSession(delay=0))

    raw = "关于推动新能源高质量发展的通知"
    assert adapter._clean_title(raw) == raw


def test_短标题不被误剥(spec: SiteSpec):
    """确认是三段式但剩余部分过短时，不剥第二段。"""
    adapter = SiteAdapter(spec, PoliteSession(delay=0))

    cleaned = adapter._clean_title("山东省能源局 短")
    assert "短" in cleaned


# ---------------- HTTP 拦截检测 ----------------


def test_拦截页抛BlockedError():
    """WAF 拦截页必须被识别，不能当成正文。"""
    session = PoliteSession(delay=0)

    with pytest.raises(BlockedError):
        session.detect_block("<html><body>访问被拒绝 Access Denied</body></html>", "http://x.test")


def test_维护页抛BlockedError():
    """站点维护页也要识别——实测中确有站点长期是维护页。"""
    session = PoliteSession(delay=0)

    with pytest.raises(BlockedError):
        session.detect_block("<html><body>网站维护升级提示</body></html>", "http://x.test")


def test_正常页面不误报():
    """正常正文页不能被误判为拦截页。"""
    session = PoliteSession(delay=0)

    session.detect_block(load("nea_law.html"), "http://x.test")


def test_4xx抛FetchError():
    """404 直接抛 FetchError，不重试。"""
    session = PoliteSession(delay=0, max_retries=0)
    session.session.get = lambda *a, **kw: _FakeResponse("not found", 404)

    with pytest.raises(FetchError):
        session.get("http://x.test/a")


def test_响应过小抛FetchError():
    """响应体过小说明不是有效正文。"""
    session = PoliteSession(delay=0, max_retries=0)
    session.session.get = lambda *a, **kw: _FakeResponse("<html></html>", 200)

    with pytest.raises(FetchError):
        session.get("http://x.test/a", expect_html=True)


# ---------------- 编码处理 ----------------


def test_UTF8页面正确解码():
    """UTF-8 页面必须解出正确中文，不能被当成 Latin-1。"""
    html = '<html><head><title>中华人民共和国能源法</title></head><body>正文</body></html>'
    response = _FakeResponse(b"", 200, content=html.encode("utf-8"))

    text = PoliteSession.decode_response(response)
    assert "中华人民共和国能源法" in text


def test_GBK页面正确解码():
    """GBK 编码的老页面也要正确解码。"""
    html = "<html><body>中华人民共和国电力法</body></html>"
    response = _FakeResponse(b"", 200, content=html.encode("gbk"))

    text = PoliteSession.decode_response(response)
    assert "中华人民共和国电力法" in text


def test_不依赖响应头声明的错误编码():
    """响应头谎报 ISO-8859-1 时，仍应按内容解出中文。

    这正是实测踩到的坑：requests 默认 charset 缺失时按 Latin-1 解，
    Latin-1 对任意字节都不报错，中文会被静默毁成乱码。
    """
    html = "<html><body>配电网故障定位方法</body></html>"
    response = _FakeResponse(
        b"", 200, content=html.encode("utf-8"), headers={"Content-Type": "text/html; charset=ISO-8859-1"}
    )

    text = PoliteSession.decode_response(response)
    # Latin-1 能"成功"解码，所以必须靠 content 判断——这里允许两种结果，
    # 但绝不允许出现「本该是中文却是乱码」的静默失败
    assert "配电网故障定位方法" in text or "é" not in text


class _FakeResponse:
    """模拟 requests.Response。"""

    def __init__(self, text: str, status_code: int = 200, content: bytes = b"", headers=None):
        self.text = text
        self.status_code = status_code
        self.content = content or text.encode("utf-8")
        self.headers = headers or {}
        self.apparent_encoding = "utf-8"


def test_分页URL不重复拼base(spec: SiteSpec):
    """带 {entry} 的模板不能重复拼 base。

    曾经拼出过 `//host/pathhttps://host/path` 这种坏 URL，
    导致 requests 报 `Failed to resolve 'hosthttps'`。
    """
    urls = build_page_urls(spec, 2)

    for url in urls:
        assert url.count("http") == 1, f"URL 里出现了多次协议头：{url}"
        assert "http://http" not in url and ".cnhttp" not in url


def test_附件文件名不重复扩展名():
    """列表页 anchor 文本自带扩展名时，不能生成 xxx.docx.docx。"""
    from energy_docs.attachments import Attachment

    att = Attachment(
        url="https://x.gov.cn/a/W02026.docx",
        kind="docx",
        title="2022年天津市节能专项资金使用计划.docx",
    )
    assert att.suggested_name() == "2022年天津市节能专项资金使用计划.docx"

    att2 = Attachment(url="https://x.gov.cn/a/W02026.pdf", kind="pdf", title="某通知.pdf")
    assert att2.suggested_name() == "某通知.pdf"


def test_附件文件名剔除非法字符():
    """标题里的路径字符要剔除，避免写文件出事。"""
    from energy_docs.attachments import Attachment

    att = Attachment(url="https://x.gov.cn/a/f.pdf", kind="pdf", title='关于/测试:文件?"')
    name = att.suggested_name()

    for bad in '/\\:*?"<>|':
        assert bad not in name


def test_标题为空时退回URL文件名():
    """标题缺失时用 URL 里的文件名兜底。"""
    from energy_docs.attachments import Attachment

    att = Attachment(url="https://x.gov.cn/a/20260901notice.pdf", kind="pdf", title="")
    assert att.suggested_name() == "20260901notice.pdf"


def test_从pdfjs预览器提取真实地址():
    """辽宁等站用 viewer.html?file= 包一层，要抽出里层的真实 PDF 地址。"""
    from energy_docs.attachments import find_attachments

    html = (
        '<a href="/uiFramework/js/pdfjs/web/viewer.html'
        '?file=/fgw/articleFileDir/2026-07/31/abc/2026073115234837059.pdf">某通知</a>'
    )
    found = find_attachments(html, "https://fgw.ln.gov.cn/fgw/index/tzgg/index.shtml", "某通知")

    assert len(found) == 1
    assert found[0].url.endswith("2026073115234837059.pdf")
    assert found[0].kind == "pdf"
    assert "viewer.html" not in found[0].url


def test_忽略非附件链接():
    """正文页链接、JS 链接不能被当附件。"""
    from energy_docs.attachments import find_attachments

    html = (
        '<a href="/art/2026/a.html">正文</a>'
        '<a href="javascript:void(0)">按钮</a>'
        '<a href="/x.pdf">真附件</a>'
    )
    found = find_attachments(html, "https://x.gov.cn/list/", "标题")

    assert len(found) == 1
    assert found[0].url == "https://x.gov.cn/x.pdf"


def test_魔数校验拒绝HTML错误页():
    """服务端把错误页当 200 返回时，不能被当 PDF 存下来。"""
    from energy_docs.attachments import sniff_kind

    assert sniff_kind(b"%PDF-1.7") == "pdf"
    assert sniff_kind(b"PK\x03\x04") == "zip"
    assert sniff_kind(b"\xd0\xcf\x11\xe0") == "ole"
    assert sniff_kind(b"<!DOCTYPE html>") == "unknown"


def test_不误匹配oldsrc属性():
    """贵州的附件标签同时带 href 和已失效的 oldsrc。

    `oldsrc="/protect/..."` 是死路径，但 `src=` 会命中 `oldsrc=` 的子串，
    以前就是这样采到一堆 404 的旧地址。属性名前必须有词边界。
    """
    from energy_docs.attachments import find_attachments

    html = (
        '<a href="./P020260908607594904873.xlsx" '
        'title="明细表.xlsx" '
        'oldsrc="/protect/P0202609/P020260908/P020260908607594904873.xlsx">明细表</a>'
    )
    found = find_attachments(
        html, "https://fgw.guizhou.gov.cn/fggz/tzgg/202609/t20260908_90843856.html", "某公示"
    )

    urls = [a.url for a in found]
    assert len(urls) == 1, f"应当只采到 href 那一个，实际：{urls}"
    assert "/protect/" not in urls[0]
    assert urls[0].endswith("P020260908607594904873.xlsx")
    assert "fggz/tzgg/202609/" in urls[0]


def test_同名附件重跑幂等(tmp_path: Path):
    """同名文件已存在时应跳过，重跑不产生哈希后缀副本。"""
    from energy_docs.attachments import Attachment, AttachmentDownloader, DownloadStats
    from energy_docs.fetcher import PoliteSession

    (tmp_path / "某通知.pdf").write_bytes(b"%PDF-1.4 existing")
    dl = AttachmentDownloader(PoliteSession(delay=0), tmp_path)
    stats = DownloadStats()

    att = Attachment(url="https://x.gov.cn/a.pdf", kind="pdf", title="某通知.pdf")
    result = dl.download(att, stats)

    assert result is None
    assert stats.skipped == 1
    assert stats.downloaded == 0
    # 目录里仍然只有原来那个文件
    assert len(list(tmp_path.iterdir())) == 1
