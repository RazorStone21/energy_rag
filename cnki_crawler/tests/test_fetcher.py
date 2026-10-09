"""HTTP 层测试，重点是**反爬检测**。

这是本项目最关键的一道防线：知网触发风控时返回的是 HTTP 200 + 验证码页。
如果检测失效，验证码页会被当成正常页面交给解析器，最终静默写入垃圾数据。
所以这里对检测逻辑做独立覆盖。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from cnki_crawler.backends.pc import load_cookies
from cnki_crawler.fetcher import BlockedError, FetchError, LoginRequiredError, PoliteSession

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> str:
    """读取 fixture 文件内容。"""
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeResponse:
    """模拟 requests.Response，只提供被测代码用到的属性。"""

    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


@pytest.fixture
def session() -> PoliteSession:
    """构造一个不限速的会话，避免测试变慢。"""
    return PoliteSession(delay=0.0, max_retries=1, backoff_base=0.0)


# ---------------- 反爬检测 ----------------


def test_风控页抛BlockedError(session: PoliteSession):
    """验证码页必须被识别并抛错，绝不静默通过。"""
    with pytest.raises(BlockedError) as excinfo:
        session._detect_block(load("blocked.html"), "https://example.test/x")

    assert "风控" in str(excinfo.value)


def test_登录页抛LoginRequiredError(session: PoliteSession):
    """登录页要单独识别，PC 后端据此降级到 wap。"""
    with pytest.raises(LoginRequiredError):
        session._detect_block(load("login.html"), "https://example.test/x")


def test_正常页面不误报(session: PoliteSession):
    """正常详情页不能被误判为风控页。"""
    session._detect_block(load("ja.html"), "https://example.test/x")


def test_风控页即使允许登录页也要拦截(session: PoliteSession):
    """allow_login_page=True 只放行登录页，风控页仍必须拦截。"""
    session.session.get = lambda *a, **kw: FakeResponse(load("blocked.html"))

    with pytest.raises(BlockedError):
        session.get("https://example.test/x", allow_login_page=True)


def test_登录页在允许模式下原样返回(session: PoliteSession):
    """PC 后端要自己判断登录页，所以这种模式下不抛异常。"""
    session.session.get = lambda *a, **kw: FakeResponse(load("login.html"))

    text = session.get("https://example.test/x", allow_login_page=True)
    assert "登录" in text


# ---------------- 状态码处理 ----------------


def test_4xx直接抛FetchError(session: PoliteSession):
    """404 等客户端错误不重试，直接抛 FetchError。"""
    session.session.get = lambda *a, **kw: FakeResponse("not found", status_code=404)

    with pytest.raises(FetchError):
        session.get("https://example.test/x")


def test_正常响应返回文本(session: PoliteSession):
    """正常响应应原样返回。"""
    session.session.get = lambda *a, **kw: FakeResponse(load("ja.html"))

    assert "人文经济学" in session.get("https://example.test/x")


def test_网络异常重试耗尽后抛FetchError(session: PoliteSession):
    """连接错误重试耗尽后抛 FetchError，并保留原因。"""

    def boom(*args, **kwargs):
        """总是抛连接错误，用于测试重试耗尽。"""
        raise requests.ConnectionError("boom")

    session.session.get = boom

    with pytest.raises(FetchError):
        session.get("https://example.test/x")


# ---------------- cookie 载入 ----------------


def test_载入netscape格式cookie(tmp_path: Path):
    """Netscape 制表符格式（浏览器扩展导出的格式）。"""
    path = tmp_path / "cookies.txt"
    path.write_text(
        "# Netscape HTTP Cookie File\n"
        ".cnki.net\tTRUE\t/\tFALSE\t0\tEcp_ClientId\tabc123\n"
        ".cnki.net\tTRUE\t/\tFALSE\t0\tSID_kns\txyz789\n",
        encoding="utf-8",
    )

    cookies = load_cookies(path)
    assert cookies == {"Ecp_ClientId": "abc123", "SID_kns": "xyz789"}


def test_载入请求头格式cookie(tmp_path: Path):
    """直接粘贴的 `k=v; k=v` 字符串。"""
    path = tmp_path / "raw.txt"
    path.write_text("Ecp_ClientId=abc123; SID_kns=xyz789", encoding="utf-8")

    cookies = load_cookies(path)
    assert cookies == {"Ecp_ClientId": "abc123", "SID_kns": "xyz789"}


def test_空cookie文件返回空字典(tmp_path: Path):
    """解析不出内容时返回空字典，而不是抛异常。"""
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    assert load_cookies(path) == {}
