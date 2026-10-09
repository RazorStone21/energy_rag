"""PC 后端：用**用户自己的浏览器 cookie** 访问 `kns.cnki.net`，获取完整摘要。

## 重要说明（请先读完再用）

1. **cookie 是你自己的访问凭证。** 本模块只是把它附在请求头里，不做任何
   伪造或绕过。请用自己的账号，遵守知网的服务条款。
2. **本模块的选择器和端点未经真实 cookie 验证。** 开发环境下 PC 搜索接口
   被滑块验证码拦截，无法取得有效会话来实测，因此这里按知网 PC 站的已知
   页面结构实现。**如果它不工作，会静默降级到 wap 后端**，不会中断爬取，
   也不会写入错误数据。首次使用请用 `--limit 1` 验证。
3. **cookie 会过期。** 失效时 PC 后端抛 `LoginRequiredError`，
   `ResilientBackend` 会捕获并整轮降级到 wap。
"""

from __future__ import annotations

import logging
from pathlib import Path

from bs4 import BeautifulSoup

from ..fetcher import FetchError, LoginRequiredError, PoliteSession
from ..parsers import ArticleRecord

logger = logging.getLogger(__name__)

# PC 站文章摘要页。dbcode=CJFD 表示期刊全文数据库。
_PC_ARTICLE_URL = "https://kns.cnki.net/kcms2/article/abstract?dbcode=CJFD&filename={article_id}"

# 桌面端 UA
_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# 摘要容器：知网 PC 站历史上用过多个 id/class，逐个尝试
_ABSTRACT_SELECTORS = (
    "#ChDivSummary",
    "span#ChDivSummary",
    ".abstract-text",
    ".brief .abstract",
    "#abstract",
)

# 登录页特征：命中即说明 cookie 失效
_LOGIN_MARKERS = ("个人登录", "机构登录", "用户登录", "login-box", "登录知网")


def load_cookies(path: str | Path) -> dict[str, str]:
    """从文件载入 cookie。

    支持两种格式：
    1. **Netscape cookie 文件**（浏览器扩展 / curl / yt-dlp 导出的制表符分隔格式）。
    2. **请求头字符串**，形如 `Ecp_ClientId=xxx; SID=yyy`。

    参数：
        path: cookie 文件路径。

    返回：
        cookie 名值对字典；解析不出内容时返回空字典。
    """
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    cookies: dict[str, str] = {}

    for line in raw.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # Netscape 格式：以 httpOnly_ 开头或注释行
        stripped = line.lstrip("#")
        if line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue
        parts = stripped.split("\t")
        if len(parts) >= 7:
            # domain, flag, path, secure, expiration, name, value
            cookies[parts[5].strip()] = parts[6].strip()

    # 回退：按请求头字符串解析
    if not cookies:
        for chunk in raw.replace("\n", ";").split(";"):
            if "=" in chunk:
                name, _, value = chunk.partition("=")
                name, value = name.strip(), value.strip()
                if name and value:
                    cookies[name] = value

    return cookies


class PcBackend:
    """用浏览器 cookie 从 PC 站抓取完整题录。"""

    name = "pc"

    def __init__(
        self,
        cookies: dict[str, str],
        delay: float = 2.0,
        timeout: int = 20,
        max_retries: int = 2,
        backoff_base: float = 5.0,
    ) -> None:
        """初始化（内部自建桌面 UA 的会话）。

        参数：
            cookies: 浏览器导出的 cookie。
            delay: 请求间隔秒数。
            timeout: 请求超时秒数。
            max_retries: 重试次数。
            backoff_base: 退避基数。
        """
        self.session = PoliteSession(
            delay=delay,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            user_agent=_DESKTOP_UA,
            cookies=cookies,
        )

    def fetch_article(self, article_id: str) -> ArticleRecord | None:
        """抓取一条完整题录。

        参数：
            article_id: 文章 ID，如 DLXT202613007。

        返回：
            ArticleRecord（`backend` = "pc"，摘要标记为完整）；
            页面结构不符时返回 None，交由调用方降级。

        异常：
            fetcher.LoginRequiredError: cookie 失效。
            fetcher.BlockedError: 触发风控。
        """
        url = _PC_ARTICLE_URL.format(article_id=article_id)
        try:
            # allow_login_page=True：登录页要由我们识别后抛 LoginRequiredError，
            # 而不是当成普通页面解析出空记录。
            html = self.session.get(url, allow_login_page=True)
        except FetchError as exc:
            logger.debug("PC 后端请求失败 %s：%s", article_id, exc)
            return None

        if self._looks_like_login(html):
            raise LoginRequiredError(f"PC 站返回登录页，cookie 可能已过期：{article_id}")

        record = self._parse(html, article_id)
        if record is None:
            return None
        record.backend = "pc"
        return record

    @staticmethod
    def _looks_like_login(html: str) -> bool:
        """判断页面是否为登录页。"""
        head = html[:8000]
        return any(marker in head for marker in _LOGIN_MARKERS) and "ChDivSummary" not in head

    @staticmethod
    def _parse(html: str, article_id: str) -> ArticleRecord | None:
        """解析 PC 站摘要页。

        参数：
            html: 页面 HTML。
            article_id: 文章 ID。

        返回：
            ArticleRecord；关键字段缺失时返回 None。
        """
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # pragma: no cover
            soup = BeautifulSoup(html, "html.parser")

        title = ""
        for selector in (".wx-tit h1", "h1", ".wx-tit"):
            node = soup.select_one(selector)
            if node and node.get_text(strip=True):
                title = node.get_text(strip=True)
                break
        if not title:
            return None

        record = ArticleRecord(article_id=article_id, title=title)

        # 完整摘要
        for selector in _ABSTRACT_SELECTORS:
            node = soup.select_one(selector)
            if node and node.get_text(strip=True):
                record.abstract = node.get_text(" ", strip=True)
                record.abstract_truncated = False
                break

        # 关键词
        for node in soup.select(".keywords a, p.keywords a, .kw a"):
            text = node.get_text(strip=True)
            if text:
                record.keywords.append(text)

        # 基金（wap 站没有这个字段，只有 PC 后端能拿到）
        fund_node = soup.select_one(".funds, p.funds, .fund")
        if fund_node:
            record.fund = fund_node.get_text(strip=True)

        # 作者与机构
        for node in soup.select(".author a, .wx-tit .author a"):
            text = node.get_text(strip=True)
            if text:
                record.authors.append(text)
        for node in soup.select(".orgn a, .author .orgn a"):
            text = node.get_text(strip=True)
            if text:
                record.affiliations.append(text)

        return record
