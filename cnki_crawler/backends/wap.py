"""wap 后端：抓取 `wap.cnki.net` 的公开文章详情页。

这是零配置的默认后端，无需登录。代价是**摘要被知网服务端截断到约 110 字**，
解析结果里 `abstract_truncated` 恒为 True。需要完整摘要请配置 PC 后端。
"""

from __future__ import annotations

from ..fetcher import PoliteSession
from ..parsers import ArticleRecord, parse_article_page

# 文章详情页地址模板
_ARTICLE_URL = "https://wap.cnki.net/touch/web/Journal/Article/{article_id}.html"

# 期刊主页地址模板
_JOURNAL_URL = "https://wap.cnki.net/touch/web/Journal/{pykm}"

# 期次目录页地址模板：PYKM + 年 + 两位期号
_ISSUE_URL = "https://wap.cnki.net/touch/web/Journal/List/{pykm}{year}{issue:02d}.html"


class WapBackend:
    """从 wap 站抓取题录，无需登录。"""

    name = "wap"

    def __init__(self, session: PoliteSession) -> None:
        """初始化。

        参数：
            session: 已配置限速的 HTTP 会话。
        """
        self.session = session

    def fetch_article(self, article_id: str) -> ArticleRecord:
        """抓取并解析一条题录。

        参数：
            article_id: 文章 ID，如 DLXT202613007。

        返回：
            ArticleRecord，`backend` 字段标记为 "wap"。

        异常：
            fetcher.BlockedError: 触发风控。
            parsers.ParseError: 页面结构不符。
        """
        html = self.session.get(_ARTICLE_URL.format(article_id=article_id))
        record = parse_article_page(html, article_id=article_id)
        record.backend = "wap"
        return record

    def fetch_journal(self, pykm: str) -> str:
        """抓取期刊主页原始 HTML。"""
        return self.session.get(_JOURNAL_URL.format(pykm=pykm))

    def fetch_issue(self, pykm: str, year: int, issue: int) -> str:
        """抓取期次目录页原始 HTML。"""
        return self.session.get(_ISSUE_URL.format(pykm=pykm, year=year, issue=issue))
