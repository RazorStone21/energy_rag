"""后端抽象：题录抓取来源的统一切面。

知网有两个可达的入口，能力不同：

- **wap 后端**（`wap.cnki.net`）：无需登录，开箱即用，但摘要被服务端
  截断到约 110 字。
- **PC 后端**（`kns.cnki.net`）：需要用户自己浏览器的 cookie，能拿到
  完整摘要，但 cookie 会过期。

`ResilientBackend` 把两者串起来：优先用 PC 拿完整数据，任何失败都降级到
wap，并在记录的 `backend` 字段里如实标注这条数据到底来自哪。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from ..fetcher import LoginRequiredError, PoliteSession
from ..parsers import ArticleRecord
from .pc import PcBackend, load_cookies
from .wap import WapBackend

logger = logging.getLogger(__name__)

__all__ = ["Backend", "PcBackend", "ResilientBackend", "WapBackend", "build_backend", "load_cookies"]


class Backend(Protocol):
    """题录抓取后端的协议。"""

    name: str

    def fetch_article(self, article_id: str) -> ArticleRecord | None:
        """抓取一条题录，失败返回 None。"""
        ...


class ResilientBackend:
    """优先 PC 后端、失败自动降级到 wap 的组合后端。"""

    def __init__(self, pc: PcBackend | None, wap: WapBackend) -> None:
        """初始化。

        参数：
            pc: PC 后端；None 表示不可用（未配置 cookie）。
            wap: wap 后端，必须提供，用作兜底。
        """
        self.pc = pc
        self.wap = wap
        self.name = "pc+wap" if pc else "wap"
        self._pc_disabled = pc is None

    def fetch_article(self, article_id: str) -> ArticleRecord | None:
        """抓取题录。PC 不可用时自动降级到 wap。

        参数：
            article_id: 文章 ID，如 DLXT202613007。

        返回：
            ArticleRecord；两个后端都失败时返回 None。
        """
        if self.pc and not self._pc_disabled:
            try:
                record = self.pc.fetch_article(article_id)
                if record is not None:
                    return record
            except LoginRequiredError:
                # cookie 过期是常态，不是错误。降级并记住，避免后续每篇都撞一次。
                logger.warning("PC 后端 cookie 已失效，本次运行后续全部降级到 wap 后端")
                self._pc_disabled = True
            except Exception as exc:  # noqa: BLE001 - 后端故障不应中断整轮爬取
                logger.warning("PC 后端抓取 %s 失败，改用 wap：%s", article_id, exc)

        return self.wap.fetch_article(article_id)


def build_backend(
    session: PoliteSession,
    cookie_file: str | Path | None = None,
    delay: float = 2.0,
) -> ResilientBackend:
    """按配置构造后端组合。

    参数：
        session: wap 请求会话。
        cookie_file: 浏览器导出的 cookie 文件路径；不存在则不启用 PC 后端。
        delay: PC 会话的请求间隔秒数。

    返回：
        ResilientBackend。没有可用 cookie 时只含 wap 后端。
    """
    wap = WapBackend(session)

    cookies: dict[str, str] = {}
    if cookie_file:
        path = Path(cookie_file)
        if path.exists():
            cookies = load_cookies(path)
            if cookies:
                logger.info("已从 %s 载入 %d 条 cookie，启用 PC 后端", path, len(cookies))
            else:
                logger.warning("cookie 文件存在但未解析出内容：%s", path)
        else:
            logger.info("未找到 cookie 文件 %s，仅使用 wap 后端", path)

    if not cookies:
        return ResilientBackend(None, wap)

    return ResilientBackend(PcBackend(cookies, delay=delay), wap)
