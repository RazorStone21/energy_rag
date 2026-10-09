"""HTTP 会话层：限速、重试退避、以及反爬拦截检测。

本模块最重要的职责是**反爬检测**。知网在触发风控时会返回 HTTP 200，
但响应体是滑块验证码页或登录页。如果不加识别，爬虫会把验证码页当成
正常页面交给解析器，最终静默地往数据库里写入一堆垃圾记录——这比直接
报错危险得多。因此所有响应都必须先过 `_detect_block`。
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# 风控页特征：出现任意一个就说明被拦截了
_BLOCK_MARKERS = (
    "安全验证",
    "captchaType=blockPuzzle",
    "/verify/home",
)

# 登录页特征：说明该资源需要登录态（PC 后端据此降级到 wap 后端）
_LOGIN_MARKERS = (
    "知网(CNKI)-登录",
    "/touch/usercenter/passport/login",
)

# 这些状态码值得重试（服务端临时问题）
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class CrawlerError(Exception):
    """爬虫基础异常。"""


class BlockedError(CrawlerError):
    """触发了知网风控（验证码/滑块）。

    这是致命错误，必须立即中止：继续请求只会加深风控，而且拿到的
    页面没有解析价值。
    """


class LoginRequiredError(CrawlerError):
    """目标资源要求登录态。

    PC 后端在 cookie 缺失或过期时抛出，调用方应据此降级到 wap 后端。
    """


class FetchError(CrawlerError):
    """重试耗尽后仍未取得有效响应。"""


class PoliteSession:
    """带限速与反爬检测的 HTTP 会话。

    单线程、单一会话、固定 UA。所有请求之间强制间隔 `delay` 秒。
    """

    def __init__(
        self,
        delay: float = 2.0,
        timeout: int = 20,
        max_retries: int = 3,
        backoff_base: float = 5.0,
        user_agent: str | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        """初始化会话。

        参数：
            delay: 两次请求之间的最小间隔秒数。
            timeout: 单次请求超时秒数。
            max_retries: 网络错误/5xx 的最大重试次数。
            backoff_base: 指数退避基数，第 n 次重试等待 backoff_base * 2**n 秒。
            user_agent: 自定义 UA，默认用移动端 UA。
            cookies: 预置 cookie（PC 后端用）。
        """
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base

        self.session = requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept"] = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        self.session.headers["Accept-Language"] = "zh-CN,zh;q=0.9"
        if cookies:
            self.session.cookies.update(cookies)

        self._last_request_at = 0.0
        self.request_count = 0

    def _throttle(self) -> None:
        """确保与上一次请求间隔至少 delay 秒。"""
        if self.delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _detect_block(self, text: str, url: str) -> None:
        """检测响应体是否为风控页或登录页，是则抛异常。

        参数：
            text: 响应体文本。
            url: 请求地址，用于错误信息。

        异常：
            BlockedError: 命中风控特征。
            LoginRequiredError: 命中登录页特征。
        """
        # 风控页很小（约 2KB），先做长度预筛以降低开销
        head = text[:4096]
        for marker in _BLOCK_MARKERS:
            if marker in head:
                raise BlockedError(
                    f"知网返回风控页（命中特征 {marker!r}），已中止。"
                    f"URL={url}。建议提高 network.delay 或稍后再试。"
                )
        for marker in _LOGIN_MARKERS:
            if marker in head:
                raise LoginRequiredError(f"该资源需要登录态（命中特征 {marker!r}）。URL={url}")

    def get(self, url: str, allow_login_page: bool = False, **kwargs) -> str:
        """GET 请求，带限速、重试与反爬检测。

        参数：
            url: 目标地址。
            allow_login_page: 为 True 时登录页不抛异常，而是原样返回文本
                （PC 后端用它来判断 cookie 是否失效）。
            **kwargs: 透传给 requests。

        返回：
            响应体文本。

        异常：
            BlockedError: 触发风控。
            LoginRequiredError: 需要登录且 allow_login_page=False。
            FetchError: 重试耗尽。
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                self._last_request_at = time.monotonic()
                self.request_count += 1
            except requests.RequestException as exc:
                last_error = exc
                self._last_request_at = time.monotonic()
                if attempt < self.max_retries:
                    wait = self.backoff_base * (2**attempt)
                    logger.warning("请求失败(%d/%d)，%.1fs 后重试：%s", attempt + 1, self.max_retries, wait, exc)
                    time.sleep(wait)
                    continue
                raise FetchError(f"请求重试耗尽：{url}") from exc

            # 风控检测先于状态码判断：风控页本身就是 HTTP 200
            if not allow_login_page:
                self._detect_block(response.text, url)
            else:
                # 即便允许登录页，风控页也必须拦下
                text_head = response.text[:4096]
                for marker in _BLOCK_MARKERS:
                    if marker in text_head:
                        raise BlockedError(f"知网返回风控页（命中特征 {marker!r}），已中止。URL={url}")

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "服务端返回 %d(%d/%d)，%.1fs 后重试：%s",
                    response.status_code,
                    attempt + 1,
                    self.max_retries,
                    wait,
                    url,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code}：{url}")

            return response.text

        raise FetchError(f"请求重试耗尽：{url}") from last_error

    def close(self) -> None:
        """关闭底层会话。"""
        self.session.close()

    def __enter__(self) -> PoliteSession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
