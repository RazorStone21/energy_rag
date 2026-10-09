"""HTTP 层：限速、重试退避、拦截检测。

中文政府站点常见两类拦截，都会返回 HTTP 200 或 403：

1. **WAF 拦截页**（云 WAF 返回的「访问被拒绝」「请稍后重试」页面）；
2. **站点维护页**（实测中「中国科技论文在线」长期是维护提示页）。

这类页面如果被当成正常内容交给抽取器，产出的就是垃圾文本。所以在这一层
就要识别并抛错，而不是让它流到下游。
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

# 拦截页特征。命中即抛 BlockedError。
_BLOCK_MARKERS = (
    "访问被拒绝",
    "Access Denied",
    "403 Forbidden",
    "您的访问过于频繁",
    "请稍后再试",
    "网站维护",
    "维护升级",
    "系统维护中",
    "正在维护",
)

# 页面太短通常意味着拿到的不是正文（错误页、跳转页）
MIN_USEFUL_BYTES = 512

# 这几种编码能"成功"解码任意字节序列，永远不会抛 UnicodeDecodeError。
# 如果响应头声明了它们，不能盲信——否则 UTF-8 的中文会被静默解成乱码
# （`中华人民共和国` → `ä¸­åäººæ°'å...`），而且没有任何异常提示。
# 实测政府站里确实有声明 ISO-8859-1 却发 UTF-8 的情况。
_UNRELIABLE_CHARSETS = frozenset(
    {"iso-8859-1", "iso8859-1", "latin1", "latin-1", "windows-1252", "cp1252",
     "ascii", "us-ascii", "ansi_x3.4-1968"}
)

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class FetchError(Exception):
    """重试耗尽后仍无法取得有效响应。"""


class BlockedError(FetchError):
    """被站点拦截或站点不可用。

    这是致命错误，应立即中止该数据源的采集。
    """


class PoliteSession:
    """带限速与拦截检测的 HTTP 会话。"""

    def __init__(
        self,
        delay: float = 2.0,
        timeout: int = 25,
        max_retries: int = 3,
        backoff_base: float = 5.0,
        user_agent: str | None = None,
    ) -> None:
        """初始化。

        参数：
            delay: 两次请求之间的最小间隔秒数。
            timeout: 单次请求超时秒数。
            max_retries: 网络错误与 5xx 的最大重试次数。
            backoff_base: 指数退避基数。
            user_agent: 自定义 UA。
        """
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base

        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        self.session.headers["Accept"] = "text/html,application/xhtml+xml,*/*;q=0.8"
        self.session.headers["Accept-Language"] = "zh-CN,zh;q=0.9"

        self._last_at = 0.0
        self.request_count = 0

    def _throttle(self) -> None:
        """确保与上一次请求间隔至少 delay 秒。"""
        if self.delay <= 0:
            return
        remaining = self.delay - (time.monotonic() - self._last_at)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def detect_block(text: str, url: str) -> None:
        """检测响应体是否为拦截页/维护页，是则抛 BlockedError。

        参数：
            text: 响应体文本。
            url: 请求地址，用于错误信息。

        异常：
            BlockedError: 命中拦截特征。
        """
        head = text[:4096]
        for marker in _BLOCK_MARKERS:
            if marker in head:
                raise BlockedError(
                    f"站点返回拦截/维护页（命中特征 {marker!r}）。URL={url}。"
                    "建议稍后再试或提高 delay。"
                )

    @staticmethod
    def decode_response(response) -> str:
        """按正确的编码解码响应体。

        **这一步不能交给 `response.text`。** requests 在响应头没有声明 charset
        时会退回 ISO-8859-1，而 Latin-1 对任意字节序列都不会报错——结果是中文页面
        被静默解码成乱码（`中华人民共和国` → `ä¸­åäººæ°'å...`），
        而且不会有任何异常提示。政府站大量不声明 charset，踩这个坑的概率很高。

        解码顺序：响应头声明的 charset → UTF-8 → charset_normalizer 的猜测 →
        GB18030（GBK/GB2312 的超集）。全部失败才用替换模式兜底。

        参数：
            response: requests 的响应对象。

        返回：
            解码后的文本。
        """
        raw = response.content

        content_type = response.headers.get("Content-Type", "")
        declared = ""
        if "charset=" in content_type.lower():
            declared = content_type.lower().split("charset=", 1)[1].split(";")[0].strip()

        candidates: list[str] = []

        # 声明的编码可信时才优先用它；Latin-1 家族放到最后兜底
        if declared and declared not in _UNRELIABLE_CHARSETS:
            candidates.append(declared)

        # 现代政府站绝大多数是 UTF-8
        candidates.append("utf-8")

        apparent = response.apparent_encoding
        if apparent:
            candidates.append(apparent)

        candidates.append("gb18030")

        if declared:
            candidates.append(declared)

        seen: set[str] = set()
        for encoding in candidates:
            key = encoding.lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue

        return raw.decode("utf-8", errors="replace")

    def get(self, url: str, expect_html: bool = True, **kwargs) -> str:
        """GET 请求，带限速、重试与拦截检测。

        参数：
            url: 目标地址。
            expect_html: 为 True 时要求响应体达到最小长度（防止拿到空白页）。
            **kwargs: 透传给 requests。

        返回：
            响应体文本。

        异常：
            BlockedError: 被拦截。
            FetchError: 重试耗尽或响应不可用。
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                self._last_at = time.monotonic()
                self.request_count += 1
            except requests.RequestException as exc:
                last_error = exc
                self._last_at = time.monotonic()
                if attempt < self.max_retries:
                    wait = self.backoff_base * (2**attempt)
                    logger.warning(
                        "请求失败(%d/%d)，%.1fs 后重试：%s",
                        attempt + 1, self.max_retries, wait, exc,
                    )
                    time.sleep(wait)
                    continue
                raise FetchError(f"请求重试耗尽：{url}") from exc

            # 解码必须显式做，不能依赖 response.text（见 decode_response 说明）
            text = self.decode_response(response)
            self.detect_block(text, url)

            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "服务端返回 %d(%d/%d)，%.1fs 后重试：%s",
                    response.status_code, attempt + 1, self.max_retries, wait, url,
                )
                time.sleep(wait)
                continue

            if response.status_code >= 400:
                raise FetchError(f"HTTP {response.status_code}：{url}")

            if expect_html and len(response.content) < MIN_USEFUL_BYTES:
                raise FetchError(
                    f"响应体过小（{len(response.content)} 字节），不是有效正文：{url}"
                )

            return text

        raise FetchError(f"请求重试耗尽：{url}") from last_error

    def close(self) -> None:
        """关闭底层会话。"""
        self.session.close()

    def __enter__(self) -> PoliteSession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
