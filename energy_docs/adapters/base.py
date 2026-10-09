"""站点适配器：配置驱动的通用采集器。

## 为什么不做「一站一个适配器」

全国有几十个能源主管机构网站（国家能源局 + 30 个省级能源局/发改委）。逐站硬编码
是不可维护的。

实测发现这些站绝大多数落在少数几个 CMS 家族里（大汉版通、拓尔思、TRS），
链接和分页模式高度雷同：

    山东能源局   /art/2026/9/20/art_59966_10313615.html
    浙江发改委   /col/col1629218/art/2026/art_<hash>.html
    山西能源局   ./zfxxgk/fdzdgknr/sjwj/5/t5_3.shtml

所以设计成：**一份站点清单（配置）+ 一个会自动尝试多种分页策略的通用适配器**。
新增一个站点通常只需要往 config.toml 里加几行，不用写代码。

只有遇到真正不兼容的站（比如纯 JS 渲染的）才需要单独写适配器。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from ..extract import (
    ExtractError,
    clean_text,
    extract_links,
    extract_main_text,
    find_data_proxy,
)
from ..fetcher import BlockedError, FetchError, PoliteSession

logger = logging.getLogger(__name__)

# 分页 URL 生成策略。按顺序尝试，第一个能产出新内容的胜出。
# 中文政府 CMS 的分页基本逃不出这几种。
PAGINATION_TEMPLATES = (
    "{dir}/index_{page}.html",
    "{dir}/index_{page}.shtml",
    "{dir}/index_{page}.htm",
    "{entry}?pageNum={page}",
    "{entry}?page={page}",
    "{dir}/list_{page}.html",
)


@dataclass
class SiteSpec:
    """一个站点的采集配置。"""

    name: str
    """站点名称，写入文档的 source 字段。"""

    base: str
    """站点根地址，形如 https://www.nea.gov.cn。"""

    entry: str
    """列表页入口路径，形如 /nyflfg/index.htm。"""

    link_pattern: str
    """内容页 URL 的正则（对绝对化后的 URL 匹配）。"""

    category: str = ""
    """栏目名，写入文档的 category 字段。"""

    max_pages: int = 40
    """最多翻多少页。防止分页策略失效时无限翻。"""

    min_chars: int = 120
    """正文最小字符数，低于此值视为抽取失败。"""

    def entry_url(self) -> str:
        """返回列表页的绝对地址。"""
        return urljoin(self.base, self.entry)


@dataclass
class CrawlResult:
    """一次站点采集的结果。"""

    source: str
    pages: int = 0
    found: int = 0
    saved: int = 0
    skipped: int = 0
    failed: int = 0
    blocked: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """返回人类可读的摘要。"""
        parts = [f"翻页 {self.pages}", f"发现 {self.found}", f"入库 {self.saved}"]
        if self.skipped:
            parts.append(f"跳过 {self.skipped}")
        if self.failed:
            parts.append(f"失败 {self.failed}")
        if self.blocked:
            parts.append("** 被拦截 **")
        return "，".join(parts)


def build_page_urls(spec: SiteSpec, page: int) -> list[str]:
    """为一个页码生成候选 URL 列表。

    参数：
        spec: 站点配置。
        page: 页码，从 2 开始（第 1 页用入口地址）。

    返回：
        候选 URL 列表。
    """
    parsed = urlparse(spec.entry_url())
    directory = parsed.path.rsplit("/", 1)[0]
    context = {
        "dir": directory,
        "page": page,
        "entry": spec.entry_url(),
    }
    base = f"{parsed.scheme}://{parsed.netloc}"
    seen: set[str] = set()
    urls: list[str] = []
    for template in PAGINATION_TEMPLATES:
        # `{entry}` 模板里已经是完整地址，不能再拼 base——
        # 否则会拼出 `//host/pathhttps://host/path` 这种坏 URL
        if "{entry}" in template:
            candidate = template.format(**context)
        else:
            candidate = base + template.format(**context)
        if candidate not in seen:
            seen.add(candidate)
            urls.append(candidate)
    return urls


class SiteAdapter:
    """配置驱动的站点采集器。"""

    def __init__(self, spec: SiteSpec, session: PoliteSession) -> None:
        """初始化。

        参数：
            spec: 站点配置。
            session: 限速后的 HTTP 会话。
        """
        self.spec = spec
        self.session = session
        self._link_re = re.compile(spec.link_pattern)
        self._working_page_templates: int | None = None
        # 大汉 CMS 的分页接口模板（含 {page} 占位），入口页解析后填上
        self._data_proxy: str | None = None

    # ---------- 列表 ----------

    def list_page(self, url: str) -> list[tuple[str, str]]:
        """抓取并解析一个列表页。

        参数：
            url: 列表页地址。

        返回：
            (标题, 绝对URL) 列表。取不到时返回空列表。
        """
        try:
            html = self.session.get(url, expect_html=False)
        except (FetchError, BlockedError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("列表页抓取失败 %s：%s", url, exc)
            return []

        # 顺便记下官方分页接口——它比猜 index_2.html 可靠得多
        if self._data_proxy is None:
            proxy = find_data_proxy(html)
            if proxy:
                self._data_proxy = urljoin(url, proxy)
                logger.debug("%s：发现分页接口 %s", self.spec.name, self._data_proxy[:90])

        items = extract_links(html, self.spec.link_pattern, base=url)
        # 兜底：有些站用相对路径且不带前缀，直接用正则全文匹配 href
        if not items:
            items = self._regex_fallback(html, url)
        return items

    def _regex_fallback(self, html: str, base: str) -> list[tuple[str, str]]:
        """extract_links 没命中时的兜底：直接在 href 上跑正则。"""
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        for match in re.finditer(r'href=["\']([^"\']+)["\'][^>]*>([^<]{4,80})', html):
            href, text = match.group(1), match.group(2).strip()
            url = urljoin(base, href)
            if self._link_re.search(url) and url not in seen:
                seen.add(url)
                results.append((clean_text(text), url))
        return results

    def iter_listings(self) -> Iterator[tuple[str, str]]:
        """从入口页开始翻页，逐条产出内容页链接。

        分页策略：先试第 2 页的各个候选 URL，哪个能返回**新的**内容链接就固定用它。

        生成：
            (标题, 绝对URL)。
        """
        seen_urls: set[str] = set()

        first = self.list_page(self.spec.entry_url())
        for title, url in first:
            if url not in seen_urls:
                seen_urls.add(url)
                yield title, url
        if not first:
            logger.warning("%s：入口页没解析到任何内容链接，请检查 link_pattern", self.spec.name)
            return

        for page in range(2, self.spec.max_pages + 1):
            items = self._fetch_page_candidates(page, seen_urls)
            if not items:
                break
            for title, url in items:
                if url not in seen_urls:
                    seen_urls.add(url)
                    yield title, url

    def _fetch_page_candidates(
        self, page: int, seen_urls: set[str]
    ) -> list[tuple[str, str]]:
        """尝试为一个页码找到能返回新内容的 URL。

        参数：
            page: 页码。
            seen_urls: 已见过的 URL 集合，用于判断「新内容」。

        返回：
            新链接列表；所有候选都失败时返回空列表。
        """
        # 有官方分页接口时优先用它。但实测部分大汉站的分页接口对非浏览器请求
        # 一律返回空白（需要会话校验），所以拿不到内容时要放弃它并退回常规策略，
        # 否则每个站都会白白浪费一次请求。
        if self._data_proxy:
            try:
                items = self.list_page(self._data_proxy.format(page=page))
            except BlockedError:
                raise
            except FetchError:
                items = []
            fresh = [(t, u) for t, u in items if u not in seen_urls]
            if fresh:
                return fresh
            logger.debug("%s：分页接口无返回，改用常规分页策略", self.spec.name)
            self._data_proxy = None

        candidates = build_page_urls(self.spec, page)
        # 已确定可用模板时只试它
        if self._working_page_templates is not None:
            candidates = [candidates[self._working_page_templates]]

        for index, url in enumerate(candidates):
            try:
                items = self.list_page(url)
            except BlockedError:
                raise
            except FetchError:
                continue

            fresh = [(t, u) for t, u in items if u not in seen_urls]
            if fresh:
                self._working_page_templates = index
                return fresh

        return []

    # ---------- 详情 ----------

    def fetch_document(self, title: str, url: str):
        """抓取并抽取一篇文档的正文。

        参数：
            title: 列表页上的标题（详情页标题为空时兜底）。
            url: 文档地址。

        返回：
            Document；抽取失败时返回 None。
        """
        from ..store import Document

        try:
            html = self.session.get(url, expect_html=False)
        except BlockedError:
            raise
        except FetchError as exc:
            logger.debug("详情页抓取失败 %s：%s", url, exc)
            return None

        try:
            extracted = extract_main_text(html, min_chars=self.spec.min_chars)
        except ExtractError as exc:
            logger.debug("正文抽取失败 %s：%s", url, exc)
            return None

        return Document(
            url=url,
            title=self._clean_title(extracted.title) or title,
            source=self.spec.name,
            cms=self.spec.category or "generic",
            category=self.spec.category,
            text=extracted.text,
            selector=extracted.selector,
        )

    def _clean_title(self, raw: str) -> str:
        """剥掉标题里混入的站点名与栏目名。

        很多政府站的 `<title>` 是 `{站点名} {栏目名} {文章标题}` 三段式，
        例如「山东省能源局 动态要闻 局主要负责同志调研…」。

        剥站名是可靠的（配置里有）。剥栏目名用了一个保守的判据：**只有确认标题
        以站点名开头时**（说明确实是三段式）才再剥一段，避免把「关于 2026 年…」
        这类正常标题的开头误剥掉。

        参数：
            raw: 抽取到的原始标题。

        返回：
            清洗后的标题。
        """
        title = (raw or "").strip()
        name = self.spec.name

        # 配置里的站点名常带栏目后缀，如「山东省能源局-通知公告」，
        # 而页面上的前缀只有「山东省能源局」。所以要把后缀也拆出来做别名。
        aliases = {name}
        for sep in ("-", "_", "·"):
            if sep in name:
                aliases.add(name.split(sep, 1)[0].strip())

        stripped = re.sub(r"[省市自治区发展和改革委员会]+", "", name)
        if stripped and stripped != name:
            aliases.add(stripped)

        matched_site = False
        for alias in aliases:
            for sep in (" ", "　", "-", "_", "|"):
                if title.startswith(alias + sep):
                    title = title[len(alias) + 1 :].strip()
                    matched_site = True
                    break
            if matched_site:
                break

        # 确认是三段式后，再剥掉栏目名那一段
        if matched_site:
            parts = title.split(" ", 1)
            if len(parts) == 2 and 2 <= len(parts[0]) <= 8 and len(parts[1]) >= 8:
                title = parts[1].strip()

        return title or raw
