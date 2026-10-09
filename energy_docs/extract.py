"""正文抽取：从政府站页面里剥出真正的正文。

中文政府网站用的 CMS 五花八门（TRS、新华云、大汉、拓尔思……），正文容器
的 class 各不相同。所以这里不写死某一个选择器，而是：

1. 先试一批常见的正文容器选择器；
2. 都不中时，按**文本密度**在所有块级元素里挑最像正文的那个；
3. 最后清掉导航、页脚、脚本残留和发布信息行。

抽取结果会带上抽取方式（哪个选择器命中的），便于发现站点改版。
"""

from __future__ import annotations

import html as html_mod
import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# 常见正文容器选择器，按优先级排序
CONTENT_SELECTORS = (
    "#Zoom",
    ".TRS_Editor",
    ".trs_editor",
    "#xw_content",
    "#content",
    ".content",
    "#detail",
    ".detail",
    ".detail-content",
    ".article-content",
    ".article",
    "#article",
    ".view",
    ".zoom",
    ".main-content",
    ".con",
    "#con",
)

# 需要整块丢弃的元素
_DROP_TAGS = ("script", "style", "nav", "header", "footer", "form", "noscript", "iframe")

# 需要丢弃的容器 class/id 关键词
_DROP_PATTERNS = re.compile(
    r"(nav|menu|breadcrumb|crumb|footer|header|sidebar|share|comment|"
    r"related|recommend|advert|\bads?\b|copyright|bottom|top-bar)",
    re.I,
)

# 页面噪声行：发布时间、来源、浏览量、分享等
_NOISE_LINE = re.compile(
    r"^\s*(发布时间|发布日期|来源|责任编辑|浏览|点击|分享到|打印|关闭|"
    r"上一篇|下一篇|相关(阅读|链接|文章)|扫一扫|字体|大中小|"
    r"【打印】|【关闭】|【字体)[:：]?.*$"
)

# 纯符号/空白行
_EMPTY_LINE = re.compile(r"^[\s　\-—_=*·.、，,;；:：|/\\<>\[\]()（）【】]*$")

# 抬头/尾部的编辑信息
_TAIL_NOISE = re.compile(r"(版权所有|京ICP备|政府网站标识码|网站地图|联系我们)")

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t　]+")
_NL_RE = re.compile(r"\n{3,}")


@dataclass
class ExtractedDoc:
    """正文抽取结果。"""

    title: str
    text: str
    selector: str
    """命中的正文选择器；空串表示走了密度兜底。"""

    @property
    def chars(self) -> int:
        """正文有效字符数。"""
        return len(self.text)


class ExtractError(Exception):
    """正文无法抽取。"""


def _soup(html: str) -> BeautifulSoup:
    """构造 BeautifulSoup 对象。"""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover
        return BeautifulSoup(html, "html.parser")


def _prune(soup: BeautifulSoup) -> None:
    """就地删除脚本、导航、页脚等非正文元素。"""
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue
        ident = " ".join(
            str(tag.get(attr, "")) for attr in ("class", "id", "role")
        )
        if ident and _DROP_PATTERNS.search(ident):
            tag.decompose()


# 文章标题选择器，按优先级排序。
# 很多政府站没有 h1，标题只在 <title> 里，而 <title> 往往带着
# 「站点名 栏目名」前缀，所以要优先找更精确的标题元素。
TITLE_SELECTORS = (
    "h1",
    ".article-title",
    ".art_title",
    ".artTitle",
    ".content_title",
    ".content-title",
    ".news_title",
    ".news-title",
    "#title",
    ".title",
)


# 标题容器里常见的尾巴：`标题发布时间：2024-11-09来源：中国人大网大中小`
_TITLE_TAIL_MARKERS = (
    "发布时间", "发布日期", "来源：", "来源:", "大中小", "【打印】", "【关闭】",
    "责任编辑", "浏览次数", "分享到",
)


def _strip_title_noise(raw: str) -> str:
    """砍掉标题里粘着的发布信息。

    很多政府站的标题容器把「标题 + 发布时间 + 来源 + 大中小」塞在同一个
    元素里，取 text 会把它们一起带出来。

    参数：
        raw: 原始标题文本。

    返回：
        只含标题的文本。
    """
    cut = len(raw)
    for marker in _TITLE_TAIL_MARKERS:
        index = raw.find(marker)
        if 0 < index < cut:
            cut = index
    return raw[:cut].strip()


def page_title(soup: BeautifulSoup) -> str:
    """取页面标题，去掉站点与栏目前缀/后缀。

    参数：
        soup: 已解析的页面。

    返回：
        清洗后的标题。
    """
    raw = ""
    for selector in TITLE_SELECTORS:
        node = soup.select_one(selector)
        if node:
            text = node.get_text(strip=True)
            # 标题选择器可能命中导航容器，太长的多半不是标题
            if 4 <= len(text) <= 120:
                raw = text
                break
    if not raw:
        node = soup.select_one("title")
        raw = node.get_text(strip=True) if node else ""

    raw = _strip_title_noise(raw)

    # 后缀：「标题---国家能源局」「标题_山东省能源局」
    raw = re.split(r"\s*[-—_|]{1,3}\s*(?:[^\s]{2,12}(?:局|网|站|委|厅|部|政府))\s*$", raw)[0]
    return raw.strip() or raw


def clean_text(raw: str) -> str:
    """清洗正文文本。

    参数：
        raw: 原始文本。

    返回：
        清洗后的文本。
    """
    text = html_mod.unescape(raw or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)

    lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or _EMPTY_LINE.match(stripped):
            continue
        if _NOISE_LINE.match(stripped) or _TAIL_NOISE.search(stripped):
            continue
        lines.append(stripped)

    return _NL_RE.sub("\n\n", "\n".join(lines)).strip()


def _density_score(node) -> int:
    """给一个候选容器打分：正文字符数减去链接文字占比的惩罚。

    导航区往往全是链接，正文区链接少。所以用「非链接文本量」作为分数。

    参数：
        node: BeautifulSoup 节点。

    返回：
        分数，越大越像正文。
    """
    text = node.get_text(strip=True)
    if not text:
        return 0
    link_chars = sum(len(a.get_text(strip=True)) for a in node.find_all("a"))
    return max(0, len(text) - 2 * link_chars)


def extract_main_text(html: str, min_chars: int = 80) -> ExtractedDoc:
    """从页面里抽取正文。

    参数：
        html: 页面 HTML。
        min_chars: 正文最小字符数，低于此值认为抽取失败。

    返回：
        ExtractedDoc。

    异常：
        ExtractError: 抽不出足够长的正文。
    """
    soup = _soup(html)
    _prune(soup)
    title = page_title(soup)

    # 1. 先试常见正文容器
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if node is None:
            continue
        text = clean_text(node.get_text("\n"))
        if len(text) >= min_chars:
            return ExtractedDoc(title=title, text=text, selector=selector)

    # 2. 密度兜底：在所有 div/article/section 里挑非链接文本最多的
    best_node = None
    best_score = 0
    for tag in soup.find_all(["article", "div", "section", "td"]):
        score = _density_score(tag)
        if score > best_score:
            best_score = score
            best_node = tag

    if best_node is not None:
        text = clean_text(best_node.get_text("\n"))
        if len(text) >= min_chars:
            logger.debug("正文走密度兜底，得分 %d", best_score)
            return ExtractedDoc(title=title, text=text, selector="")

    # 3. 最后退到整页文本
    text = clean_text(soup.get_text("\n"))
    if len(text) >= min_chars:
        logger.debug("正文退到整页文本")
        return ExtractedDoc(title=title, text=text, selector="body")

    raise ExtractError(f"抽不出正文（最长候选仅 {len(text)} 字符）")


# CDATA 数据岛：大汉版通等 CMS 把列表数据塞在 <record><![CDATA[...]]></record> 里
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)


def extract_links(html: str, pattern: str, base: str = "") -> list[tuple[str, str]]:
    """从列表页抽取符合条件的链接。

    同时处理两种形态：

    1. **标准 `<a href="...">`** —— 常规静态列表页；
    2. **CDATA 数据岛里的 `<a href='...'>`** —— 大汉版通等 CMS 会把列表数据
       以 `<record><![CDATA[ <a href='...'>标题</a> ]]></record>` 的形式内嵌在
       页面里。这类内容被 BeautifulSoup 当作**纯文本**，标准解析抓不到，
       所以要把 CDATA 段单独当 HTML 再解析一遍。

    参数：
        html: 列表页 HTML。
        pattern: 链接地址需匹配的正则（对绝对化后的 URL 匹配）。
        base: 相对链接的基准地址。

    返回：
        (标题, 绝对URL) 列表，按出现顺序去重。
    """
    from urllib.parse import urljoin

    regex = re.compile(pattern)
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    def collect(soup: BeautifulSoup) -> None:
        """把 soup 里所有匹配的链接收进 results。"""
        for anchor in soup.find_all("a", href=True):
            text = anchor.get_text(strip=True)
            if not text or len(text) < 4:
                continue
            url = urljoin(base, anchor["href"].strip())
            if not regex.search(url) or url in seen:
                continue
            seen.add(url)
            results.append((text, url))

    collect(_soup(html))

    # 再扫一遍 CDATA 数据岛
    for cdata in _CDATA_RE.findall(html):
        if "<a" in cdata.lower():
            collect(_soup(cdata))

    return results


# 大汉 CMS 的 ajax 参数对象：var param_745545 = {col:1,webid:355,...};
_JS_PARAM_RE = re.compile(
    r"var\s+(param_\d+)\s*=\s*\{(.*?)\}\s*;", re.S
)


def _parse_js_params(body: str) -> dict[str, str]:
    """解析大汉 CMS 的 JS 参数对象体。

    形如 `col:1,webid:355,path:'http://...',columnid:59960`。

    参数：
        body: 花括号内的内容。

    返回：
        参数名到值的映射。
    """
    params: dict[str, str] = {}
    for match in re.finditer(r"(\w+)\s*:\s*(?:'([^']*)'|\"([^\"]*)\"|([^,}]+))", body):
        key = match.group(1)
        value = match.group(2) or match.group(3) or match.group(4) or ""
        params[key] = value.strip()
    return params


def find_data_proxy(html: str) -> str | None:
    """找出大汉 CMS 的分页数据接口（dataproxy.jsp）。

    ## 为什么不能只抄页面里那个 URL

    页面 `<nextgroup>` 里贴的 dataproxy URL 本身是**不完整**的——直接带
    `page=2` 请求它会返回空白。真正的参数分散在页面脚本里：

        var param_745545 = {col:1,webid:355,path:'...',columnid:59960,
                            sourceContentType:1,unitid:'745545',...};

    所以这里把 JS 参数对象解析出来，和接口路径拼成一条完整可用的 URL。
    页码用 `{page}` 占位，调用方格式化。

    参数：
        html: 列表页 HTML。

    返回：
        含 `{page}` 占位的完整接口 URL；没找到返回 None。
    """
    proxy_match = re.search(r"/module/web/jpage/dataproxy\.jsp", html)
    if not proxy_match:
        return None

    path = proxy_match.group(0)

    # 找到紧邻的 JS 参数对象
    params: dict[str, str] = {}
    for match in _JS_PARAM_RE.finditer(html):
        body = match.group(2)
        if "columnid" in body and "webid" in body:
            params = _parse_js_params(body)
            break

    # 没解析到参数对象时，退回页面里那条 URL（可能不完整，但聊胜于无）
    if not params:
        inline = re.search(r"/module/web/jpage/dataproxy\.jsp\?[^\"'\s>]+", html)
        if not inline:
            return None
        url = html_mod.unescape(inline.group(0))
        return re.sub(r"([?&])page=\d+", r"\1page={page}", url)

    query = "&".join(f"{k}={v}" for k, v in params.items() if k != "col")
    return f"{path}?page={{page}}&{query}"
