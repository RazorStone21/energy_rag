"""知网 wap 页面解析器：期刊页、期次页、文章详情页。

所有解析函数都是**纯函数**（输入 HTML 字符串，输出数据结构），不涉及网络，
因此可以完全离线用 fixtures 做单元测试。

选择器基于 2026-09-22 实测的真实页面结构。知网改版时这里会最先失效，
所以每个函数在关键字段缺失时都会抛 `ParseError` 而不是返回半条数据——
宁可报错，也不要往库里塞残缺记录。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from bs4 import BeautifulSoup

# 文章 ID：如 DLXT202613007、DYLC2026092000F
_ARTICLE_ID_RE = re.compile(r"/Journal/Article/([A-Za-z0-9]+)\.html")

# 期次页/详情页的 title 末尾形态：{刊名}{年}年{期}期-手机知网
_TITLE_TAIL_RE = re.compile(r"(?P<journal>[^-]+?)(?P<year>\d{4})年(?P<issue>\d{2})期-手机知网\s*$")

# 详情页 title 形态：{文章标题}-{刊名}{年}年{期}期-手机知网
_TITLE_DETAIL_RE = re.compile(
    r"^(?P<article_title>.+)-(?P<journal>[^-]+?)(?P<year>\d{4})年(?P<issue>\d{2})期-手机知网\s*$"
)

# 摘要截断标志：知网服务端把摘要截到约 110 字，末尾补省略号
_TRUNCATION_SUFFIXES = ("...", "…")

# PDF 大小文本：下载PDF版(1610K)
_PDF_SIZE_RE = re.compile(r"下载PDF版\s*\(\s*(\d+)\s*K\s*\)", re.I)

# 学科代码文本：C042;I140
_SUBJECT_CODE_RE = re.compile(r"^[A-Z]\d{3}(;[A-Z]\d{3})*$")


class ParseError(Exception):
    """页面结构与预期不符，解析失败。"""


@dataclass
class ArticleRef:
    """列表页里的一条文章引用（只有 ID 和标题等信息，尚无完整题录）。"""

    article_id: str
    title: str
    section: str = ""
    authors: list[str] = field(default_factory=list)
    date: str = ""


@dataclass
class IssueInfo:
    """期次页解析结果。"""

    pykm: str
    year: int
    issue: int
    journal: str
    articles: list[ArticleRef] = field(default_factory=list)


@dataclass
class JournalMeta:
    """期刊页解析结果。"""

    pykm: str
    name: str
    subject_codes: str = ""
    online_first: list[ArticleRef] = field(default_factory=list)


@dataclass
class ArticleRecord:
    """一篇论文的题录。

    `abstract` 在 wap 后端下是**服务端截断的约 110 字预览**，
    此时 `abstract_truncated` 为 True。不要把它当作完整摘要使用。
    """

    article_id: str
    title: str
    journal: str = ""
    pykm: str = ""
    year: int = 0
    issue: int = 0
    authors: list[str] = field(default_factory=list)
    affiliations: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    abstract: str = ""
    abstract_truncated: bool = False
    # 基金项目。wap 页面不提供，仅 PC 后端能拿到。
    fund: str = ""
    cited_count: int = 0
    download_count: int = 0
    pdf_size_kb: int = 0
    url: str = ""
    # 数据来源后端："wap" 或 "pc"。如实记录，便于判断摘要是否完整。
    backend: str = "wap"

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的字典。"""
        return asdict(self)


def _soup(html: str) -> BeautifulSoup:
    """构造 BeautifulSoup 对象，优先用 lxml，回退到标准库解析器。"""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover - 环境缺 lxml 时的兜底
        return BeautifulSoup(html, "html.parser")


def _text(node) -> str:
    """取节点的去空白文本，节点为空时返回空串。"""
    if node is None:
        return ""
    return node.get_text(strip=True)


def _split_semicolon(raw: str) -> list[str]:
    """按中文/英文分号切分并清洗，去掉空项和尾部残留分隔符。"""
    if not raw:
        return []
    parts = re.split(r"[；;]", raw)
    return [p.strip() for p in parts if p.strip()]


def _append_unique(target: list[str], values: list[str]) -> None:
    """把 values 中尚未出现过的项依次追加进 target（去重，保持顺序）。

    知网同一机构会在"作者行内单位"和"机构"字段各出现一次，
    且后者的文本带尾部分号，所以比较前要先规范化。
    """
    seen = {_normalize_token(v) for v in target}
    for value in values:
        key = _normalize_token(value)
        if key and key not in seen:
            target.append(key)
            seen.add(key)


def _normalize_token(raw: str) -> str:
    """规范化用于去重比较的文本：去空白、去尾部分号。"""
    return (raw or "").strip().rstrip("；;").strip()


def parse_article_id(url: str) -> str:
    """从文章链接里提取文章 ID。

    参数：
        url: 形如 `//wap.cnki.net/touch/web/Journal/Article/DLXT202613007.html`。

    返回：
        文章 ID，如 `DLXT202613007`；匹配失败返回空串。
    """
    match = _ARTICLE_ID_RE.search(url or "")
    return match.group(1) if match else ""


def parse_article_page(html: str, article_id: str = "") -> ArticleRecord:
    """解析文章详情页，返回题录。

    参数：
        html: 详情页 HTML。
        article_id: 文章 ID，可从链接得到；若为空则尝试从页面推断。

    返回：
        ArticleRecord。

    异常：
        ParseError: 标题缺失（说明拿到的不是详情页）。
    """
    soup = _soup(html)

    title_node = soup.select_one(".c-card__title2")
    title = _text(title_node)
    if not title:
        # 兜底：从 title 标签解析
        title_match = _TITLE_DETAIL_RE.match(_text(soup.title))
        if title_match:
            title = title_match.group("article_title").strip()
    if not title:
        raise ParseError("详情页缺少标题，页面结构可能已改版")

    record = ArticleRecord(article_id=article_id, title=title)

    # 刊名、年、期
    book_title = _text(soup.select_one(".c-book__title"))
    record.journal = book_title
    tail = _TITLE_TAIL_RE.search(_text(soup.title))
    if tail:
        if not record.journal:
            record.journal = tail.group("journal").strip()
        record.year = int(tail.group("year"))
        record.issue = int(tail.group("issue"))

    # 作者与机构（作者行内）
    author_box = soup.select_one(".c-card__author")
    if author_box:
        _append_unique(record.authors, [_text(a) for a in author_box.select('a[href*="/Scholar/Index/"]')])
        _append_unique(
            record.affiliations,
            [_text(a) for a in author_box.select('a[href*="/Organization/List/"]')],
        )

    # 摘要
    record.abstract = _text(soup.select_one(".c-card__aritcle"))
    if record.abstract.endswith(_TRUNCATION_SUFFIXES):
        record.abstract_truncated = True
        record.abstract = record.abstract.rstrip(".…").strip()
    elif record.abstract:
        # 知网偶发不加省略号但同样截断；wap 后端一律标记为截断
        record.abstract_truncated = True

    # 机构 / 领域 / 关键词：标签-内容成对出现
    for item in soup.select(".c-card__paper-item"):
        label = _text(item.select_one(".c-card__paper-name"))
        content = item.select_one(".c-card__paper-content")
        if content is None:
            continue
        if "机" in label and "构" in label:
            _append_unique(record.affiliations, [_text(a) for a in content.select("a")])
        elif "领" in label and "域" in label:
            _append_unique(record.subjects, [_text(a) for a in content.select("a")])
        elif "关键词" in label:
            _append_unique(record.keywords, [_text(a) for a in content.select("a")])

    # 被引 / 下载
    record.cited_count = _to_int(_text(soup.select_one(".c-card__statics-resource")))
    record.download_count = _to_int(_text(soup.select_one(".c-card__statics-download")))

    # PDF 大小
    size_match = _PDF_SIZE_RE.search(html)
    if size_match:
        record.pdf_size_kb = int(size_match.group(1))

    # 期刊代码：详情页底部一般有 NetStartList/{PYKM}.html 链接
    pykm_link = soup.select_one('a[href*="Journal/NetStartList/"]')
    if pykm_link:
        pykm_match = re.search(r"/NetStartList/([A-Za-z0-9]+)\.html", pykm_link.get("href", ""))
        if pykm_match:
            record.pykm = pykm_match.group(1)

    if not article_id:
        detail_link = soup.select_one('a[href*="/Journal/Article/"]')
        if detail_link:
            record.article_id = parse_article_id(detail_link.get("href", ""))

    record.url = f"https://wap.cnki.net/touch/web/Journal/Article/{record.article_id}.html"
    return record


def parse_issue_page(html: str, pykm: str = "") -> IssueInfo:
    """解析期次页，返回该期全部文章引用。

    参数：
        html: 期次页 HTML。
        pykm: 期刊代码，用于回填。

    返回：
        IssueInfo。

    异常：
        ParseError: 未找到任何文章条目。
    """
    soup = _soup(html)

    tail = _TITLE_TAIL_RE.search(_text(soup.title))
    if not tail:
        raise ParseError("期次页 title 结构不符，无法确定年/期")
    journal = tail.group("journal").strip()
    year = int(tail.group("year"))
    issue = int(tail.group("issue"))

    info = IssueInfo(pykm=pykm, year=year, issue=issue, journal=journal)

    # 目录按栏目分组：每个 .c-catalog__subtitle 下跟一组 .c-catalog__item
    current_section = ""
    for node in soup.select(".c-catalog__subtitle, a.c-catalog__item"):
        classes = node.get("class") or []
        if "c-catalog__subtitle" in classes:
            current_section = _text(node)
            continue
        article_id = parse_article_id(node.get("href", ""))
        title = _text(node.select_one(".c-catalog__item-div")) or _text(node)
        if article_id and title:
            info.articles.append(ArticleRef(article_id, title, current_section))

    if not info.articles:
        raise ParseError(f"期次页未解析出任何文章：{journal} {year}年{issue}期")

    return info


def parse_journal_page(html: str, pykm: str = "") -> JournalMeta:
    """解析期刊页，返回期刊元数据与网络首发列表。

    参数：
        html: 期刊页 HTML。
        pykm: 期刊代码。

    返回：
        JournalMeta。

    异常：
        ParseError: 刊名缺失。
    """
    soup = _soup(html)

    name = _text(soup.select_one(".c-book__title"))
    if not name:
        # 兜底：从 title 标签取（形如 "电力系统自动化-手机知网"）
        name = re.sub(r"-手机知网\s*$", "", _text(soup.title)).strip()
    if not name:
        raise ParseError(f"期刊页缺少刊名：PYKM={pykm}")

    meta = JournalMeta(pykm=pykm, name=name)

    # 学科代码：页面上形如 C042;I140 的短文本
    for node in soup.find_all(string=_SUBJECT_CODE_RE):
        code_text = node.strip()
        if code_text:
            meta.subject_codes = code_text
            break

    # 网络首发：a.c-catalog__item-web 内含标题、作者、日期三个子节点
    for link in soup.select("a.c-catalog__item-web"):
        article_id = parse_article_id(link.get("href", ""))
        if not article_id:
            continue
        title = _text(link.select_one(".c-catalog__item-div"))
        if not title:
            continue
        authors = _split_semicolon(_text(link.select_one(".c-catalog__item-name")))
        date = _text(link.select_one(".c-catalog__item-time"))
        if any(ref.article_id == article_id for ref in meta.online_first):
            continue
        meta.online_first.append(ArticleRef(article_id, title, "网络首发", authors, date))

    return meta


def _to_int(raw: str) -> int:
    """把页面上的计数文本转成整数，失败返回 0。"""
    if not raw:
        return 0
    match = re.search(r"\d+", raw.replace(",", ""))
    return int(match.group()) if match else 0
