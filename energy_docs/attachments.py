"""附件发现与下载：把政府站上的 PDF / DOC / DOCX / XLS 原件抓下来。

## 为什么需要单独一个模块

实测发现中文政府站有两种发文形态，需要区别对待：

1. **附件直发**：列表页的每条就是一个附件直链（辽宁发 PDF、天津发 DOCX）。
   这类站命中率接近 100%，是主要产出源。
2. **文章页带附件**：列表页链到 HTML 正文，正文页里再挂附件。
   抽样 56 篇的结果是只有 7% 带 PDF、23% 带 Office 附件。

所以下载器要同时从**列表页**和**文章页**里找附件。

## 附件链接的两种写法

- 直链：`href=".../xxx.pdf"`
- pdfjs 预览器包装：`viewer.html?file=/fgw/articleFileDir/.../xxx.pdf`
  ——真实文件地址在 `file=` 参数里，是相对路径，必须解析出来。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup

from .fetcher import BlockedError, FetchError, PoliteSession

logger = logging.getLogger(__name__)

# 要抓的附件类型
ATTACHMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".wps", ".ofd")

# pdfjs 预览器的真实文件参数
_PDFJS_RE = re.compile(r"viewer\.html\?file=([^\"'&]+)", re.I)

# 直链附件。两条约束都是踩过坑才加上的：
#
# 1. **必须限定在属性里**。早期版本只匹配「引号包着以 .pdf 结尾的串」，
#    结果把 `title="xxx.pdf"`、JS 数组里的裸文件名也当成链接，拼出一堆假地址。
# 2. **属性名前必须有词边界**。贵州的附件标签同时带 `href="./P0202...xlsx"`
#    和已失效的 `oldsrc="/protect/P0202...xlsx"`；没有 `\b` 时 `src=` 会命中
#    `oldsrc=`，于是采到一堆 404 的旧路径。
_DIRECT_RE = re.compile(
    r"""\b(?:href|src|data-file|data-url|data-src)\s*=\s*["']([^"']+\.(?:"""
    + "|".join(s.lstrip(".") for s in ATTACHMENT_SUFFIXES)
    + r"""))["']""",
    re.I,
)

# 文件名里不允许的字符
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 文件类型魔数，用于校验下到的确实是对应格式
_MAGIC = {
    b"%PDF": "pdf",
    b"\xd0\xcf\x11\xe0": "ole",  # 旧版 doc/xls
    b"PK\x03\x04": "zip",  # docx/xlsx（OOXML 是 zip）
}


@dataclass
class Attachment:
    """一个待下载的附件。"""

    url: str
    """绝对地址。"""

    kind: str
    """预期类型：pdf / docx / xlsx …"""

    title: str = ""
    """所属文章标题，用于命名。"""

    source_page: str = ""
    """发现它的页面地址。"""

    def suggested_name(self) -> str:
        """建议的文件名（含扩展名）。

        优先用「文章标题 + 原扩展名」，标题为空时退回 URL 里的文件名。

        返回：
            安全的文件名。
        """
        extension = Path(urlparse(self.url).path).suffix.lower()
        if extension not in ATTACHMENT_SUFFIXES:
            extension = f".{self.kind}"

        base = _UNSAFE_RE.sub("", (self.title or "").strip())
        # 标题常常自带扩展名（列表页 anchor 文本就是「xxx.docx」），
        # 不先去掉就会生成 `xxx.docx.docx`
        base = re.sub(
            r"\.(?:" + "|".join(s.lstrip(".") for s in ATTACHMENT_SUFFIXES) + r")$",
            "",
            base,
            flags=re.I,
        )
        base = re.sub(r"\s+", "", base)[:80].strip(".-")
        if not base:
            base = _UNSAFE_RE.sub("", unquote(Path(urlparse(self.url).path).stem))[:80]
        if not base:
            base = "attachment"

        return f"{base}{extension}"


@dataclass
class DownloadStats:
    """下载统计。"""

    found: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_written: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """返回人类可读的摘要。"""
        mb = self.bytes_written / 1024 / 1024
        lines = [
            f"发现附件      {self.found}",
            f"已下载        {self.downloaded}",
            f"跳过（已存在）{self.skipped}",
            f"失败          {self.failed}",
            f"总大小        {self.bytes_written:,} 字节 ({mb:.1f} MB)",
        ]
        if self.by_kind:
            kinds = "，".join(f"{k} {v}" for k, v in sorted(self.by_kind.items()))
            lines.append(f"按类型        {kinds}")
        return "\n".join(lines)


def find_attachments(html: str, base_url: str, title: str = "") -> list[Attachment]:
    """从页面里找出所有附件链接。

    标题优先取**锚文本**——列表页直挂附件时，锚文本就是通知标题，用它命名才有
    意义；一律用传入的 `title`（往往是站点名）会产出
    `天津市发展改革委-通知公告-09e5f5a7.docx` 这种零信息量的文件名。

    参数：
        html: 页面 HTML。
        base_url: 页面地址，用于把相对链接绝对化。
        title: 兜底标题（锚文本为空时用）。

    返回：
        Attachment 列表，按 URL 去重。
    """
    found: dict[str, Attachment] = {}

    def add(raw_url: str, anchor_text: str = "") -> None:
        """把一个候选链接收进来。"""
        candidate = raw_url.strip().strip("\"'")
        if not candidate or candidate.startswith(("javascript:", "mailto:", "#")):
            return
        url = urljoin(base_url, candidate)
        if not url.lower().startswith(("http://", "https://")):
            return
        suffix = Path(urlparse(url).path).suffix.lower()
        if suffix not in ATTACHMENT_SUFFIXES:
            return
        if url in found:
            return
        found[url] = Attachment(
            url=url,
            kind=suffix.lstrip("."),
            title=(anchor_text or title).strip(),
            source_page=base_url,
        )

    # 1. 优先走 DOM：能拿到锚文本当标题
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:  # pragma: no cover
        soup = BeautifulSoup(html, "html.parser")

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = anchor.get_text(strip=True) or anchor.get("title", "")
        viewer = _PDFJS_RE.search(href)
        if viewer:
            add(viewer.group(1), text)
            continue
        if Path(urlparse(href).path).suffix.lower() in ATTACHMENT_SUFFIXES:
            add(href, text)

    # 2. 兜底走正则：覆盖 src / data-* 属性里的附件（这类没有锚文本可用）
    for match in _PDFJS_RE.finditer(html):
        add(match.group(1))
    for match in _DIRECT_RE.finditer(html):
        add(match.group(1))

    return list(found.values())


def sniff_kind(payload: bytes) -> str:
    """按魔数判断文件真实类型。

    参数：
        payload: 文件开头若干字节。

    返回：
        `pdf` / `ole` / `zip` / `unknown`。
    """
    for magic, kind in _MAGIC.items():
        if payload.startswith(magic):
            return kind
    return "unknown"


class AttachmentDownloader:
    """附件下载器。"""

    def __init__(self, session: PoliteSession, out_dir: str | Path) -> None:
        """初始化。

        参数：
            session: 限速后的 HTTP 会话。
            out_dir: 附件保存目录。
        """
        self.session = session
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()

    def download(self, attachment: Attachment, stats: DownloadStats) -> Path | None:
        """下载一个附件。

        参数：
            attachment: 附件信息。
            stats: 统计对象，就地更新。

        返回：
            落盘路径；失败或跳过时返回 None。
        """
        if attachment.url in self._seen:
            stats.skipped += 1
            return None
        self._seen.add(attachment.url)

        target = self.out_dir / attachment.suggested_name()

        # 同名文件已存在就跳过（重跑时幂等，不会产生一堆哈希后缀的副本）
        if target.exists():
            stats.skipped += 1
            return None

        try:
            payload = self._fetch_bytes(attachment.url)
        except BlockedError:
            raise
        except FetchError as exc:
            stats.failed += 1
            logger.warning("附件下载失败 %s：%s", attachment.url[:90], exc)
            return None

        if not payload:
            stats.failed += 1
            return None

        # 校验真实类型，防止把 HTML 错误页当 PDF 存下来
        kind = sniff_kind(payload[:8])
        if kind == "unknown":
            stats.failed += 1
            logger.warning(
                "附件内容既不是 PDF 也不是 Office 文档，已丢弃：%s", attachment.url[:90]
            )
            return None

        target.write_bytes(payload)
        stats.downloaded += 1
        stats.bytes_written += len(payload)
        stats.by_kind[attachment.kind] = stats.by_kind.get(attachment.kind, 0) + 1
        return target

    def _fetch_bytes(self, url: str) -> bytes:
        """以二进制方式下载。

        附件可能很大，所以不复用 fetcher 的文本解码路径。

        参数：
            url: 附件地址。

        返回：
            文件字节内容。
        """
        self.session._throttle()
        response = self.session.session.get(url, timeout=self.session.timeout)
        self.session._last_at = __import__("time").monotonic()
        self.session.request_count += 1

        if response.status_code >= 400:
            raise FetchError(f"HTTP {response.status_code}：{url}")

        return response.content


def _short_hash(text: str) -> str:
    """取 URL 的短哈希。

    保留此函数用于排查同名附件——重跑时的幂等由「同名即跳过」保证，
    真出现同名不同内容需要人工介入时，可以用它手工消歧。

    参数：
        text: 任意文本（通常是 URL）。

    返回：
        8 位十六进制哈希。
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
