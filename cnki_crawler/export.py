"""导出题录，供 energy_rag 摄入。

## 为什么要导成 Markdown

energy_rag 的摄入管道（`src/ingestion.py`）有几条硬约束，直接决定了导出格式：

1. 只读 `data/gov_doc/` 的**第一层**（用 `iterdir()` 而非 `rglob()`），子目录被忽略；
2. 只认 `.pdf` / `.txt` / `.md` / `.docx` / `.xlsx`，**不认 JSON/JSONL**；
3. 用 `path.name`（仅文件名）作为去重、删除和增量构建的键，所以**文件名必须
   全局唯一且稳定**——改名等于新增文档；
4. **没有 sidecar 元数据机制**，元数据只能写进文件正文。

因此每条题录导出为一个 `.md` 文件：文件名用知网文章 ID 保证唯一与稳定，
题录字段以列表形式写进正文顶部，再放摘要与关键词。
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from .store import Store

logger = logging.getLogger(__name__)

# 文件名里不允许出现的字符（跨平台保守起见）
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 标题在文件名里最多保留多少字符
_TITLE_SLUG_LIMIT = 40


class ExportError(Exception):
    """导出失败。"""


def build_filename(record: dict, ascii_only: bool = False) -> str:
    """由题录生成唯一且稳定的文件名。

    形态：`{文章ID}-{标题片段}-{年份}-cn.md`

    文章 ID 来自知网且全局唯一，因此文件名天然不会碰撞；标题片段只是为了
    人工浏览时便于识别。

    参数：
        record: 题录字典。
        ascii_only: 为 True 时丢弃非 ASCII 字符（文件名变成纯 ASCII）。

    返回：
        文件名（不含目录）。
    """
    article_id = record.get("article_id") or "unknown"
    year = record.get("year") or 0

    slug = _UNSAFE_CHARS.sub("", (record.get("title") or "").strip())
    slug = re.sub(r"\s+", "-", slug)[:_TITLE_SLUG_LIMIT].strip("-")
    if ascii_only:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-")

    parts = [article_id]
    if slug:
        parts.append(slug)
    parts.append(str(year))
    parts.append("cn")

    return "-".join(parts) + ".md"


def render_markdown(record: dict) -> str:
    """把一条题录渲染成 Markdown 正文。

    参数：
        record: 题录字典。

    返回：
        Markdown 文本。
    """
    lines: list[str] = []
    title = record.get("title") or "(无标题)"
    lines.append(f"# {title}")
    lines.append("")

    def field(label: str, value) -> None:
        """写一行「标签: 值」，值为空则跳过。"""
        if not value:
            return
        if isinstance(value, (list, tuple)):
            if not value:
                return
            value = "；".join(str(v) for v in value)
        lines.append(f"- **{label}**：{value}")

    field("作者", record.get("authors"))
    field("机构", record.get("affiliations"))
    field("期刊", record.get("journal"))
    field("年卷期", _format_issue(record))
    field("关键词", record.get("keywords"))
    field("学科领域", record.get("subjects"))
    field("基金", record.get("fund"))

    cited = record.get("cited_count") or 0
    downloads = record.get("download_count") or 0
    if cited or downloads:
        field("被引/下载", f"{cited} / {downloads}")

    field("知网ID", record.get("article_id"))
    field("来源", record.get("url"))

    lines.append("")
    lines.append("## 摘要")
    lines.append("")

    abstract = (record.get("abstract") or "").strip()
    if abstract:
        lines.append(abstract)
    else:
        lines.append("(公开页面未提供摘要)")

    # 诚实标注：截断的摘要必须写清楚，不能让人以为是完整摘要
    if record.get("abstract_truncated"):
        lines.append("")
        lines.append(
            "> 注：以上摘要摘自知网公开题录页，该页面**服务端截断**了摘要"
            "（约 110 字），此处并非完整摘要。"
        )

    lines.append("")
    return "\n".join(lines)


def export_markdown(store: Store, out_dir: str | Path, ascii_only: bool = False) -> int:
    """把库中全部题录导出为 Markdown 文件。

    参数：
        store: 存储对象。
        out_dir: 输出目录（通常是 energy_rag 的 `data/gov_doc/`）。
        ascii_only: 文件名是否只保留 ASCII。

    返回：
        写出的文件数。

    异常：
        ExportError: 输出目录不存在，或出现文件名碰撞。
    """
    target = Path(out_dir)
    if not target.exists():
        raise ExportError(f"输出目录不存在：{target}。请先创建，避免误写路径。")

    written = 0
    seen_names: dict[str, str] = {}

    for record in store.iter_articles():
        name = build_filename(record, ascii_only=ascii_only)

        # 文件名是 energy_rag 的去重键，碰撞会静默覆盖数据，必须拦下
        article_id = record.get("article_id", "")
        if name in seen_names and seen_names[name] != article_id:
            raise ExportError(
                f"文件名碰撞：{name} 同时对应 {seen_names[name]} 与 {article_id}"
            )
        seen_names[name] = article_id

        (target / name).write_text(render_markdown(record), encoding="utf-8")
        written += 1

    logger.info("已导出 %d 个 Markdown 文件到 %s", written, target)
    return written


def export_jsonl(store: Store, out_path: str | Path) -> int:
    """把库中全部题录导出为 JSONL。

    参数：
        store: 存储对象。
        out_path: 输出文件路径。

    返回：
        写出的记录数。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in store.iter_articles():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_csv(store: Store, out_path: str | Path) -> int:
    """把库中全部题录导出为 CSV（utf-8-sig，Excel 可直接打开）。

    参数：
        store: 存储对象。
        out_path: 输出文件路径。

    返回：
        写出的记录数。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "article_id", "title", "authors", "affiliations", "keywords", "subjects",
        "journal", "pykm", "year", "issue", "abstract", "abstract_truncated",
        "fund", "cited_count", "download_count", "pdf_size_kb", "url", "backend",
    ]

    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in store.iter_articles():
            row = dict(record)
            for key in ("authors", "affiliations", "keywords", "subjects"):
                row[key] = "；".join(row.get(key) or [])
            writer.writerow(row)
            count += 1
    return count


def _format_issue(record: dict) -> str:
    """把年/期拼成人类可读的字符串。"""
    year = record.get("year") or 0
    issue = record.get("issue") or 0
    if year and issue:
        return f"{year}年第{issue}期"
    if year:
        return f"{year}年"
    return ""
