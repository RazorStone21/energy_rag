"""导出：把库里的文档写成训练语料。

输出两种形态：
- **每篇一个 .txt**：文件名是 URL 哈希，稳定且唯一；
- **单个大文件**：所有文档用分隔标记拼起来，适合直接喂给语言模型。

同时统计字节数，让「离 GB 还有多远」可量化。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from .store import CorpusStats, Store

logger = logging.getLogger(__name__)

# 文件名里不允许的字符
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def render_document(record: dict) -> str:
    """把一篇文档渲染成语料文本。

    参数：
        record: 来自 store.iter_documents 的字典。

    返回：
        渲染后的文本。
    """
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    if not text:
        return ""

    meta_bits: list[str] = []
    if record.get("source"):
        meta_bits.append(f"来源：{record['source']}")
    if record.get("category"):
        meta_bits.append(f"栏目：{record['category']}")
    if record.get("url"):
        meta_bits.append(f"原文：{record['url']}")

    blocks: list[str] = []
    if meta_bits:
        blocks.append("，".join(meta_bits))
    if title:
        blocks.append(f"# {title}")
    blocks.append(text)

    return "\n\n".join(blocks)


def write_text_corpus(
    store: Store,
    out_dir: str | Path | None,
    single_file: str | Path | None = None,
    source: str | None = None,
    min_chars: int = 0,
    stats: CorpusStats | None = None,
) -> CorpusStats:
    """导出纯文本语料。

    参数：
        store: 存储对象。
        out_dir: 每篇一个 .txt 的输出目录；None 表示不写分篇文件。
        single_file: 若指定，额外把所有文档拼进这一个文件。
        source: 只导出某个来源；None 表示全部。
        min_chars: 只导出正文长度不低于该值的文档（过滤短公告）。
        stats: 已有统计对象，传入则在其上累加。

    返回：
        CorpusStats。
    """
    if stats is None:
        stats = CorpusStats()

    directory = Path(out_dir) if out_dir else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)

    sink = None
    if single_file:
        single_path = Path(single_file)
        single_path.parent.mkdir(parents=True, exist_ok=True)
        sink = single_path.open("w", encoding="utf-8")

    try:
        for record in store.iter_documents(source=source):
            if min_chars and (record.get("chars") or 0) < min_chars:
                continue

            text = render_document(record)
            if not text:
                continue

            payload = text + "\n"
            size = len(payload.encode("utf-8"))

            stats.docs += 1
            stats.chars += len(text)
            stats.bytes_written += size
            name = record.get("source") or "未知来源"
            stats.by_source[name] = stats.by_source.get(name, 0) + 1

            if directory:
                doc_id = _UNSAFE_RE.sub("", str(record.get("doc_id") or "unknown"))
                (directory / f"{doc_id}.txt").write_text(payload, encoding="utf-8")
            if sink:
                sink.write(f"===== {record.get('doc_id')} =====\n{payload}\n")
    finally:
        if sink:
            sink.close()

    return stats


def write_jsonl(store: Store, out_path: str | Path, source: str | None = None) -> int:
    """导出为 JSONL。

    参数：
        store: 存储对象。
        out_path: 输出路径。
        source: 只导出某个来源。

    返回：
        写出的记录数。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in store.iter_documents(source=source):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
