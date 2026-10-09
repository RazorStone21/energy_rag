"""存储层：SQLite 主库 + JSONL 逐条落盘。

和 patent_corpus 同样的思路：SQLite 负责查询和断点续传，JSONL 负责
「进程被杀也不丢已抓数据」。去重键是文档 URL 的哈希（政府站的 URL 稳定）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    url        TEXT NOT NULL,
    title      TEXT DEFAULT '',
    source     TEXT DEFAULT '',
    cms        TEXT DEFAULT '',
    category   TEXT DEFAULT '',
    published  TEXT DEFAULT '',
    text       TEXT DEFAULT '',
    chars      INTEGER DEFAULT 0,
    selector   TEXT DEFAULT '',
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS crawl_state (
    source   TEXT NOT NULL,
    url      TEXT NOT NULL,
    status   TEXT DEFAULT 'pending',
    note     TEXT DEFAULT '',
    seen_at  TEXT,
    PRIMARY KEY (source, url)
);

CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_chars ON documents(chars);
"""


def _now() -> str:
    """返回当前时间的 ISO 字符串。"""
    return datetime.now().isoformat(timespec="seconds")


def doc_id_for(url: str) -> str:
    """由 URL 生成稳定的文档 ID。

    参数：
        url: 文档地址。

    返回：
        16 位十六进制哈希前缀。
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


@dataclass
class Document:
    """一篇采集到的文档。"""

    url: str
    title: str = ""
    source: str = ""
    cms: str = ""
    category: str = ""
    published: str = ""
    text: str = ""
    selector: str = ""

    @property
    def doc_id(self) -> str:
        """稳定文档 ID。"""
        return doc_id_for(self.url)

    @property
    def chars(self) -> int:
        """正文字符数。"""
        return len(self.text)

    def to_dict(self) -> dict:
        """转成可序列化字典。"""
        data = self.__dict__.copy()
        data["doc_id"] = self.doc_id
        data["chars"] = self.chars
        return data


@dataclass
class CorpusStats:
    """采集统计。"""

    docs: int = 0
    chars: int = 0
    bytes_written: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """返回人类可读的统计摘要。"""
        mb = self.bytes_written / 1024 / 1024
        lines = [
            f"文档数        {self.docs}",
            f"总字符数      {self.chars:,}",
            f"总字节数      {self.bytes_written:,} ({mb:.1f} MB)",
        ]
        if self.docs:
            avg = self.bytes_written / self.docs
            lines.append(f"平均每篇      {avg:,.0f} 字节")
            lines.append(f"按当前体量，到 1GB 约需 {1024**3 / avg:,.0f} 篇")
        if self.by_source:
            lines.append("")
            lines.append("按来源：")
            for source, count in sorted(self.by_source.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:6d}  {source}")
        return "\n".join(lines)


class Store:
    """文档存储。"""

    def __init__(self, db_path: str | Path, jsonl_path: str | Path | None = None) -> None:
        """打开（必要时创建）数据库。

        参数：
            db_path: SQLite 文件路径。
            jsonl_path: JSONL 追加文件路径。
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ---------- 抓取状态（断点续传） ----------

    def already_seen(self, source: str, url: str) -> bool:
        """判断某个 URL 是否已处理过。"""
        row = self.conn.execute(
            "SELECT 1 FROM crawl_state WHERE source = ? AND url = ? AND status = 'done'",
            (source, url),
        ).fetchone()
        return row is not None

    def mark_seen(self, source: str, url: str, status: str = "done", note: str = "") -> None:
        """记录某个 URL 的处理状态。"""
        self.conn.execute(
            """
            INSERT INTO crawl_state (source, url, status, note, seen_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, url) DO UPDATE SET
                status = excluded.status, note = excluded.note, seen_at = excluded.seen_at
            """,
            (source, url, status, note, _now()),
        )
        self.conn.commit()

    # ---------- 文档 ----------

    def add_document(self, doc: Document) -> None:
        """写入一篇文档，同时追加到 JSONL。"""
        payload = doc.to_dict()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (doc_id, url, title, source, cms, category, published, text, chars, selector, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.doc_id, doc.url, doc.title, doc.source, doc.cms, doc.category,
                doc.published, doc.text, doc.chars, doc.selector, _now(),
            ),
        )
        self.conn.commit()

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def has_document(self, doc_id: str) -> bool:
        """判断文档是否已入库。"""
        row = self.conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def iter_documents(self, source: str | None = None) -> Iterator[dict]:
        """遍历文档，可按来源过滤。"""
        sql = "SELECT * FROM documents"
        params: tuple = ()
        if source:
            sql += " WHERE source = ?"
            params = (source,)
        sql += " ORDER BY source, published DESC, doc_id"

        for row in self.conn.execute(sql, params):
            yield dict(row)

    def stats(self) -> dict:
        """返回总体统计。"""
        total = self.conn.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(chars),0) AS s FROM documents"
        ).fetchone()
        sources = self.conn.execute(
            "SELECT COUNT(DISTINCT source) AS c FROM documents"
        ).fetchone()["c"]
        seen = self.conn.execute(
            "SELECT COUNT(*) AS c FROM crawl_state WHERE status='done'"
        ).fetchone()["c"]
        return {
            "documents": total["c"],
            "chars": total["s"],
            "sources": sources,
            "urls_seen": seen,
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
