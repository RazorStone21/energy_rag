"""存储层：SQLite 主库 + JSONL 逐条落盘。

设计要点：
1. **双写**。每条题录同时写 SQLite 和 JSONL。JSONL 是追加写的，
   进程被 Ctrl-C 或被杀掉时已抓数据不会丢；SQLite 提供查询和断点续传。
2. **断点续传**。`issues` 表用 (pykm, year, issue) 做主键并记录 status，
   重跑时跳过已完成的期次。
3. **幂等**。题录以 article_id 为主键，重复抓取用 INSERT OR REPLACE 覆盖。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .parsers import ArticleRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS journals (
    pykm          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    subject_codes TEXT DEFAULT '',
    source        TEXT DEFAULT 'seed',
    updated_at    TEXT
);

CREATE TABLE IF NOT EXISTS issues (
    pykm          TEXT NOT NULL,
    year          INTEGER NOT NULL,
    issue         INTEGER NOT NULL,
    journal       TEXT DEFAULT '',
    article_count INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'pending',
    fetched_at    TEXT,
    PRIMARY KEY (pykm, year, issue)
);

CREATE TABLE IF NOT EXISTS articles (
    article_id         TEXT PRIMARY KEY,
    pykm               TEXT DEFAULT '',
    journal            TEXT DEFAULT '',
    year               INTEGER DEFAULT 0,
    issue              INTEGER DEFAULT 0,
    title              TEXT NOT NULL,
    authors            TEXT DEFAULT '[]',
    affiliations       TEXT DEFAULT '[]',
    keywords           TEXT DEFAULT '[]',
    subjects           TEXT DEFAULT '[]',
    abstract           TEXT DEFAULT '',
    abstract_truncated INTEGER DEFAULT 1,
    fund               TEXT DEFAULT '',
    cited_count        INTEGER DEFAULT 0,
    download_count     INTEGER DEFAULT 0,
    pdf_size_kb        INTEGER DEFAULT 0,
    url                TEXT DEFAULT '',
    backend            TEXT DEFAULT 'wap',
    crawled_at         TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_pykm ON articles(pykm, year, issue);
CREATE INDEX IF NOT EXISTS idx_articles_year ON articles(year);
"""


def _now() -> str:
    """返回当前时间的 ISO 字符串。"""
    return datetime.now().isoformat(timespec="seconds")


class Store:
    """题录存储。同时维护 SQLite 与 JSONL。"""

    def __init__(self, db_path: str | Path, jsonl_path: str | Path | None = None) -> None:
        """打开（必要时创建）数据库。

        参数：
            db_path: SQLite 文件路径。
            jsonl_path: JSONL 追加文件路径；None 表示不写 JSONL。
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

    # ---------- 期刊 ----------

    def upsert_journal(self, pykm: str, name: str, subject_codes: str = "", source: str = "seed") -> None:
        """写入或更新期刊元数据。"""
        self.conn.execute(
            """
            INSERT INTO journals (pykm, name, subject_codes, source, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pykm) DO UPDATE SET
                name = excluded.name,
                subject_codes = CASE WHEN excluded.subject_codes != ''
                                     THEN excluded.subject_codes ELSE journals.subject_codes END,
                updated_at = excluded.updated_at
            """,
            (pykm, name, subject_codes, source, _now()),
        )
        self.conn.commit()

    def journals(self, source: str | None = None) -> list[dict]:
        """列出已知期刊，可按来源过滤。"""
        sql = "SELECT * FROM journals"
        params: tuple = ()
        if source:
            sql += " WHERE source = ?"
            params = (source,)
        sql += " ORDER BY pykm"
        return [dict(row) for row in self.conn.execute(sql, params)]

    # ---------- 期次 ----------

    def upsert_issue(self, pykm: str, year: int, issue: int, journal: str, article_count: int) -> None:
        """记录一个期次。已有记录不会被重置回 pending。"""
        self.conn.execute(
            """
            INSERT INTO issues (pykm, year, issue, journal, article_count, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(pykm, year, issue) DO UPDATE SET
                journal = excluded.journal,
                article_count = excluded.article_count
            """,
            (pykm, year, issue, journal, article_count),
        )
        self.conn.commit()

    def mark_issue_done(self, pykm: str, year: int, issue: int) -> None:
        """把期次标记为已完成（断点续传靠这个）。"""
        self.conn.execute(
            "UPDATE issues SET status = 'done', fetched_at = ? WHERE pykm = ? AND year = ? AND issue = ?",
            (_now(), pykm, year, issue),
        )
        self.conn.commit()

    def is_issue_done(self, pykm: str, year: int, issue: int) -> bool:
        """判断某期次是否已抓完。"""
        row = self.conn.execute(
            "SELECT status FROM issues WHERE pykm = ? AND year = ? AND issue = ?",
            (pykm, year, issue),
        ).fetchone()
        return bool(row and row["status"] == "done")

    def issues_for(self, pykm: str, min_year: int) -> list[dict]:
        """列出某期刊指定年份之后的所有期次。"""
        rows = self.conn.execute(
            "SELECT * FROM issues WHERE pykm = ? AND year >= ? ORDER BY year DESC, issue DESC",
            (pykm, min_year),
        )
        return [dict(row) for row in rows]

    # ---------- 题录 ----------

    def add_article(self, record: ArticleRecord) -> None:
        """写入一条题录，同时追加到 JSONL。"""
        payload = asdict(record)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO articles (
                article_id, pykm, journal, year, issue, title, authors, affiliations,
                keywords, subjects, abstract, abstract_truncated, fund, cited_count,
                download_count, pdf_size_kb, url, backend, crawled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.article_id,
                record.pykm,
                record.journal,
                record.year,
                record.issue,
                record.title,
                json.dumps(record.authors, ensure_ascii=False),
                json.dumps(record.affiliations, ensure_ascii=False),
                json.dumps(record.keywords, ensure_ascii=False),
                json.dumps(record.subjects, ensure_ascii=False),
                record.abstract,
                1 if record.abstract_truncated else 0,
                record.fund,
                record.cited_count,
                record.download_count,
                record.pdf_size_kb,
                record.url,
                record.backend,
                _now(),
            ),
        )
        self.conn.commit()

        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def has_article(self, article_id: str) -> bool:
        """判断题录是否已存在。"""
        row = self.conn.execute(
            "SELECT 1 FROM articles WHERE article_id = ?", (article_id,)
        ).fetchone()
        return row is not None

    def article(self, article_id: str) -> dict | None:
        """按 ID 取一条题录。"""
        row = self.conn.execute(
            "SELECT * FROM articles WHERE article_id = ?", (article_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None

    def iter_articles(self, pykm: str | None = None, min_year: int | None = None) -> Iterator[dict]:
        """遍历题录，可按期刊代码和最小年份过滤。"""
        sql = "SELECT * FROM articles"
        clauses: list[str] = []
        params: list = []
        if pykm:
            clauses.append("pykm = ?")
            params.append(pykm)
        if min_year:
            clauses.append("year >= ?")
            params.append(min_year)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY pykm, year DESC, issue DESC, article_id"

        cursor = self.conn.execute(sql, params)
        for row in cursor:
            yield _row_to_dict(row)

    def stats(self) -> dict:
        """返回总体统计，供 status 子命令使用。"""
        total = self.conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
        journals = self.conn.execute("SELECT COUNT(*) AS c FROM journals").fetchone()["c"]
        done = self.conn.execute("SELECT COUNT(*) AS c FROM issues WHERE status='done'").fetchone()["c"]
        pending = self.conn.execute("SELECT COUNT(*) AS c FROM issues WHERE status='pending'").fetchone()["c"]
        truncated = self.conn.execute(
            "SELECT COUNT(*) AS c FROM articles WHERE abstract_truncated = 1"
        ).fetchone()["c"]
        return {
            "articles": total,
            "journals": journals,
            "issues_done": done,
            "issues_pending": pending,
            "abstract_truncated": truncated,
        }

    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    """把 SQLite 行转成字典，并把 JSON 字段反序列化。"""
    data = dict(row)
    for key in ("authors", "affiliations", "keywords", "subjects"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except json.JSONDecodeError:
                data[key] = []
    if "abstract_truncated" in data:
        data["abstract_truncated"] = bool(data["abstract_truncated"])
    return data
