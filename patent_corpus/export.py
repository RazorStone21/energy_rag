"""专利题录的结构化抽取与语料输出。

输出两种形态：

- **JSONL**：保留全部字段，便于后续做结构化分析或重新切片。
- **纯文本语料**：每件专利一个 `.txt`，标题 + 摘要 + 权利要求书（+ 说明书），
  适合直接喂给语言模型做领域适应训练。

同时统计字符数与字节数，让「离 GB 还有多远」这件事可量化。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .domain import classify_domain, clean_claims, clean_text
from .loader import LoadedTable

logger = logging.getLogger(__name__)

# 文件名里不允许的字符
_UNSAFE_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# 语料里各段的分隔标记
_SECTION_TEMPLATE = {
    "title": "【发明名称】",
    "abstract": "【摘要】",
    "claims": "【权利要求书】",
    "description": "【说明书】",
}


@dataclass
class PatentRecord:
    """一件专利的结构化记录。"""

    pub_number: str = ""
    app_number: str = ""
    title: str = ""
    abstract: str = ""
    claims: str = ""
    description: str = ""
    ipc: str = ""
    """主分类号，领域判定依据。"""
    ipc_all: str = ""
    """完整分类号列表，仅作参考，不参与领域判定。"""
    applicant: str = ""
    inventor: str = ""
    app_date: str = ""
    pub_date: str = ""
    patent_type: str = ""
    domain_reason: str = ""
    source_file: str = ""

    def to_dict(self) -> dict:
        """转成可序列化字典。"""
        return asdict(self)

    @property
    def doc_id(self) -> str:
        """稳定的文档 ID，优先用公开号。"""
        for candidate in (self.pub_number, self.app_number):
            cleaned = _UNSAFE_RE.sub("", str(candidate or "").strip())
            if cleaned:
                return cleaned
        return f"row-{abs(hash(self.title)) % 10**10}"

    def render_text(self, include_description: bool = True) -> str:
        """渲染成训练语料文本。

        参数：
            include_description: 是否包含说明书全文。

        返回：
            拼接好的文本；无有效内容时返回空串。
        """
        blocks: list[str] = []
        for key in ("title", "abstract", "claims"):
            value = getattr(self, key)
            if value:
                blocks.append(f"{_SECTION_TEMPLATE[key]}\n{value}")
        if include_description and self.description:
            blocks.append(f"{_SECTION_TEMPLATE['description']}\n{self.description}")

        if not blocks:
            return ""

        meta_bits = [f"公开号：{self.pub_number}"] if self.pub_number else []
        if self.ipc:
            meta_bits.append(f"IPC：{self.ipc}")
        if self.applicant:
            meta_bits.append(f"申请人：{self.applicant}")
        if meta_bits:
            blocks.insert(0, "，".join(meta_bits))

        return "\n\n".join(blocks)


@dataclass
class CorpusStats:
    """语料统计。

    字段分两组，职责不同：
    - **筛选组**（rows_*、with_*、reasons）由 `build_corpus` 填；
    - **产出组**（written、chars、bytes_written）由 `write_text_corpus` 填。
    这样两边各自累加，不会互相覆盖，也不会双重计数。
    """

    # 筛选组
    rows_seen: int = 0
    rows_included: int = 0
    rows_skipped: int = 0
    with_claims: int = 0
    with_description: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    # 产出组
    written: int = 0
    chars: int = 0
    bytes_written: int = 0

    def summary(self) -> str:
        """返回人类可读的统计摘要。"""
        mb = self.bytes_written / 1024 / 1024
        lines = [
            f"读取记录      {self.rows_seen}",
            f"落入领域      {self.rows_included}",
            f"筛除          {self.rows_skipped}",
            f"含权利要求书  {self.with_claims}",
            f"含说明书      {self.with_description}",
            f"写出文档      {self.written}",
            f"总字符数      {self.chars:,}",
            f"总字节数      {self.bytes_written:,} ({mb:.1f} MB)",
        ]
        if self.bytes_written and self.written:
            avg = self.bytes_written / self.written
            lines.append(f"平均每件      {avg:,.0f} 字节")
            lines.append(f"按当前每件体量，到 1GB 约需 {1024**3 / avg:,.0f} 件")
        return "\n".join(lines)


def iter_records(table: LoadedTable) -> Iterator[PatentRecord]:
    """从一张表里逐行抽出专利记录。

    参数：
        table: 已读取的导出表。

    生成：
        PatentRecord（尚未做领域筛选）。
    """
    cols = table.columns
    frame = table.frame

    def cell(row, key: str) -> str:
        """取某规范字段的值，列不存在时返回空串。"""
        column = cols.get(key)
        if not column or column not in row.index:
            return ""
        return clean_text(row[column])

    for _, row in frame.iterrows():
        yield PatentRecord(
            pub_number=cell(row, "pub_number"),
            app_number=cell(row, "app_number"),
            title=cell(row, "title"),
            abstract=cell(row, "abstract"),
            claims=clean_claims(cell(row, "claims")),
            description=cell(row, "description"),
            ipc=cell(row, "ipc"),
            ipc_all=cell(row, "ipc_all"),
            applicant=cell(row, "applicant"),
            inventor=cell(row, "inventor"),
            app_date=cell(row, "app_date"),
            pub_date=cell(row, "pub_date"),
            patent_type=cell(row, "patent_type"),
            source_file=table.path.name,
        )


def build_corpus(
    tables: list[LoadedTable],
    keywords_only: bool = False,
) -> tuple[list[PatentRecord], CorpusStats]:
    """把多张导出表处理成一个领域内的专利记录列表。

    参数：
        tables: 已读取的表列表。
        keywords_only: 为 True 时跳过 IPC 判定，只用关键词（调试用）。

    返回：
        (记录列表, 统计)。统计里的字节数是按语料渲染后的实际大小估算的。
    """
    records: list[PatentRecord] = []
    stats = CorpusStats()
    seen_ids: set[str] = set()

    for table in tables:
        for record in iter_records(table):
            stats.rows_seen += 1

            if not record.title and not record.abstract:
                stats.rows_skipped += 1
                stats.reasons["无标题且无摘要"] = stats.reasons.get("无标题且无摘要", 0) + 1
                continue

            if keywords_only:
                from .domain import has_domain_keyword

                hit = has_domain_keyword(record.title, record.abstract, record.claims)
                in_domain = bool(hit)
                reason = f"关键词「{hit}」" if hit else "无关键词"
            else:
                in_domain, reason = classify_domain(
                    record.ipc, record.title, record.abstract, record.claims
                )

            if not in_domain:
                stats.rows_skipped += 1
                stats.reasons[reason] = stats.reasons.get(reason, 0) + 1
                continue

            # 同一件专利用不同批次导出时可能重复
            doc_id = record.doc_id
            if doc_id in seen_ids:
                stats.rows_skipped += 1
                stats.reasons["重复"] = stats.reasons.get("重复", 0) + 1
                continue
            seen_ids.add(doc_id)

            record.domain_reason = reason
            records.append(record)
            stats.rows_included += 1
            if record.claims:
                stats.with_claims += 1
            if record.description:
                stats.with_description += 1

    return records, stats


def write_jsonl(records: list[PatentRecord], out_path: str | Path) -> int:
    """写出 JSONL。

    参数：
        records: 记录列表。
        out_path: 输出文件路径。

    返回：
        写出的记录数。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def write_text_corpus(
    records: list[PatentRecord],
    out_dir: str | Path | None,
    include_description: bool = True,
    single_file: str | Path | None = None,
    stats: CorpusStats | None = None,
) -> CorpusStats:
    """写出纯文本训练语料。

    参数：
        records: 记录列表。
        out_dir: 每件一个 .txt 的输出目录；为 None 时只写 single_file。
        include_description: 是否包含说明书全文。
        single_file: 若指定，额外把所有文档拼进这一个文件。
        stats: 已有的统计对象（通常来自 build_corpus）。传入则在其上累加，
            这样「读取/筛除」这类上游计数不会被丢掉。

    返回：
        更新后的 CorpusStats（字节数按实际写出内容统计）。
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
        for record in records:
            text = record.render_text(include_description=include_description)
            if not text:
                continue

            payload = text + "\n"
            size = len(payload.encode("utf-8"))

            stats.written += 1
            stats.chars += len(text)
            stats.bytes_written += size

            if directory:
                (directory / f"{record.doc_id}.txt").write_text(payload, encoding="utf-8")
            if sink:
                sink.write(f"===== {record.doc_id} =====\n{payload}\n")
    finally:
        if sink:
            sink.close()

    return stats
