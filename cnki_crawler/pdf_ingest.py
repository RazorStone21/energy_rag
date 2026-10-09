"""自备全文 PDF 的入库工具链。

## 合规边界

本模块**不下载任何东西**。它只处理用户自己放进目录的 PDF——那些通过
机构订阅、个人购买或其他合法途径获得的文件。爬虫部分永远不碰全文。

## 做什么

1. 扫描指定目录里的 PDF；
2. 用 pymupdf 抽取前两页文本；
3. 拿文本去已抓取的题录库里做标题匹配；
4. 匹配上的，按 energy_rag 的文件名约定复制到 `data/gov_doc/`，
   并打印对应的知网 ID 便于溯源；
5. 匹配不上的，**如实报告**，不猜、不硬塞。
"""

from __future__ import annotations

import difflib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .store import Store

logger = logging.getLogger(__name__)

# 标题匹配的相似度下限。低于这个值认为是不同文章。
MATCH_THRESHOLD = 0.82

# 抽取 PDF 前多少页用于识别标题
_HEAD_PAGES = 2

# 归一化时要去掉的字符（空白、常见标点）
_PUNCT_RE = re.compile(r"[\s　（）()【】\[\]《》<>，,。.；;：:、\-—_/\\|!！?？\"'“”‘’]+")


@dataclass
class IngestResult:
    """一轮入库的结果。"""

    matched: list[tuple[Path, Path, dict]] = field(default_factory=list)
    unmatched: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)

    def summary(self) -> str:
        """返回人类可读的摘要。"""
        return (
            f"匹配成功 {len(self.matched)}，未匹配 {len(self.unmatched)}，"
            f"读取失败 {len(self.errors)}"
        )


def normalize_title(text: str) -> str:
    """归一化标题用于比较：去掉空白与标点，转小写。

    参数：
        text: 原始标题。

    返回：
        归一化后的字符串。
    """
    return _PUNCT_RE.sub("", text or "").lower()


def extract_text_head(pdf_path: Path, pages: int = _HEAD_PAGES) -> str:
    """抽取 PDF 前若干页的文本。

    参数：
        pdf_path: PDF 路径。
        pages: 抽取页数。

    返回：
        拼接后的文本；读取失败抛异常。
    """
    import fitz  # pymupdf，延迟导入避免无 PDF 需求时报错

    chunks: list[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for index in range(min(pages, doc.page_count)):
            chunks.append(doc.load_page(index).get_text())
    return "\n".join(chunks)


def match_record(text_head: str, records: list[dict]) -> tuple[dict | None, float]:
    """在题录库里找与 PDF 文本最匹配的一条。

    先用包含关系快速命中，再用 difflib 计算相似度兜底。

    参数：
        text_head: PDF 前几页的文本。
        records: 候选题录列表。

    返回：
        (匹配到的题录, 相似度)。都不达标时返回 (None, 最高相似度)。
    """
    normalized_head = normalize_title(text_head)
    best: dict | None = None
    best_ratio = 0.0

    for record in records:
        title = normalize_title(record.get("title", ""))
        if not title or len(title) < 8:
            continue

        # 标题完整出现在 PDF 正文里，基本可以确定是同一篇
        if title in normalized_head:
            return record, 1.0

        ratio = difflib.SequenceMatcher(None, title, normalized_head[: len(title) * 3]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = record

    if best is not None and best_ratio >= MATCH_THRESHOLD:
        return best, best_ratio
    return None, best_ratio


def ingest_pdfs(
    pdf_dir: str | Path,
    store: Store,
    out_dir: str | Path,
    move: bool = False,
    dry_run: bool = False,
) -> IngestResult:
    """扫描 PDF 目录，匹配题录后按约定命名复制到语料目录。

    参数：
        pdf_dir: 用户存放自备 PDF 的目录（递归扫描）。
        store: 题录库。
        out_dir: 语料输出目录（energy_rag 的 `data/gov_doc/`）。
        move: True 表示移动而非复制原文件。
        dry_run: 只报告不落盘。

    返回：
        IngestResult。
    """
    source_dir = Path(pdf_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"PDF 目录不存在：{source_dir}")

    target_dir = Path(out_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"语料目录不存在：{target_dir}")

    # 一次性载入题录，避免每个 PDF 都查一遍库
    records = list(store.iter_articles())
    logger.info("题录库共 %d 条，开始匹配 %s 下的 PDF", len(records), source_dir)

    result = IngestResult()

    for pdf_path in sorted(source_dir.rglob("*.pdf")):
        try:
            text_head = extract_text_head(pdf_path)
        except Exception as exc:  # noqa: BLE001 - 单个坏文件不该中断整批
            result.errors.append((pdf_path, str(exc)))
            logger.warning("读取失败 %s：%s", pdf_path.name, exc)
            continue

        record, ratio = match_record(text_head, records)
        if record is None:
            result.unmatched.append(pdf_path)
            logger.warning("未匹配到题录（最高相似度 %.2f）：%s", ratio, pdf_path.name)
            continue

        # 沿用 export.py 的命名规则，保证两条路径产出的文件名一致
        from export import build_filename

        filename = build_filename(record).removesuffix(".md") + ".pdf"
        destination = target_dir / filename

        if dry_run:
            logger.info("[试运行] %s -> %s", pdf_path.name, filename)
            result.matched.append((pdf_path, destination, record))
            continue

        shutil.move(str(pdf_path), str(destination)) if move else shutil.copy2(
            pdf_path, destination
        )
        logger.info("已入库 %s -> %s（相似度 %.2f）", pdf_path.name, filename, ratio)
        result.matched.append((pdf_path, destination, record))

    return result
