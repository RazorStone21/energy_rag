"""爬取编排：期刊 → 年份 → 期次 → 文章。

爬取图：

    Journal/{PYKM}                        期刊主页（校验刊名、拿学科代码）
      └─ Journal/List/{PYKM}{年}{期}      期次目录（拿到该期全部文章 ID）
           └─ Journal/Article/{ID}        文章详情（题录）

每一步都落盘，`issues` 表的 status 字段决定断点续传时跳过哪些期次。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from tqdm import tqdm

from .backends import ResilientBackend
from .backends.wap import WapBackend
from .fetcher import BlockedError, CrawlerError, FetchError, PoliteSession
from .parsers import ParseError, parse_issue_page, parse_journal_page
from .store import Store

logger = logging.getLogger(__name__)

# 单个年份内最多尝试多少期。周刊一年最多 52 期，52 足够覆盖。
MAX_ISSUES_PER_YEAR = 52

# 连续多少期取不到就认为该年份的期号已枚举完
CONSECUTIVE_MISS_LIMIT = 3


@dataclass
class CrawlStats:
    """一轮爬取的统计。"""

    journals: int = 0
    issues_done: int = 0
    issues_skipped: int = 0
    articles_new: int = 0
    articles_failed: int = 0
    blocked: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """返回人类可读的统计摘要。"""
        parts = [
            f"期刊 {self.journals}",
            f"期次完成 {self.issues_done}",
            f"期次跳过 {self.issues_skipped}",
            f"新增题录 {self.articles_new}",
        ]
        if self.articles_failed:
            parts.append(f"失败 {self.articles_failed}")
        if self.blocked:
            parts.append("** 触发了风控 **")
        return "，".join(parts)


class Crawler:
    """题录爬取器。"""

    def __init__(
        self,
        store: Store,
        session: PoliteSession,
        backend: ResilientBackend,
        wap: WapBackend,
        abort_on_block: bool = True,
    ) -> None:
        """初始化。

        参数：
            store: 存储对象。
            session: HTTP 会话。
            backend: 题录抓取后端（PC + wap 组合）。
            wap: wap 后端，用于抓期刊页与期次页（这两个 PC 站没有对应接口）。
            abort_on_block: 触发风控时是否立即中止。
        """
        self.store = store
        self.session = session
        self.backend = backend
        self.wap = wap
        self.abort_on_block = abort_on_block

    def crawl_journal(
        self,
        pykm: str,
        years: int = 10,
        max_issues: int = 0,
        progress: bool = True,
    ) -> CrawlStats:
        """爬取一本期刊近若干年的全部题录。

        参数：
            pykm: 期刊代码。
            years: 爬取最近多少年。
            max_issues: 本次最多处理的期次数量，0 表示不限（冒烟测试用）。
            progress: 是否显示进度条。

        返回：
            CrawlStats。

        异常：
            fetcher.BlockedError: 触发风控且 abort_on_block 为 True。
        """
        stats = CrawlStats()

        # 1. 期刊主页：校验代码有效并更新刊名
        try:
            journal_html = self.wap.fetch_journal(pykm)
            meta = parse_journal_page(journal_html, pykm=pykm)
            self.store.upsert_journal(pykm, meta.name, meta.subject_codes, source="seed")
            logger.info("开始爬取 %s《%s》", pykm, meta.name)
        except ParseError as exc:
            logger.error("期刊 %s 无法解析，跳过：%s", pykm, exc)
            stats.errors.append(f"{pykm}: {exc}")
            return stats

        stats.journals = 1
        current_year = datetime.now().year
        min_year = current_year - years + 1

        # 2. 逐年份枚举期次
        issue_budget = max_issues if max_issues > 0 else float("inf")
        issued = 0

        for year in range(current_year, min_year - 1, -1):
            if issued >= issue_budget:
                break
            consecutive_miss = 0

            for issue in range(1, MAX_ISSUES_PER_YEAR + 1):
                if issued >= issue_budget:
                    break
                if consecutive_miss >= CONSECUTIVE_MISS_LIMIT:
                    break

                # 断点续传：已完成的期次直接跳过
                if self.store.is_issue_done(pykm, year, issue):
                    stats.issues_skipped += 1
                    continue

                try:
                    self._crawl_issue(pykm, year, issue, stats, progress)
                    issued += 1
                    consecutive_miss = 0
                except _IssueMissing:
                    consecutive_miss += 1
                except BlockedError:
                    stats.blocked = True
                    logger.error("触发知网风控，已中止。已完成：%s", stats.summary())
                    if self.abort_on_block:
                        raise
                    return stats
                except CrawlerError as exc:
                    stats.articles_failed += 1
                    stats.errors.append(f"{pykm} {year}年{issue}期: {exc}")
                    logger.warning("期次抓取失败 %s %d-%d：%s", pykm, year, issue, exc)
                    consecutive_miss += 1

        return stats

    def _crawl_issue(
        self,
        pykm: str,
        year: int,
        issue: int,
        stats: CrawlStats,
        progress: bool,
    ) -> None:
        """抓取单个期次：先取目录，再逐篇取题录。

        异常：
            _IssueMissing: 该期次不存在（期号枚举到头了）。
        """
        try:
            html = self.wap.fetch_issue(pykm, year, issue)
        except FetchError as exc:
            raise _IssueMissing(f"{pykm} {year}-{issue} 不存在") from exc

        try:
            info = parse_issue_page(html, pykm=pykm)
        except ParseError as exc:
            raise _IssueMissing(f"{pykm} {year}-{issue} 无文章") from exc

        self.store.upsert_issue(pykm, year, issue, info.journal, len(info.articles))

        # 补全期次页码：目录里没有，用期刊代码回填到每条题录
        iterator = info.articles
        if progress:
            iterator = tqdm(
                info.articles,
                desc=f"{info.journal} {year}年{issue}期",
                leave=False,
                unit="篇",
            )

        for ref in iterator:
            if self.store.has_article(ref.article_id):
                continue
            try:
                record = self.backend.fetch_article(ref.article_id)
            except BlockedError:
                raise
            except CrawlerError as exc:
                stats.articles_failed += 1
                stats.errors.append(f"{ref.article_id}: {exc}")
                logger.warning("题录抓取失败 %s：%s", ref.article_id, exc)
                continue

            if record is None:
                stats.articles_failed += 1
                continue

            # 目录页已知的字段优先补全，避免详情页缺失时丢信息
            if not record.title:
                record.title = ref.title
            if not record.pykm:
                record.pykm = pykm
            if not record.journal:
                record.journal = info.journal
            if not record.year:
                record.year = year
            if not record.issue:
                record.issue = issue

            self.store.add_article(record)
            stats.articles_new += 1

        self.store.mark_issue_done(pykm, year, issue)
        stats.issues_done += 1


class _IssueMissing(Exception):
    """内部信号：该期次不存在，用于控制期号枚举的终止。"""
