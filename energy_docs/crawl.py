"""采集编排：站点 → 列表页 → 详情页 → 入库。

每篇文档入库后立刻写 JSONL，进程被 Ctrl-C 也不丢数据。
`crawl_state` 表记录已处理的 URL，重跑时跳过（断点续传）。
"""

from __future__ import annotations

import logging

from tqdm import tqdm

from .adapters.base import CrawlResult, SiteAdapter, SiteSpec
from .fetcher import BlockedError, PoliteSession
from .store import Store

logger = logging.getLogger(__name__)


class Crawler:
    """按站点清单逐站采集。"""

    def __init__(self, store: Store, session: PoliteSession, abort_on_block: bool = True) -> None:
        """初始化。

        参数：
            store: 存储对象。
            session: HTTP 会话。
            abort_on_block: 被拦截时是否立即中止。
        """
        self.store = store
        self.session = session
        self.abort_on_block = abort_on_block

    def crawl_site(
        self,
        spec: SiteSpec,
        max_docs: int = 0,
        progress: bool = True,
    ) -> CrawlResult:
        """采集一个站点。

        参数：
            spec: 站点配置。
            max_docs: 本次最多入库多少篇，0 为不限（冒烟测试用）。
            progress: 是否显示进度条。

        返回：
            CrawlResult。

        异常：
            fetcher.BlockedError: 被拦截且 abort_on_block 为 True。
        """
        result = CrawlResult(source=spec.name)
        adapter = SiteAdapter(spec, self.session)

        logger.info("开始采集《%s》", spec.name)
        iterator = adapter.iter_listings()
        if progress:
            iterator = tqdm(iterator, desc=spec.name, unit="篇", leave=False)

        saved_this_run = 0
        try:
            for title, url in iterator:
                result.found += 1

                if max_docs and saved_this_run >= max_docs:
                    break

                if self.store.already_seen(spec.name, url):
                    result.skipped += 1
                    continue

                try:
                    doc = adapter.fetch_document(title, url)
                except BlockedError:
                    result.blocked = True
                    logger.error("《%s》采集过程中被拦截，已停止该站", spec.name)
                    if self.abort_on_block:
                        raise
                    return result
                except Exception as exc:  # noqa: BLE001 - 单篇失败不该中断整站
                    result.failed += 1
                    result.errors.append(f"{url}: {exc}")
                    self.store.mark_seen(spec.name, url, "failed", str(exc)[:200])
                    continue

                if doc is None:
                    result.failed += 1
                    self.store.mark_seen(spec.name, url, "failed", "抽取失败")
                    continue

                self.store.add_document(doc)
                self.store.mark_seen(spec.name, url)
                result.saved += 1
                saved_this_run += 1
        except BlockedError:
            result.blocked = True
            raise

        logger.info("《%s》完成：%s", spec.name, result.summary())
        return result

    def crawl_all(
        self,
        specs: list[SiteSpec],
        max_docs: int = 0,
        progress: bool = True,
    ) -> tuple[list[CrawlResult], bool]:
        """采集多个站点。

        参数：
            specs: 站点配置列表。
            max_docs: 每站最多入库多少篇。
            progress: 是否显示进度条。

        返回：
            (各站结果列表, 是否有站被拦截)。
        """
        results: list[CrawlResult] = []
        any_blocked = False

        for spec in specs:
            try:
                results.append(self.crawl_site(spec, max_docs=max_docs, progress=progress))
            except BlockedError as exc:
                any_blocked = True
                logger.error("《%s》被拦截，跳过：%s", spec.name, exc)
                results.append(CrawlResult(source=spec.name, blocked=True, errors=[str(exc)]))
                if self.abort_on_block:
                    logger.error("已配置为遇拦截即中止，停止后续站点")
                    break

        return results, any_blocked
