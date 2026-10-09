"""命令行入口。

用法示例：

    python -m cnki_crawler status
    python -m cnki_crawler discover --probe
    python -m cnki_crawler crawl --journal DLXT --years 2
    python -m cnki_crawler crawl --all --years 5
    python -m cnki_crawler export --format md --out ../data/gov_doc
    python -m cnki_crawler pdf ingest ~/papers --out ../data/gov_doc
"""

from __future__ import annotations

import argparse
import logging
import sys

from .backends import build_backend
from .backends.wap import WapBackend
from .config import load_settings
from .crawl import Crawler
from .discovery import expand_by_probe, load_seeds
from .export import export_csv, export_jsonl, export_markdown
from .fetcher import BlockedError, CrawlerError, PoliteSession
from .pdf_ingest import ingest_pdfs
from .store import Store

logger = logging.getLogger("cnki_crawler")

# 触发风控时的退出码，便于脚本区分「正常结束」和「被拦截」
EXIT_BLOCKED = 2


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="cnki_crawler",
        description="知网能源/电力论文题录爬虫（只抓公开题录，不下载全文）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="合规边界见 README.md。",
    )
    parser.add_argument("--config", help="配置文件路径（默认用包内 config.toml）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- crawl ----
    crawl = sub.add_parser("crawl", help="爬取题录")
    crawl.add_argument("--journal", action="append", default=[], help="期刊代码，可重复指定")
    crawl.add_argument("--all", action="store_true", help="爬取配置里的全部期刊")
    crawl.add_argument("--years", type=int, help="爬取最近多少年（覆盖配置）")
    crawl.add_argument("--limit", type=int, default=0, help="最多处理多少期（冒烟测试用，0 为不限）")

    # ---- discover ----
    discover = sub.add_parser("discover", help="发现/扩展期刊")
    discover.add_argument("--probe", action="store_true", help="探测候选期刊代码并校验刊名")

    # ---- status ----
    sub.add_parser("status", help="查看爬取进度统计")

    # ---- export ----
    export = sub.add_parser("export", help="导出题录")
    export.add_argument("--format", choices=["md", "jsonl", "csv"], default="md")
    export.add_argument("--out", required=True, help="输出目录(md)或文件路径(jsonl/csv)")
    export.add_argument("--ascii-only", action="store_true", help="文件名只用 ASCII 字符")

    # ---- pdf ----
    pdf = sub.add_parser("pdf", help="自备全文 PDF 入库（不联网下载）")
    pdf_sub = pdf.add_subparsers(dest="pdf_command", required=True)
    pdf_ingest = pdf_sub.add_parser("ingest", help="扫描 PDF 并匹配题录入库")
    pdf_ingest.add_argument("pdf_dir", help="存放自备 PDF 的目录")
    pdf_ingest.add_argument("--out", required=True, help="语料输出目录")
    pdf_ingest.add_argument("--move", action="store_true", help="移动原文件而非复制")
    pdf_ingest.add_argument("--dry-run", action="store_true", help="只报告不落盘")

    return parser


def _open_store(settings) -> Store:
    """按配置打开存储。"""
    return Store(settings.paths.db, settings.paths.jsonl)


def _open_session(settings) -> PoliteSession:
    """按配置构造 HTTP 会话。"""
    return PoliteSession(
        delay=settings.network.delay,
        timeout=settings.network.timeout,
        max_retries=settings.network.max_retries,
        backoff_base=settings.network.backoff_base,
        user_agent=settings.network.user_agent or None,
    )


def cmd_crawl(args, settings) -> int:
    """执行 crawl 子命令。"""
    targets = list(args.journal)
    if args.all:
        targets = list(settings.journals)
    if not targets:
        print("请用 --journal CODE 指定期刊，或用 --all 爬取全部种子期刊", file=sys.stderr)
        return 1

    unknown = [t for t in targets if t not in settings.journals]
    if unknown:
        logger.warning("以下代码不在配置的种子表里，仍会尝试：%s", ", ".join(unknown))

    years = args.years if args.years else settings.crawl.years
    limit = args.limit or settings.crawl.max_issues

    store = _open_store(settings)
    session = _open_session(settings)
    backend = build_backend(session, settings.paths.cookie_file, delay=settings.network.delay)
    wap = WapBackend(session)
    crawler = Crawler(store, session, backend, wap, settings.crawl.abort_on_block)

    load_seeds(settings.journals, store)
    logger.info("后端：%s；目标 %d 本刊；最近 %d 年", backend.name, len(targets), years)

    total = 0
    try:
        for index, pykm in enumerate(targets, start=1):
            logger.info("[%d/%d] %s", index, len(targets), pykm)
            try:
                stats = crawler.crawl_journal(pykm, years=years, max_issues=limit)
            except BlockedError as exc:
                print(f"\n触发知网风控，已中止：{exc}", file=sys.stderr)
                print("建议：等待数小时后再试，或调大 network.delay。", file=sys.stderr)
                return EXIT_BLOCKED
            except CrawlerError as exc:
                logger.error("期刊 %s 爬取失败：%s", pykm, exc)
                continue
            total += stats.articles_new
            logger.info("  %s", stats.summary())
    except KeyboardInterrupt:
        print("\n已中断。进度已保存，重跑同一命令即可续传。", file=sys.stderr)
        return 130
    finally:
        store.close()
        session.close()

    print(f"\n完成。本次新增题录 {total} 条。")
    return 0


def cmd_discover(args, settings) -> int:
    """执行 discover 子命令。"""
    store = _open_store(settings)
    session = _open_session(settings)
    try:
        seeded = load_seeds(settings.journals, store)
        print(f"种子期刊 {seeded} 本已写入。")
        if not args.probe:
            print("加 --probe 可探测扩展更多期刊代码（会逐个请求知网并校验刊名）。")
            return 0

        known = {row["pykm"] for row in store.journals()}
        print(f"开始探测候选代码（已有 {len(known)} 本，探测间隔 {settings.network.delay}s）...")
        try:
            added = expand_by_probe(session, store, known=known)
        except BlockedError as exc:
            print(f"\n探测过程中触发风控，已停止：{exc}", file=sys.stderr)
            return EXIT_BLOCKED

        print(f"\n新增 {len(added)} 本能源/电力期刊：")
        for code, name in added:
            print(f"  {code:8s} 《{name}》")
        return 0
    finally:
        store.close()
        session.close()


def cmd_status(args, settings) -> int:
    """执行 status 子命令。"""
    store = _open_store(settings)
    try:
        stats = store.stats()
        print("爬取进度")
        print(f"  题录总数    {stats['articles']}")
        print(f"  已知期刊    {stats['journals']}")
        print(f"  期次已完成  {stats['issues_done']}")
        print(f"  期次待处理  {stats['issues_pending']}")
        if stats["articles"]:
            pct = stats["abstract_truncated"] / stats["articles"] * 100
            print(f"  摘要被截断  {stats['abstract_truncated']} ({pct:.0f}%)")

        journals = store.journals()
        if journals:
            print("\n期刊清单")
            for row in journals:
                count = sum(1 for _ in store.iter_articles(pykm=row["pykm"]))
                print(f"  {row['pykm']:8s} 《{row['name']}》  {count} 条  [{row['source']}]")
        return 0
    finally:
        store.close()


def cmd_export(args, settings) -> int:
    """执行 export 子命令。"""
    store = _open_store(settings)
    try:
        if args.format == "md":
            count = export_markdown(store, args.out, ascii_only=args.ascii_only)
            print(f"已导出 {count} 个 Markdown 文件到 {args.out}")
        elif args.format == "jsonl":
            count = export_jsonl(store, args.out)
            print(f"已导出 {count} 条到 {args.out}")
        else:
            count = export_csv(store, args.out)
            print(f"已导出 {count} 条到 {args.out}")
        return 0
    finally:
        store.close()


def cmd_pdf(args, settings) -> int:
    """执行 pdf ingest 子命令。"""
    store = _open_store(settings)
    try:
        result = ingest_pdfs(
            args.pdf_dir,
            store,
            args.out,
            move=args.move,
            dry_run=args.dry_run,
        )
        print(f"\n{result.summary()}")
        if result.unmatched:
            print("\n以下 PDF 未匹配到题录（请确认已在库中，或标题差异过大）：")
            for path in result.unmatched[:20]:
                print(f"  {path.name}")
            if len(result.unmatched) > 20:
                print(f"  ... 另有 {len(result.unmatched) - 20} 个")
        if result.errors:
            print("\n读取失败：")
            for path, err in result.errors[:10]:
                print(f"  {path.name}: {err}")
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。

    参数：
        argv: 参数列表；None 表示取 sys.argv。

    返回：
        进程退出码。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # requests 的 DEBUG 日志太吵，压掉
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    handlers = {
        "crawl": cmd_crawl,
        "discover": cmd_discover,
        "status": cmd_status,
        "export": cmd_export,
        "pdf": cmd_pdf,
    }
    handler = handlers[args.command]
    try:
        return handler(args, settings)
    except FileNotFoundError as exc:
        print(f"文件不存在：{exc}", file=sys.stderr)
        return 1
    except CrawlerError as exc:
        print(f"爬取错误：{exc}", file=sys.stderr)
        return 1
