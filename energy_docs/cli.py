"""命令行入口。

典型用法：

    # 看配置里有哪些站点
    python -m energy_docs sites

    # 摸清一个新站的链接形态（用来写 link_pattern）
    python -m energy_docs probe http://nyj.shandong.gov.cn/col/col59966/index.html

    # 冒烟测试：每站只抓 5 篇
    python -m energy_docs crawl --limit 5

    # 正式采集
    python -m energy_docs crawl

    # 导出语料
    python -m energy_docs export --out corpus/ --single all.txt --min-chars 300
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from urllib.parse import urljoin, urlparse

from .adapters.base import SiteAdapter
from .attachments import (
    ATTACHMENT_SUFFIXES,
    AttachmentDownloader,
    DownloadStats,
    find_attachments,
)
from .config import load_settings
from .crawl import Crawler
from .export import write_jsonl, write_text_corpus
from .fetcher import BlockedError, FetchError, PoliteSession
from .store import Store

logger = logging.getLogger("energy_docs")

EXIT_BLOCKED = 2


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="energy_docs",
        description="中文能源电力公开文档采集器（只采公开页面，低频礼貌请求）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="配置文件路径（默认用包内 config.toml）")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sites", help="列出配置里的站点")

    probe = sub.add_parser("probe", help="探测一个页面的链接形态（用于编写 link_pattern）")
    probe.add_argument("url", help="列表页地址")

    crawl = sub.add_parser("crawl", help="采集文档")
    crawl.add_argument("--site", action="append", default=[], help="只采指定站点，可重复")
    crawl.add_argument("--limit", type=int, default=0, help="每站最多入库多少篇（0 为不限）")

    fetch = sub.add_parser("fetch", help="下载附件原件（PDF/DOC/DOCX/XLS）")
    fetch.add_argument("--out", required=True, help="附件保存目录")
    fetch.add_argument("--site", action="append", default=[], help="只处理指定站点，可重复")
    fetch.add_argument("--limit", type=int, default=0, help="每站最多检查多少篇（0 为不限）")

    export = sub.add_parser("export", help="导出语料")
    export.add_argument("--out", help="每篇一个 .txt 的输出目录")
    export.add_argument("--single", help="把所有文档拼进这一个文件")
    export.add_argument("--jsonl", help="输出 JSONL 的路径")
    export.add_argument("--source", help="只导出指定来源")
    export.add_argument("--min-chars", type=int, default=0, help="只导出正文长度不低于此值的文档")

    sub.add_parser("status", help="查看采集进度")

    return parser


def _open_session(settings) -> PoliteSession:
    """按配置构造 HTTP 会话。"""
    return PoliteSession(
        delay=settings.network.delay,
        timeout=settings.network.timeout,
        max_retries=settings.network.max_retries,
        backoff_base=settings.network.backoff_base,
    )


def cmd_sites(args, settings) -> int:
    """列出配置里的站点。"""
    print(f"共 {len(settings.sites)} 个站点：\n")
    for spec in settings.sites:
        print(f"  {spec.name}")
        print(f"    入口  : {spec.entry_url()}")
        print(f"    匹配  : {spec.link_pattern}")
        if spec.category:
            print(f"    栏目  : {spec.category}")
        print()
    return 0


def cmd_probe(args, settings) -> int:
    """探测页面链接形态，帮助编写 link_pattern。"""
    session = _open_session(settings)
    try:
        html = session.get(args.url, expect_html=False)
    except (BlockedError, FetchError) as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        return 1
    finally:
        session.close()

    soup_links = re.findall(r'href=["\']([^"\']+)["\']', html)
    parsed = urlparse(args.url)

    # 把 URL 里的数字归一化后聚类，看出路径形态
    shapes: Counter[str] = Counter()
    samples: dict[str, str] = {}
    for href in soup_links:
        url = urljoin(args.url, href)
        if urlparse(url).netloc != parsed.netloc:
            continue
        path = urlparse(url).path
        shape = re.sub(r"\d+", "N", path)
        shape = re.sub(r"[0-9a-f]{16,}", "HASH", shape)
        shapes[shape] += 1
        samples.setdefault(shape, url)

    print(f"页面共 {len(soup_links)} 个链接，同域路径形态：\n")
    for shape, count in shapes.most_common(15):
        print(f"  x{count:4d}  {shape}")
        print(f"         例：{samples[shape]}")

    print("\n把上面最像内容页的那条抄成正则填进 config.toml 的 link_pattern。")
    print("例如形态 `/art/N/N/N/art_N_N.html` 对应正则：")
    print(r"    link_pattern = '/art/\d{4}/\d{1,2}/\d{1,2}/art_\d+_\d+\.html'")
    return 0


def cmd_crawl(args, settings) -> int:
    """执行采集。"""
    specs = settings.sites
    if args.site:
        wanted = set(args.site)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"没有匹配的站点：{', '.join(args.site)}", file=sys.stderr)
            return 1

    store = Store(settings.paths.db, settings.paths.jsonl)
    session = _open_session(settings)
    crawler = Crawler(store, session, abort_on_block=settings.crawl.abort_on_block)

    print(f"准备采集 {len(specs)} 个站点，每站上限 {args.limit or '不限'} 篇\n")
    try:
        results, any_blocked = crawler.crawl_all(
            specs, max_docs=args.limit, progress=True
        )
    except KeyboardInterrupt:
        print("\n已中断。进度已保存，重跑同一命令即可续传。", file=sys.stderr)
        return 130
    finally:
        store.close()
        session.close()

    print("\n采集结果：")
    for result in results:
        print(f"  {result.source:32s} {result.summary()}")
        for err in result.errors[:3]:
            print(f"      ! {err[:110]}")

    return EXIT_BLOCKED if any_blocked else 0


def cmd_fetch(args, settings) -> int:
    """下载站点的附件原件。

    同时处理两种发文形态：
    1. 列表页直接挂附件（辽宁发 PDF、天津发 DOCX）——命中率最高；
    2. 列表页链到 HTML 正文，正文里再挂附件——只有约 7%~23% 的文章有。

    参数：
        args: 解析后的命令行参数。
        settings: 配置。

    返回：
        进程退出码。
    """
    specs = settings.sites
    if args.site:
        wanted = set(args.site)
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            print(f"没有匹配的站点：{', '.join(args.site)}", file=sys.stderr)
            return 1

    store = Store(settings.paths.db, None)
    session = _open_session(settings)
    downloader = AttachmentDownloader(session, args.out)
    stats = DownloadStats()

    print(f"从 {len(specs)} 个站点收集附件 -> {args.out}\n")
    try:
        for spec in specs:
            adapter = SiteAdapter(spec, session)
            print(f"── {spec.name}")

            # 形态 1：列表页自身就挂着附件
            try:
                list_html = session.get(spec.entry_url(), expect_html=False)
                direct = find_attachments(list_html, spec.entry_url(), spec.name)
                for att in direct:
                    stats.found += 1
                    downloader.download(att, stats)
                if direct:
                    print(f"     列表页直挂附件 {len(direct)} 个")
            except BlockedError as exc:
                print(f"     被拦截：{exc}", file=sys.stderr)
                break
            except FetchError as exc:
                logger.debug("列表页抓取失败：%s", exc)

            # 形态 2：进内容页找附件
            examined = 0
            for title, url in adapter.iter_listings():
                if args.limit and examined >= args.limit:
                    break
                if store.already_seen("attachment", url):
                    continue

                # 列表项本身就是附件（辽宁/天津模式），直接下载，别当网页去抓
                lowered = url.lower().split("?")[0]
                if lowered.endswith(ATTACHMENT_SUFFIXES):
                    from .attachments import Attachment

                    examined += 1
                    stats.found += 1
                    downloader.download(
                        Attachment(url=url, kind=lowered.rsplit(".", 1)[-1], title=title),
                        stats,
                    )
                    store.mark_seen("attachment", url)
                    continue

                examined += 1
                try:
                    html = session.get(url, expect_html=False)
                except BlockedError:
                    print("     被拦截，停止该站", file=sys.stderr)
                    break
                except FetchError:
                    continue

                for att in find_attachments(html, url, title):
                    stats.found += 1
                    downloader.download(att, stats)
                store.mark_seen("attachment", url)

            print(f"     检查内容页 {examined} 篇，累计下载 {stats.downloaded} 个")
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
    finally:
        store.close()
        session.close()

    print(f"\n{stats.summary()}")
    return 0


def cmd_export(args, settings) -> int:
    """执行导出。"""
    if not (args.out or args.single or args.jsonl):
        print("请至少指定 --out / --single / --jsonl 之一", file=sys.stderr)
        return 1

    store = Store(settings.paths.db, None)
    try:
        if args.jsonl:
            count = write_jsonl(store, args.jsonl, source=args.source)
            print(f"已写出 JSONL：{count} 篇 -> {args.jsonl}")

        if args.out or args.single:
            stats = write_text_corpus(
                store,
                args.out,
                single_file=args.single,
                source=args.source,
                min_chars=args.min_chars,
            )
            print(f"\n{stats.summary()}")
    finally:
        store.close()
    return 0


def cmd_status(args, settings) -> int:
    """查看采集进度。"""
    store = Store(settings.paths.db, None)
    try:
        stats = store.stats()
        print("采集进度")
        print(f"  已入库文档  {stats['documents']}")
        print(f"  覆盖来源    {stats['sources']}")
        print(f"  已处理 URL  {stats['urls_seen']}")
        print(f"  总字符数    {stats['chars']:,}")

        by_source: Counter[str] = Counter()
        chars: Counter[str] = Counter()
        for record in store.iter_documents():
            name = record.get("source") or "未知"
            by_source[name] += 1
            chars[name] += record.get("chars") or 0
        if by_source:
            print("\n按来源：")
            for name, count in by_source.most_common():
                print(f"  {name:34s} {count:6d} 篇   {chars[name]:>10,} 字")
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
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    handlers = {
        "sites": cmd_sites,
        "probe": cmd_probe,
        "crawl": cmd_crawl,
        "fetch": cmd_fetch,
        "export": cmd_export,
        "status": cmd_status,
    }
    return handlers[args.command](args, settings)
