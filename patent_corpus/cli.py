"""命令行入口。

典型用法：

    # 1. 先看列识别对不对（务必先做这一步）
    python -m patent_corpus inspect ~/patent_exports/

    # 2. 确认无误后构建语料
    python -m patent_corpus build ~/patent_exports/ \\
        --out corpus/ --jsonl patents.jsonl --single corpus-all.txt
"""

from __future__ import annotations

import argparse
import logging
import sys

from .domain import classify_domain
from .export import build_corpus, write_jsonl, write_text_corpus
from .loader import COLUMN_ALIASES, LoadError, collect_files, load_table

logger = logging.getLogger("patent_corpus")

# 检查阶段每张表最多看多少行
_INSPECT_ROWS = 3


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="patent_corpus",
        description="把佰腾/专利之星的导出文件转成中文能源电力训练语料",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="检查列识别与领域筛选结果（不产出文件）")
    inspect.add_argument("source", help="导出文件或目录")
    inspect.add_argument("--limit", type=int, default=200, help="最多检查多少行")

    build = sub.add_parser("build", help="构建语料")
    build.add_argument("source", help="导出文件或目录")
    build.add_argument("--out", help="每件一个 .txt 的输出目录")
    build.add_argument("--jsonl", help="输出 JSONL 的路径")
    build.add_argument("--single", help="额外把所有文档拼进这一个文件")
    build.add_argument("--no-description", action="store_true", help="不含说明书全文")
    build.add_argument("--keywords-only", action="store_true", help="只用关键词判定领域（调试用）")

    return parser


def cmd_inspect(args) -> int:
    """执行 inspect 子命令：报告列识别与领域筛选情况。"""
    files = collect_files(args.source)
    if not files:
        print(f"在 {args.source} 下没找到受支持的导出文件", file=sys.stderr)
        return 1

    print(f"找到 {len(files)} 个文件\n")

    total_rows = 0
    total_in = 0
    reasons: dict[str, int] = {}

    for path in files:
        try:
            table = load_table(path)
        except LoadError as exc:
            print(f"✗ {exc}\n")
            continue

        print(f"── {path.name}  ({table.row_count} 行)")
        print("   列识别：")
        for canonical in COLUMN_ALIASES:
            actual = table.columns.get(canonical)
            mark = "✓" if actual else "·"
            print(f"     {mark} {canonical:14s} -> {actual or '（未识别）'}")

        if table.unmatched_headers:
            preview = table.unmatched_headers[:8]
            print(f"   未使用的列：{preview}")

        # 抽样看领域判定
        shown = 0
        for record in iter_records_sample(table, args.limit):
            total_rows += 1
            if not record.title and not record.abstract:
                continue
            in_domain, reason = classify_domain(
                record.ipc, record.title, record.abstract, record.claims
            )
            if in_domain:
                total_in += 1
            reasons[reason] = reasons.get(reason, 0) + 1

            if shown < _INSPECT_ROWS:
                flag = "✓ 收录" if in_domain else "· 筛除"
                print(f"\n   [{flag}] {reason}")
                print(f"     标题: {record.title[:70]}")
                if record.ipc:
                    print(f"     IPC : {record.ipc[:70]}")
                if record.abstract:
                    print(f"     摘要: {record.abstract[:70]}")
                shown += 1
        print()

    if total_rows:
        rate = total_in / total_rows * 100
        print(f"抽样 {total_rows} 条，落入能源电力领域 {total_in} 条（{rate:.0f}%）")
        print("\n筛选依据分布：")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {count:6d}  {reason}")
        if rate < 5:
            print(
                "\n⚠ 收录率很低。请确认导出时勾选了 IPC 分类号字段——"
                "没有 IPC 的话判定会退化成关键词匹配，准确率会明显下降。"
            )
    return 0


def cmd_build(args) -> int:
    """执行 build 子命令：产出语料。"""
    if not (args.out or args.jsonl or args.single):
        print("请至少指定 --out / --jsonl / --single 之一", file=sys.stderr)
        return 1

    files = collect_files(args.source)
    if not files:
        print(f"在 {args.source} 下没找到受支持的导出文件", file=sys.stderr)
        return 1

    tables = []
    for path in files:
        try:
            tables.append(load_table(path))
        except LoadError as exc:
            print(f"✗ 跳过 {exc}", file=sys.stderr)

    if not tables:
        print("没有可用的输入表", file=sys.stderr)
        return 1

    records, stats = build_corpus(tables, keywords_only=args.keywords_only)
    print(f"\n处理 {len(tables)} 个文件，{stats.rows_seen} 行")
    print(f"落入能源电力领域 {stats.rows_included} 件，筛除 {stats.rows_skipped} 件")

    if not records:
        print("\n没有产出任何记录。先用 inspect 确认列识别与领域判定。", file=sys.stderr)
        return 1

    if args.jsonl:
        count = write_jsonl(records, args.jsonl)
        print(f"已写出 JSONL：{count} 条 -> {args.jsonl}")

    if args.out or args.single:
        # 传入同一份 stats，让「读取/筛除」的上游计数和「写出字节数」合并展示
        write_text_corpus(
            records,
            args.out,
            include_description=not args.no_description,
            single_file=args.single,
            stats=stats,
        )

    print(f"\n{stats.summary()}")

    print("\n筛选依据分布（前 10）：")
    for reason, count in sorted(stats.reasons.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {count:6d}  {reason}")
    return 0


def iter_records_sample(table, limit: int):
    """从表里抽样若干条记录，用于 inspect。

    参数：
        table: 已读取的表。
        limit: 最多取多少行。

    生成：
        PatentRecord。
    """
    from .export import iter_records

    for index, record in enumerate(iter_records(table)):
        if index >= limit:
            break
        yield record


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
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(message)s",
    )

    try:
        if args.command == "inspect":
            return cmd_inspect(args)
        return cmd_build(args)
    except LoadError as exc:
        print(f"读取错误：{exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"路径错误：{exc}", file=sys.stderr)
        return 1
