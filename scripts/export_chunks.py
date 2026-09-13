"""导出 chunks.pkl 为可读格式（JSON / CSV / Markdown 预览 / JSONL）。

chunks.pkl 里存的是 langchain 的 Document 对象，直接 pickle.load 需要
安装 langchain_core 的 Python 环境。本脚本在读取后
把每个片段转换成 {source, page, type, content} 字典，便于查看或交给
Excel / pandas / 评测脚本使用。

用法：
    python -m scripts.export_chunks                       # 默认导出 JSON 到 data/chunks_latest.json
    python -m scripts.export_chunks --format csv          # 导出 CSV（Excel 可直接打开）
    python -m scripts.export_chunks --format md           # 导出 Markdown 预览
    python -m scripts.export_chunks --format jsonl        # 导出 JSONL（每行一条）
    python -m scripts.export_chunks --stats               # 只打印统计信息，不导出
    python -m scripts.export_chunks --preview 5           # 只打印前 5 条到控制台
    python -m scripts.export_chunks --type table          # 只导出表格类 chunk
    python -m scripts.export_chunks --source 绿证          # 只导出文件名包含「绿证」的 chunk
    python -m scripts.export_chunks --limit 100 -o out.json   # 只导出前 100 条，指定输出路径
"""

import argparse
import json

from src.config import load_settings

LOCATION_FIELDS = (
    "heading_path",
    "line_start",
    "line_end",
    "block_kind",
    "block_index",
    "paragraph_start",
    "paragraph_end",
    "table_index",
    "sheet_name",
    "sheet_state",
    "row_start",
    "row_end",
    "column_start",
    "column_end",
    "header_row_start",
    "header_row_end",
)


def _export_record(document):
    """整理通用字段，并在文档确有章节或行号时保留这些信息。"""
    record = {
        "source": document.metadata.get("source", ""),
        "page": document.metadata.get("page", ""),
        "type": document.metadata.get("type", "text"),
        "content": document.page_content,
    }
    for field in LOCATION_FIELDS:
        if field in document.metadata:
            record[field] = document.metadata[field]
    return record


def _location_label(record):
    """按实际存在的信息显示页码、标题和所属原文块行号。"""
    parts = []
    if record.get("page") not in (None, "", 0, "0", "?"):
        parts.append(f"第{record['page']}页")
    if record.get("heading_path"):
        parts.append(f"标题: {record['heading_path']}")
    if record.get("line_start") is not None and record.get("line_end") is not None:
        parts.append(f"所属原文块: 第{record['line_start']}—{record['line_end']}行")
    if record.get("paragraph_start") is not None and record.get("paragraph_end") is not None:
        parts.append(f"所属原文段落: 第{record['paragraph_start']}—{record['paragraph_end']}段")
    if record.get("table_index") is not None:
        parts.append(f"第{record['table_index']}个表格")
    if record.get("sheet_name"):
        parts.append(f"工作表: {record['sheet_name']}")
        parts.append(f"第{record['row_start']}—{record['row_end']}行")
        parts.append(f"{record['column_start']}—{record['column_end']}列")
        if record.get("header_row_start") is not None:
            parts.append(f"表头: 第{record['header_row_start']}—{record['header_row_end']}行")
    return " | ".join(parts)


def load_chunks(settings):
    """通过 ChunkStore 读取片段，把每条结果整理成来源、页码、类型和正文的字典。"""
    if not settings.chunks_path.exists():
        raise SystemExit(
            f"[错误] 未找到 {settings.chunks_path}，请先运行 `python main.py build` 建索引"
        )
    from src.storage.chunks import ChunkStore

    chunks = ChunkStore(settings.chunks_path, settings.manifest_path).load_chunks()
    return [_export_record(chunk) for chunk in chunks]


def filter_chunks(chunks, type_=None, source=None, limit=None):
    """按内容类型和文件名包含的文字筛选片段，再按 limit 限制数量。"""
    if type_:
        chunks = [c for c in chunks if c["type"] == type_]
    if source:
        chunks = [c for c in chunks if source in c["source"]]
    if limit is not None:
        chunks = chunks[:limit]
    return chunks


def print_stats(chunks):
    """打印类型分布与来源（文件）分布。"""
    from collections import Counter

    types = Counter(c["type"] for c in chunks)
    sources = Counter(c["source"] for c in chunks)
    print(f"chunk 总数: {len(chunks)}")
    print("类型分布:", dict(types))
    print(f"来源文件数: {len(sources)}")
    for src, n in sources.most_common():
        print(f"  {n:5d}  {src}")


def _type_label(t: str) -> str:
    """将片段类型转换为导出报告中使用的中文标签。"""
    return {"table": "表格", "figure": "图表", "text": "正文"}.get(t, "正文")


def to_markdown(chunks) -> str:
    """按来源文件分组，把片段正文、页码和类型整理为 Markdown 文本。"""
    from collections import defaultdict

    by_source = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c)

    lines = ["# 切分后的 chunks 预览", f"共 {len(chunks)} 个 chunk", ""]
    for src, items in by_source.items():
        lines.append(f"## {src}（{len(items)} chunks）")
        lines.append("")
        for i, c in enumerate(items, 1):
            label = _type_label(c["type"])
            location = _location_label(c)
            lines.append(f"### [{i}] {location}（{label}）")
            lines.append("")
            lines.append(c["content"])
            lines.append("")
    return "\n".join(lines)


def export(chunks, fmt, output):
    """把片段写到 output 文件，fmt 可选 json、csv、md 或 jsonl。"""
    if fmt == "json":
        data = json.dumps(chunks, ensure_ascii=False, indent=2)
        output.write_text(data, encoding="utf-8")
    elif fmt == "jsonl":
        output.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
            encoding="utf-8",
        )
    elif fmt == "csv":
        import csv

        # 带 BOM 的 UTF-8 便于 Excel 识别中文；newline 避免 Windows 下出现额外空行。
        with open(output, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["source", "page", "type", "content"]
            for field in LOCATION_FIELDS:
                if any(field in chunk for chunk in chunks):
                    fieldnames.append(field)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(chunks)
    elif fmt == "md":
        output.write_text(to_markdown(chunks), encoding="utf-8")
    else:  # pragma: no cover
        raise SystemExit(f"[错误] 未知格式 {fmt}，可选：json / csv / md / jsonl")


def main():
    """读取导出参数，筛选片段并展示统计、预览或写出所选格式。"""
    parser = argparse.ArgumentParser(description="导出 chunks.pkl 为可读格式")
    parser.add_argument(
        "--format",
        choices=["json", "csv", "md", "jsonl"],
        default="json",
        help="导出格式（默认 json）",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="输出路径（默认 data/chunks_latest.<ext>）"
    )
    parser.add_argument(
        "--type", dest="type_", choices=["text", "table", "figure"], help="只导出指定类型"
    )
    parser.add_argument("--source", type=str, default=None, help="只导出文件名包含该子串的 chunk")
    parser.add_argument("--limit", type=int, default=None, help="只导出前 N 条")
    parser.add_argument("--stats", action="store_true", help="只打印统计信息，不导出")
    parser.add_argument("--preview", type=int, default=None, help="只打印前 N 条到控制台，不写文件")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    settings = load_settings(args.config, args.data_root)

    chunks = load_chunks(settings)
    chunks = filter_chunks(chunks, type_=args.type_, source=args.source, limit=args.limit)

    # 统计和预览只在控制台显示；若同时指定，统计优先，不写导出文件。
    if args.stats:
        print_stats(chunks)
        return

    if args.preview is not None:
        for c in chunks[: args.preview]:
            location = _location_label(c)
            print(f"[{c['type']}] {c['source']} {location}".rstrip())
            print(c["content"])
            print("-" * 60)
        print(f"（共 {len(chunks)} 条）")
        return

    output = args.output
    if output is None:
        ext = {"json": "json", "csv": "csv", "md": "md", "jsonl": "jsonl"}[args.format]
        output = settings.chunks_path.parent / f"chunks_latest.{ext}"
    else:
        output = __import__("pathlib").Path(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    export(chunks, args.format, output)
    print(f"[导出] {len(chunks)} 个 chunk -> {output}")


if __name__ == "__main__":
    main()
