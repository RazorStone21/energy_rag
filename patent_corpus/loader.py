"""专利导出文件的读取与列名识别。

不同专利数据库（佰腾、专利之星、智慧芽、incoPat）导出的表头各不相同，
而且同一家在改版时也会变。所以这里不硬编码列名，而是维护一张**别名表**，
对表头做归一化后模糊匹配。

匹配不上的列会被如实报告出来，而不是猜——猜错列会把权利要求书当摘要用，
产出的语料就废了。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# 列名别名表：规范字段 -> 可能的表头写法（归一化后比较）
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "title": (
        "标题", "名称", "专利名称", "发明名称", "发明创造名称", "专利标题",
        "title", "inventiontitle", "patenttitle",
    ),
    "abstract": (
        "摘要", "专利摘要", "摘要中文", "中文摘要", "摘要内容",
        "abstract", "abstracttext",
    ),
    "claims": (
        "权利要求", "权利要求书", "主权项", "权利要求书全文",
        "claims", "claim", "claims全文",
    ),
    "description": (
        "说明书", "说明书全文", "专利说明书", "描述", "说明书文本",
        "description", "specification", "fulldescription",
    ),
    "pub_number": (
        "公开号", "公开公告号", "公告号", "专利号", "公开(公告)号", "公开（公告）号",
        "publicationnumber", "pubno", "pubnumber", "patentnumber",
    ),
    "app_number": (
        "申请号", "申请号内部", "申请号(内部)", "applicationnumber", "appno",
    ),
    # 领域判定用主分类号，**不是**完整分类号。原因见 domain.py 的模块说明：
    # 实测中「一种NFC近场供电解锁的U形锁」主分类号是 E05B67/22（锁），
    # 但副分类号里带 H02J50/10（无线供电）。用完整分类号会把这类无关专利
    # 全部误收进来。所以 ipc 只认主分类号。
    "ipc": (
        "主分类号", "ipc主分类", "主ipc", "ipc分类号主",
        "mainipc", "primaryipc",
    ),
    # 完整分类号仅作参考字段保留，不参与领域判定
    "ipc_all": (
        "分类号", "ipc分类号", "ipc", "ipc分类", "ipcnumber", "ipcclass",
    ),
    "applicant": (
        "申请人", "申请专利权人", "专利权人", "申请人名称",
        "applicant", "assignee",
    ),
    "inventor": ("发明人", "设计人", "inventor", "inventors"),
    "app_date": ("申请日", "applicationdate", "appdate"),
    "pub_date": ("公开日", "公开公告日", "公告日", "publicationdate", "pubdate"),
    "patent_type": ("专利类型", "类型", "patenttype", "type"),
    "status": ("法律状态", "法律状态代码", "status", "legalstatus"),
}

# 归一化时去掉的字符
_JUNK_RE = re.compile(r"[\s　()（）\[\]【】<>《》:：,，;；、\-_/\\|.!！?？\"'“”‘’]+")

# 支持的文件后缀。.xls（旧版 BIFF 格式）需要 xlrd。
SUPPORTED_SUFFIXES = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt")


class LoadError(Exception):
    """读取或识别失败。"""


@dataclass
class LoadedTable:
    """一个导出文件的读取结果。"""

    path: Path
    frame: pd.DataFrame
    columns: dict[str, str | None]
    """规范字段 -> 实际列名。匹配不到的字段值为 None。"""

    row_count: int = 0
    unmatched_headers: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """至少要有标题或摘要，否则这份表没有利用价值。"""
        return bool(self.columns.get("title") or self.columns.get("abstract"))


def normalize_header(raw: str) -> str:
    """归一化表头，用于模糊匹配。

    参数：
        raw: 原始表头文本。

    返回：
        去掉空白与标点、转小写的字符串。
    """
    return _JUNK_RE.sub("", str(raw or "")).lower()


def detect_columns(headers: list[str]) -> tuple[dict[str, str | None], list[str]]:
    """把实际表头映射到规范字段。

    匹配策略：先精确（归一化后相等），再包含（表头含别名或别名含表头），
    最后取最长别名的匹配以避免「摘要」误配到「摘要附图」这类长表头。

    参数：
        headers: 实际表头列表。

    返回：
        (规范字段到实际列名的映射, 未匹配上的表头列表)。
    """
    normalized = {h: normalize_header(h) for h in headers}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        norm_aliases = [normalize_header(a) for a in aliases]
        best: tuple[int, str] | None = None

        for header, norm in normalized.items():
            if header in used or not norm:
                continue
            for alias in norm_aliases:
                if norm == alias:
                    score = 1000 + len(alias)
                elif alias in norm or norm in alias:
                    score = len(alias)
                else:
                    continue
                if best is None or score > best[0]:
                    best = (score, header)

        if best is not None:
            mapping[canonical] = best[1]
            used.add(best[1])
        else:
            mapping[canonical] = None

    unmatched = [h for h in headers if h not in used]
    return mapping, unmatched


def load_table(path: str | Path, sheet: int | str = 0) -> LoadedTable:
    """读取一个导出文件。

    参数：
        path: 文件路径（.xlsx/.xlsm/.csv/.tsv/.txt）。
        sheet: Excel 工作表名或序号。

    返回：
        LoadedTable。

    异常：
        LoadError: 后缀不支持、读取失败或表头无法识别。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise LoadError(f"文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise LoadError(f"{file_path.name}：不支持的后缀 {suffix}")

    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            # .xls（旧版 BIFF）由 xlrd 支撑，未安装时给出可操作的提示
            try:
                frame = pd.read_excel(file_path, sheet_name=sheet, dtype=str)
            except ImportError as exc:
                raise LoadError(
                    f"{file_path.name}：读取旧版 .xls 需要 xlrd。"
                    "请执行 pip install xlrd，或在导出时改选 .xlsx / .csv。"
                ) from exc
        else:
            # 专利导出常见 GBK，先试 utf-8，失败再退 GBK
            try:
                frame = pd.read_csv(file_path, dtype=str, sep=None, engine="python")
            except UnicodeDecodeError:
                frame = pd.read_csv(
                    file_path, dtype=str, sep=None, engine="python", encoding="gbk"
                )
    except Exception as exc:  # noqa: BLE001 - 统一转成 LoadError
        raise LoadError(f"{file_path.name} 读取失败：{exc}") from exc

    frame.columns = [str(c).strip() for c in frame.columns]
    mapping, unmatched = detect_columns(list(frame.columns))

    table = LoadedTable(
        path=file_path,
        frame=frame,
        columns=mapping,
        row_count=len(frame),
        unmatched_headers=unmatched,
    )

    if not table.usable:
        raise LoadError(
            f"{file_path.name}：没有识别出标题或摘要列。"
            f"实际表头为 {list(frame.columns)[:12]}。"
            "请在 config.toml 的 [columns] 段里手动指定列名映射。"
        )

    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        logger.info("%s：未识别到这些字段 %s", file_path.name, ", ".join(missing))

    return table


def collect_files(source: str | Path, pattern: str = "**/*") -> list[Path]:
    """收集目录下所有受支持的导出文件。

    参数：
        source: 文件或目录路径。
        pattern: 目录下的 glob 模式。

    返回：
        文件路径列表（已排序）。

    异常：
        LoadError: 路径不存在。
    """
    root = Path(source)
    if not root.exists():
        raise LoadError(f"路径不存在：{root}")
    if root.is_file():
        return [root]

    files = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files
