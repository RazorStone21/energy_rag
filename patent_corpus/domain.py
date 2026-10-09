"""领域筛选与文本清洗。

## 为什么按 IPC 分类号筛

专利的 IPC 分类号是人工标引的，比关键词可靠得多——用「电力」做关键词会漏掉
写「电网」「配网」「变流器」的，也会误收「电力猫」这类消费电子。

## 必须用「主分类号」，不能用完整分类号

这是从真实导出数据里发现的坑。实测样本：

    一种NFC近场供电解锁的U形锁
      主分类号：E05B67/22（锁）
      分类号  ：E05B67/22;E05B67/18;E05B47/00;E05B67/00;H02J50/10

    一种揉锤按摩机芯和按摩器械
      主分类号：A61H7/00（按摩）
      分类号  ：A61H7/00;A61H23/02;H02K7/075;H02K7/116

这两件**都不是电力专利**，只是一个用了无线供电、一个用了电动机，所以在副分类
里带了 H02 号。如果拿完整分类号做判定，这类无关专利会被成批误收进语料。

所以领域判定**只看主分类号**（`PatentRecord.ipc`）。完整分类号保留在
`ipc_all` 字段里供人工复核，但不参与筛选。

## 为什么分「严格」和「宽泛」两类

像 `H01L`（半导体器件）本身覆盖所有芯片，直接收进来会把光伏以外的半导体
专利全带进来；`F23`（燃烧设备）也包括锅炉以外的各种燃烧器。所以这类**宽泛
分类必须配合关键词确认**才能判定属于能源电力领域。
"""

from __future__ import annotations

import html
import re

# 严格能源/电力分类：命中即是目标领域
STRICT_IPC: dict[str, str] = {
    "H02": "发电、变电、配电",
    "H02J": "供电或配电系统",
    "H02K": "电机",
    "H02M": "变流装置",
    "H02N": "静电电机等",
    "H02P": "电机控制",
    "H02S": "光伏发电系统",
    "H01M": "电池、燃料电池",
    "F03D": "风力发电机",
    "F03B": "水力机械",
    "F03G": "其他能源机械",
    "F01K": "蒸汽机动力装置",
    "F01D": "汽轮机",
    "F02C": "燃气轮机",
    "F22B": "蒸汽发生",
    "F22G": "蒸汽过热",
    "F24S": "太阳能集热器",
    "F24T": "地热能利用",
    "G21": "核物理、核工程",
    "G21B": "聚变反应堆",
    "G21C": "裂变反应堆",
    "G21D": "核发电",
    "C10L": "燃料",
}

# 宽泛分类：跨领域通用，必须配合关键词确认
BROAD_IPC: dict[str, str] = {
    "H01L": "半导体器件（含光伏，也含其他芯片）",
    "H01G": "电容器（含超级电容）",
    "H05B": "电热",
    "F23": "燃烧设备",
    "F24H": "流体加热器",
    "F28D": "换热设备",
    "B60L": "电动车辆",
    "H01B": "电缆导体",
    "G01R": "电变量测量",
    "G05F": "电压电流调节",
    "C01B": "非金属元素（含制氢）",
    "C25B": "电解制氢",
}

# 领域关键词：用于宽泛分类的确认，以及无 IPC 记录的兜底
DOMAIN_KEYWORDS: tuple[str, ...] = (
    "电力", "电网", "配电", "变电", "输电", "发电", "电能", "电量",
    "电压", "电流", "母线", "变压器", "变流器", "逆变器", "换流",
    "光伏", "太阳能", "风电", "风能", "风力", "储能", "电池",
    "锂电", "燃料电池", "氢能", "制氢", "电解", "核电", "核能", "反应堆",
    "汽轮机", "燃气轮机", "锅炉", "余热", "热电", "热泵", "地热",
    "新能源", "可再生", "充电桩", "充电站", "微网", "微电网",
    "智能电网", "特高压", "直流输电", "无功", "继电保护", "调度",
    "发电厂", "电站", "机组", "电机", "励磁", "绝缘", "开关柜",
)

# IPC 解析：抽出部+大类+小类，如 "H02J 3/38" -> "H02J"
_IPC_RE = re.compile(r"([A-H]\d{2}[A-Z])")

# 清洗时要去除的控制字符
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 连续空白
_WS_RE = re.compile(r"[ \t ]+")

# 连续换行
_NL_RE = re.compile(r"\n{3,}")


class DomainError(Exception):
    """领域筛选配置错误。"""


def parse_ipc_codes(raw: str) -> list[str]:
    """从 IPC 字段文本里提取所有分类号。

    参数：
        raw: 原始 IPC 文本，可能形如 `H02J3/38;H02J13/00` 或 `H02J 3/38, H02M 7/48`。

    返回：
        去重后的分类号列表，如 `["H02J", "H02M"]`。
    """
    if not raw:
        return []
    codes = _IPC_RE.findall(raw.upper())
    # 保序去重
    seen: set[str] = set()
    result: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def _prefix_hit(code: str, table: dict[str, str]) -> str | None:
    """判断分类号是否命中表中的某个前缀，返回**最长**的命中前缀。

    必须取最长前缀：`H02` 和 `H02J` 都在表里，若按字典序先命中 `H02`，
    判定依据就会粗化成「发电、变电、配电」而不是精确的「供电或配电系统」。
    """
    matches = [prefix for prefix in table if code.startswith(prefix)]
    return max(matches, key=len) if matches else None


def has_domain_keyword(*texts: str) -> str | None:
    """检查文本里是否出现领域关键词，返回命中的第一个。"""
    joined = " ".join(t for t in texts if t)
    for keyword in DOMAIN_KEYWORDS:
        if keyword in joined:
            return keyword
    return None


def classify_domain(ipc_raw: str, *texts: str) -> tuple[bool, str]:
    """判断一条专利是否属于能源/电力领域。

    判定顺序：
    1. IPC 命中严格分类 → 直接相关；
    2. IPC 命中宽泛分类 → 需要关键词确认；
    3. 没有可用 IPC → 退回关键词判定。

    参数：
        ipc_raw: IPC 分类号字段。
        *texts: 用于关键词确认的文本（标题、摘要等）。

    返回：
        (是否相关, 判定依据说明)。
    """
    codes = parse_ipc_codes(ipc_raw)
    keyword = has_domain_keyword(*texts)

    for code in codes:
        hit = _prefix_hit(code, STRICT_IPC)
        if hit:
            return True, f"IPC {hit}（{STRICT_IPC[hit]}）"

    broad_hits = []
    for code in codes:
        hit = _prefix_hit(code, BROAD_IPC)
        if hit:
            broad_hits.append(hit)

    if broad_hits:
        if keyword:
            return True, f"IPC {broad_hits[0]}（{BROAD_IPC[broad_hits[0]]}）+ 关键词「{keyword}」"
        return False, f"IPC {broad_hits[0]} 属宽泛分类但无领域关键词"

    if not codes:
        if keyword:
            return True, f"无 IPC，关键词「{keyword}」"
        return False, "无 IPC 且无领域关键词"

    return False, f"IPC {codes[0]} 不在能源电力范围"


def clean_text(raw: object) -> str:
    """清洗专利文本。

    处理：HTML 实体、控制字符、多余空白、页码残留。

    参数：
        raw: 原始文本（可能是 NaN、None 或数字）。

    返回：
        清洗后的文本；无有效内容时返回空串。
    """
    if raw is None:
        return ""
    text = str(raw)
    # pandas 的缺失值会变成字符串 "nan"
    if text.strip().lower() in ("nan", "none", "", "nat"):
        return ""

    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    # 去掉每行首尾空白，再合并多余空行
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def clean_claims(raw: object) -> str:
    """清洗权利要求书文本，保留编号结构。

    权利要求书的编号（1. 2. 3.）是有意义的层级信息，不能像段落一样合并掉。

    参数：
        raw: 原始权利要求文本。

    返回：
        清洗后的文本。
    """
    return clean_text(raw)


def normalize_for_claims_split(text: str) -> list[str]:
    """把权利要求书按权项切分。

    参数：
        text: 清洗后的权利要求文本。

    返回：
        每一项权利要求的文本列表。
    """
    if not text:
        return []
    # 权项通常以行首的「数字.」或「数字、」开头
    parts = re.split(r"\n(?=\s*\d{1,3}\s*[.、．])", text)
    return [p.strip() for p in parts if p.strip()]
