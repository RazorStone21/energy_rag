"""期刊发现：种子表 + 代码探测扩展。

## 为什么需要严格的刊名校验

知网的期刊代码是拼音缩写，且存在模糊匹配。实测中探测 `SDDL` 返回的是
《中学政史地》、`YNDL` 返回《云南地理环境研究》、`NCDL` 返回《南昌大学
学报(理科版)》——**它们都返回 HTTP 200**。

如果只看状态码就收下，语料库会被无关文献污染。因此所有探测结果都必须
通过 `is_energy_power_journal` 的刊名关键词校验才会入库。
"""

from __future__ import annotations

import logging

from .fetcher import BlockedError, CrawlerError, LoginRequiredError, PoliteSession
from .parsers import ParseError, parse_journal_page
from .store import Store

logger = logging.getLogger(__name__)

# 能源/电力领域的刊名关键词。命中任意一个才认为是目标领域的期刊。
ENERGY_POWER_KEYWORDS = (
    "电力",
    "电网",
    "电气",
    "电机",
    "电工",
    "能源",
    "储能",
    "光伏",
    "太阳能",
    "风电",
    "风能",
    "核电",
    "核动力",
    "核科学",
    "动力工程",
    "热能",
    "高压",
    "变流",
    "发电",
    "输变电",
    "供用电",
    "继电",
    "自动化设备",
    "清洁能源",
    "可再生",
)

# 探测候选池：能源/电力领域中文期刊的常见拼音缩写。
# 命中率不高是正常的——`--probe` 会逐个请求并校验刊名，
# 只有确实解析到目标领域期刊的代码才会入库。
PROBE_CANDIDATES = (
    # 电力系统 / 电网
    "DWJS", "DWFX", "DWDY", "DLXT", "DLZD", "DLAZ", "DLJS", "DLQC", "DLXJ",
    "ZGDL", "GDDL", "ZJDL", "JSDL", "HNDL", "SCDL", "XJDL", "NXDL", "GSDL",
    "SHDJ", "HBDL", "DNDL", "AHDL", "FJDL", "JXDL", "SDDL", "HEDL", "HNLD",
    # 电工 / 电机
    "DGJS", "DJKZ", "DQJS", "DQXB", "DJDQ", "DYCJ",
    # 高压 / 变流
    "GDYJ", "GDYZ", "GYDQ",
    # 动力 / 热能
    "RNDL", "DLGC", "RNXB", "TYXB",
    # 新能源 / 储能
    "CNKX", "CNJS", "KZNY", "ZGLN", "TYNY", "FNYN", "TYNB", "ZDSL",
    # 核电
    "HDLG", "YZJS", "YZKX", "HKDL",
    # 综合 / 学报
    "ZGDC", "JDQW", "DWDQ", "JSND", "DLZD",
)


def is_energy_power_journal(name: str) -> bool:
    """判断刊名是否属于能源/电力领域。

    参数：
        name: 期刊名称。

    返回：
        True 表示命中领域关键词。
    """
    return any(keyword in name for keyword in ENERGY_POWER_KEYWORDS)


def load_seeds(settings_journals: dict[str, str], store: Store) -> int:
    """把配置里的种子期刊写入存储。

    参数：
        settings_journals: 配置的 {代码: 刊名} 映射。
        store: 存储对象。

    返回：
        写入的期刊数量。
    """
    for pykm, name in settings_journals.items():
        store.upsert_journal(pykm, name, source="seed")
    return len(settings_journals)


def probe_code(session: PoliteSession, code: str) -> str | None:
    """探测一个期刊代码，返回刊名。

    参数：
        session: HTTP 会话。
        code: 候选期刊代码。

    返回：
        刊名；代码无效或页面结构不符时返回 None。

    异常：
        fetcher.BlockedError: 触发风控，应立即中止。
    """
    from .backends.wap import _JOURNAL_URL

    try:
        html = session.get(_JOURNAL_URL.format(pykm=code))
    except LoginRequiredError:
        return None
    except (CrawlerError, ParseError) as exc:
        logger.debug("探测 %s 失败：%s", code, exc)
        return None

    try:
        meta = parse_journal_page(html, pykm=code)
    except ParseError:
        return None
    return meta.name or None


def expand_by_probe(
    session: PoliteSession,
    store: Store,
    candidates: tuple[str, ...] = PROBE_CANDIDATES,
    known: set[str] | None = None,
) -> list[tuple[str, str]]:
    """批量探测候选代码，把命中的能源/电力期刊写入存储。

    参数：
        session: HTTP 会话。
        store: 存储对象。
        candidates: 候选代码池。
        known: 已知代码集合，用于跳过重复探测。

    返回：
        新增的 (代码, 刊名) 列表。

    异常：
        fetcher.BlockedError: 触发风控，立即向上抛出让调用方中止。
    """
    known = set(known or ())
    added: list[tuple[str, str]] = []

    for code in candidates:
        if code in known:
            continue
        try:
            name = probe_code(session, code)
        except BlockedError:
            # 风控是全局性的，继续探测只会更糟
            logger.error("探测过程中触发知网风控，已停止扩展。已完成 %d 个", len(added))
            raise

        if not name:
            continue
        if not is_energy_power_journal(name):
            logger.info("跳过 %s（《%s》不在能源/电力领域）", code, name)
            continue

        store.upsert_journal(code, name, source="probe")
        known.add(code)
        added.append((code, name))
        logger.info("新增期刊 %s = 《%s》", code, name)

    return added
