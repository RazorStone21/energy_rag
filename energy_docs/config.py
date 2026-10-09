"""配置加载：读取 config.toml 并构造站点清单。

相对路径一律相对于配置文件所在目录解析。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .adapters.base import SiteSpec

DEFAULT_CONFIG = Path(__file__).parent / "config.toml"

# 请求间隔下限：政府站承载能力有限
MIN_DELAY = 1.5


@dataclass
class PathSettings:
    """各类路径（已解析为绝对路径）。"""

    data_dir: Path
    db: Path
    jsonl: Path


@dataclass
class NetworkSettings:
    """网络参数。"""

    delay: float = 2.0
    timeout: int = 25
    max_retries: int = 3
    backoff_base: float = 5.0


@dataclass
class CrawlSettings:
    """采集行为参数。"""

    abort_on_block: bool = True
    max_pages: int = 40


@dataclass
class Settings:
    """完整配置。"""

    paths: PathSettings
    network: NetworkSettings
    crawl: CrawlSettings
    sites: list[SiteSpec]


def load_settings(config_path: str | Path | None = None) -> Settings:
    """读取并校验配置。

    参数：
        config_path: 配置文件路径；None 表示用包内 config.toml。

    返回：
        Settings。

    异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置非法（间隔过低、站点清单为空或字段缺失）。
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    base = path.parent

    raw_paths = raw.get("paths", {})
    paths = PathSettings(
        data_dir=_resolve(base, raw_paths.get("data_dir", "data")),
        db=_resolve(base, raw_paths.get("db", "data/energy_docs.db")),
        jsonl=_resolve(base, raw_paths.get("jsonl", "data/documents.jsonl")),
    )

    raw_net = raw.get("network", {})
    delay = float(raw_net.get("delay", 2.0))
    if delay < MIN_DELAY:
        raise ValueError(
            f"network.delay={delay} 过低（下限 {MIN_DELAY} 秒）。"
            "政府网站承载能力有限，请勿设置更低的值。"
        )
    network = NetworkSettings(
        delay=delay,
        timeout=int(raw_net.get("timeout", 25)),
        max_retries=int(raw_net.get("max_retries", 3)),
        backoff_base=float(raw_net.get("backoff_base", 5.0)),
    )

    raw_crawl = raw.get("crawl", {})
    crawl = CrawlSettings(
        abort_on_block=bool(raw_crawl.get("abort_on_block", True)),
        max_pages=int(raw_crawl.get("max_pages", 40)),
    )

    sites = _load_sites(raw.get("sites") or [], crawl.max_pages)
    if not sites:
        raise ValueError(f"配置文件 {path} 里没有任何 [[sites]]，没有可采集的站点")

    return Settings(paths=paths, network=network, crawl=crawl, sites=sites)


def _load_sites(raw_sites: list[dict], max_pages: int) -> list[SiteSpec]:
    """把配置里的站点段转成 SiteSpec 列表。

    参数：
        raw_sites: TOML 里的 [[sites]] 数组。
        max_pages: 默认最大翻页数。

    返回：
        SiteSpec 列表。

    异常：
        ValueError: 必填字段缺失或正则非法。
    """
    import re

    specs: list[SiteSpec] = []
    for index, item in enumerate(raw_sites, start=1):
        missing = [k for k in ("name", "base", "entry", "link_pattern") if not item.get(k)]
        if missing:
            raise ValueError(f"第 {index} 个 [[sites]] 缺少必填字段：{', '.join(missing)}")

        pattern = str(item["link_pattern"])
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"站点「{item['name']}」的 link_pattern 不是合法正则：{exc}") from exc

        specs.append(
            SiteSpec(
                name=str(item["name"]),
                base=str(item["base"]).rstrip("/"),
                entry=str(item["entry"]),
                link_pattern=pattern,
                category=str(item.get("category", "")),
                max_pages=int(item.get("max_pages", max_pages)),
                min_chars=int(item.get("min_chars", 120)),
            )
        )
    return specs


def _resolve(base: Path, value: str) -> Path:
    """把相对路径按配置文件目录解析为绝对路径。"""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()
