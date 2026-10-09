"""配置加载：读取 config.toml 并做基本校验。

相对路径一律相对于**配置文件所在目录**解析，这样从任何工作目录运行
结果都一致。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# 默认配置文件位置
DEFAULT_CONFIG = Path(__file__).parent / "config.toml"

# 请求间隔下限：低于这个值就不再是礼貌爬取
MIN_DELAY = 1.0


@dataclass
class PathSettings:
    """各类路径。已解析为绝对路径。"""

    data_dir: Path
    db: Path
    jsonl: Path
    cookie_file: Path


@dataclass
class NetworkSettings:
    """网络参数。"""

    delay: float = 2.0
    timeout: int = 20
    max_retries: int = 3
    backoff_base: float = 5.0
    user_agent: str = ""


@dataclass
class CrawlSettings:
    """爬取行为参数。"""

    years: int = 10
    abort_on_block: bool = True
    max_issues: int = 0


@dataclass
class Settings:
    """完整配置。"""

    paths: PathSettings
    network: NetworkSettings
    crawl: CrawlSettings
    journals: dict[str, str] = field(default_factory=dict)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """读取并校验配置文件。

    参数：
        config_path: 配置文件路径；None 表示用模块同目录下的 config.toml。

    返回：
        Settings 对象。

    异常：
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置项非法（如请求间隔过低、期刊表为空）。
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
        db=_resolve(base, raw_paths.get("db", "data/cnki.db")),
        jsonl=_resolve(base, raw_paths.get("jsonl", "data/articles.jsonl")),
        cookie_file=_resolve(base, raw_paths.get("cookie_file", "data/cookies.txt")),
    )

    raw_net = raw.get("network", {})
    delay = float(raw_net.get("delay", 2.0))
    if delay < MIN_DELAY:
        raise ValueError(
            f"network.delay={delay} 过低（下限 {MIN_DELAY} 秒）。"
            "降低间隔会显著提高被知网封禁的风险，请勿设置更低的值。"
        )
    network = NetworkSettings(
        delay=delay,
        timeout=int(raw_net.get("timeout", 20)),
        max_retries=int(raw_net.get("max_retries", 3)),
        backoff_base=float(raw_net.get("backoff_base", 5.0)),
        user_agent=str(raw_net.get("user_agent", "")),
    )

    raw_crawl = raw.get("crawl", {})
    crawl = CrawlSettings(
        years=int(raw_crawl.get("years", 10)),
        abort_on_block=bool(raw_crawl.get("abort_on_block", True)),
        max_issues=int(raw_crawl.get("max_issues", 0)),
    )

    journals = {str(k): str(v) for k, v in (raw.get("journals") or {}).items()}
    if not journals:
        raise ValueError(f"配置文件 {path} 的 [journals] 段为空，没有可爬的期刊")

    return Settings(paths=paths, network=network, crawl=crawl, journals=journals)


def _resolve(base: Path, value: str) -> Path:
    """把配置里的相对路径按配置文件目录解析为绝对路径。"""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()
