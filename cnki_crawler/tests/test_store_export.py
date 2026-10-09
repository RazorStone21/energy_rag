"""存储与导出测试。

导出部分重点验证两件事：
1. **文件名唯一且稳定**——它是 energy_rag 的去重键，改名等于新增文档。
2. **截断摘要被如实标注**——不能让人误以为拿到了完整摘要。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cnki_crawler.export import (
    ExportError,
    build_filename,
    export_markdown,
    render_markdown,
)
from cnki_crawler.parsers import ArticleRecord
from cnki_crawler.store import Store


@pytest.fixture
def record() -> ArticleRecord:
    """构造一条测试题录。"""
    return ArticleRecord(
        article_id="DLXT202613007",
        title="基于大语言模型和深度强化学习联合决策的配电网重构方法",
        journal="电力系统自动化",
        pykm="DLXT",
        year=2026,
        issue=13,
        authors=["张三", "李四"],
        affiliations=["华北电力大学"],
        keywords=["配电网重构", "深度强化学习"],
        subjects=["电力系统及其自动化"],
        abstract="针对配电网重构问题，提出一种联合决策方法",
        abstract_truncated=True,
        cited_count=12,
        download_count=340,
        url="https://wap.cnki.net/touch/web/Journal/Article/DLXT202613007.html",
    )


# ---------------- 存储 ----------------


def test_题录写入与读回(tmp_path: Path, record: ArticleRecord):
    """写入后应能完整读回，JSON 字段正确反序列化。"""
    with Store(tmp_path / "t.db", tmp_path / "t.jsonl") as store:
        store.add_article(record)
        got = store.article("DLXT202613007")

    assert got is not None
    assert got["title"] == record.title
    assert got["authors"] == ["张三", "李四"]
    assert got["keywords"] == ["配电网重构", "深度强化学习"]
    assert got["abstract_truncated"] is True


def test_同时写JSONL(tmp_path: Path, record: ArticleRecord):
    """JSONL 应逐条追加，进程被杀也不丢已抓数据。"""
    jsonl = tmp_path / "t.jsonl"
    with Store(tmp_path / "t.db", jsonl) as store:
        store.add_article(record)
        store.add_article(ArticleRecord(article_id="X1", title="第二篇"))

    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert "DLXT202613007" in lines[0]


def test_断点续传标记(tmp_path: Path):
    """期次完成标记应持久化，重跑时能跳过。"""
    with Store(tmp_path / "t.db") as store:
        store.upsert_issue("DLXT", 2026, 13, "电力系统自动化", 20)
        assert store.is_issue_done("DLXT", 2026, 13) is False

        store.mark_issue_done("DLXT", 2026, 13)
        assert store.is_issue_done("DLXT", 2026, 13) is True


def test_重复写入不产生重复记录(tmp_path: Path, record: ArticleRecord):
    """同一 article_id 重复写入应覆盖而非新增。"""
    with Store(tmp_path / "t.db") as store:
        store.add_article(record)
        store.add_article(record)
        assert store.stats()["articles"] == 1


# ---------------- 文件名 ----------------


def test_文件名包含文章ID保证唯一(record: ArticleRecord):
    """文件名必须以文章 ID 开头，这是唯一性的来源。"""
    name = build_filename(record.to_dict())

    assert name.startswith("DLXT202613007-")
    assert name.endswith("-2026-cn.md")


def test_文件名稳定(record: ArticleRecord):
    """同样的题录必须生成同样的文件名——改名等于新增文档。"""
    assert build_filename(record.to_dict()) == build_filename(record.to_dict())


def test_文件名剔除非法字符(record: ArticleRecord):
    """标题里的路径分隔符等非法字符必须被剔除。"""
    data = record.to_dict()
    data["title"] = '含/斜杠:和*星号的标题?"<>|'
    name = build_filename(data)

    assert "/" not in name
    assert ":" not in name
    assert "*" not in name
    assert "?" not in name


def test_纯ASCII文件名模式(record: ArticleRecord):
    """--ascii-only 模式下文件名不含非 ASCII 字符。"""
    name = build_filename(record.to_dict(), ascii_only=True)

    assert name.isascii()


# ---------------- Markdown 渲染 ----------------


def test_渲染包含题录字段(record: ArticleRecord):
    """正文应包含标题、作者、期刊等关键字段。"""
    md = render_markdown(record.to_dict())

    assert record.title in md
    assert "张三" in md
    assert "电力系统自动化" in md
    assert "2026年第13期" in md


def test_截断摘要被明确标注(record: ArticleRecord):
    """截断摘要必须带免责说明，这是诚实性要求。"""
    md = render_markdown(record.to_dict())

    assert "截断" in md
    assert "并非完整摘要" in md


def test_完整摘要不加截断说明(record: ArticleRecord):
    """PC 后端拿到的完整摘要不应出现截断说明。"""
    data = record.to_dict()
    data["abstract_truncated"] = False

    md = render_markdown(data)
    assert "并非完整摘要" not in md


# ---------------- 导出 ----------------


def test_导出markdown到目录(tmp_path: Path, record: ArticleRecord):
    """应导出为 md 文件，且目录不存在时报错而非静默创建。"""
    out = tmp_path / "gov_doc"
    out.mkdir()

    with Store(tmp_path / "t.db") as store:
        store.add_article(record)
        count = export_markdown(store, out)

    assert count == 1
    files = list(out.glob("*.md"))
    assert len(files) == 1
    assert files[0].name.startswith("DLXT202613007-")


def test_输出目录不存在时抛错(tmp_path: Path, record: ArticleRecord):
    """输出目录必须已存在，避免误写到意料之外的路径。"""
    with Store(tmp_path / "t.db") as store:
        store.add_article(record)
        with pytest.raises(ExportError):
            export_markdown(store, tmp_path / "不存在的目录")
