"""记忆存储的无模型回归测试；使用临时目录验证往返、容错和路径安全。"""

from __future__ import annotations

import os

import pytest

from src.storage.memory import (
    SESSION_ID,
    MemoryStore,
    Turn,
    escape_line,
    new_session_id,
    parse_session,
    render_session,
    sanitize_label,
    unescape_line,
)


def store_at(path, enabled=True):
    """在临时目录里创建记忆存储，避免碰到仓库的真实数据目录。"""
    return MemoryStore(path / "memory", enabled=enabled)


def write_session(store, turns, title="测试会话"):
    """新建一份会话并写入给定的消息，返回会话 id。"""
    session = store.create_session(title)
    session.turns = list(turns)
    store.save_session(session)
    return session.id


def test_line_escaping_is_invertible():
    """验证转义与还原互为逆运算，边界样式与反斜杠都不会被吃掉。"""
    cases = [
        "## 助手",
        "\\## 助手",
        "\\\\",
        "\\",
        "# 标题",
        "   ## 用户",
        "    ## 助手",
        "## 助手 #",
        "##\t助手",
        "### 参考来源",
        "### 参考来源 #",
        "普通文本",
        "## 装机目标",
        "行中的 ### 参考来源 不是边界",
        "   ",
        "",
        "全角　## 助手",
    ]
    for value in cases:
        assert unescape_line(escape_line(value)) == value


def test_session_round_trip_keeps_content_byte_for_byte():
    """验证会话说回文件再读回来，标题、消息和来源都逐字节一致。"""
    turns = [
        Turn("user", "装机目标？"),
        Turn("assistant", "## 装机目标\n\n到 2030 年。", ["a.pdf 第27页（正文）"]),
        # 整行就是边界样式，必须原样保留而不是被当成新消息。
        Turn("assistant", "### 参考来源\n\n这是我编的标题。"),
        Turn("assistant", "\\## 助手\n\n反斜杠开头。"),
        Turn("assistant", "   ## 用户\n\n缩进三空格。"),
        Turn("assistant", "## 助手 #\n\n带收尾井号。"),
        Turn("assistant", "```\n## 用户\n```"),
        Turn("assistant", ""),
        Turn("assistant", "结尾没有换行"),
    ]
    session = parse_session(render_session("测试会话", turns, 0), "x", 0)
    assert session.title == "测试会话"
    assert len(session.turns) == len(turns)
    for original, loaded in zip(turns, session.turns):
        assert (original.role, original.content) == (loaded.role, loaded.content)
    assert session.turns[1].sources == ["a.pdf 第27页（正文）"]


def test_session_file_round_trips_through_the_store(tmp_path):
    """验证经过真实文件读写后内容仍然一致，且列表能给出标题和条数。"""
    store = store_at(tmp_path)
    session_id = write_session(
        store,
        [Turn("user", "问题"), Turn("assistant", "回答\n第二行")],
    )
    loaded = store.load_session(session_id)
    assert [turn.content for turn in loaded.turns] == ["问题", "回答\n第二行"]
    summaries = store.list_sessions()
    assert [(item.id, item.title, item.message_count) for item in summaries] == [
        (session_id, "测试会话", 2)
    ]


def test_file_never_contains_carriage_returns(tmp_path):
    """验证写入前统一换行，读取时的归一化不会改动正文。"""
    store = store_at(tmp_path)
    session_id = store.create_session("换行").id
    store.append_exchange(session_id, "问题\r\n带 CRLF", "回答\r带 CR")
    loaded = store.load_session(session_id)
    assert loaded.turns[0].content == "问题\n带 CRLF"
    assert loaded.turns[1].content == "回答\n带 CR"
    assert "\r" not in store.session_path(session_id).read_text(encoding="utf-8")


def test_empty_answer_only_records_the_question(tmp_path):
    """验证没有生成回答时只记用户那一轮，不写入空的助手段落。"""
    store = store_at(tmp_path)
    session_id = store.create_session("空回答").id
    store.append_exchange(session_id, "问题", "", [])
    assert [turn.role for turn in store.load_session(session_id).turns] == ["user"]


def test_session_ids_are_generated_and_validated(tmp_path):
    """验证会话 id 形状固定，恶意构造的 id 一律拒绝且不会越出会话目录。"""
    store = store_at(tmp_path)
    assert SESSION_ID.match(new_session_id())
    for bad in [
        "../../etc/passwd",
        "..%2F..%2Fetc",
        "a/b",
        "a\\b",
        "/etc/passwd",
        "",
        "   ",
        ".",
        "..",
        ".hidden",
        "a\0b",
        "C:evil",
        "a" * 200,
        # 大写十六进制不是允许的形状，避免大小写不敏感的路径撞名。
        "20260101-000000-ZZZZZZ",
        # 全角字符与同形字数字不在字符集里。
        "２０２６０１０１-００００００-abcdef",
        "20260101-000000-abcdef.md",
    ]:
        with pytest.raises(ValueError):
            store.session_path(bad)


def test_session_path_stays_inside_the_sessions_directory(tmp_path):
    """验证合法 id 解析出的路径始终落在会话目录内。"""
    store = store_at(tmp_path)
    path = store.session_path("20260101-000000-abcdef")
    assert path.parent == store.sessions_dir.resolve()


def test_temp_files_and_broken_files_do_not_break_the_list(tmp_path):
    """验证写入残留不进列表，手工改坏的文件只标记而不让列表失败。"""
    store = store_at(tmp_path)
    good = write_session(store, [Turn("user", "问题")])
    sessions = store.sessions_dir
    # atomic_write 的临时文件也在这个目录里，不能当成会话。
    (sessions / f"{good}.md.abc123").write_text("临时残留", encoding="utf-8")
    (sessions / "随手放的文件.md").write_text("不是会话", encoding="utf-8")
    (sessions / "20260101-000000-broken.md").write_bytes("## 用户\n\n没有标题".encode("gbk"))
    ids = [item.id for item in store.list_sessions()]
    assert good in ids
    assert "随手放的文件" not in ids
    assert f"{good}.md" not in ids


def test_hand_edited_file_with_wrong_encoding_does_not_raise(tmp_path):
    """验证按其它编码另存的文件不会抛异常，只是解析不出消息，列表里仍然在。

    坏字节按替换字符读出，边界样式因此匹配不上；这里要守的是「不报错」，
    而不是假装能还原编码。
    """
    store = store_at(tmp_path)
    session_id = "20260101-000000-abcdef"
    store.sessions_dir.mkdir(parents=True, exist_ok=True)
    (store.sessions_dir / f"{session_id}.md").write_bytes("## 用户\n\n中文内容".encode("gbk"))
    loaded = store.load_session(session_id)
    assert loaded is not None
    assert loaded.turns == []
    assert [item.id for item in store.list_sessions()] == [session_id]
    assert store.list_sessions()[0].message_count == 0


def test_session_id_collisions_do_not_overwrite(tmp_path):
    """验证同一秒内连续新建不会互相覆盖。"""
    store = store_at(tmp_path)
    first = store.create_session("第一条")
    second = store.create_session("第二条")
    assert first.id != second.id
    assert len(store.list_sessions()) == 2
    assert store.load_session(first.id).title == "第一条"


def test_disabled_store_touches_nothing(tmp_path):
    """验证关闭记忆后既不读也不写，连目录都不创建。"""
    store = store_at(tmp_path, enabled=False)
    store.write_memory("不该落盘")
    assert not store.dir.exists()
    assert store.read_memory() == ""
    assert store.list_sessions() == []
    assert store.create_session("会话").id
    assert not store.dir.exists()


def test_memory_file_is_written_atomically(tmp_path, monkeypatch):
    """验证保存长期记忆时先写临时文件再替换，中途失败不会留下半份内容。"""
    store = store_at(tmp_path)
    store.write_memory("第一版")
    import src.storage.memory as module

    def fail(path, data):
        """模拟写入失败，用于确认原文件不被破坏。"""
        raise OSError("磁盘已满")

    monkeypatch.setattr(module, "atomic_write", fail)
    with pytest.raises(OSError):
        store.write_memory("第二版")
    assert store.memory_path.read_text(encoding="utf-8") == "第一版"
    assert os.listdir(store.dir) == ["memory.md"]


def test_source_labels_are_single_line_and_plain(tmp_path):
    """验证来源标签被压成单行并去掉反引号，不会撑破列表项。"""
    assert sanitize_label("a.pdf\n第3页") == "a.pdf 第3页"
    assert sanitize_label("`a.pdf` 第3页") == "'a.pdf' 第3页"
    store = store_at(tmp_path)
    session_id = store.create_session("标签").id
    store.append_exchange(session_id, "问题", "回答", ["a.pdf\n\t第3页（正文）"])
    loaded = store.load_session(session_id)
    assert loaded.turns[1].sources == ["a.pdf 第3页（正文）"]
    raw = store.session_path(session_id).read_text(encoding="utf-8")
    assert raw.count("1. a.pdf 第3页（正文）") == 1


def test_edited_title_is_kept_and_file_name_wins_as_id(tmp_path):
    """验证标题写到文件里，而 id 始终以文件名为准，手工改名不会出现两个 id。"""
    store = store_at(tmp_path)
    session_id = write_session(store, [Turn("user", "问题")], title="原标题")
    path = store.session_path(session_id)
    renamed = path.with_name("20260101-000000-rename.md")
    path.rename(renamed)
    loaded = store.load_session("20260101-000000-rename")
    assert loaded.title == "原标题"
    assert loaded.id == "20260101-000000-rename"


def test_parse_never_raises_on_arbitrary_text():
    """验证任意文本都能解析，不会抛异常，没有标题时退回会话 id。"""
    cases = ["", "没有边界", "## 用户\n只有用户", "\n\n\n", "# 只有标题", "## 助手\n\n回答"]
    for text in cases:
        session = parse_session(text, "20260101-000000-abcdef", 0)
        assert session.id == "20260101-000000-abcdef"
    assert parse_session("没有标题", "abc", 0).title == "abc"
    assert parse_session("# 有标题", "abc", 0).title == "有标题"
