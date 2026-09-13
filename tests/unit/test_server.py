"""Web 接口层的无模型回归测试；使用替身验证事件编码、顺序和错误处理。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.config import ConversationSettings, MemorySettings
from src.schemas import AnswerResult, ContextBundle, SearchHit
from src.server.demo import DemoMemoryStore, apply_demo
from src.server.events import encode_event, sources_payload
from src.server.service import RagService
from src.storage.chunks import ChunkStore
from src.storage.memory import SESSION_ID, MemoryStore


def doc(content="正文内容", source="a.pdf", page=1, type_="text"):
    """创建只包含正文和元数据的轻量片段，供无模型测试使用。"""
    return SimpleNamespace(
        page_content=content, metadata={"source": source, "page": page, "type": type_}
    )


def decode_events(frames):
    """把 SSE 报文解析回事件名和数据，用于验证编码结果和事件顺序。"""
    events = []
    for frame in frames:
        if frame.startswith(":"):
            continue
        name = "message"
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        events.append((name, json.loads("\n".join(data))))
    return events


def asking_pipeline(hits=(), answer="答案", timings=None, error=None):
    """构造按回调顺序产出来源、提示词和文本块的问答管线替身。"""
    result = AnswerResult(
        answer=answer,
        evidence=list(hits),
        prompt="",
        timings=timings if timings is not None else {"generation": 1.0},
    )

    def ask(question, with_rerank=True, hybrid=True, on_prompt=None, on_token=None, **rest):
        """按真实管线的回调顺序产出来源、提示词和文本块，可选地抛出异常。"""
        if error is not None:
            raise error
        # 没有命中片段时真实管线直接返回空答案，既不回调 on_context 也不调用生成模型。
        if not hits:
            return result
        context = ContextBundle(prompt=f"提示词：{question}", evidence=list(hits))
        if rest.get("on_context"):
            rest["on_context"](context)
        if on_token is not None:
            on_token(answer)
        result.prompt = context.prompt
        return result

    pipeline = Mock()
    pipeline.ask.side_effect = ask
    return pipeline


def fake_runtime(pipeline=None, chunk_store=None, ingestion=None, memory_store=None):
    """组装只包含服务层需要的字段的运行环境替身，不加载任何模型。

    记忆存储用演示模式的内存实现：服务层的落盘路径照样走一遍，
    但测试不会往仓库的 data/memory 里写任何文件。
    """
    settings = SimpleNamespace(
        model_ids={"generation": "Qwen/Qwen3-8B", "embedding": "BAAI/bge-m3"},
        milvus=SimpleNamespace(collection="gov_docs"),
        doc_dir=Path("/tmp/docs"),
        prompt_template="{context}|{question}|{history}|{memory}",
        conversation=ConversationSettings(),
        memory=MemorySettings(),
    )
    return SimpleNamespace(
        settings=settings,
        pipeline=pipeline or Mock(),
        chunk_store=chunk_store or Mock(),
        ingestion=ingestion or Mock(),
        memory_store=memory_store or DemoMemoryStore(seed=False),
        release_models=Mock(),
    )


def test_event_encoding_round_trips_text_with_newlines():
    """正文里的换行不能拆断报文，编码后的数据必须能原样解析回来。"""
    payload = {"text": '第一行\n第二行\r\n第三行 "引号" \\反斜杠'}
    frame = encode_event("token", payload)
    assert frame.startswith("event: token\n")
    assert frame.endswith("\n\n")
    # 一条事件的数据只占一行，载荷里的换行不会把它拆成两条事件。
    body = frame.split("\n")[1:-2]
    assert len(body) == 1 and body[0].startswith("data: ")
    assert decode_events([frame]) == [("token", payload)]


def test_source_payload_keeps_prompt_order_and_missing_scores():
    """验证来源序号与提示词片段一致，没有产生分数的阶段返回 null 而不是零。"""
    hits = [
        SearchHit(document=doc("正文", page=3), dense_score=0.5, rerank_score=0.9),
        SearchHit(document=doc("表格", type_="table"), bm25_score=7.5),
    ]
    payload = sources_payload(hits)
    assert [item["index"] for item in payload] == [1, 2]
    assert payload[0]["label"] == "a.pdf 第3页"
    assert payload[0]["type_label"] == "正文"
    assert payload[0]["scores"] == {"dense": 0.5, "bm25": None, "rrf": None, "rerank": 0.9}
    assert payload[1]["type_label"] == "表格"
    assert payload[1]["scores"]["bm25"] == 7.5


def test_chat_events_deliver_sources_before_tokens():
    """验证事件顺序为排队、来源、提示词、逐词元、结束。"""
    hit = SearchHit(document=doc("正文", page=2), dense_score=0.5)
    service = RagService(fake_runtime(pipeline=asking_pipeline([hit], answer="答")))
    events = decode_events(service.chat_events("问题"))
    assert [name for name, _ in events] == ["queued", "sources", "prompt", "token", "done"]
    assert events[1][1][0]["index"] == 1
    assert events[2][1]["prompt"] == "提示词：问题"
    assert events[3][1]["text"] == "答"
    assert events[4][1]["answer"] == "答"
    assert events[4][1]["demo"] is False


def test_chat_events_send_empty_sources_when_nothing_retrieved():
    """验证没有命中片段时仍发送空来源事件，并补齐缺失的阶段耗时。"""
    service = RagService(
        fake_runtime(pipeline=asking_pipeline([], answer="", timings={"retrieval": 0.2}))
    )
    events = decode_events(service.chat_events("问题"))
    assert [name for name, _ in events] == ["queued", "sources", "done"]
    assert events[1][1] == []
    assert events[2][1]["answer"] == ""
    assert events[2][1]["timings"] == {
        "rewrite": None,
        "retrieval": 0.2,
        "rerank": None,
        "context": None,
        "generation": None,
    }


def test_chat_events_report_failures_as_events():
    """验证管线异常转成 error 事件，生成器本身不向上抛出。"""
    service = RagService(fake_runtime(pipeline=asking_pipeline(error=RuntimeError("索引未就绪"))))
    events = decode_events(service.chat_events("问题"))
    assert [name for name, _ in events] == ["queued", "error"]
    assert "索引未就绪" in events[1][1]["message"]


def test_history_comes_from_the_stored_conversation():
    """验证历史由服务端从会话记录读取，而不是由前端传入。"""
    runtime = fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())]))
    store = runtime.memory_store
    session = store.create_session("追问的会话")
    store.append_exchange(session.id, "上一问", "上一答")
    service = RagService(runtime)
    list(service.chat_events("追问", conversation_id=session.id))
    assert runtime.pipeline.ask.call_args.kwargs["history"] == [
        ("user", "上一问"),
        ("assistant", "上一答"),
    ]


def test_exchange_is_appended_to_the_conversation():
    """验证一轮问答结束后，会话记录里多出用户与助手两条消息。"""
    runtime = fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())], answer="回答"))
    store = runtime.memory_store
    session = store.create_session("待追加")
    service = RagService(runtime)
    events = decode_events(service.chat_events("问题", conversation_id=session.id))
    assert events[-1][1]["conversation_id"] == session.id
    stored = store.load_session(session.id)
    assert [(turn.role, turn.content) for turn in stored.turns] == [
        ("user", "问题"),
        ("assistant", "回答"),
    ]


def test_memory_reaches_the_prompt_and_survives_a_failed_append():
    """验证长期记忆进入提示词，且落盘失败时事件流仍以 done 结束。"""
    runtime = fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())]))
    runtime.memory_store.write_memory("## 用户偏好\n- 关注储能")
    store = runtime.memory_store
    session = store.create_session("会话")
    store.append_exchange = Mock(side_effect=OSError("磁盘已满"))
    service = RagService(runtime)
    events = decode_events(service.chat_events("问题", conversation_id=session.id))
    prompt = next(data["prompt"] for name, data in events if name == "prompt")
    assert prompt.startswith("提示词：问题")
    # 追加失败只写日志：事件流仍以 done 结束，前端不会把已经收到的答案清掉。
    assert [name for name, _ in events][-1] == "done"
    assert runtime.pipeline.ask.call_args.kwargs["memory"] == "## 用户偏好\n- 关注储能"


def test_busy_state_reports_queueing_to_the_client():
    """验证模型被占用时状态报告忙碌，并发请求先收到排队事件再等待。"""
    service = RagService(fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())])))
    assert service.busy is False
    service._lock.acquire()
    stream = service.chat_events("问题")
    try:
        assert service.busy is True
        # 首帧在启动生成线程之前产生，这里只取第一帧后关闭生成器。
        assert decode_events([next(stream)]) == [("queued", {"busy": True})]
    finally:
        stream.close()
        service._lock.release()


def test_status_reads_index_without_loading_models(tmp_path):
    """验证状态接口只读取片段缓存和清单，不加载模型也不连接向量库。"""
    store = ChunkStore(tmp_path / "chunks.pkl", tmp_path / "build_manifest.json")
    store.publish(
        [doc("正文", source="a.pdf", page=1), doc("表格", source="b.xlsx", type_="table")],
        {"a.pdf": "hash-a", "b.xlsx": "hash-b"},
    )
    runtime = fake_runtime(chunk_store=store)
    runtime.pipeline = Mock(side_effect=AssertionError("状态接口不应触碰问答管线"))
    status = RagService(runtime).status()
    assert status["ready"] is True
    assert status["demo"] is False
    assert status["chunk_count"] == 2
    assert status["sources"] == [
        {"name": "a.pdf", "chunks": 1},
        {"name": "b.xlsx", "chunks": 1},
    ]
    assert status["built_at_ms"] and status["reason"] is None
    runtime.pipeline.assert_not_called()


def test_status_reports_unfinished_write(tmp_path):
    """验证存在未完成写入标记时状态接口说明原因，而不是抛错。"""
    store = ChunkStore(tmp_path / "chunks.pkl", tmp_path / "build_manifest.json")
    store.publish([doc()], {"a.pdf": "hash"})
    store.pending_path.write_text("{}", encoding="utf-8")
    status = RagService(fake_runtime(chunk_store=store)).status()
    assert status["ready"] is False
    assert "索引上次写入未完成" in status["reason"]


def test_status_reports_missing_index(tmp_path):
    """验证没有片段缓存时状态接口提示需要先执行入库。"""
    store = ChunkStore(tmp_path / "chunks.pkl", tmp_path / "build_manifest.json")
    status = RagService(fake_runtime(chunk_store=store)).status()
    assert status["ready"] is False
    assert "还没有索引文件" in status["reason"]


def test_build_releases_models_and_records_result(tmp_path):
    """验证入库前释放模型显存，并把结果记录到状态里供断线客户端查询。"""
    ingestion = Mock()
    ingestion.build.return_value = SimpleNamespace(processed=["a.pdf"], removed=[], failed={})
    store = ChunkStore(tmp_path / "chunks.pkl", tmp_path / "build_manifest.json")
    runtime = fake_runtime(ingestion=ingestion, chunk_store=store)
    service = RagService(runtime)
    events = decode_events(service.build_events(incremental=True))
    assert [name for name, _ in events] == ["queued", "done"]
    assert events[1][1]["processed"] == ["a.pdf"]
    ingestion.build.assert_called_once_with(incremental=True)
    runtime.release_models.assert_called_once()
    assert service.status()["last_build"]["processed"] == ["a.pdf"]


def test_demo_runtime_serves_events_without_models():
    """验证演示替身走同一条事件路径，来源使用真实的元数据字段。"""
    runtime = fake_runtime()
    apply_demo(runtime)
    service = RagService(runtime, demo=True)
    events = decode_events(service.chat_events("储能目标"))
    names = [name for name, _ in events]
    assert names[0] == "queued" and names[-1] == "done"
    assert "sources" in names and names.count("token") > 1
    labels = [source["label"] for source in events[names.index("sources")][1]]
    assert any("工作表: 分省目标" in label for label in labels)
    assert events[-1][1]["demo"] is True


def test_demo_pipeline_respects_retrieval_switches():
    """验证演示模式也会按检索开关决定哪些阶段产生分数。"""
    runtime = fake_runtime()
    apply_demo(runtime)
    service = RagService(runtime, demo=True)
    events = decode_events(service.chat_events("问题", hybrid=False, rerank=False))
    sources = next(data for name, data in events if name == "sources")
    assert sources[0]["scores"] == {"dense": 0.72, "bm25": None, "rrf": None, "rerank": None}


def test_demo_chunk_store_supports_status_reads():
    """验证演示替身提供的索引状态字段完整，不依赖真实索引文件。"""
    runtime = fake_runtime()
    apply_demo(runtime)
    status = RagService(runtime, demo=True).status()
    assert status["ready"] is True and status["demo"] is True
    assert status["chunk_count"] == 3
    assert len(status["sources"]) == 3


def test_app_rejects_requests_while_busy(tmp_path):
    """验证 HTTP 层在模型被占用时返回 409 和中文说明。"""
    fastapi = pytest.importorskip("fastapi")
    assert fastapi  # 只用于在没有安装 Web 依赖时跳过本用例
    from fastapi.testclient import TestClient

    from src.server.app import create_app

    front = tmp_path / "front"
    (front / "assets").mkdir(parents=True)
    (front / "index.html").write_text("<html></html>", encoding="utf-8")
    store = ChunkStore(tmp_path / "chunks.pkl", tmp_path / "build_manifest.json")
    service = RagService(fake_runtime(chunk_store=store))
    client = TestClient(create_app(service, front_dir=front))

    status = client.get("/api/status").json()
    assert status["ready"] is False and "还没有索引文件" in status["reason"]
    assert client.post("/api/chat", json={"question": "   "}).status_code == 422

    service._lock.acquire()
    try:
        response = client.post("/api/chat", json={"question": "问题"})
        assert response.status_code == 409
        assert "正在进行" in response.json()["detail"]
    finally:
        service._lock.release()


def test_app_streams_sse_and_serves_frontend(tmp_path):
    """验证问答返回事件流，首页与静态资源可以正常访问。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.server.app import create_app, resolve_front_dir

    front = tmp_path / "front"
    (front / "assets").mkdir(parents=True)
    (front / "index.html").write_text("<html>首页</html>", encoding="utf-8")
    (front / "assets" / "app.js").write_text("// 脚本", encoding="utf-8")
    runtime = fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())]))
    client = TestClient(create_app(RagService(runtime), front_dir=front))

    assert resolve_front_dir(front) == front.resolve()
    # 显式目录不可用时回退到项目自带的 front/，两个位置都没有才会报错。
    assert (resolve_front_dir(tmp_path / "缺失目录") / "index.html").is_file()

    assert "首页" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200

    response = client.post("/api/chat", json={"question": "储能目标"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in response.text and "event: done" in response.text


def client_app(tmp_path, memory_store=None):
    """搭建带前端目录的测试客户端，记忆存储默认用真实实现写到临时目录。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.server.app import create_app

    front = tmp_path / "front"
    (front / "assets").mkdir(parents=True, exist_ok=True)
    (front / "index.html").write_text("<html></html>", encoding="utf-8")
    runtime = fake_runtime(pipeline=asking_pipeline([SearchHit(document=doc())]))
    if memory_store is None:
        memory_store = MemoryStore(tmp_path / "memory")
    runtime.memory_store = memory_store
    return TestClient(create_app(RagService(runtime), front_dir=front)), runtime


def test_conversation_endpoints_round_trip(tmp_path):
    """验证会话接口的列表、新建、详情、删除，以及非法 id 的拒绝。"""
    client, _ = client_app(tmp_path)

    assert client.get("/api/conversations").json() == {
        "total": 0,
        "enabled": True,
        "items": [],
    }
    created = client.post("/api/conversations", json={"title": "会话标题"}).json()
    session_id = created["id"]
    assert SESSION_ID.match(session_id)
    assert client.get("/api/conversations").json()["total"] == 1

    detail = client.get(f"/api/conversations/{session_id}").json()
    assert detail["title"] == "会话标题" and detail["messages"] == []

    assert client.delete(f"/api/conversations/{session_id}").json() == {"removed": True}
    assert client.delete(f"/api/conversations/{session_id}").status_code == 404
    assert client.get(f"/api/conversations/{session_id}").status_code == 404

    for bad in ["abc", "20260101-000000-ZZZZZZ", "20260101-000000-abcdef.md"]:
        assert client.get(f"/api/conversations/{bad}").status_code == 400
        assert client.delete(f"/api/conversations/{bad}").status_code == 400


def test_chat_writes_the_exchange_into_the_session_file(tmp_path):
    """验证提问后会话 markdown 里多出这一轮，来源以位置标签的形式落盘。"""
    client, _ = client_app(tmp_path)
    session_id = client.post("/api/conversations", json={"title": "记录"}).json()["id"]
    response = client.post(
        "/api/chat",
        json={"question": "储能目标", "conversation_id": session_id},
    )
    assert f'"conversation_id": "{session_id}"' in response.text

    detail = client.get(f"/api/conversations/{session_id}").json()
    assert [(item["role"], item["content"]) for item in detail["messages"]] == [
        ("user", "储能目标"),
        ("assistant", "答案"),
    ]
    # 记忆文件只存来源的位置标签，不存片段正文。
    assert detail["messages"][1]["sources"] == [
        {"index": 1, "label": "a.pdf 第1页"},
    ]
    raw = (tmp_path / "memory" / "sessions" / f"{session_id}.md").read_text(encoding="utf-8")
    assert "## 用户" in raw and "### 参考来源" in raw


def test_chat_rejects_malformed_conversation_id(tmp_path):
    """验证提问里带的非法会话 id 会被拒绝，而不是被当成没有历史。"""
    client, _ = client_app(tmp_path)
    response = client.post(
        "/api/chat",
        json={"question": "问题", "conversation_id": "../../etc/passwd"},
    )
    assert response.status_code == 400
    assert "会话 id 不合法" in response.json()["detail"]


def test_chat_rejects_stale_clients_sending_history(tmp_path):
    """验证缓存了旧前端的浏览器带 history 字段时会明确报错，而不是静默丢掉多轮上下文。"""
    client, _ = client_app(tmp_path)
    assert client.post("/api/chat", json={"question": "问题", "history": []}).status_code == 422


def test_memory_endpoints_read_and_write(tmp_path):
    """验证长期记忆的读取与整份覆盖保存。"""
    client, _ = client_app(tmp_path)
    empty = client.get("/api/memory").json()
    assert empty["enabled"] is True and empty["content"] == "" and empty["exists"] is False

    saved = client.put("/api/memory", json={"content": "# 长期记忆\n- 关注储能"}).json()
    assert saved["content"] == "# 长期记忆\n- 关注储能" and saved["exists"] is True
    assert client.get("/api/memory").json()["content"] == saved["content"]
    assert (tmp_path / "memory" / "memory.md").read_text(encoding="utf-8") == saved["content"]


def test_memory_endpoints_report_disabled(tmp_path):
    """验证关闭记忆后接口如实报告，且拒绝写入，也不创建目录。"""
    client, _ = client_app(tmp_path, memory_store=MemoryStore(tmp_path / "memory", enabled=False))
    assert client.get("/api/memory").json()["enabled"] is False
    assert client.put("/api/memory", json={"content": "内容"}).status_code == 409
    assert client.post("/api/conversations", json={"title": "会话"}).status_code == 409
    assert not (tmp_path / "memory").exists()
