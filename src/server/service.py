"""把 Runtime 的问答与入库包装成 Web 服务使用的事件流；本模块不依赖 Web 框架。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import Counter

from ..bootstrap import Runtime
from .events import KEEPALIVE_FRAME, done_payload, encode_event, error_payload, sources_payload

# 生成期间可能长时间没有输出（首次提问还要加载模型），用注释帧保持连接不被中断。
KEEPALIVE_SECONDS = 15


class RagService:
    """持有 Runtime 与一把串行锁，把问答和入库转成前端可以逐条接收的事件。"""

    def __init__(self, runtime: Runtime, demo: bool = False):
        """保存运行环境；创建过程不加载模型，也不连接向量库。"""
        self.runtime = runtime
        self.demo = demo
        # 项目不支持并发构建或在线上切换索引，因此问答与入库共用一把锁串行执行。
        self._lock = threading.Lock()
        self._building = False
        self._warm = False
        self._last_build = None
        self._counts = {}
        self._count_revision = None
        # 正在生成的那一次会话，删除时要拒绝，否则在途的追加会把文件写回来。
        self._active_conversation = None

    @property
    def busy(self):
        """报告当前是否有问答或入库占用着模型。"""
        return self._lock.locked()

    def status(self) -> dict:
        """汇总索引状态、来源片段数和运行标志；不加载模型，也不连接向量库。

        只读取片段缓存文件的版本信息和构建清单，清单是普通 JSON，不需要反序列化全部片段。
        """
        payload = {
            "ready": False,
            "reason": None,
            "busy": self.busy,
            "building": self._building,
            "warm": self._warm,
            "demo": self.demo,
            "last_build": self._last_build,
            "models": dict(self.runtime.settings.model_ids),
            "collection": self.runtime.settings.milvus.collection,
            "doc_dir": str(self.runtime.settings.doc_dir),
            "sources": [],
            "chunk_count": 0,
            "built_at_ms": None,
        }
        try:
            # revision 内部已经检查过写入中断标记，这里不再重复调用 assert_ready。
            revision = self.runtime.chunk_store.revision()
            manifest = self.runtime.chunk_store.load_manifest()
            if revision is None:
                payload["reason"] = "还没有索引文件，请先在文档目录执行一次入库。"
                return payload
            payload["built_at_ms"] = revision[1] // 1_000_000
        except Exception as exc:
            payload["reason"] = str(exc)
            return payload
        payload["ready"] = True
        try:
            counts = self._source_counts(revision)
        except Exception as exc:
            # 片段统计失败不影响索引可用性，返回来源列表但把原因告诉界面。
            counts = {}
            payload["reason"] = f"读取片段统计失败：{exc}"
        payload["sources"] = [
            {"name": name, "chunks": counts.get(name, 0)} for name in sorted(manifest)
        ]
        payload["chunk_count"] = sum(counts.values())
        return payload

    def _history_turns(self, conversation_id):
        """读取会话记录里的全部消息，交给提示词组件按配置截取最近若干轮。

        这里不再截断：轮数与字数上限由 context_builder 一处决定，避免两个地方各截一次。
        """
        if not conversation_id:
            return []
        try:
            session = self.runtime.memory_store.load_session(conversation_id)
        except ValueError:
            # id 形状不合法时按没有历史处理，具体错误由会话接口负责返回。
            return []
        if session is None:
            return []
        return [(turn.role, turn.content) for turn in session.turns]

    def _record_exchange(self, conversation_id, question, result):
        """把这一轮问答写进会话记录；写失败只记日志，不影响已经生成的回答。"""
        if not conversation_id:
            return
        try:
            labels = [item["label"] for item in sources_payload(result.evidence)]
            self.runtime.memory_store.append_exchange(
                conversation_id,
                question,
                result.answer,
                labels,
            )
        except Exception as exc:
            # 落盘失败不能变成 error 事件：答案已经流式发给用户了，
            # 这时把 done 换成 error 只会让界面把内容清掉。
            logging.warning("保存会话记录失败：%s", exc)

    def _source_counts(self, revision) -> dict:
        """按片段缓存的版本号缓存每个来源的片段数，避免每次请求都反序列化全部片段。"""
        if revision != self._count_revision:
            chunks = self.runtime.chunk_store.load_chunks()
            self._counts = dict(Counter(chunk.metadata.get("source", "?") for chunk in chunks))
            self._count_revision = revision
        return self._counts

    def warmup(self) -> None:
        """在后台线程里预先加载模型，减少第一次提问前的等待；失败只记录不抛出。

        演示模式的问答不经过模型，预热反而会去加载真实权重，因此直接跳过。
        """
        if self.demo:
            return
        components = (
            getattr(self.runtime, "embedder", None),
            getattr(self.runtime, "reranker", None),
            getattr(self.runtime, "generator", None),
        )

        def run():
            """依次加载各组件，忽略没有加载方法的替身对象。"""
            for component in components:
                load = getattr(component, "load", None)
                if load is None:
                    continue
                try:
                    load()
                except Exception:
                    # 预热只是优化，失败不影响后续按需加载。
                    pass

        threading.Thread(target=run, daemon=True).start()

    def chat_events(self, question, hybrid=True, rerank=True, conversation_id=None):
        """把一次问答转成 SSE 事件序列：排队、来源、提示词、逐词元、结束或错误。

        生成过程放在独立线程里，回调只能把结果放进队列，因此模型所在的生成器始终留在
        那个线程；客户端断开时不会在事件循环线程上做收尾，也就不会卡住整个服务。
        本方法只产出事件，不抛出异常，错误以 error 事件结束。
        """
        events: queue.Queue = queue.Queue()
        finished = object()
        delivered = False

        def on_context(bundle):
            """在生成开始前送出参考来源和提示词，让界面先显示回答依据。"""
            nonlocal delivered
            delivered = True
            events.put(("sources", sources_payload(bundle.evidence)))
            events.put(("prompt", {"prompt": bundle.prompt}))

        def produce():
            """在后台线程完成问答，无论成功失败都放入结束标记。"""
            try:
                if not self._lock.acquire(blocking=False):
                    # 抢不到锁说明另有请求在用模型，先告诉前端正在排队再等待。
                    events.put(("queued", {"busy": True}))
                    self._lock.acquire()
                try:
                    self._active_conversation = conversation_id
                    # 记忆与历史都在锁内读取，避免和并发的追加读到写了一半的状态。
                    memory_text = self.runtime.memory_store.read_memory()
                    result = self.runtime.pipeline.ask(
                        question,
                        with_rerank=rerank,
                        hybrid=hybrid,
                        history=self._history_turns(conversation_id),
                        memory=memory_text,
                        on_context=on_context,
                        on_token=lambda piece: events.put(("token", {"text": piece})),
                    )
                    self._record_exchange(conversation_id, question, result)
                finally:
                    self._active_conversation = None
                    self._lock.release()
                self._warm = True
                # 没有命中片段时管线不会调用 on_context，这里补一个空来源事件，
                # 让前端收到的事件顺序保持一致。
                if not delivered:
                    events.put(("sources", []))
                events.put(("done", done_payload(result, self.demo, conversation_id)))
            except Exception as exc:
                events.put(("error", error_payload(exc)))
            finally:
                events.put(finished)

        yield encode_event("queued", {"busy": self.busy})
        try:
            thread = threading.Thread(target=produce, daemon=True)
            thread.start()
        except Exception as exc:
            yield encode_event("error", error_payload(f"无法启动生成线程：{exc}"))
            return
        yield from self._drain(events, finished)

    def build_events(self, incremental: bool = True):
        """执行一次入库并推送开始与结果；入库没有进度回调，只能给出首尾状态。"""
        events: queue.Queue = queue.Queue()
        finished = object()

        def produce():
            """在后台线程完成入库，结束后记录结果供断线的客户端查询。"""
            try:
                if not self._lock.acquire(blocking=False):
                    events.put(("queued", {"busy": True}))
                    self._lock.acquire()
                try:
                    self._building = True
                    # 入库主要用嵌入模型，先释放生成和视觉模型腾出显存；
                    # 之后的第一次提问需要重新加载这些模型。
                    self.runtime.release_models()
                    result = self.runtime.ingestion.build(incremental=incremental)
                finally:
                    self._building = False
                    self._lock.release()
                self._last_build = {
                    "incremental": incremental,
                    "processed": list(result.processed),
                    "removed": list(result.removed),
                    "failed": dict(result.failed),
                    "finished_at_ms": int(time.time() * 1000),
                }
                events.put(("done", self._last_build))
            except Exception as exc:
                self._last_build = {
                    "incremental": incremental,
                    "processed": [],
                    "removed": [],
                    "failed": {"__error__": str(exc)},
                    "finished_at_ms": int(time.time() * 1000),
                }
                events.put(("error", error_payload(exc)))
            finally:
                events.put(finished)

        yield encode_event("queued", {"busy": self.busy})
        try:
            thread = threading.Thread(target=produce, daemon=True)
            thread.start()
        except Exception as exc:
            yield encode_event("error", error_payload(f"无法启动入库线程：{exc}"))
            return
        yield from self._drain(events, finished)

    def _drain(self, events: queue.Queue, finished):
        """按顺序转发后台线程放入的事件，空闲时发送注释帧保活。

        调用方提前关闭生成器时（客户端断开）本循环结束，后台线程仍会把这一轮跑完，
        期间锁不释放，因此界面上的「停止」只能停止接收，不能中断模型推理。
        """
        while True:
            try:
                item = events.get(timeout=KEEPALIVE_SECONDS)
            except queue.Empty:
                yield KEEPALIVE_FRAME
                continue
            if item is finished:
                return
            name, payload = item
            try:
                frame = encode_event(name, payload)
            except (TypeError, ValueError):
                # 单条事件无法编码时跳过它，不因为一条坏事件中断整轮问答。
                continue
            yield frame

    def check_conversation_id(self, session_id) -> None:
        """校验会话 id 的形状，不合法时抛出 ValueError，供路由转成 400。"""
        self.runtime.memory_store.session_path(session_id)

    def memory_status(self) -> dict:
        """返回长期记忆的开关、注入上限与全文，供界面显示和编辑。"""
        store = self.runtime.memory_store
        settings = self.runtime.settings.memory
        return {
            "enabled": bool(store.enabled),
            "max_chars": settings.max_chars,
            "content": store.read_memory(),
            "path": str(store.memory_path),
            "exists": store.memory_exists(),
        }

    def write_memory(self, content: str) -> dict:
        """覆盖保存长期记忆并返回保存后的状态；功能关闭时拒绝写入。"""
        if not self.runtime.memory_store.enabled:
            raise RuntimeError("记忆功能已在配置中关闭")
        self.runtime.memory_store.write_memory(content)
        return self.memory_status()

    def create_conversation(self, title: str = "") -> dict:
        """新建一份会话记录，返回它的摘要供界面立即选中。"""
        if not self.runtime.memory_store.enabled:
            raise RuntimeError("记忆功能已在配置中关闭")
        session = self.runtime.memory_store.create_session(title)
        return {
            "id": session.id,
            "title": session.title,
            "created_at_ms": session.created_at_ms,
            "updated_at_ms": session.updated_at_ms,
            "message_count": 0,
            "parse_error": False,
        }

    def list_conversations(self, limit: int | None = None) -> dict:
        """按修改时间从新到旧列出会话摘要，总数与本次返回条数分开给出。"""
        summaries = self.runtime.memory_store.list_sessions()
        selected = summaries[:limit] if limit else summaries
        return {
            "total": len(summaries),
            "enabled": bool(self.runtime.memory_store.enabled),
            "items": [
                {
                    "id": item.id,
                    "title": item.title,
                    "created_at_ms": item.created_at_ms,
                    "updated_at_ms": item.updated_at_ms,
                    "message_count": item.message_count,
                    "parse_error": item.parse_error,
                }
                for item in selected
            ],
        }

    def load_conversation(self, session_id: str) -> dict | None:
        """读取一份会话的完整消息；id 不合法时抛 ValueError，不存在时返回 None。

        记忆文件只存参考来源的位置标签，不存片段正文，所以这里的来源没有正文和分数。
        """
        session = self.runtime.memory_store.load_session(session_id)
        if session is None:
            return None
        return {
            "id": session.id,
            "title": session.title,
            "created_at_ms": session.created_at_ms,
            "updated_at_ms": session.updated_at_ms,
            "messages": [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "sources": [
                        {"index": index, "label": label}
                        for index, label in enumerate(turn.sources, start=1)
                    ],
                }
                for turn in session.turns
            ],
        }

    def delete_conversation(self, session_id: str) -> bool:
        """删除一份会话；正在生成回答的那一次会拒绝，避免在途追加把文件写回来。"""
        if session_id == self._active_conversation:
            raise RuntimeError("这次会话正在生成回答，请等它结束后再删除")
        return self.runtime.memory_store.delete_session(session_id)
