"""演示模式的内存替身：没有模型依赖时也能预览界面，返回内容全部是编造的。"""

from __future__ import annotations

import time
from pathlib import Path

from ..context_builder import ContextBuilder
from ..schemas import AnswerResult, BuildResult, SearchHit
from ..storage.memory import SESSION_ID, Session, SessionSummary, Turn, new_session_id

# 演示片段使用真实的元数据字段，让来源标签、序号和位置展示走与真实流程相同的代码路径。
DEMO_DOCUMENTS = (
    {
        "content": (
            "到 2030 年，全国新增新型储能装机容量要达到 1.2 亿千瓦以上，"
            "其中独立储能和共享储能占比不低于 40%。\n"
            "各地应完善容量电价机制，保障储能设施的合理收益。"
        ),
        "metadata": {
            "source": "13-china-new-energy-storage-development-report-2026-cn.pdf",
            "page": 27,
            "type": "text",
        },
    },
    {
        "content": (
            "| 省份 | 2025 年装机（万千瓦） | 2030 年目标（万千瓦） | 年均增速 |\n"
            "| --- | --- | --- | --- |\n"
            "| 山东 | 820 | 2400 | 23.9% |\n"
            "| 江苏 | 610 | 1800 | 24.2% |\n"
            "| 广东 | 480 | 1600 | 27.2% |"
        ),
        "metadata": {
            "source": "15-china-new-energy-system-15th-five-year-plan-2026-cn.xlsx",
            "type": "table",
            "sheet_name": "分省目标",
            "sheet_state": "visible",
            "row_start": 2,
            "row_end": 5,
            "column_start": "A",
            "column_end": "D",
            "header_row_start": 2,
            "header_row_end": 2,
        },
    },
    {
        "content": (
            "绿电交易应与绿证核发衔接，避免环境权益重复计算；"
            "跨省区交易的新能源电量在受端省份计入消纳责任权重。"
        ),
        "metadata": {
            "source": "06-china-green-electricity-certificate-development-report-2024-cn.pdf",
            "page": 41,
            "type": "text",
        },
    },
)

DEMO_ANSWER = (
    "根据提供的文档片段，可以整理出以下几点：\n\n"
    "1. **装机目标**：到 2030 年，全国新增新型储能装机容量要达到 1.2 亿千瓦以上，"
    "其中独立储能和共享储能占比不低于 40%（来源：新能源储能发展报告，第 27 页）。\n"
    "2. **分省安排**：山东、江苏、广东三省 2030 年目标分别为 2400、1800、1600 万千瓦，"
    "年均增速均在 24% 左右（来源：新型能源体系十五五规划，工作表「分省目标」）。\n"
    "3. **配套机制**：各地应完善容量电价机制保障储能合理收益；"
    "绿电交易要与绿证核发衔接，避免环境权益重复计算。\n\n"
    "需要注意的是，涉及具体省份的项目安排还应以主管部门发布的正式文件为准。"
)

DEMO_BUILD = {
    "processed": ["01-china-accelerating-new-power-system-construction-action-plan-2024-cn.pdf"],
    "removed": [],
    "failed": {},
}


class DemoDocument:
    """演示片段，只包含正文和元数据，与 DocumentLike 的要求一致。"""

    def __init__(self, content: str, metadata: dict):
        """保存正文与元数据，供提示词和来源面板共用。"""
        self.page_content = content
        self.metadata = dict(metadata)


def demo_hits(with_rerank: bool = True, hybrid: bool = True) -> list[SearchHit]:
    """构造演示用的检索结果，分数随检索开关变化，让界面上的开关有可见效果。"""
    hits = []
    for position, item in enumerate(DEMO_DOCUMENTS):
        dense = 0.72 - position * 0.08
        bm25 = 8.4 - position * 2.1
        rrf = 1 / 61 + (1 / 62 if position == 0 else 0)
        hits.append(
            SearchHit(
                document=DemoDocument(item["content"], item["metadata"]),
                # 只用向量召回时没有 BM25 和融合分数，跳过重排时没有重排分数，
                # 与真实流程中哪些阶段会产生分数保持一致。
                dense_score=dense,
                bm25_score=bm25 if hybrid else None,
                rrf_score=rrf if hybrid else None,
                rerank_score=0.93 - position * 0.15 if with_rerank else None,
            )
        )
    return hits


class DemoPipeline:
    """演示用问答流程：返回编造的片段和答案，不加载任何模型。"""

    def __init__(self, context_builder: ContextBuilder):
        """保存提示词组装器，让演示提示词与真实流程的格式一致。"""
        self.context_builder = context_builder

    def ask(
        self,
        query,
        with_rerank=True,
        hybrid=True,
        on_prompt=None,
        on_token=None,
        on_context=None,
        history=None,
        memory=None,
    ):
        """按真实流程的回调顺序产出来源、提示词和答案文本块。"""
        hits = demo_hits(with_rerank=with_rerank, hybrid=hybrid)
        context = self.context_builder.build(query, hits, history=history, memory=memory)
        if on_context:
            on_context(context)
        if on_prompt:
            on_prompt(context.prompt)
        pieces = [DEMO_ANSWER[index : index + 6] for index in range(0, len(DEMO_ANSWER), 6)]
        if on_token is not None:
            # 逐块返回并加入间隔，让前端的流式渲染效果接近真实生成。
            for piece in pieces:
                time.sleep(0.03)
                on_token(piece)
        return AnswerResult(
            answer=DEMO_ANSWER,
            evidence=context.evidence,
            prompt=context.prompt,
            timings={
                "retrieval": 0.18,
                "rerank": 0.42 if with_rerank else 0.0,
                "context": 0.002,
                "generation": 4.6,
            },
        )


class DemoIngestion:
    """演示用入库流程：不读写任何文件，只返回一个成功结果。"""

    def build(self, doc_dir=None, max_files=None, save=True, incremental=False, only=None):
        """返回固定的入库结果，字段与真实 BuildResult 一致。"""
        return BuildResult(
            vector_store=None,
            chunks=[],
            processed=list(DEMO_BUILD["processed"]),
            removed=list(DEMO_BUILD["removed"]),
            failed=dict(DEMO_BUILD["failed"]),
        )


class DemoChunkStore:
    """演示用片段存储：提供状态接口需要的版本号、清单和片段列表。"""

    def revision(self):
        """返回固定的版本号元组，让状态接口认为索引已经就绪。"""
        return ("demo/chunks.pkl", 1_760_000_000_000_000_000, 0, 1024, 1)

    def load_manifest(self):
        """返回编造的文件清单，只用于界面展示。"""
        return {item["metadata"]["source"]: "demo-sha256" for item in DEMO_DOCUMENTS}

    def load_chunks(self):
        """返回演示片段，供状态接口统计每个来源的片段数。"""
        return [DemoDocument(item["content"], item["metadata"]) for item in DEMO_DOCUMENTS]


DEMO_MEMORY = "# 长期记忆\n\n## 用户偏好\n- 关注新型储能与电力市场\n- 回答尽量给出文件名和页码\n"

DEMO_SESSION_TITLE = "演示会话：新型储能装机目标"


class DemoMemoryStore:
    """演示用的记忆存储：数据只放在内存里，不读写任何文件。"""

    def __init__(self, seed: bool = True):
        """准备内存中的记忆与会话表；seed 为假时不放示例会话，供测试从空开始。"""
        self.enabled = True
        self.memory_path = Path("演示模式/不写入磁盘/memory.md")
        self._memory = DEMO_MEMORY
        self._sessions: dict[str, Session] = {}
        if seed:
            self._seed_session()

    def _seed_session(self):
        """放一条已经问过的会话，用于展示历史记录和参考来源的样子。"""
        started = int(time.time() * 1000) - 3_600_000
        session_id = new_session_id()
        session = Session(
            id=session_id,
            title=DEMO_SESSION_TITLE,
            created_at_ms=started,
            updated_at_ms=started,
            turns=[
                Turn("user", "新型储能的装机目标是多少？"),
                Turn("assistant", DEMO_ANSWER, ["13-新能源储能发展报告.pdf 第27页（正文）"]),
            ],
        )
        self._sessions[session_id] = session

    @property
    def sessions_dir(self) -> Path:
        """演示模式没有真的目录，返回记忆文件所在的位置。"""
        return self.memory_path.parent

    def read_memory(self) -> str:
        """返回演示用的长期记忆全文。"""
        return self._memory

    def write_memory(self, text: str) -> None:
        """把长期记忆保存在内存里，不写磁盘。"""
        self._memory = str(text)

    def memory_exists(self) -> bool:
        """演示记忆始终视为已存在。"""
        return True

    def session_path(self, session_id) -> Path:
        """按与真实存储相同的规则校验会话 id，但不产生真实路径。"""
        if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
            raise ValueError("会话 id 不合法")
        return self.memory_path.parent / f"{session_id}.md"

    def create_session(self, title: str) -> Session:
        """在内存里新建一份会话记录。"""
        started = int(time.time() * 1000)
        session = Session(
            id=new_session_id(),
            title=title or "新对话",
            created_at_ms=started,
            updated_at_ms=started,
        )
        self._sessions[session.id] = session
        return session

    def list_sessions(self, limit: int | None = None) -> list[SessionSummary]:
        """按修改时间从新到旧列出演示会话。"""
        summaries = [
            SessionSummary(
                id=session.id,
                title=session.title,
                created_at_ms=session.created_at_ms,
                updated_at_ms=session.updated_at_ms,
                message_count=len(session.turns),
                parse_error=False,
            )
            for session in self._sessions.values()
        ]
        summaries.sort(key=lambda item: item.updated_at_ms, reverse=True)
        return summaries[:limit] if limit else summaries

    def load_session(self, session_id: str) -> Session | None:
        """按 id 取出演示会话；id 不合法时抛出 ValueError。"""
        self.session_path(session_id)
        return self._sessions.get(session_id)

    def append_exchange(self, session_id, question, answer="", sources=()):
        """把一轮问答追加到内存里的会话记录。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.turns.append(Turn("user", str(question)))
        if str(answer).strip():
            session.turns.append(Turn("assistant", str(answer), [str(label) for label in sources]))
        session.updated_at_ms = int(time.time() * 1000)
        return session

    def save_session(self, session: Session) -> None:
        """把会话写回内存表，签名与真实存储保持一致。"""
        self._sessions[session.id] = session

    def delete_session(self, session_id: str) -> bool:
        """从内存表里删掉一份会话。"""
        self.session_path(session_id)
        return self._sessions.pop(session_id, None) is not None


def apply_demo(runtime) -> None:
    """把 Runtime 的问答、入库、索引状态和记忆存储换成内存替身，不改动其余组件。"""
    runtime.pipeline = DemoPipeline(
        ContextBuilder(
            runtime.settings.prompt_template,
            runtime.settings.conversation,
            runtime.settings.memory,
        )
    )
    runtime.ingestion = DemoIngestion()
    runtime.chunk_store = DemoChunkStore()
    runtime.memory_store = DemoMemoryStore()
