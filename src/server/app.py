"""提供问答、入库和静态文件接口的 FastAPI 应用；只有本模块依赖 Web 框架。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .service import RagService

BUSY_MESSAGE = "已有问答或索引构建正在进行，请稍后再试"


class ChatRequest(BaseModel):
    """一次问答请求：问题、两个检索开关，以及要记入哪一份会话记录。

    历史由服务端从会话记录里读取，因此这里不再接受 history。
    多余字段直接报错：忽略它们会让缓存了旧前端的浏览器静默地丢掉多轮上下文。
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    hybrid: bool = True
    rerank: bool = True
    conversation_id: str | None = None


class ConversationRequest(BaseModel):
    """新建会话记录时给出的标题，留空表示用默认标题。"""

    title: str = ""


class MemoryRequest(BaseModel):
    """长期记忆的全文，保存时整份覆盖。"""

    content: str = ""


class BuildRequest(BaseModel):
    """入库请求，只允许选择是否增量，不接受会替换整个索引的参数。"""

    incremental: bool = True


def resolve_front_dir(explicit: str | Path | None = None) -> Path:
    """定位前端目录：优先使用显式路径，否则按源码位置推断，找不到时给出中文提示。"""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    # 本文件位于 src/server/ 下，上两级就是项目根目录，front 与 src 同级。
    candidates.append(Path(__file__).resolve().parents[2] / "front")
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate.resolve()
    tried = "、".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"没有找到前端页面，请检查 {tried} 或使用 --front-dir 指定目录")


def _sse_response(events) -> StreamingResponse:
    """把事件生成器包装成 SSE 响应，禁用缓存并让反向代理不要缓冲。"""
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def create_app(service: RagService, front_dir: str | Path | None = None) -> FastAPI:
    """创建 FastAPI 应用并挂载前端目录；创建过程不加载模型，也不连接向量库。"""
    directory = resolve_front_dir(front_dir)
    app = FastAPI(
        title="能源文档 RAG 问答系统",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    @app.get("/api/status")
    async def status():
        """返回索引状态与运行标志；实现里只读文件，不触发模型加载。"""
        return service.status()

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """接收问题并返回 SSE 事件流；已有任务占用模型时直接拒绝。"""
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能为空")
        if service.busy:
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        if request.conversation_id is not None:
            try:
                service.check_conversation_id(request.conversation_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        events = service.chat_events(
            question,
            hybrid=request.hybrid,
            rerank=request.rerank,
            conversation_id=request.conversation_id,
        )
        return _sse_response(events)

    @app.post("/api/build")
    async def build(request: BuildRequest):
        """触发一次入库并返回事件流；入库期间问答同样会被拒绝。"""
        if service.busy:
            raise HTTPException(status_code=409, detail=BUSY_MESSAGE)
        return _sse_response(service.build_events(incremental=request.incremental))

    @app.get("/api/conversations")
    async def conversations(limit: int | None = None):
        """列出会话记录，按最近修改从新到旧；limit 只影响返回条数，总数照常给出。"""
        return service.list_conversations(limit=limit)

    @app.post("/api/conversations")
    async def create_conversation(request: ConversationRequest):
        """新建一份会话记录；只有第一条提问时才会真正落盘。"""
        try:
            return service.create_conversation(request.title.strip()[:60])
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/conversations/{session_id}")
    async def conversation_detail(session_id: str):
        """读取一份会话的完整消息；id 不合法返回 400，不存在返回 404。"""
        try:
            session = service.load_conversation(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if session is None:
            raise HTTPException(status_code=404, detail="没有找到这份会话记录")
        return session

    @app.delete("/api/conversations/{session_id}")
    async def delete_conversation(session_id: str):
        """删除一份会话记录；正在生成回答的那一次会被拒绝。"""
        try:
            removed = service.delete_conversation(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail="没有找到这份会话记录")
        return {"removed": True}

    @app.get("/api/memory")
    async def read_memory():
        """返回长期记忆的开关、注入上限与全文。"""
        return service.memory_status()

    @app.put("/api/memory")
    async def update_memory(request: MemoryRequest):
        """整份覆盖长期记忆；功能关闭时拒绝写入。"""
        try:
            return service.write_memory(request.content)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"保存长期记忆失败：{exc}") from exc

    @app.get("/")
    async def index():
        """返回前端首页。"""
        return FileResponse(directory / "index.html")

    app.mount("/static", StaticFiles(directory=directory / "assets"), name="static")
    return app
