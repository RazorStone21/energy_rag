"""把问答与入库结果编码成 SSE 报文；本模块只依赖标准库，便于脱离 Web 框架测试。"""

from __future__ import annotations

import json
import math

from ..context_builder import ELEMENT_TYPE_LABELS, source_label
from ..schemas import SearchHit

KEEPALIVE_FRAME = ": keepalive\n\n"
TIMING_STAGES = ("rewrite", "retrieval", "rerank", "context", "generation")
SCORE_NAMES = {
    "dense_score": "dense",
    "bm25_score": "bm25",
    "rrf_score": "rrf",
    "rerank_score": "rerank",
}


def encode_event(name: str, payload) -> str:
    """把事件名和数据编码成一条 SSE 报文。

    json.dumps 不转义换行，而片段正文和模型输出几乎必然含换行，
    直接写进 data 字段会把一条事件拆成两条非法事件，因此逐行加前缀。
    """
    text = json.dumps(payload, ensure_ascii=False).replace("\r", "\\r")
    lines = "".join(f"data: {line}\n" for line in text.split("\n"))
    return f"event: {name}\n{lines}\n"


def _json_safe(value):
    """把元数据值收敛成可以 JSON 序列化的形式，无法识别的类型转为字符串。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        # JSON 没有 NaN 和无穷的表示，异常分数按「该阶段未产生分数」处理。
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def source_payload(index: int, hit: SearchHit) -> dict:
    """把一条检索片段整理成来源面板需要的字段，序号与提示词里的片段编号一致。"""
    document = hit.document
    metadata = dict(document.metadata)
    scores = {}
    for field, name in SCORE_NAMES.items():
        value = getattr(hit, field, None)
        scores[name] = _json_safe(value)
    return {
        "index": index,
        "label": source_label(metadata),
        "source": _json_safe(metadata.get("source", "?")),
        "type": _json_safe(metadata.get("type", "text")),
        "type_label": ELEMENT_TYPE_LABELS.get(metadata.get("type"), "正文"),
        "metadata": {str(key): _json_safe(value) for key, value in metadata.items()},
        "content": str(document.page_content),
        "scores": scores,
    }


def sources_payload(evidence) -> list[dict]:
    """按提示词中的顺序给全部片段编号，供界面显示参考来源。"""
    return [source_payload(index, hit) for index, hit in enumerate(evidence, start=1)]


def timings_payload(timings) -> dict:
    """补齐缺失的阶段耗时，避免前端把没有该阶段的 null 显示成 0 或 undefined。"""
    timings = timings or {}
    return {stage: timings.get(stage) for stage in TIMING_STAGES}


def done_payload(result, demo: bool = False, conversation_id=None) -> dict:
    """整理一次问答的最终结果；答案取自管线返回值，与流式收到的文本一致。

    会话 id 一并回传，前端据此把这次问答归到对应的会话记录上。
    """
    return {
        "answer": result.answer,
        "sources": sources_payload(result.evidence),
        "timings": timings_payload(result.timings),
        "demo": demo,
        "conversation_id": conversation_id,
    }


def error_payload(message: str) -> dict:
    """把失败原因整理成给用户看的中文说明。"""
    return {"message": str(message)}
