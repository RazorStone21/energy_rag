"""定义解析、检索和问答结果的字段；保存片段时仍可使用 LangChain Document。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .interfaces import VectorStore


class DocumentLike(Protocol):
    """说明片段必须有 page_content 和 metadata 两个属性，LangChain Document 可直接使用。"""

    @property
    def page_content(self) -> str:
        """返回片段文字，可以是正文、Markdown 表格或图片描述。"""
        ...

    @property
    def metadata(self) -> dict:
        """返回来源文件、页码、内容类型和图片位置等信息。"""
        ...


@dataclass(frozen=True)
class SearchHit:
    """一条检索片段及各阶段分数；None 表示没有该阶段分数，不等于零分。"""

    document: DocumentLike
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


@dataclass
class ContextBundle:
    """同时保存发给模型的提示词和其中使用的片段，方便查看回答依据。"""

    prompt: str
    evidence: list[SearchHit]


@dataclass
class AnswerResult:
    """保存答案、使用的片段、提示词和耗时；是否调用模型由问答流程决定。"""

    answer: str
    evidence: list[SearchHit]
    prompt: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    # 实际用于检索和重排的问题；没有改写时与用户原话相同，提示词里的【问题】始终是原话。
    search_query: str = ""


@dataclass
class ParseResult:
    """区分正文、表格、图片描述及解析错误，避免把失败误认为空文档成功入库。"""

    texts: list[DocumentLike] = field(default_factory=list)
    tables: list[DocumentLike] = field(default_factory=list)
    figures: list[DocumentLike] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BuildResult:
    """保存入库结果：processed 是成功文件，removed 是清理的文件，failed 记录失败原因。"""

    vector_store: VectorStore
    chunks: list[DocumentLike]
    processed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def document_key(document) -> tuple:
    """用来源、位置、类型和正文识别重复片段，不合并出现在不同章节的相同文字。"""
    meta = document.metadata
    return (
        str(meta.get("source", "")),
        str(meta.get("page", "")),
        str(meta.get("type", "text")),
        str(meta.get("bbox", "")),
        str(meta.get("heading_path", "")),
        str(meta.get("line_start", "")),
        str(meta.get("line_end", "")),
        str(meta.get("block_index", "")),
        str(meta.get("sheet_name", "")),
        str(meta.get("row_start", "")),
        str(meta.get("row_end", "")),
        str(meta.get("column_start", "")),
        str(meta.get("column_end", "")),
        document.page_content,
    )


def positive_int(value, name: str) -> int:
    """校验正整数参数并返回原值；布尔值虽然属于整数子类，也必须拒绝。"""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def valid_query(query: str) -> str:
    """校验问题为非空字符串，并去除首尾空白。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    return query.strip()
