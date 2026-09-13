"""列出各组件需要提供的方法及输入输出，具体实现只需符合这些方法要求。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .schemas import DocumentLike, ParseResult, SearchHit


class Parser(Protocol):
    def parse(self, path: Path, on_status=None) -> ParseResult:
        """解析一个文件，返回正文、表格、图片描述及阶段错误。

        on_status 是可选函数，接收一句描述当前动作的文字（如「正在描述第 3 张图表」），
        供调用方显示进度。逐张图表上报还是只报一次由实现决定；
        不需要进度的实现可以忽略这个参数。
        """
        ...


class Chunker(Protocol):
    def split(self, parsed: ParseResult) -> list[DocumentLike]:
        """把解析结果整理成可入库的片段，保留内容结构、来源和页码等信息。"""
        ...


class VectorStore(Protocol):
    def search(self, query: str, k: int) -> list[SearchHit]:
        """根据问题查找最多 k 个相似片段，并返回对应的向量检索分数。"""
        ...

    def add(self, chunks: list[DocumentLike]) -> None:
        """将非空片段集合写入向量索引。"""
        ...

    def delete_sources(self, names: list[str]) -> None:
        """按来源文件删除旧片段，不决定哪些来源需要更新。"""
        ...

    def replace_all(self, chunks: list[DocumentLike]) -> None:
        """使用完整的新片段集合替换现有向量索引。"""
        ...


class LexicalIndex(Protocol):
    def search(self, query: str, k: int) -> list[tuple[DocumentLike, float]]:
        """按关键词匹配程度返回片段及其 BM25 分数。"""
        ...


class QueryRewriter(Protocol):
    def rewrite(self, query: str, history=None) -> str:
        """结合历史对话，把依赖上下文的追问改写成可以独立检索的问题。

        没有历史、判断不需要改写或改写失败时，返回的问题应与传入的完全相同。
        """
        ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int | None = None, hybrid: bool = True) -> list[SearchHit]:
        """按配置查找相关片段，返回后续可以重排的结果列表。"""
        ...


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """计算问题与片段的匹配分数，验证输出后返回排序结果。

        原候选保持不变，新结果同时保留向量、BM25 和 RRF 分数。
        """
        ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str:
        """接收提示词，返回生成的答案字符串。"""
        ...

    def generate_stream(self, prompt: str):
        """接收提示词，按生成顺序逐块产出新增的答案文本。"""
        ...
