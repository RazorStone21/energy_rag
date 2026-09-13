"""使用 BM25 按关键词检索片段；磁盘上的片段缓存变化后重新建立 BM25 索引。"""

from __future__ import annotations

import math
from collections import Counter

from ..schemas import DocumentLike, positive_int


def tokenize(text: str) -> list[str]:
    """用 jieba 的搜索模式拆分文本，返回去掉空白项后的词列表。"""
    import jieba

    return [
        token
        for token in jieba.lcut_for_search(text)
        if token.strip()
    ]  # fmt: skip


class BM25Index:
    """保存片段的分词统计，并根据问题计算和排序 BM25 分数。"""

    def __init__(self, docs, tokenizer=None, k1=1.5, b=0.75):
        """统计各片段的词频和长度，以及每个词出现在多少个片段中，供 BM25 评分使用。"""
        self.documents = list(docs)
        self.tokenizer = tokenizer or tokenize
        self.document_tokens = [
            self.tokenizer(document.page_content)
            for document in self.documents
        ]  # fmt: skip
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.term_frequencies = [Counter(tokens) for tokens in self.document_tokens]

        self.document_count = len(self.documents)
        self.average_document_length = (
            sum(self.document_lengths) / max(self.document_count, 1)
        )  # fmt: skip

        # 同一片段内重复出现的词，只为“包含该词的片段数”贡献一次计数。
        self.document_frequencies = Counter()
        for tokens in self.document_tokens:
            self.document_frequencies.update(set(tokens))

        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        # 保留 BM25 公式中的参数名：k1 控制词频影响，b 控制片段长度的修正程度。
        self.k1 = k1
        self.b = b

    def _idf(self, term: str) -> float:
        """计算词的 IDF 权重：包含这个词的片段越少，权重越大。"""
        matching_document_count = self.document_frequencies.get(term, 0)
        numerator = self.document_count - matching_document_count + 0.5
        denominator = matching_document_count + 0.5
        return math.log(numerator / denominator + 1)

    def _score_document(self, document_index: int, query_tokens: list[str]) -> float:
        """根据查询词在当前片段中的出现次数和片段长度，计算这个片段的 BM25 分数。"""
        score = 0.0
        term_frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        for term in query_tokens:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            # 有词命中才计算长度修正，避免所有片段都为空时除以零。
            length_normalization = (
                1 - self.b
                + self.b * document_length / self.average_document_length
            )  # fmt: skip
            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * length_normalization
            term_weight = numerator / denominator
            score += self._idf(term) * term_weight
        return score

    def search(self, query: str, k: int = 20) -> list[tuple[DocumentLike, float]]:
        """返回最多 k 个分数大于零的 (片段, 分数)，按分数从高到低排列。"""
        positive_int(k, "k")
        query_tokens = self.tokenizer(query)
        scored_documents = []

        for document_index in range(self.document_count):
            score = self._score_document(document_index, query_tokens)
            if score > 0:
                scored_documents.append((score, document_index))

        scored_documents.sort(
            key=lambda item: item[0],
            reverse=True,
        )
        top_documents = scored_documents[:k]
        return [
            (self.documents[document_index], score)
            for score, document_index in top_documents
        ]  # fmt: skip


class CachedBM25:
    """重复使用已建立的 BM25 索引，只在片段缓存文件变化后重建。"""

    def __init__(self, chunk_store, index_factory=BM25Index):
        """保存片段读取对象和 BM25 创建函数，首次查询时再建立索引。"""
        self.chunk_store = chunk_store
        self.index_factory = index_factory
        self._revision = None
        self._index = None

    def get_index(self):
        """按缓存文件状态刷新索引；文件删除或加载失败时清除旧内存索引。"""
        try:
            revision = self.chunk_store.revision()
            if revision is None:
                self._revision = None
                self._index = None
                return None

            cache_changed = revision != self._revision
            if self._index is None or cache_changed:
                chunks = self.chunk_store.load_chunks()
                self._index = self.index_factory(chunks)
                self._revision = revision
        except Exception:
            # 读取或重建失败后清除旧索引，防止后续查询继续使用过期片段。
            self._revision = None
            self._index = None
            raise
        return self._index

    def search(self, query: str, k: int) -> list[tuple[DocumentLike, float]]:
        """使用最新的 BM25 索引查询；片段缓存不存在时返回空列表。"""
        index = self.get_index()
        if index is None:
            return []
        return index.search(query, k=k)
