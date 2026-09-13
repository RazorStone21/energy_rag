"""分别进行向量搜索和 BM25 搜索，再合并排名；同一片段在每路结果中只计分一次。"""

from __future__ import annotations

import logging
from dataclasses import replace

from ..interfaces import LexicalIndex, VectorStore
from ..schemas import SearchHit, document_key, positive_int, valid_query

logger = logging.getLogger(__name__)


def unique_hits(hits):
    """删除重复片段，只保留第一次出现的结果及其顺序。"""
    seen = set()
    result = []
    for hit in hits:
        key = document_key(hit.document)
        if key not in seen:
            seen.add(key)
            result.append(hit)
    return result


def rrf_fuse(dense, sparse, k, rrf_k):
    """合并 dense 向量结果和 sparse BM25 结果，按 RRF 分数取前 k 条。

    每路按排名贡献 1 / (rrf_k + 排名)，相同片段累加贡献，不直接相加原始分数。
    """
    positive_int(k, "k")
    positive_int(rrf_k, "rrf_k")
    merged = {}
    for branch in (dense, sparse):
        # 分支内先去重；同一片段可从两条分支各得一次分，不能因重复返回而刷分。
        for rank, hit in enumerate(unique_hits(branch), start=1):
            key = document_key(hit.document)
            previous = merged.get(key)
            if previous is None:
                merged[key] = replace(hit, rrf_score=1.0 / (rrf_k + rank))
            else:
                # 先遍历向量结果，再遍历 BM25；再次遇到相同片段时补上 BM25 分数。
                merged[key] = replace(
                    previous,
                    bm25_score=hit.bm25_score,
                    rrf_score=previous.rrf_score + 1.0 / (rrf_k + rank),
                )
    return sorted(merged.values(), key=lambda hit: hit.rrf_score, reverse=True)[:k]


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        settings,
        readiness_check=None,
    ):
        """接收向量搜索、BM25 和检索配置，也可传入查询前检查索引的函数。"""
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.settings = settings
        self.readiness_check = readiness_check

    def retrieve(self, query, k=None, hybrid=True):
        """分别执行向量搜索和 BM25 搜索，再按配置合并结果。

        k 覆盖向量检索数量和合并后保留数量；BM25 数量仍由 bm25_top_k 决定。
        一路失败时仍可使用另一路的结果；发生错误且两路都没有结果时抛出异常。
        """
        query = valid_query(query)
        if k is None:
            dense_k = self.settings.dense_top_k
            fusion_k = self.settings.fusion_top_k
        else:
            dense_k = positive_int(k, "k")
            fusion_k = k
        if self.readiness_check:
            self.readiness_check()
        # 调用参数和配置都开启时才使用 BM25，任意一处关闭就只做向量搜索。
        use_hybrid = hybrid and self.settings.hybrid_enabled
        dense, sparse, errors = [], [], []
        try:
            dense = self.vector_store.search(query, dense_k)
        except Exception as exc:
            if not use_hybrid:
                raise
            errors.append(exc)
            logger.warning("Dense retrieval failed: %s", exc)
        if use_hybrid:
            try:
                lexical_results = self.lexical_index.search(
                    query,
                    k=self.settings.bm25_top_k,
                )
                sparse = [
                    SearchHit(document=doc, bm25_score=float(score))
                    for doc, score in lexical_results
                ]
            except Exception as exc:
                errors.append(exc)
                logger.warning("BM25 retrieval failed: %s", exc)
        if errors and not dense and not sparse:
            # 运行错误与正常的空召回必须区分，调用方才能判断是否需要修复服务。
            raise RuntimeError("Retrieval failed with no usable fallback results") from errors[0]
        if use_hybrid:
            return rrf_fuse(dense, sparse, fusion_k, self.settings.rrf_k)
        return unique_hits(dense)[:dense_k]
