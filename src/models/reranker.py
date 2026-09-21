"""重排模型适配器；排序时保留原有召回分数与文档元数据。"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from numbers import Real

from ..schemas import positive_int, valid_query


class Reranker:
    def __init__(self, model_path, factory=None, max_length=None):
        """保存重排模型路径与最大词元数；可传入自定义加载函数，此时还不加载权重。"""
        self.model_path = model_path
        self._factory = factory
        # FlagEmbedding 默认把候选截到 512 个词元，中文大约 750 字，超出的正文不参与打分；
        # 长片段只按开头排序会明显偏离实际相关性，因此默认放宽到模型支持的 8192。
        self.max_length = max_length
        self._model = None
        # 预热线程和首个请求可能几乎同时进来，加载要串行，否则同一份权重会被加载两次。
        self._lock = threading.Lock()

    def load(self):
        """首次重排时加载模型，后续调用复用同一个实例。

        加载全程持锁：并发的第二个调用会等第一个加载完再复用同一个实例。
        """
        with self._lock:
            if self._model is None:
                factory = self._factory
                if factory is None:
                    from FlagEmbedding import FlagReranker

                    factory = FlagReranker
                self._model = factory(str(self.model_path), use_fp16=True)
            return self._model

    def rerank(self, query, hits, top_k):
        """计算问题与片段的匹配分数，验证输出后返回排序结果。

        原候选保持不变，新结果同时保留向量、BM25 和 RRF 分数。
        """
        query = valid_query(query)
        positive_int(top_k, "top_k")
        if not hits:
            return []

        pairs = [
            [query, hit.document.page_content]
            for hit in hits
        ]  # fmt: skip

        model = self.load()
        raw_scores = model.compute_score(
            pairs,
            normalize=True,
            max_length=self.max_length,
        )

        # 只有一个候选时模型可能返回单个数字，这里统一转成分数列表。
        if isinstance(raw_scores, Real):
            scores = [float(raw_scores)]
        else:
            scores = [float(score) for score in raw_scores]

        # 每个候选必须对应一个有效数字；数量不符或出现 NaN、无穷大会让排序不可靠。
        if len(scores) != len(hits):
            raise ValueError("Reranker returned invalid scores")
        for score in scores:
            if not math.isfinite(score):
                raise ValueError("Reranker returned invalid scores")

        # 复制检索结果并补上重排分数，原结果和已有的向量、BM25、RRF 分数保持不变。
        ranked_hits = []
        for hit, score in zip(hits, scores):
            ranked_hit = replace(
                hit,
                rerank_score=score,
            )
            ranked_hits.append(ranked_hit)

        sorted_hits = sorted(
            ranked_hits,
            key=lambda hit: hit.rerank_score,
            reverse=True,
        )
        return sorted_hits[:top_k]

    def release(self):
        """移除本组件持有的重排模型引用。"""
        with self._lock:
            self._model = None
