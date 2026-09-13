"""组织问答步骤：检索片段、重新排序、整理提示词，再调用模型生成答案。"""

from __future__ import annotations

from time import perf_counter

from .interfaces import Generator, QueryRewriter, Reranker, Retriever
from .schemas import AnswerResult, valid_query


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        context_builder,
        generator: Generator,
        settings,
        rewriter: QueryRewriter | None = None,
    ):
        """接收问答各阶段组件与检索参数，流程本身不加载模型。

        rewriter 是可选的：不传时问答流程与没有查询改写时完全一致。
        """
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.generator = generator
        self.settings = settings
        self.rewriter = rewriter

    def retrieve(self, query, k=None, with_rerank=True, hybrid=True):
        """查找相关片段，返回带分数、来源和页码等信息的结果，不生成答案。

        k 传给检索器；启用重排时，最终最多保留 context_top_k 条。
        """
        query = valid_query(query)
        hits = self.retriever.retrieve(query, k=k, hybrid=hybrid)
        if with_rerank and hits:
            hits = self.reranker.rerank(query, hits, self.settings.context_top_k)
        return hits

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
        """执行检索到生成的完整流程，返回答案、证据、提示词和阶段耗时。

        没有相关片段时返回空答案，不调用生成模型。
        on_prompt 是可选函数，接收实际使用的提示词，例如用来打印提示词。
        on_token 是可选函数，接收生成过程中新增的文本块，例如用来边生成边显示。
        on_context 是可选函数，在生成前接收 ContextBundle，供调用方提前展示来源。
        history 是 (role, text) 列表：配置了改写器时，用它把追问里的指代还原成
        独立问句，因此会间接影响检索词；没有改写器时只用于提示词里的【历史对话】。
        memory 是长期记忆全文，作为背景参考注入，不参与检索。
        """
        query = valid_query(query)
        # perf_counter 的前后差值是耗时秒数；首次调用时也包括相应模型的加载时间。
        timings = {}
        start = perf_counter()
        # 改写只决定「拿什么词去检索」：检索和重排用改写后的问题，
        # 而提示词里的【问题】始终保留用户的原话，界面显示的依据才不会和提问对不上。
        search_query = query
        if self.rewriter is not None:
            search_query = self.rewriter.rewrite(query, history=history)
            # 没有改写器时整个阶段都不记录，前端会显示“—”，与「改写了但耗时为 0」区分开。
            timings["rewrite"] = perf_counter() - start
        start = perf_counter()
        hits = self.retriever.retrieve(search_query, hybrid=hybrid)
        timings["retrieval"] = perf_counter() - start
        start = perf_counter()
        if with_rerank and hits:
            hits = self.reranker.rerank(search_query, hits, self.settings.context_top_k)
        timings["rerank"] = perf_counter() - start
        if not hits:
            return AnswerResult(answer="", evidence=[], timings=timings, search_query=search_query)
        start = perf_counter()
        # 未启用重排时，直接将检索结果交给上下文组件，不额外截取 context_top_k。
        context = self.context_builder.build(query, hits, history=history, memory=memory)
        timings["context"] = perf_counter() - start
        # 先给出证据，调用方才能在生成开始前展示来源，而不是等答案生成完。
        if on_context:
            on_context(context)
        if on_prompt:
            on_prompt(context.prompt)
        start = perf_counter()
        # 只有需要实时显示时才走流式路径，其余调用方仍拿到一次性返回的完整答案。
        if on_token is None:
            answer = self.generator.generate(context.prompt)
        else:
            answer = self._stream_answer(context.prompt, on_token)
        timings["generation"] = perf_counter() - start
        return AnswerResult(answer, context.evidence, context.prompt, timings, search_query)

    def _stream_answer(self, prompt, on_token):
        """逐块转发模型输出，边交给 on_token 显示边累积成完整答案。"""
        pieces = []
        for piece in self.generator.generate_stream(prompt):
            pieces.append(piece)
            on_token(piece)
        return "".join(pieces).strip()
