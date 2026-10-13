"""重排模型适配器；排序时保留原有召回分数与文档元数据。"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from numbers import Real

from ..schemas import positive_int, valid_query

# Qwen3-Reranker 判断「文档是否满足查询」时使用的默认检索指令。
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# 一次前向打分的最多候选条数。分批是为了把峰值显存和单批大小绑定，
# 而不是和全部候选（fusion_top_k=20）绑定；分数与整批一次前向完全一致。
# 取 2：末尾 logits 张量按 batch×seq×vocab 物化，4 条 × 8192 时 logits 单张就约
# 10G，生成评测里嵌入/重排/生成三模型已占约 35G，只剩约 13G 给前向激活。
_SCORE_BATCH = 2


class _Qwen3Reranker:
    """Qwen3-Reranker 适配器：把 transformers 模型包装成旧 FlagReranker 的接口。

    对外暴露 compute_score(pairs, normalize, max_length)，与旧加载方式对齐，
    Reranker.rerank 里的打分、校验与排序逻辑因此完全不用改。
    """

    # Qwen3-Reranker 的固定前后缀：system 段是固定的判定指令，末尾 assistant 段
    # 带一个空的 <think> 块；中间夹 <Instruct>/<Query>/<Document> 三行正文。
    # 直接拼前后缀等价于模型自带的 chat_template，但绕开了 apply_chat_template
    # 对 system/query/document 这些非标准 role 的处理差异。
    _PREFIX = (
        '<|im_start|>system\n'
        'Judge whether the Document meets the requirements based on the Query and the '
        'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        '<|im_start|>user\n'
    )
    _SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._prefix_tokens = tokenizer.encode(self._PREFIX, add_special_tokens=False)
        self._suffix_tokens = tokenizer.encode(self._SUFFIX, add_special_tokens=False)

    def _format_pair(self, query, document):
        """按 Qwen3-Reranker 的输入格式拼出一条 query-document 对。"""
        return (
            f"<Instruct>: {DEFAULT_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def compute_score(self, pairs, normalize=True, max_length=None):
        """返回每条候选的相关性分数（模型输出 yes 的概率），顺序与 pairs 一致。

        normalize 仅为兼容旧接口保留：这里的分数本来就是 0~1 的概率。

        逐批打分而不是把全部候选塞进一次前向：重排取末尾 logits，因果模型各序列
        之间没有跨序列注意力，分批结果与整批完全一致；但一批 20 条 × 8192 token
        的 MLP 中间激活会直接打爆 48G 卡（生成模型同时驻留时更紧），分批后峰值
        只与单批大小相关。
        """
        import torch

        tokenizer = self._tokenizer
        max_length = max_length or 8192
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        all_scores = []
        for start in range(0, len(pairs), _SCORE_BATCH):
            chunk = pairs[start : start + _SCORE_BATCH]
            texts = [self._format_pair(query, document) for query, document in chunk]
            # 先把正文按「总长 - 前后缀」截断，再拼上前缀/后缀，最后统一左填充。
            inputs = tokenizer(
                texts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
            )
            for i, ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = self._prefix_tokens + ids + self._suffix_tokens
            # 正文已在上面按「总长 - 前后缀」截断，这里只需左填充到批次内最长序列。
            inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits[:, -1, :]
            # 在 [no, yes] 两个词元上做 softmax，取 yes 的概率作为相关分数。
            scores = torch.log_softmax(
                torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1), dim=1
            )[:, 1].exp()
            all_scores.extend(scores.tolist())
        return all_scores


def _load_qwen3_reranker(model_path):
    """从本地目录加载 Qwen3-Reranker，返回带 compute_score 接口的适配器。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    # 与嵌入模型一致的修复：重排只做一次前向取末尾 logits，不需要 KV cache。
    # 默认 use_cache=True 会给整批每个 token 建 36 层 KV cache，20 条候选 × 8192
    # token 时约 15 GiB，直接把 48G 卡打爆（见 embedder._disable_use_cache 的注释）。
    model.config.use_cache = False
    return _Qwen3Reranker(model, tokenizer)


class Reranker:
    def __init__(self, model_path, factory=None, max_length=None):
        """保存重排模型路径与最大词元数；可传入自定义加载函数，此时还不加载权重。"""
        self.model_path = model_path
        self._factory = factory
        # 单条候选参与打分的最大词元数；超出的正文不参与打分，长片段只按开头排序
        # 会明显偏离实际相关性，因此默认放宽到 8192（Qwen3-Reranker 支持 32K）。
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
                    self._model = _load_qwen3_reranker(str(self.model_path))
                else:
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
