"""延迟加载嵌入模型，语义切分与向量存储共享同一组件实例。"""

from __future__ import annotations

import threading

# Qwen3-Embedding 的 query 检索指令前缀；document 侧不加任何前缀。
# 指令通过 query_encode_kwargs 传给 sentence-transformers 的 encode，只在 query
# 编码时生效，入库的文档向量与语义切分使用的文档向量都不受影响。
QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
    "Query: "
)


def _disable_use_cache(embeddings) -> None:
    """关掉嵌入模型前向的 KV cache，避免长句嵌入时显存被它吃爆。

    sentence-transformers 的 encode 只用最后隐藏态做池化，根本不需要各层的
    past_key_values；但 transformers 默认 use_cache=True，会给整批每个 token 建
    36 层 KV cache。短句无所谓，一旦某份文档里有几千 token 的超长"句子"（政务
    清单条目以换行分隔、拆句器把它们粘成一整句），256 句一批的 KV cache 就要
    十几 GiB 显存，直接把 48G 卡打爆（0162 负面清单实测）。
    """
    client = getattr(embeddings, "client", None)
    if client is None:
        client = getattr(embeddings, "_client", None)
    if client is None:
        return
    module = client[0] if hasattr(client, "__getitem__") else None
    auto_model = getattr(module, "auto_model", None) if module is not None else None
    if auto_model is not None:
        auto_model.config.use_cache = False


class Embedder:
    def __init__(self, settings, factory=None):
        """保存嵌入模型配置；可传入自定义加载函数，此时还不加载权重。"""
        self.settings = settings
        self._factory = factory
        self._model = None
        # 预热线程和首个请求可能几乎同时进来，加载要串行，否则同一份权重会被加载两次。
        self._lock = threading.Lock()

    def load(self):
        """首次调用时加载嵌入模型，后续调用返回同一个实例。

        加载全程持锁：并发的第二个调用会等第一个加载完再复用同一个实例，
        而不是各自加载一份，在显存紧张时表现为直接 OOM。
        """
        with self._lock:
            if self._model is None:
                factory = self._factory
                if factory is None:
                    from langchain_huggingface import HuggingFaceEmbeddings

                    factory = HuggingFaceEmbeddings
                # 配置中的 path 是本地模型目录；model_name 是加载器的参数名，也接受目录路径。
                self._model = factory(
                    model_name=str(self.settings.path),
                    # 嵌入模型原生是 bfloat16：以 fp32 加载会占约 16G，问答时要与
                    # 32B 生成、4B 重排同时驻留，这里按原生 bf16（约 8G）才留得下余量。
                    # torch_dtype 要传给 transformers 的 from_pretrained，必须嵌进
                    # SentenceTransformer 自己的 model_kwargs 里；放在顶层它不接收。
                    model_kwargs={
                        "device": self.settings.device,
                        "model_kwargs": {"torch_dtype": "bfloat16"},
                    },
                    # batch_size 只影响文档侧批量编码；默认 32 会让 4B 嵌入模型在
                    # 48G 卡上吃不饱，构建（切分 + 入库）正是靠这里批量嵌入提速。
                    # 文档侧按 token 截断到 1024：MinerU 解析出的表格与 list 块不经
                    # 语义切分、整块入库，一份巨型名录/表格能到数万 token，tokenizer
                    # 按整批最长句填充后，256 条一批的 MLP 中间激活随 seq 线性暴涨
                    # （关掉 use_cache 也救不了，见 _disable_use_cache 之上的注释）。
                    # 实测 256×2048 仍会把 gate/up 投影撑到 20G 而 OOM，256×1024 峰值
                    # 约 25G 才安全。截断只影响向量，入库仍保存原文，检索时仍能取回
                    # 完整表格。
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 256,
                        "processing_kwargs": {
                            "text": {"max_length": 1024, "truncation": True}
                        },
                    },
                    # langchain-milvus 直接调用返回对象的 embed_query，因此 query 指令
                    # 必须配在 HuggingFaceEmbeddings 上，而不是在 Embedder.embed_query 里
                    # 手动拼接。normalize_embeddings 同时写进 query 侧，避免新版把
                    # query_encode_kwargs 整体替换掉 encode_kwargs 时丢失归一化。
                    query_encode_kwargs={
                        "normalize_embeddings": True,
                        "prompt": QUERY_INSTRUCTION,
                    },
                )
                _disable_use_cache(self._model)
            return self._model

    def embed_documents(self, texts):
        """把一批文本转换为向量，并由嵌入模型把向量长度归一化为 1。"""
        return self.load().embed_documents(texts)

    def embed_query(self, text):
        """生成查询向量，使用与文档向量相同的嵌入模型。"""
        return self.load().embed_query(text)

    def release(self):
        """移除本组件持有的模型引用，允许后续重新加载。"""
        with self._lock:
            self._model = None
