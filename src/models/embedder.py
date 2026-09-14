"""延迟加载嵌入模型，语义切分与向量存储共享同一组件实例。"""

from __future__ import annotations


class Embedder:
    def __init__(self, settings, factory=None):
        """保存嵌入模型配置；可传入自定义加载函数，此时还不加载权重。"""
        self.settings = settings
        self._factory = factory
        self._model = None

    def load(self):
        """首次调用时加载嵌入模型，后续调用返回同一个实例。"""
        if self._model is None:
            factory = self._factory
            if factory is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                factory = HuggingFaceEmbeddings
            # 配置中的 path 是本地模型目录；model_name 是加载器的参数名，也接受目录路径。
            self._model = factory(
                model_name=str(self.settings.path),
                model_kwargs={"device": self.settings.device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._model

    def embed_documents(self, texts):
        """把一批文本转换为向量，并由嵌入模型把向量长度归一化为 1。"""
        return self.load().embed_documents(texts)

    def embed_query(self, text):
        """生成查询向量，使用与文档向量相同的嵌入模型。"""
        return self.load().embed_query(text)

    def release(self):
        """移除本组件持有的模型引用，允许后续重新加载。"""
        self._model = None
