"""通过 LangChain 读写 Milvus，负责片段的保存、搜索和删除。"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from ..schemas import SearchHit, positive_int


def source_expr(names):
    """把文件名列表转换为 Milvus 的 source 筛选条件，处理文件名中的特殊字符。"""
    return "source in " + json.dumps(list(names), ensure_ascii=False)


class MilvusStore:
    def __init__(self, embedder, settings, factory=None):
        """保存嵌入组件和 Milvus 配置；测试时可传入 factory 替代真实数据库类。"""
        self.embedder = embedder
        self.settings = settings
        self._factory = factory
        self._backend = None

    def _options(self):
        """把 MilvusSettings 整理成创建 Milvus 对象需要的参数。

        使用本地 .db 文件时，先创建存放它的目录；复制参数以免修改原配置。
        """
        uri = self.settings.connection_args.get("uri", "")
        if uri and "://" not in uri and str(uri).endswith(".db"):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return dict(
            collection_name=self.settings.collection,
            # 不同格式的来源字段不同，用动态字段保存页码、标题和行号等可选信息。
            # 此选项用于新建集合；已有固定字段的集合需通过全量 build 重建。
            enable_dynamic_field=True,
            connection_args=deepcopy(self.settings.connection_args),
            index_params=deepcopy(self.settings.index_params),
            search_params=deepcopy(self.settings.search_params),
        )

    def _get_factory(self):
        """取得创建数据库对象所用的类，默认使用 LangChain 的 Milvus 类。"""
        if self._factory is None:
            from langchain_milvus import Milvus

            self._factory = Milvus
        return self._factory

    @property
    def backend(self):
        """第一次使用时创建 Milvus 对象，后续操作继续使用这个对象。"""
        if self._backend is None:
            self._backend = self._get_factory()(
                embedding_function=self.embedder.load(),
                **self._options(),
            )
        return self._backend

    def search(self, query, k):
        """按向量相似度查找最多 k 个片段，返回片段及对应分数。

        使用 HNSW 时，让搜索参数 ef 至少等于需要返回的数量 k。
        """
        positive_int(k, "k")
        params = deepcopy(self.settings.search_params)
        if "ef" in params.get("params", {}):
            params["params"]["ef"] = max(params["params"]["ef"], k)
        results = self.backend.similarity_search_with_score(query, k=k, param=params)
        return [SearchHit(document=doc, dense_score=float(score)) for doc, score in results]

    def replace_all(self, chunks):
        """删除旧集合并用传入的片段重新建库；片段为空时停止，防止误删数据。"""
        if not chunks:
            raise ValueError("Refusing to replace an index with no chunks")
        self._backend = self._get_factory().from_documents(
            chunks, self.embedder.load(), drop_old=True, **self._options()
        )

    def validate_update(self):
        """增量写入前检查集合支持动态字段，旧集合先重建，避免删除后才发现字段不兼容。

        langchain_milvus 的 collection.schema 只暴露 fields，取不到动态字段开关
        （它对应的是 pymilvus 客户端的 schema，没有这个属性），
        因此改走底层 MilvusClient 的 describe_collection。
        """
        client = getattr(self.backend, "client", None)
        if client is None:
            # 测试替身或未来的实现可能没有 client；没有可查的信息就不拦，交给写入时暴露。
            return
        info = client.describe_collection(collection_name=self.settings.collection) or {}
        # 只有服务端明确回答「不支持」才拦截；字段缺失时不要凭猜测阻断增量更新。
        if info.get("enable_dynamic_field") is False:
            raise RuntimeError(
                "现有 Milvus 集合不支持动态字段，请先运行 python main.py build 全量重建"
            )

    def add(self, chunks):
        """把新片段添加到 Milvus；没有片段时不执行写入。"""
        if chunks:
            self.backend.add_documents(chunks)

    def delete_sources(self, names):
        """根据 source 字段删除指定文件的片段；文件名列表为空时不操作。"""
        if names:
            self.backend.delete(expr=source_expr(names))
