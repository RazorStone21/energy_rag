"""根据配置创建模型、解析器和存储对象，再把它们连接成入库与问答流程。"""

from __future__ import annotations

from functools import cached_property

from .chunker import Chunker
from .config import Settings
from .context_builder import ContextBuilder
from .ingestion import IngestionPipeline
from .models.embedder import Embedder
from .models.generator import Generator, VisionGenerator
from .models.reranker import Reranker
from .models.rewriter import QueryRewriter
from .parsers.excel import ExcelParser
from .parsers.markdown import MarkdownParser
from .parsers.pdf import PDFParser
from .parsers.registry import ParserRegistry
from .parsers.text import TextParser
from .parsers.word import WordParser
from .pipeline import RAGPipeline
from .retrieval.bm25 import CachedBM25
from .retrieval.hybrid import HybridRetriever
from .storage.chunks import ChunkStore
from .storage.memory import MemoryStore
from .storage.milvus import MilvusStore


class Runtime:
    def __init__(self, settings):
        """保存 settings 并创建各组件对象；实际使用模型时才加载权重。"""
        self.settings = settings
        self.embedder = Embedder(settings.embedding)
        self.reranker = Reranker(settings.reranker_path)
        self.generator = Generator(settings.generation)
        self.vision = VisionGenerator(settings.vision)
        # 改写复用问答的同一个文本模型实例，不额外占显存，也不单独加载。
        self.rewriter = QueryRewriter(
            settings.rewrite_prompt,
            settings.rewrite,
            self.generator,
        )
        self.chunk_store = ChunkStore(settings.chunks_path, settings.manifest_path)
        # 记忆存储挂在 Runtime 上，这样演示模式和测试可以像换 pipeline 一样换掉它。
        self.memory_store = MemoryStore(settings.memory_dir, settings.memory.enabled)
        self.vector_store = MilvusStore(self.embedder, settings.milvus)
        self.lexical_index = CachedBM25(self.chunk_store)
        self.context_builder = ContextBuilder(
            settings.prompt_template,
            settings.conversation,
            settings.memory,
        )
        self.chunker = Chunker(self.embedder, settings.splitting)
        # 只注册已经实现的格式；文件发现与解析共用这里的后缀列表。
        self.parser = ParserRegistry(
            {
                ".pdf": PDFParser(self.vision, settings.vision),
                ".txt": TextParser(),
                ".md": MarkdownParser(),
                ".docx": WordParser(),
                ".xlsx": ExcelParser(settings.excel),
            }
        )

    # 第一次访问时创建流程，之后直接返回已创建的对象，避免重复组装。
    @cached_property
    def pipeline(self):
        """把检索、重排、提示词和生成组件组成问答流程，创建后重复使用。"""
        retriever = HybridRetriever(
            self.vector_store,
            self.lexical_index,
            self.settings.retrieval,
            readiness_check=self.chunk_store.assert_ready,
        )
        return RAGPipeline(
            retriever,
            self.reranker,
            self.context_builder,
            self.generator,
            self.settings.retrieval,
            self.rewriter,
        )

    @cached_property
    def ingestion(self):
        """用当前 Runtime 的解析器、切分器和存储对象创建入库流程。"""
        return IngestionPipeline(
            self.parser,
            self.chunker,
            self.vector_store,
            self.chunk_store,
            self.settings.doc_dir,
            supported_suffixes=self.parser.supported_suffixes,
        )

    def release_models(self):
        """清除当前 Runtime 保存的模型和数据库对象；其他地方仍在使用的对象不会消失。"""
        for component in (self.embedder, self.reranker, self.generator, self.vision):
            component.release()
        # Milvus 对象也保存着嵌入模型，因此这里一并清除，让模型有机会释放。
        self.vector_store._backend = None

    def release_reranker(self):
        """清除重排模型引用并尝试回收内存，减少随后生成评测的显存占用。"""
        import gc
        import sys

        self.reranker.release()
        gc.collect()
        torch = sys.modules.get("torch")
        # 只清理已经加载过的 PyTorch；empty_cache 不能释放仍被其他对象使用的张量。
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_runtime(settings: Settings) -> Runtime:
    """根据 settings 创建一个新的 Runtime，由调用方决定使用多久。"""
    return Runtime(settings)
