"""读取 config.toml，检查参数，并把模型、文档和缓存路径整理到 Settings 对象中。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import positive_int

# 长期记忆注入提示词的硬上限；约 5600 词元，留给文档片段和回答足够空间。
MAX_MEMORY_CHARS = 8000


@dataclass(frozen=True)
class EmbeddingSettings:
    """嵌入模型的本地目录和运行设备，供 Embedder 使用。"""

    path: Path
    device: str


@dataclass(frozen=True)
class GenerationSettings:
    """文本模型的本地目录，以及回答长度、采样和思考模式等生成参数。"""

    path: Path
    max_new_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool


@dataclass(frozen=True)
class VisionSettings:
    """图片描述的模型和提示词，以及提取图片时使用的尺寸、清晰度限制。"""

    path: Path
    max_new_tokens: int
    prompt: str
    min_width: int
    min_height: int
    dpi: int
    # 一次送给视觉模型的图片数量。实测 12 张时单张耗时从 5.3 秒降到 1.0 秒，
    # 显存只涨一档；再往上收益递减，而整套图里最慢的一张会拖住整批。
    batch_size: int = 12

    def __post_init__(self):
        """检查批量大小；批次为 1 时等价于逐张描述。"""
        positive_int(self.batch_size, "vision.batch_size")


@dataclass(frozen=True)
class SplitSettings:
    """正文如何分句、在哪里切分，以及过滤片段时保留的最少有效字符数。"""

    threshold_type: str
    threshold_amount: float
    buffer_size: int
    sentence_regex: str
    min_chars: int


@dataclass(frozen=True)
class ExcelSettings:
    """Excel 每片的数据行数、每个数据区域的表头行数，以及工作表扫描上限。"""

    rows_per_chunk: int = 50
    header_rows: int = 1
    max_sheet_cells: int = 500_000

    def __post_init__(self):
        """验证行数和扫描上限；表头允许为 0，表示所有行都是数据。"""
        positive_int(self.rows_per_chunk, "excel.rows_per_chunk")
        positive_int(self.max_sheet_cells, "excel.max_sheet_cells")
        if (
            isinstance(self.header_rows, bool)
            or not isinstance(self.header_rows, int)
            or self.header_rows < 0
        ):
            raise ValueError("excel.header_rows 必须是非负整数")


@dataclass(frozen=True)
class ConversationSettings:
    """多轮问答带进提示词的历史长度上限，避免长对话把提示词撑大。"""

    max_turns: int = 6
    max_chars: int = 2000

    def __post_init__(self):
        """检查轮数与字数上限；0 轮表示不把历史带进提示词。"""
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 0
        ):
            raise ValueError("conversation.max_turns 必须是非负整数")
        positive_int(self.max_chars, "conversation.max_chars")


@dataclass(frozen=True)
class RewriteSettings:
    """多轮追问的查询改写开关，以及送进改写提示词的历史长度上限。

    改写只在有历史对话时触发：没有历史就无从还原指代，单轮提问因此
    完全不受影响，既不增加耗时也不改变检索词。
    """

    enabled: bool = True
    max_turns: int = 2
    max_chars: int = 500

    def __post_init__(self):
        """检查开关与历史上限；0 轮表示不送历史，此时改写没有依据会退化为原文。"""
        if not isinstance(self.enabled, bool):
            raise ValueError("rewrite.enabled 必须是布尔值")
        if (
            isinstance(self.max_turns, bool)
            or not isinstance(self.max_turns, int)
            or self.max_turns < 0
        ):
            raise ValueError("rewrite.max_turns 必须是非负整数")
        positive_int(self.max_chars, "rewrite.max_chars")


@dataclass(frozen=True)
class MemorySettings:
    """长期记忆的开关与注入上限；存放位置和文档、索引一样放在 [paths] 里。"""

    enabled: bool = True
    max_chars: int = 1000

    def __post_init__(self):
        """检查开关与注入上限；上限还受硬上限约束，超出直接拒绝。"""
        if not isinstance(self.enabled, bool):
            raise ValueError("memory.enabled 必须是布尔值")
        positive_int(self.max_chars, "memory.max_chars")
        # 模型侧不做截断（local_qwen 的 tokenizer 没有传 truncation 和 max_length），
        # 超长提示词不会被自动裁掉，所以配置里的上限是唯一防线。
        if self.max_chars > MAX_MEMORY_CHARS:
            raise ValueError(f"memory.max_chars 不能超过 {MAX_MEMORY_CHARS}")


@dataclass(frozen=True)
class RetrievalSettings:
    """向量搜索、BM25、结果合并和重排各保留多少片段，以及混合检索开关。"""

    dense_top_k: int = 20
    bm25_top_k: int = 20
    fusion_top_k: int = 20
    context_top_k: int = 5
    rrf_k: int = 60
    hybrid_enabled: bool = True

    def __post_init__(self):
        """检查候选数量和混合检索开关，禁止零值、负值及布尔值冒充数量。"""
        for name in ("dense_top_k", "bm25_top_k", "fusion_top_k", "context_top_k", "rrf_k"):
            positive_int(getattr(self, name), name)
        if not isinstance(self.hybrid_enabled, bool):
            raise ValueError("hybrid_enabled 必须是布尔值")


@dataclass(frozen=True)
class MilvusSettings:
    """创建 Milvus 对象需要的集合名、连接地址、建索引和搜索参数。"""

    collection: str
    connection_args: dict
    index_params: dict
    search_params: dict


@dataclass(frozen=True)
class Settings:
    """整个项目的配置；Runtime 把其中对应的配置交给各组件使用。"""

    data_root: Path
    doc_dir: Path
    chunks_path: Path
    manifest_path: Path
    memory_dir: Path
    embedding: EmbeddingSettings
    reranker_path: Path
    generation: GenerationSettings
    vision: VisionSettings
    splitting: SplitSettings
    retrieval: RetrievalSettings
    milvus: MilvusSettings
    prompt_template: str
    model_ids: dict[str, str]
    rewrite_prompt: str = ""
    excel: ExcelSettings = field(default_factory=ExcelSettings)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    rewrite: RewriteSettings = field(default_factory=RewriteSettings)


def _resolve_data_root(
    # config_path： config.toml 的位置，
    # configured_root：配置文件里填写的数据根目录
    # override：外部指定的、用来覆盖配置的数据根目录
    config_path: Path, configured_root: str, override: str | Path | None
) -> Path:
    """将数据根目录转换为绝对路径。

    相对路径的起点：
    - 命令行或环境变量传入：运行命令时所在的目录。
    - config.toml 中填写：config.toml 所在的目录。
    绝对路径直接使用。
    """
    if override is not None:
        root_value = override
    else:
        root_value = configured_root
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        if override is not None:
            base_directory = Path.cwd()
        else:
            base_directory = config_path.parent
        root = base_directory / root
    # 将拼接成的 /root/project/../data 转化为 /root/data
    return root.resolve()


def _validate_config(raw: dict) -> None:
    """检查生成参数、切分参数和提示词占位符，避免加载模型后才发现配置错误。"""
    generation = raw["generation"]
    vision = raw["vision"]
    splitting = raw["splitting"]
    for name, value in (
        ("max_new_tokens", generation["max_new_tokens"]),
        ("vision.max_new_tokens", vision["max_new_tokens"]),
        ("vision.min_width", vision["min_width"]),
        ("vision.min_height", vision["min_height"]),
        ("vision.dpi", vision["dpi"]),
        ("splitting.min_chars", splitting["min_chars"]),
    ):
        positive_int(value, name)
    if generation["temperature"] < 0 or not 0 < generation["top_p"] <= 1:
        raise ValueError("temperature 必须非负，top_p 必须在 (0, 1] 内")
    if not isinstance(generation["enable_thinking"], bool):
        raise ValueError("enable_thinking 必须是布尔值")
    if splitting["buffer_size"] < 0:
        raise ValueError("buffer_size 不能为负数")
    template = raw["prompts"]["rag"]
    if "{context}" not in template or "{question}" not in template:
        raise ValueError("问答模板必须包含 {context} 和 {question}")
    # {history} 和 {memory} 都是可选的：format 会忽略多余关键字参数，
    # 因此带或不带它们的模板都能通过检查。
    template.format(context="", question="", history="", memory="")

    # [rewrite] 段缺失时按默认值处理，默认是启用；启用就必须给出改写模板，
    # 否则会在加载模型之后才发现无从改写。
    if raw.get("rewrite", {}).get("enabled", True):
        rewrite_prompt = raw["prompts"].get("rewrite", "")
        if "{question}" not in rewrite_prompt:
            raise ValueError("启用查询改写时 [prompts] 的 rewrite 模板必须包含 {question}")
        rewrite_prompt.format(history="", question="")


def load_settings(
    config_path: str | Path | None = None, data_root: str | Path | None = None
) -> Settings:
    """读取 TOML 并返回 Settings 配置对象；配置文件不存在或参数无效时抛出异常。

    config_path 优先于 ENERGY_RAG_CONFIG，data_root 优先于 ENERGY_RAG_HOME。
    文件中的相对数据根目录以配置文件位置为基准，不依赖包的安装位置。
    """
    config_location = config_path or os.environ.get("ENERGY_RAG_CONFIG", "config.toml")
    path = Path(config_location).expanduser().resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)

    paths = raw["paths"]
    override = data_root
    if override is None:
        override = os.environ.get("ENERGY_RAG_HOME")
    root = _resolve_data_root(path, paths["data_root"], override)

    def resolve(value: str) -> Path:
        """将配置中的一个路径转换成绝对路径，已有绝对路径保持其位置。"""
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    _validate_config(raw)
    memory = MemorySettings(**raw.get("memory", {}))
    # 旧配置没有 paths.memory 时使用默认位置。
    memory_dir = resolve(paths.get("memory", "data/memory"))
    document_dir = resolve(paths["documents"])
    # 会话记录是 markdown，而 .md 解析器已经注册；记忆目录若落在文档目录里，
    # 下一次入库会把对话记录当成待索引文档。
    if memory_dir == document_dir or document_dir in memory_dir.parents:
        raise ValueError("paths.memory 不能放在 paths.documents 目录内")
    embedding = raw["embedding"]
    generation = raw["generation"]
    vision = raw["vision"]
    splitting = raw["splitting"]
    milvus = raw["milvus"]
    connection = dict(milvus["connection"])
    uri = connection["uri"]
    # HTTP 服务地址直接使用；Milvus Lite 文件地址和其他数据路径使用同一根目录。
    if "://" not in uri and uri.endswith(".db"):
        connection["uri"] = str(resolve(uri))
    return Settings(
        data_root=root,
        doc_dir=document_dir,
        chunks_path=resolve(paths["chunks"]),
        manifest_path=resolve(paths["manifest"]),
        memory_dir=memory_dir,
        embedding=EmbeddingSettings(
            path=resolve(embedding["path"]),
            device=embedding["device"],
        ),
        reranker_path=resolve(raw["reranker"]["path"]),
        generation=GenerationSettings(
            path=resolve(generation["path"]),
            max_new_tokens=generation["max_new_tokens"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            enable_thinking=generation["enable_thinking"],
        ),
        vision=VisionSettings(
            path=resolve(vision["path"]),
            max_new_tokens=vision["max_new_tokens"],
            prompt=raw["prompts"]["figure"],
            min_width=vision["min_width"],
            min_height=vision["min_height"],
            dpi=vision["dpi"],
            batch_size=vision.get("batch_size", VisionSettings.batch_size),
        ),
        splitting=SplitSettings(**splitting),
        retrieval=RetrievalSettings(**raw["retrieval"]),
        milvus=MilvusSettings(
            collection=milvus["collection"],
            connection_args=connection,
            index_params=milvus["index"],
            search_params=milvus["search"],
        ),
        prompt_template=raw["prompts"]["rag"],
        excel=ExcelSettings(**raw.get("excel", {})),
        # 旧配置没有 [conversation] 段时使用默认的历史长度上限。
        conversation=ConversationSettings(**raw.get("conversation", {})),
        memory=memory,
        rewrite_prompt=raw["prompts"].get("rewrite", ""),
        # 旧配置没有 [rewrite] 段时使用默认值。
        rewrite=RewriteSettings(**raw.get("rewrite", {})),
        model_ids={
            "embedding": embedding["model_id"],
            "reranker": raw["reranker"]["model_id"],
            "generation": generation["model_id"],
            "vision": vision["model_id"],
        },
    )
