"""正文语义切分与噪声过滤，不负责数据库或文件写入。"""

from __future__ import annotations

import importlib
import re
from itertools import groupby

from .config import SplitSettings
from .headings import heading_level
from .schemas import DocumentLike, ParseResult

MEANINGFUL_CHAR_PATTERN = re.compile(r"[一-鿿A-Za-z0-9]")

# 解析器标为这些类型的片段不参与句子切分，整体保留。
WHOLE_UNIT_KINDS = ("code", "list")


def meaningful_len(text: str) -> int:
    """统计中文、英文字母和数字的数量；标点与空白不计入有效字符数。"""
    return len(MEANINGFUL_CHAR_PATTERN.findall(text))


def _with_content(chunk, text):
    """复制片段并替换正文；兼容 LangChain Document 与其他实现。"""
    copy_method = getattr(chunk, "model_copy", None)
    if copy_method is not None:
        return copy_method(update={"page_content": text})
    return type(chunk)(page_content=text, metadata=dict(chunk.metadata))


# Milvus 的 text 字段在 langchain_milvus 里硬编码 max_length=65535（按字符计）。
# MinerU 解析出的巨型 list/code/section 块会整体保留或整段入库，一份名录/清单
# 能到数万字符（0162 负面清单：39752 行、每行一个字的退化提取，整段 82844 字符），
# replace_all 写库时直接报 MilvusException（VARCHAR value length exceeds max_length）。
# 这里按换行把超限整块切成多片，每片 UTF-8 字节数控制在 30000 以内：字节数与字符数
# 都远低于 65535，无论 Milvus 按哪种口径计都不会越界，还顺带让单条片段不至于大到
# 塞爆生成模型的上下文。
_MAX_CHUNK_BYTES = 30_000

# 切分逻辑版本号：改动任何会影响片段产出的逻辑（断点截断、超长切片、句合并等）
# 都要递增，让按文件缓存失效、强制重新切分，避免新旧产物混用。
_CHUNKER_VERSION = "1"


def _split_oversized(document):
    """把超过存储上限的整块按换行切成多条片段，元数据原样继承。

    list/code/section 以行为单位，换行处切分不打断结构；单行本身就超长时
    （如压缩过的代码行）按字符硬切。普通正文、表格、图片描述远小于上限，不受影响。
    """
    text = document.page_content
    if len(text.encode("utf-8")) <= _MAX_CHUNK_BYTES:
        return [document]
    # 先按换行拆成原子行；单行超限时按字符硬切，一个 UTF-8 字符最多 4 字节，
    # 取 1/4 预算字符数必不超过字节上限。
    atoms = []
    for line in text.split("\n"):
        if len(line.encode("utf-8")) <= _MAX_CHUNK_BYTES:
            atoms.append(line)
            continue
        step = _MAX_CHUNK_BYTES // 4
        atoms.extend(line[start:start + step] for start in range(0, len(line), step))
    # 按字节预算把原子行重新拼成片段，换行分隔，保持原有行结构。
    chunks, current, current_bytes = [], [], 0
    for atom in atoms:
        atom_bytes = len(atom.encode("utf-8"))
        if current and current_bytes + atom_bytes > _MAX_CHUNK_BYTES:
            chunks.append("\n".join(current))
            current, current_bytes = [], 0
        current.append(atom)
        current_bytes += atom_bytes
    if current:
        chunks.append("\n".join(current))
    return [_with_content(document, piece) for piece in chunks if piece.strip()]


def _source_key(chunk):
    """片段的来源标识，用于判断两条片段是否真的连续。

    只看来源，不看页码：同一份文档的相邻两页在正文里本来就是连着的，
    合并切分之后，标题跨页搬运反而是正确行为。
    """
    return chunk.metadata.get("source")


def _split_trailing_heading(text, sentence_regex):
    """末尾一句若是编号标题就拆出来，返回 (标题, 剩余正文)。

    只剩一句时不动：搬走它会把这一块搬空。标题本身也要求是有效标题，
    实测末尾短句里绝大多数是 PDF 提取留下的碎片，那些不能当标题搬。
    """
    marks = list(re.finditer(sentence_regex, text))
    if len(marks) < 2:
        return "", text
    cut = marks[-2].end()
    tail = text[cut:].strip()
    if heading_level(tail) is None:
        return "", text
    return tail, text[:cut].rstrip()


def restore_heading_boundaries(chunks, sentence_regex, min_chars):
    """把被切在块尾的章节标题交给下一块。

    语义断点可能正好落在「（三）推进构网型技术应用。」和它的正文之间，
    标题孤零零留在上一块末尾、正文在下一块开头，检索时问题里的关键词
    和答案就分到了两条片段上。标题属于它下面的内容，因此挪到下一块开头。

    只搬编号标题：实测末尾的短句（≤20 字）里只有约一成是编号标题，
    其余大多是被排版拆散的数字和页码残留，搬来搬去只会添乱。
    跨文件不搬——不同文件的内容本来就不连续；跨页可以搬，同一份文档的
    相邻两页在正文里是连着的，切分时已经按整篇处理。
    """
    if len(chunks) < 2:
        return list(chunks)
    texts = [chunk.page_content for chunk in chunks]
    # carried[i] 是从第 i 块末尾挪出来、要接到第 i+1 块开头的标题。
    carried = [""] * len(chunks)
    for index in range(len(chunks) - 1):
        if _source_key(chunks[index]) != _source_key(chunks[index + 1]):
            continue
        heading, rest = _split_trailing_heading(texts[index], sentence_regex)
        # 搬走标题后剩下的部分若会被长度门槛过滤掉，就等于把内容搬丢了，宁可不搬。
        if not heading or meaningful_len(rest) < min_chars:
            continue
        carried[index] = heading
        texts[index] = rest
    return [
        _with_content(chunk, (carried[index - 1] if index else "") + texts[index])
        for index, chunk in enumerate(chunks)
    ]


# 断点检测嵌入的合并句长度上限（字符）。语义切分只为找断点而嵌入，不需要整句；
# 政务清单这类文档的条目以换行而非句末标点分隔，拆句器会把它们粘成数千字符的
# "句子"，不截断的话 tokenizer 会按整批最长句填充，前向显存随序列长度暴涨而 OOM
# （0162 负面清单的 3530 字符合并句实测直接打爆 48G）。512 字符对断点检测绰绰有余，
# 且只截断这些病态超长句——正常合并句不过百来字符，不受影响。
_BREAKPOINT_EMBED_MAX_CHARS = 512


def _make_batched_chunker(base_cls):
    """给 SemanticChunker 套一层批量化：把所有版面块的句子合并后一次性嵌入。

    LangChain 的 SemanticChunker.create_documents 对每个版面块分别调用一次
    embed_documents，一块往往只有三五句，固定开销（分词、CUDA 启动、结果回传）
    远大于实际计算；千页文档有几万块时，这一项会让构建慢几十倍。这里先按与
    父类完全相同的顺序拆句、合并，把全部合并句一次性交给 embed_documents
    （从而吃满 encode 的 batch_size），再按原顺序回填，最后交给父类的断点、
    阈值与切片逻辑。输出与原生实现逐字节一致，仅合并了嵌入调用。
    """
    combine_sentences = importlib.import_module(base_cls.__module__).combine_sentences

    class BatchedSemanticChunker(base_cls):
        def create_documents(self, texts, metadatas=None):
            _metadatas = metadatas or [{}] * len(texts)
            # 预拆句并做 buffer 合并，只收集真正需要嵌入的合并句；
            # 单句块与父类 split_text 一致：直接返回、不做嵌入。
            combined_lists = []
            for text in texts:
                single = self._get_single_sentences_list(text)
                if len(single) == 1:
                    combined_lists.append(None)
                    continue
                sentences = [
                    {"sentence": x, "index": i} for i, x in enumerate(single)
                ]
                combined_lists.append(
                    [
                        s["combined_sentence"]
                        for s in combine_sentences(sentences, self.buffer_size)
                    ]
                )
            flat = [c for lst in combined_lists if lst is not None for c in lst]
            # 嵌入前截断超长合并句（见 _BREAKPOINT_EMBED_MAX_CHARS）：断点检测的嵌入
            # 只在相邻句间算距离，无需整句上下文；截断只发生在病态超长句上。
            flat = [c[:_BREAKPOINT_EMBED_MAX_CHARS] for c in flat]
            embeddings = self.embeddings.embed_documents(flat) if flat else []

            # 用预计算的嵌入回放，避免父类逐块再次嵌入；顺序与预拆分严格一致。
            original_embed = self.embeddings.embed_documents
            iterator = iter(embeddings)

            def replay(_texts):
                return [next(iterator) for _ in _texts]

            self.embeddings.embed_documents = replay
            try:
                return super().create_documents(texts, metadatas=_metadatas)
            finally:
                self.embeddings.embed_documents = original_embed

    return BatchedSemanticChunker


def _load_splitter_factory():
    """尝试从不同模块导入 SemanticChunker；都不可用时保留导入错误。"""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        try:
            from langchain_text_splitters import SemanticChunker
        except ImportError:
            from langchain.text_splitter import SemanticChunker
    return _make_batched_chunker(SemanticChunker)


class Chunker:
    """负责正文切分和统一过滤，表格及图片描述保持完整。"""

    def __init__(self, embedder, settings: SplitSettings, splitter_factory=None):
        """保存嵌入组件和切分参数；也可传入自定义切分器创建函数，用到时才创建。"""
        self.embedder = embedder
        self.settings = settings
        self._factory = splitter_factory
        self._splitter = None

    def _get_splitter(self):
        """首次使用时创建切分器，后续文件复用同一个实例。"""
        if self._splitter is None:
            factory = self._factory
            if factory is None:
                factory = _load_splitter_factory()
            self._splitter = factory(
                self.embedder,
                breakpoint_threshold_type=self.settings.threshold_type,
                breakpoint_threshold_amount=self.settings.threshold_amount,
                buffer_size=self.settings.buffer_size,
                sentence_split_regex=self.settings.sentence_regex,
            )
        return self._splitter

    @property
    def fingerprint(self) -> str:
        """切分链路指纹：切分参数、切分逻辑版本与嵌入模型一起决定片段产出。

        断点检测靠嵌入模型算句子间语义距离，换模型会让断点整体漂移；因此这里把
        嵌入模型路径也纳入指纹，与 parser.fingerprint 拼接后作为按文件缓存的键。
        """
        s = self.settings
        embed_path = str(getattr(getattr(self.embedder, "settings", None), "path", ""))
        return "chunk-" + _CHUNKER_VERSION + "-" + "|".join(
            [
                s.threshold_type,
                str(s.threshold_amount),
                str(s.buffer_size),
                s.sentence_regex,
                str(s.min_chars),
                embed_path,
            ]
        )

    def split_texts(self, docs: list[DocumentLike]) -> list[DocumentLike]:
        """按语义断点拆分正文，继承来源、页码与章节；空输入直接返回空列表。

        解析器已经按版面块切好（PDF 由 MinerU 给出段落级块，Word、Markdown
        本就按块组织），因此逐块做语义切分：切分器会把源片段的元数据复制给
        每一段，页码与章节路径自动跟着走。不再把相邻页面拼成整篇再切——
        那会让不同页的内容糊进同一段，并让页码失去意义。

        拆分后修一次标题边界：断点可能正好落在小标题和它的正文之间，
        那样标题会孤零零留在上一块末尾，见 restore_heading_boundaries。
        """
        if not docs:
            return []
        chunks = []
        for _, group in groupby(docs, key=_source_key):
            chunks.extend(self._get_splitter().split_documents(list(group)))
        return restore_heading_boundaries(
            chunks,
            self.settings.sentence_regex,
            self.settings.min_chars,
        )

    def filter(self, chunks: list[DocumentLike]) -> list[DocumentLike]:
        """保留有效字符数不少于 min_chars 的片段；只检查长度，不判断内容是否相关。"""
        return [
            chunk
            for chunk in chunks
            if meaningful_len(chunk.page_content) >= self.settings.min_chars
        ]

    def split(self, parsed: ParseResult) -> list[DocumentLike]:
        """切分普通正文，完整保留代码块、列表、表格和图片描述，再过滤噪声。

        长度门槛只用于判断「这次切分是否切出了没用的残片」，因此只作用在
        按句子切出来的正文上；表格、图片、代码块和列表本就是完整单元，
        再短也自成一条信息，按同一门槛砍掉会直接丢内容。
        """
        prose, whole_units = [], []
        for document in parsed.texts:
            # 代码块和列表都不能按句子切分：代码切开会打断结构，
            # 列表切开会把并列的几条条目分到不同片段里。
            if document.metadata.get("block_kind") in WHOLE_UNIT_KINDS:
                whole_units.append(document)
            else:
                prose.append(document)
        text_chunks = self.filter(self.split_texts(prose))
        chunks = text_chunks + whole_units + parsed.tables + parsed.figures
        # 存储上限的兜底：任何片段正文超长（巨型 list/code/section 块）都按换行切成多片，
        # 避免 replace_all 写库时撞上 Milvus text 字段的 65535 上限而整体失败。
        return [piece for chunk in chunks for piece in _split_oversized(chunk)]
