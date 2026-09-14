"""正文语义切分与噪声过滤，不负责数据库或文件写入。"""

from __future__ import annotations

import re
from itertools import groupby

from .config import SplitSettings
from .headings import heading_level
from .schemas import DocumentLike, ParseResult

MEANINGFUL_CHAR_PATTERN = re.compile(r"[一-鿿A-Za-z0-9]")

# 句末标点，允许后面跟一个收尾的引号或括号；用来判断页尾那句有没有说完。
TERMINAL_PUNCT_PATTERN = re.compile(r"[。！？；;.!?][」”』）)]?\s*$")

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


def _source_key(chunk):
    """片段的来源标识，用于判断两条片段是否真的连续。

    只看来源，不看页码：同一份文档的相邻两页在正文里本来就是连着的，
    合并切分之后，标题跨页搬运反而是正确行为。
    """
    return chunk.metadata.get("source")


def _with_page_meta(chunk, metadata):
    """复制片段并换上它第一句所属页面的元数据。

    页码和章节路径要一起换：章节路径本来就是逐页记录的，整篇合并后若沿用
    第一页的值，引用里会给出错误的章节。原文没有章节路径时连键一起去掉，
    免得留下空串让引用多出一截。
    """
    updated = dict(chunk.metadata)
    updated["page"] = metadata.get("page")
    heading = metadata.get("heading_path")
    if heading:
        updated["heading_path"] = heading
    else:
        updated.pop("heading_path", None)
    copy_method = getattr(chunk, "model_copy", None)
    if copy_method is not None:
        return copy_method(update={"metadata": updated})
    return type(chunk)(page_content=chunk.page_content, metadata=updated)


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


def _sentences(text, sentence_regex):
    """按句末标点分句并丢掉空串。

    拼接整篇和回收页码两侧必须用同一套规则：两边数出的句子数一旦对不上，
    后面的页码会整体错位。
    """
    return [sentence for sentence in re.split(sentence_regex, text) if sentence]


def _meta_at(spans, offset):
    """返回覆盖该字符偏移的页面元数据；超出末尾时取最后一页。"""
    for end, metadata in spans:
        if offset < end:
            return metadata
    return spans[-1][1]


def _merge_pages(docs, sentence_regex):
    """把同一来源的连续页面拼成整篇，并给出每一句所属页面的元数据。

    切分器对每篇文档单独调用 split_text，若把每一页各自当成一篇，跨页的段落
    必然断在页边界上。页尾那句若停在句中（末尾没有句末标点），就直接接上
    下一页的开头，让被劈开的一句复原；说完了才用换行保留段落边界。

    返回整篇文本和逐句元数据表，两者按句子顺序一一对应。
    """
    merged = ""
    spans = []  # (结束偏移, 该页元数据)，按拼接顺序排列
    for document in docs:
        text = document.page_content.strip()
        if not text:
            continue
        # 页尾那句已经说完才换行；没说完就直接接上，让被劈开的一句复原。
        # 整篇还没有内容时前面不加任何分隔。
        if merged and TERMINAL_PUNCT_PATTERN.search(merged):
            separator = "\n"
        else:
            separator = ""
        merged += separator + text
        spans.append((len(merged), document.metadata))
    sentences, metas, cursor = [], [], 0
    for sentence in _sentences(merged, sentence_regex):
        sentences.append(sentence)
        metas.append(_meta_at(spans, cursor))
        cursor += len(sentence)
    return "".join(sentences), metas


def _load_splitter_factory():
    """尝试从不同模块导入 SemanticChunker；都不可用时保留导入错误。"""
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        try:
            from langchain_text_splitters import SemanticChunker
        except ImportError:
            from langchain.text_splitter import SemanticChunker
    return SemanticChunker


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

    def _split_merged(self, docs: list[DocumentLike]) -> list[DocumentLike]:
        """把同一来源的连续页面拼成整篇再切分，然后把页码按句回填到各片段。

        切分器产出的每一块都是若干连续句子的拼接（句子之间被补了空格），
        按同一套规则重新分句就能数出这一块占用了页码表里的哪一段。块的页码
        取第一句所在页：跨页的块按它开始的位置引用，与阅读顺序一致。
        """
        merged, metas = _merge_pages(docs, self.settings.sentence_regex)
        if not merged:
            return []
        chunks = self._get_splitter().split_documents([_with_content(docs[0], merged)])
        assigned, cursor = [], 0
        for chunk in chunks:
            metadata = metas[cursor] if cursor < len(metas) else metas[-1]
            assigned.append(_with_page_meta(chunk, metadata))
            cursor += len(_sentences(chunk.page_content, self.settings.sentence_regex))
        return assigned

    def split_texts(self, docs: list[DocumentLike]) -> list[DocumentLike]:
        """按语义断点拆分正文，继承来源和页码；空输入直接返回空列表。

        带页码的来源（PDF 正文）先把相邻页面拼成整篇再切：切分器对每篇文档
        独立调用 split_text，逐页处理会让跨页的段落必然断在页边界上。
        没有页码的来源（Markdown、Word 等）本就按块组织，保持逐块切分。

        拆分后修一次标题边界：断点可能正好落在小标题和它的正文之间，
        那样标题会孤零零留在上一块末尾，见 restore_heading_boundaries。
        """
        if not docs:
            return []
        chunks = []
        for _, group in groupby(docs, key=_source_key):
            group = list(group)
            if "page" in group[0].metadata:
                chunks.extend(self._split_merged(group))
            else:
                chunks.extend(self._get_splitter().split_documents(group))
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
        return text_chunks + whole_units + parsed.tables + parsed.figures
