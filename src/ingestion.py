"""组织文档入库步骤：先解析和检查文件，再更新向量数据库与片段缓存。"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Collection
from pathlib import Path

from .interfaces import Chunker, Parser, VectorStore
from .schemas import BuildResult, DocumentLike, positive_int

logger = logging.getLogger(__name__)


def file_hash(path: str | Path) -> str:
    """按固定大小分块计算 SHA-256，避免把整份文件一次性读入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_documents(
    doc_dir: str | Path,
    supported_suffixes: Collection[str],
    max_files: int | None = None,
) -> list[Path]:
    """按名称列出目录第一层已支持的文件，忽略后缀大小写、其他格式和子目录。"""
    doc_dir = Path(doc_dir)
    if not doc_dir.is_dir():
        raise ValueError(f"Document directory does not exist: {doc_dir}")
    if max_files is not None:
        positive_int(max_files, "max_files")
    suffixes = {suffix.lower() for suffix in supported_suffixes}
    paths = []
    for path in doc_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        # Word、Excel 打开文件时生成的 ~$ 文件是锁定信息，不是可解析文档。
        if path.suffix.lower() in (".docx", ".xlsx") and path.name.startswith("~$"):
            continue
        paths.append(path)
    return sorted(paths)[:max_files]


class IngestionPipeline:
    def __init__(
        self,
        parser: Parser,
        chunker: Chunker,
        vector_store: VectorStore,
        chunk_store,
        doc_dir,
        supported_suffixes: Collection[str] = (".pdf",),
    ):
        """保存入库组件、目录和支持的格式；直接使用旧 PDF 解析器时默认只扫描 PDF。"""
        self.parser = parser
        self.chunker = chunker
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.doc_dir = Path(doc_dir)
        self.supported_suffixes = frozenset(supported_suffixes)

    def _parse_files(
        self, target_paths: list[Path], file_hashes: dict[str, str], progress=None
    ) -> tuple[dict[str, list[DocumentLike]], dict[str, str]]:
        """逐个解析和切分目标文件，返回按来源分组的成功片段及失败原因。

        解析前后比较文件哈希，防止文件在读取期间被修改；失败文件不用于替换旧片段。
        progress 是可选的上报对象，用来显示文件级和当前文件内的进度。
        """
        chunks_by_source, failed_sources = {}, {}
        for index, path in enumerate(target_paths, start=1):
            if progress is not None:
                progress.start_file(index, path.name)
            try:
                parsed = self.parser.parse(
                    path, on_status=progress.note if progress is not None else None
                )
                if parsed.errors:
                    raise RuntimeError("; ".join(parsed.errors))
                chunks = self.chunker.split(parsed)
                if not chunks:
                    raise ValueError("No usable chunks were extracted")
                if file_hash(path) != file_hashes[path.name]:
                    raise RuntimeError("File changed during parsing; retry the build")
                chunks_by_source[path.name] = chunks
                if progress is not None:
                    progress.finish_file(path.name, len(chunks))
            except Exception as exc:
                failed_sources[path.name] = str(exc)
                if progress is not None:
                    progress.fail_file(path.name, str(exc))
                logger.warning(
                    "Failed to process %s; keeping existing data: %s",
                    path.name,
                    exc,
                )

        return chunks_by_source, failed_sources

    def build(
        self,
        doc_dir: str | Path | None = None,
        max_files: int | None = None,
        save: bool = True,
        incremental: bool = False,
        only: list[str] | None = None,
        progress_factory=None,
    ) -> BuildResult:
        """执行全部重建或增量更新，返回成功处理、已删除和处理失败的文件信息。

        全量解析失败会在写索引前终止；增量失败文件保留旧数据并等待重试。
        max_files 限制处理数量；增量模式仍会检查整个目录中已删除的文件。
        only 按文件名强制重处理指定文件，不清理其他文件，即使它们已从目录删除。
        save=False 仍写入向量库，但清除本地缓存，避免新旧索引混用。
        progress_factory 接收待处理文件数并返回进度上报对象；默认不显示进度。
        """
        if doc_dir is not None:
            doc_dir = Path(doc_dir)
        else:
            doc_dir = self.doc_dir
        if max_files is not None:
            positive_int(max_files, "max_files")
        # only 是增量模式的补丁入口：全量本来就会重做所有文件，混用只会让人误解语义。
        if only is not None and (not incremental or not only):
            raise ValueError("--only requires --incremental and at least one filename")
        # 即使限制处理数量，也必须完整扫描文件列表，避免把未选中文档误判为删除。
        all_paths = list_documents(doc_dir, self.supported_suffixes)
        paths_by_name = {path.name: path for path in all_paths}
        if only is not None:
            missing = sorted(set(only) - set(paths_by_name))
            if missing:
                raise ValueError(f"--only files not found: {missing}")
        if incremental:
            # 先确认缓存目录可用，避免解析完一堆文件才发现结果保存不了。
            self.chunk_store.assert_ready()
            # 没有旧清单就无法比较文件变化；普通增量转为全量，指定 only 时则拒绝执行。
            if not self.chunk_store.manifest_path.exists():
                if only is not None:
                    raise ValueError("--only requires a previous full build manifest")
                logger.warning("No manifest found; performing a full build")
                incremental = False

        # 清单记录"上次成功写入索引时每个文件的哈希"，是判断文件是否变化的唯一基准。
        manifest = {}
        if incremental:
            manifest = self.chunk_store.load_manifest()
        # 片段缓存缺失就拼不出完整增量结果，只能先做一次全量。
        if incremental and not self.chunk_store.chunks_path.exists():
            raise RuntimeError(
                "Chunk cache is missing; run a full build before incremental updates"
            )
        existing_chunks = []
        if incremental:
            existing_chunks = self.chunk_store.load_chunks()
        if only is not None:
            # 按传入顺序去重；同一个来源在本次构建中只处理一次。
            selected_paths = [paths_by_name[name] for name in dict.fromkeys(only)]
        else:
            # 全量和普通增量都按目录顺序取前 max_files 个（None 表示全部）。
            selected_paths = all_paths[:max_files]
        # 这里先算一遍哈希；解析结束后会再算一次，用来发现解析期间被改动的文件。
        file_hashes = {path.name: file_hash(path) for path in selected_paths}
        removed_sources = []
        if incremental and only is None:
            # only 模式只负责指定的几个文件，不承担清理职责。
            # 只有文件确实不在磁盘上才清理，不能因某种格式暂未注册而删除它的旧索引。
            removed_sources = [
                name
                for name in manifest
                if not (doc_dir / name).is_file()
            ]  # fmt: skip
        # 全量和 only 模式即使哈希没变也重新处理；普通增量只处理新增或内容改变的文件。
        force_reprocess = not incremental or only is not None
        # 过滤出真正要重新解析的文件：哈希与清单一致的旧文件直接沿用已有片段。
        target_paths = [
            path
            for path in selected_paths
            if force_reprocess or manifest.get(path.name) != file_hashes[path.name]
        ]
        # 全量会用 replace_all 整体覆盖索引，一个可解析文档都没有时报错，避免把索引清空。
        if not incremental and not target_paths:
            raise ValueError("No supported documents found; the existing index was not changed")
        # 增量且无新增、无修改、无删除：索引和缓存都保持原样，只把现状回报给调用方。
        if incremental and not target_paths and not removed_sources:
            return BuildResult(vector_store=self.vector_store, chunks=existing_chunks)

        # 进度要在算出待处理文件数之后才建得出来，所以这里传的是工厂而不是实例。
        progress = progress_factory(len(target_paths)) if progress_factory is not None else None
        try:
            chunks_by_source, failed_sources = self._parse_files(
                target_paths, file_hashes, progress
            )

            # 全量要求所有选中文档都解析成功，否则整体替换会顺手丢掉失败文档的旧片段。
            if failed_sources and not incremental:
                raise RuntimeError(
                    f"Full build aborted before modifying the index: {failed_sources}"
                )
            # 按来源顺序展平成本次新增或更新的片段，供后面写入向量库。
            new_chunks = []
            for chunks in chunks_by_source.values():
                new_chunks.extend(chunks)
            # 增量里全是失败文件且没有文件被删除：不写索引也不改缓存，只回报失败原因。
            if not chunks_by_source and not removed_sources:
                return BuildResult(
                    vector_store=self.vector_store,
                    chunks=[],
                    failed=failed_sources,
                )

            # 增量只替换成功解析的文件，并清理已删除文件；失败文件保留旧内容和哈希供下次重试。
            sources_to_replace = set(chunks_by_source) | set(removed_sources)
            if incremental:
                preserved_chunks = [
                    chunk
                    for chunk in existing_chunks
                    if chunk.metadata.get("source") not in sources_to_replace
                ]
                merged_chunks = preserved_chunks + new_chunks
            else:
                merged_chunks = new_chunks
            new_manifest = {
                name: digest
                for name, digest in manifest.items()
                if name not in sources_to_replace
            }  # fmt: skip
            # 只有解析成功的来源写入新哈希；失败的来源保留旧哈希，下次增量仍会被判定为待处理。
            new_manifest.update({name: file_hashes[name] for name in chunks_by_source})

            # 所有待更新文档先解析完，再触碰索引；解析失败不会先删掉旧片段。
            # 数据库、缓存和清单分开写入，可能只有部分成功；标记让查询发现未完成的更新。
            if progress is not None:
                progress.start_phase("写入索引")
            if incremental:
                # 存储组件可在写入前检查集合格式；检查失败时不创建标记，也不删除旧片段。
                validate_update = getattr(self.vector_store, "validate_update", None)
                if validate_update is not None:
                    validate_update()
            # begin_write 落下"更新中"标记，中途失败时查询侧能看出索引和缓存可能不一致。
            self.chunk_store.begin_write()
            if incremental:
                # 增量：先按来源清掉旧片段（含已删除文件），再写入本次解析结果。
                self.vector_store.delete_sources(sorted(sources_to_replace))
                self.vector_store.add(new_chunks)
            else:
                # 全量：用本次结果整体替换，不保留任何旧片段。
                self.vector_store.replace_all(new_chunks)
            if progress is not None:
                progress.start_phase("保存缓存")
            if save:
                # publish 写入片段缓存与清单，并清除"更新中"标记。
                self.chunk_store.publish(merged_chunks, new_manifest)
            else:
                # 不保存时宁可丢弃缓存：本地缓存与向量库不一致比下次重建更危险。
                self.chunk_store.discard_cache()
            return BuildResult(
                vector_store=self.vector_store,
                chunks=new_chunks,
                processed=list(chunks_by_source),
                removed=removed_sources,
                failed=failed_sources,
            )
        finally:
            # 中途抛异常时也要把进度条擦掉，否则报错信息会叠在进度条上。
            if progress is not None:
                progress.close()
