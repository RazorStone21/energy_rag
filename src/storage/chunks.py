"""读写 Chunk 缓存和文件哈希清单，并用 .pending 文件标记尚未完成的写入。"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path

# 清单里除“文件名 → 哈希”外的保留键，记录产出这份索引的解析链路指纹。
MANIFEST_PARSER_KEY = "__parser__"


def atomic_write(path, data):
    """先把数据写到同目录的临时文件，写完后再替换目标文件，避免直接覆盖到一半。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ChunkStore:
    def __init__(self, chunks_path, manifest_path):
        """保存缓存和清单的路径，并把清单后缀改为 .pending 作为写入标记路径。"""
        self.chunks_path = Path(chunks_path)
        self.manifest_path = Path(manifest_path)
        self.pending_path = self.manifest_path.with_suffix(".pending")

    def assert_ready(self):
        """检查是否存在 .pending 文件；存在时停止读取，因为上次写入可能还没完成。"""
        if self.pending_path.exists():
            raise RuntimeError(
                "索引上次写入未完成或正在构建，请在构建结束后查询；"
                "若构建失败，请修复错误并使用同一配置执行 python main.py build 全量重建。"
            )

    def revision(self):
        """返回缓存文件的路径、时间和大小等信息，供 BM25 判断是否需要重新建索引。"""
        self.assert_ready()
        try:
            stat = self.chunks_path.stat()
        except FileNotFoundError:
            return None
        return (
            str(self.chunks_path.resolve()),
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
            stat.st_ino,
        )

    def load_chunks(self):
        """检查索引状态后读取并校验片段列表，缓存不存在时返回空列表。"""
        self.assert_ready()
        if not self.chunks_path.exists():
            return []
        with self.chunks_path.open("rb") as stream:
            chunks = pickle.load(stream)
        if not isinstance(chunks, list) or any(
            not isinstance(getattr(c, "page_content", None), str)
            or not isinstance(getattr(c, "metadata", None), dict)
            for c in chunks
        ):
            raise ValueError("Invalid chunks cache; rebuild the index")
        return chunks

    def load_manifest(self):
        """读取并检查“文件名 → 文件哈希”清单，文件不存在时返回空字典。

        清单里还有一条保留项记录解析链路指纹（见 parser_fingerprint），它不是文件，
        因此不在这里返回，否则会被当成"磁盘上已删除的来源"。
        """
        self.assert_ready()
        if not self.manifest_path.exists():
            return {}
        result = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in result.items()
        ):
            raise ValueError("Invalid build manifest")
        return {name: digest for name, digest in result.items() if name != MANIFEST_PARSER_KEY}

    def parser_fingerprint(self) -> str:
        """读取上次构建使用的解析链路指纹；没有清单或旧清单没有该字段时返回空串。

        空串表示"这套索引来路不明"，与当前指纹比对必然不等，从而触发一次全量重建。
        """
        self.assert_ready()
        if not self.manifest_path.exists():
            return ""
        try:
            result = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        if not isinstance(result, dict):
            return ""
        value = result.get(MANIFEST_PARSER_KEY)
        return value if isinstance(value, str) else ""

    def begin_write(self):
        """创建 .pending 文件，表示本次索引更新尚未完成。"""
        # 标记只用于发现未完成的写入，不能阻止多个进程同时写入，也不能撤销数据库修改。
        atomic_write(self.pending_path, b"index update in progress\n")

    def publish(self, chunks, manifest, parser_fingerprint=""):
        """把片段和文件哈希清单保存到磁盘，两个文件都写完后再删除 .pending 标记。

        parser_fingerprint 一并写进清单：下次增量构建据此判断索引是否由同一套
        解析链路产出，不是的话必须全量重建。
        """
        chunk_bytes = pickle.dumps(list(chunks))
        stored = dict(manifest)
        stored[MANIFEST_PARSER_KEY] = parser_fingerprint
        manifest_text = json.dumps(
            stored,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        manifest_bytes = manifest_text.encode("utf-8")
        # 先准备好两个文件的内容再写入；即使第一个已替换，第二个失败时仍保留 .pending。
        atomic_write(self.chunks_path, chunk_bytes)
        atomic_write(self.manifest_path, manifest_bytes)
        # 任一步写入失败都会保留标记，提醒下次读取前先全量重建。
        self.pending_path.unlink(missing_ok=True)

    def discard_cache(self):
        """删除旧缓存、文件哈希清单和写入标记，避免 BM25 继续使用过期片段。"""
        self.chunks_path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        self.pending_path.unlink(missing_ok=True)
