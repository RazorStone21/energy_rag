"""按文件持久化切好的片段，让全量构建可以断点续跑、出错只重解析出错的文件。"""

from __future__ import annotations

import pickle
from pathlib import Path

from .chunks import atomic_write


class ParseCache:
    """把每份文件解析+切分后的片段，按「文件哈希 + 解析指纹」缓存到磁盘。

    与 chunks.pkl / build_manifest.json 不同，这里在每份文件解析成功后**立即**落盘，
    而不是等索引全部写完。因此全量构建即使在最后写索引（replace_all）时失败，
    已解析文件的片段也还留在缓存里，下次构建直接复用，不必从头重解析一遍。
    指纹不一致（换了解析器、切分逻辑或嵌入模型）时缓存失效，强制重新解析。
    """

    def __init__(self, directory: Path):
        """保存缓存目录；目录不存在时由首次写入创建。"""
        self.directory = Path(directory)

    def _path(self, name: str) -> Path:
        """返回某个来源名对应的缓存文件路径；文件名直接做定位，内容里另存校验信息。"""
        return self.directory / (name + ".pkl")

    def get(self, name: str, digest: str, fingerprint: str):
        """命中且校验通过时返回片段列表；哈希或指纹不符、或缓存损坏时返回 None。"""
        path = self._path(name)
        if not path.exists():
            return None
        try:
            with path.open("rb") as stream:
                record = pickle.load(stream)
            if not isinstance(record, dict):
                return None
            if record.get("digest") != digest or record.get("fingerprint") != fingerprint:
                return None
            chunks = record.get("chunks")
            if not isinstance(chunks, list):
                return None
            # 片段必须是带正文和元数据的对象，缺任一字段的缓存视为损坏，重新解析。
            if any(
                not isinstance(getattr(c, "page_content", None), str)
                or not isinstance(getattr(c, "metadata", None), dict)
                for c in chunks
            ):
                return None
            return chunks
        except (OSError, pickle.PickleError, EOFError, ValueError, AttributeError):
            return None

    def put(self, name: str, digest: str, fingerprint: str, chunks) -> None:
        """把某份文件解析+切分后的片段连同校验信息写入缓存。"""
        record = {
            "digest": digest,
            "fingerprint": fingerprint,
            "chunks": list(chunks),
        }
        atomic_write(self._path(name), pickle.dumps(record))

    def clear(self) -> None:
        """清空缓存目录下所有片段，仅在确需强制全量重新解析时使用。"""
        for path in self.directory.glob("*.pkl"):
            path.unlink(missing_ok=True)
