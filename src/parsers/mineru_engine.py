"""MinerU 调用边界：进程内解析一份 PDF，返回中性的版面块序列。

MinerU 的模型在它自己的单例里缓存，一个进程只加载一次；解析结果留在内存，
不写任何中间目录，因此不需要把输出目录和文档名对齐。图表裁剪图以 data URL
的形式随内容一起返回，直接解码成 PIL.Image 交给视觉模型。

调用方必须是真实脚本（`main.py` 满足）：MinerU 内部用 spawn 进程池加载 PDF 图片，
子进程会按路径重新导入 `__main__`，从 stdin 或交互式解释器调用会直接崩在
`FileNotFoundError: <stdin>`。
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 每页重复、或只是导航的块：进索引只会污染检索，直接丢弃。
SKIPPED_TYPES = frozenset(
    {
        "header",
        "footer",
        "page_number",
        "page_footnote",
        "page_aside_text",
        "aside_text",
        "index",
    }
)

# 交给视觉模型描述的块类型。
FIGURE_TYPES = frozenset({"image", "chart"})


class MineruUnavailable(RuntimeError):
    """MinerU 未安装或模型未就绪：整套 PDF 解析都用不了，不是单份文件的问题。"""


class MineruParseError(RuntimeError):
    """单份文档解析失败。"""


@dataclass
class MineruBlock:
    """一个版面块；各字段按块类型填写。

    kind 是中性取值，不直接用 MinerU 的类型名：text（含标题）、table、figure、
    code、list。标题另外带 level，正文带 heading_path 由转换层维护。
    """

    kind: str
    page: int
    index_on_page: int
    bbox: tuple[float, float, float, float] | None
    text: str = ""
    level: int | None = None
    caption: str = ""
    image: object | None = None
    table_html: str = ""


_warp_patched = False


def _patch_mineru_warp() -> None:
    """修复 MinerU 对退化检测框的裁图崩溃（不改 site-packages）。

    MinerU 的 OCR 检测偶尔会给出宽或高为 0 的退化四边形（四个角点重合或共线），
    此时 ``mineru.model.ocr.image.get_rotate_crop_image`` 与印章矫正里的
    ``seal_det_warp.Homography`` 会把 (0,0) 或负尺寸传给 docvortex 的
    ``warp_image``，后者直接抛 "Unsupported warp settings"，整份 PDF 解析失败
    （0127 那份 49MB 的 mee 报告即因此挂掉）。

    这里在项目自己的代码里各包一层：文字识别裁图遇退化框返回空裁剪，调用方
    ``rotate_vertical_crop_if_needed`` 对空裁剪原样返回；印章矫正把展开尺寸钳到
    1 像素。两者都不会中断整篇解析，丢掉的只是退化框（原本就没有可读内容）。
    """
    global _warp_patched
    if _warp_patched:
        return
    import numpy as np
    from mineru.model.ocr import image as ocr_image
    from mineru.model.ocr import seal_det_warp

    # 1) 文字识别裁图：退化四边形（宽或高为 0）返回空裁剪。
    original_crop = ocr_image.get_rotate_crop_image

    def safe_crop(img, points):
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
        width = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
        height = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
        if width <= 0 or height <= 0:
            return img[:0, :0].copy()
        return original_crop(img, points)

    ocr_image.get_rotate_crop_image = safe_crop

    # 2) 印章矫正：展开尺寸不足 1 像素时钳到 1。
    original_homo = seal_det_warp.Homography

    def safe_homography(
        image,
        img_points,
        world_width,
        world_height,
        interpolation=None,
        ratio_width=1.0,
        ratio_height=1.0,
    ):
        world_width = max(1.0, float(np.nan_to_num(world_width)))
        world_height = max(1.0, float(np.nan_to_num(world_height)))
        return original_homo(
            image,
            img_points,
            world_width,
            world_height,
            interpolation=interpolation,
            ratio_width=ratio_width,
            ratio_height=ratio_height,
        )

    seal_det_warp.Homography = safe_homography
    _warp_patched = True


def decode_image(data_url: str):
    """把 MinerU 的图片字段解码成 PIL.Image；不是 data URL 或解码失败时返回 None。

    版面块里的 `img_path` 在进程内解析时是内联的 data URL（`data:image/jpeg;base64,…`），
    图片可能很大，因此解码失败只跳过这一张图，不拖垮整份文档。
    """
    if not isinstance(data_url, str) or ";base64," not in data_url:
        return None
    try:
        payload = base64.b64decode(data_url.split(";base64,", 1)[1])
        from PIL import Image

        image = Image.open(io.BytesIO(payload))
        # 必须先读进内存：MinerU 的字节流随解析结果存活，惰性图片会在后续访问时失效。
        image.load()
        return image
    except (binascii.Error, ValueError, OSError) as exc:
        logger.debug("图表解码失败，跳过该图：%s", exc)
        return None


class MineruEngine:
    """懒加载 MinerU，解析整份文档，并按文件指纹复用上一次结果。

    text / table / figure 三个消费者共用同一次解析：MinerU 是"跑一次出全部产物"，
    按类型分别调用会把同一份 PDF 解析三遍。
    """

    def __init__(self, settings):
        """保存 [mineru] 配置；此时不加载模型、不碰显存。"""
        self.settings = settings
        self._cached_key = None
        self._cached_items = None
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        """连续解析失败的文件数；达到上限时调用方应中止构建而不是逐份失败。"""
        return self._consecutive_failures

    def available(self) -> bool:
        """只检查依赖是否装好，不加载权重。

        环境变量必须先设好再 import：MinerU 在 import 时读 MINERU_HOME 并缓存模型根目录，
        晚一步设置就会把模型目录锁到默认的 ~/.mineru，之后所有就绪检查都找不到模型。
        """
        self._prepare_environment()
        try:
            import mineru  # noqa: F401
        except Exception as exc:  # pragma: no cover - 环境缺失时的分支
            logger.warning("MinerU 不可用：%s", exc)
            return False
        return True

    def _prepare_environment(self) -> None:
        """在 import mineru 之前把模型目录与来源落到环境变量上。

        MinerU 在 import 时读这些值，晚设无效；用 setdefault 保证命令行或环境里
        已有的设置优先。
        """
        os.environ.setdefault("MINERU_HOME", str(self.settings.home))
        os.environ.setdefault("MINERU_MODEL_SOURCE", self.settings.model_source)
        # MinerU 用 loguru 直接写 stderr，压到 WARNING 以免刷屏。
        os.environ.setdefault("LOGURU_LEVEL", "WARNING")

    def required_repos(self):
        """返回当前档位需要的模型仓库列表（basic 档只有版面/OCR/表格那一组）。"""
        self._prepare_environment()
        from mineru.model.registry import model_repos_for_tier

        return model_repos_for_tier(self.settings.tier)

    def preflight(self) -> None:
        """检查模型是否就绪，缺失时给出可执行的修复命令。"""
        if not self.available():
            raise MineruUnavailable(
                "未安装 MinerU：请先安装 mineru[torch]（见 pyproject.toml 的 runtime 依赖）"
            )
        from mineru.model.download import verify_model_repo

        missing = [
            repo.local_name or repo.name
            for repo in self.required_repos()
            if not verify_model_repo(repo).ready
        ]
        if missing:
            raise MineruUnavailable(
                "MinerU 模型未就绪：%s。运行 python -m scripts.download_models --models mineru"
                % "、".join(missing)
            )

    def download_models(self, on_status=None) -> None:
        """下载当前档位需要的模型，供 scripts/download_models.py 调用。"""
        self._prepare_environment()
        from mineru.model.download import download_model_repo, verify_model_repo

        for repo in self.required_repos():
            name = repo.local_name or repo.name
            if verify_model_repo(repo).ready:
                if on_status is not None:
                    on_status(f"MinerU 模型 {name} 已存在")
                continue
            if on_status is not None:
                on_status(f"下载 MinerU 模型 {name}")
            download_model_repo(repo, source=self.settings.model_source)

    def parse_items(self, path: Path) -> list[dict]:
        """解析一份 PDF，返回 MinerU 的内容项列表（content_list）。

        同一文件（路径 + mtime + 大小一致）重复调用直接复用，不再跑第二遍。
        单份失败抛 MineruParseError，由调用方决定报错还是中止整轮构建。
        """
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key == self._cached_key:
            return self._cached_items

        self.preflight()
        import mineru
        from mineru.render import render_content_list

        _patch_mineru_warp()

        try:
            # image_analysis=False：不做 MinerU 自带的图片分析，图表由项目自己的
            # 视觉模型描述，口径与既有评测保持一致。
            result = mineru.parse(
                str(path),
                tier=self.settings.tier,
                page_range="all",
                image_analysis=False,
            )
            items = render_content_list(result.middle_json)
        except Exception as exc:  # noqa: BLE001 - 统一包装成单文件级错误
            self._consecutive_failures += 1
            raise MineruParseError(f"{path.name}: {exc}") from exc

        self._consecutive_failures = 0
        self._cached_key, self._cached_items = key, items
        return items

    def release(self) -> None:
        """丢弃缓存并归还 MinerU 占用的显存，供入库结束后调用。"""
        self._cached_key = None
        self._cached_items = None
        self._cached_parser_release()
        _empty_cuda_cache()

    @staticmethod
    def _cached_parser_release() -> None:
        """调用 MinerU 自己的模型释放钩子；接口不存在时静默跳过。"""
        import sys

        for module_name, attribute in (
            ("mineru.model.runtime", "clean_memory"),
            ("mineru.model.runtime.memory", "clean_memory"),
        ):
            module = sys.modules.get(module_name)
            clean = getattr(module, attribute, None) if module else None
            if callable(clean):
                try:
                    clean("cuda")
                except Exception:  # pragma: no cover - 释放失败不影响主流程
                    logger.debug("MinerU 显存清理失败", exc_info=True)
                return


def _empty_cuda_cache() -> None:
    """回收 Python 垃圾并让 PyTorch 归还缓存显存；未加载 torch 时什么也不做。"""
    import gc
    import sys

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
