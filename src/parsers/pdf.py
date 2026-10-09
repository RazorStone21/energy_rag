"""用 MinerU 解析 PDF：正文、表格与图表位置都来自它的版面分析。

MinerU 一次调用就给出整份文档的版面块（正文、表格、图表各自的 bbox 与裁剪图），
因此这里不再按"正文/表格/图片"分别提取——那会把同一份 PDF 解析三遍。图表描述
交给项目自己的视觉模型，见 mineru_pdf。
"""

from __future__ import annotations

from pathlib import Path

from ..schemas import ParseResult
from . import mineru_pdf
from .mineru_engine import MineruParseError, MineruUnavailable


class PDFParser:
    """把 PDF 交给 MinerU，转成正文、表格与图表片段。"""

    def __init__(
        self,
        vision,
        vision_settings,
        mineru,
        mineru_settings,
        document_factory=None,
    ):
        """保存视觉模型、MinerU 引擎与各自的配置；此处不加载任何模型。"""
        self.vision = vision
        self.vision_settings = vision_settings
        self.mineru = mineru
        self.mineru_settings = mineru_settings
        self.document_factory = document_factory

    @property
    def fingerprint(self) -> str:
        """PDF 解析链路的版本：换档位会改变版面块，因此要写进指纹。"""
        return f"mineru-{self.mineru_settings.tier}"

    def parse(self, path, on_status=None) -> ParseResult:
        """解析一份 PDF；单份失败写进 errors，环境不可用则抛出。

        on_status 是可选函数，接收一句描述当前动作的文字，用于显示解析进度。
        单份文件的异常不写 errors 之外的通道：入库流程据此保留该文件的旧索引，
        与"整份文件读不出来"的既有语义一致。MineruUnavailable 是个例外——
        它表示模型缺失或连续失败，是环境问题而不是文档问题，必须让整轮构建
        停下来，否则会逐份失败到最后一无所获。
        """
        path = Path(path)
        result = ParseResult()
        if not self.mineru_settings.enabled:
            result.errors.append("mineru: 解析已在配置中关闭（[mineru].enabled = false）")
            return result
        if self.mineru.consecutive_failures >= self.mineru_settings.max_consecutive_failures:
            raise MineruUnavailable(
                f"连续 {self.mineru.consecutive_failures} 份文件解析失败，"
                "已中止本轮构建；修复环境后重跑（已成功的文件会按指纹跳过）"
            )
        try:
            if on_status is not None:
                on_status("MinerU 解析版面")
            items = self.mineru.parse_items(path)
        except MineruParseError as exc:
            result.errors.append(f"mineru: {exc}")
            return result
        if not items:
            result.errors.append("mineru: 没有解析出任何版面块")
            return result
        if on_status is not None:
            on_status("整理正文、表格与图表")
        return mineru_pdf.blocks_to_parse_result(
            items,
            path.name,
            self.vision,
            self.vision_settings,
            self.mineru_settings,
            document_factory=self.document_factory,
            on_status=on_status,
        )

    def release(self):
        """归还 MinerU 占用的显存，供入库结束后调用。"""
        self.mineru.release()
