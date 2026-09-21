"""验证入库进度的显示与降级行为；用内存流替代终端，不依赖真实终端。"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.ingestion import IngestionPipeline
from src.progress import BuildProgress, format_duration
from src.schemas import ParseResult
from src.storage.chunks import ChunkStore


class FakeTty(io.StringIO):
    """既是可读写的内存流，又自称是终端，用来走交互式渲染分支。"""

    def isatty(self) -> bool:
        """让进度条按终端渲染，走交互式分支。"""
        return True


class Recording:
    """记录收到的每一次上报，替代真实进度条。"""

    def __init__(self, total):
        """保存文件总数并准备记录事件。"""
        self.total = total
        self.events = []

    def start_file(self, index, name):
        """记录一个文件开始处理。"""
        self.events.append(("start", index, name))

    def note(self, text):
        """记录一条附加说明。"""
        self.events.append(("note", text))

    def finish_file(self, name, count):
        """记录一个文件处理完成及其片段数。"""
        self.events.append(("finish", name, count))

    def fail_file(self, name, reason):
        """记录一个文件处理失败及原因。"""
        self.events.append(("fail", name, reason))

    def start_phase(self, name):
        """记录进入新的处理阶段。"""
        self.events.append(("phase", name))

    def close(self):
        """记录进度显示结束。"""
        self.events.append(("close",))


def test_format_duration_switches_to_hours_only_when_needed():
    """不足一小时只显示分秒，超过才带上小时，避免出现 0:05:07 这种冗余。"""
    assert format_duration(0) == "00:00"
    assert format_duration(65) == "01:05"
    assert format_duration(3599) == "59:59"
    assert format_duration(3600) == "1:00:00"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(-5) == "00:00"


def test_non_interactive_output_is_one_line_per_file():
    """输出被重定向时不能刷控制符，改成每个文件一行普通日志。"""
    stream = io.StringIO()
    progress = BuildProgress(3, stream=stream)

    progress.start_file(1, "a.pdf")
    progress.note("正在描述第 2 张图表")
    # 这是「正在处理」，不该产生输出；只有结束和失败才各占一行。
    assert stream.getvalue() == ""

    progress.finish_file("a.pdf", 12)
    progress.start_file(2, "b.pdf")
    progress.fail_file("b.pdf", "解析失败")

    text = stream.getvalue()
    assert text == "[1/3] a.pdf  12 个片段  00:00\n[2/3] 失败：b.pdf —— 解析失败\n"
    assert "\033[" not in text


def test_interactive_render_draws_bar_and_clears_before_finishing():
    """终端下画进度条，结束时把进度条擦掉，后面的输出才不会被顶乱。"""
    stream = FakeTty()
    progress = BuildProgress(4, stream=stream)
    progress.start_file(1, "a.pdf")
    progress.note("正在描述第 3 张图表")

    drawn = stream.getvalue()
    assert "0/4" in drawn
    assert "a.pdf" in drawn and "正在描述第 3 张图表" in drawn
    # 一个文件都还没完成，进度条应该整体是空的。
    assert "░░" in drawn and "██" not in drawn

    progress.close()
    assert "\033[2K" in stream.getvalue()  # 用过清行控制符把画出的内容擦掉


def test_bar_fills_in_proportion_to_completed_files():
    """进度条长度跟着完成比例走，全部完成时没有空格。"""
    stream = FakeTty()
    progress = BuildProgress(2, stream=stream)
    progress.finish_file("a.pdf", 1)
    halfway, _ = progress._lines()
    assert halfway.count("█") > 0 and halfway.count("░") > 0

    progress.finish_file("b.pdf", 1)
    done, _ = progress._lines()
    assert done.count("░") == 0
    assert "2/2" in done


def test_eta_is_hidden_until_the_first_file_finishes():
    """还没有文件完成时估不出剩余时间，此时不能显示剩余时长或算出荒唐的数字。"""
    stream = FakeTty()
    progress = BuildProgress(10, stream=stream)
    progress.start_file(1, "a.pdf")
    head, _ = progress._lines()
    assert "剩余" not in head

    progress.finish_file("a.pdf", 1)
    head, _ = progress._lines()
    assert "剩余约" in head


def test_ingestion_reports_each_file_and_phase(tmp_path):
    """入库流程按文件上报解析进度，并在进入写索引、保存缓存前切换阶段。"""
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.pdf").write_bytes(b"a1")
    store = ChunkStore(tmp_path / "state" / "chunks.pkl", tmp_path / "state" / "manifest.json")

    parser = Mock()

    def parse(path, on_status=None):
        """模拟解析器逐张图表上报进度，并返回一条可入库的正文。"""
        on_status("正在描述第 1 张图表")
        return ParseResult(
            texts=[
                SimpleNamespace(
                    page_content="能源政策正文内容足够长",
                    metadata={"source": path.name, "page": 1, "type": "text"},
                )
            ]
        )

    parser.parse.side_effect = parse
    chunker = Mock()
    chunker.split.side_effect = lambda parsed: parsed.texts

    pipeline = IngestionPipeline(parser, chunker, Mock(), store, directory)
    recorder = Recording(1)
    result = pipeline.build(progress_factory=lambda total: recorder)

    assert result.processed == ["a.pdf"]
    assert recorder.events == [
        ("start", 1, "a.pdf"),
        ("note", "正在描述第 1 张图表"),
        ("finish", "a.pdf", 1),
        ("phase", "写入索引"),
        ("phase", "保存缓存"),
        ("close",),
    ]


def test_ingestion_closes_progress_even_when_the_build_fails(tmp_path):
    """解析失败时也要收尾，否则报错信息会叠在没擦掉的进度条上。"""
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.pdf").write_bytes(b"a1")
    store = ChunkStore(tmp_path / "state" / "chunks.pkl", tmp_path / "state" / "manifest.json")

    parser = Mock()
    parser.parse.side_effect = lambda path, on_status=None: ParseResult(errors=["OCR 不可用"])

    pipeline = IngestionPipeline(parser, Mock(), Mock(), store, directory)
    recorder = Recording(1)
    with pytest.raises(RuntimeError, match="Full build aborted"):
        pipeline.build(progress_factory=lambda total: recorder)

    assert recorder.events[-1] == ("close",)
    assert ("fail", "a.pdf", "OCR 不可用") in recorder.events
