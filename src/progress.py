"""在终端里显示入库进度：整体一条进度条，下一行是当前文件的实时状态。

只依赖标准库，流程层因此不必引入进度条库。进度写 stderr，
这样把 stdout 重定向到文件时不会混进结果；输出不是终端时自动退化为
每个文件一行的普通日志，不刷满屏的控制符。

单文件解析动辄几分钟（图表要逐张调用视觉模型），所以除了文件级进度，
还需要一行实时状态，否则界面会长时间一动不动，看起来像卡住了。
"""

from __future__ import annotations

import shutil
import sys
import time

# 进度条主体宽度；终端太窄时会再压缩。
_BAR_WIDTH = 24
_MIN_BAR_WIDTH = 10
# 状态行宽度上限，超出就截断，避免自动换行把已经画好的行顶乱。
_STATUS_WIDTH = 96


def format_duration(seconds: float) -> str:
    """把秒数格式化成 分:秒；超过一小时显示 时:分:秒。"""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class BuildProgress:
    """收集入库过程中的进度并刷新终端显示。

    所有方法在非终端环境下也能安全调用，只是退化为按行输出。
    """

    def __init__(self, total: int, stream=None):
        """total 是本次要处理的文件总数；默认写到 stderr。"""
        self.total = max(1, total)
        self.stream = stream if stream is not None else sys.stderr
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started_at = time.perf_counter()
        self.file_started_at = self.started_at
        self.done = 0
        self.phase = "解析文件"
        self.current = ""
        self.status = ""
        self._drawn_lines = 0

    # ---------------- 对外接口 ----------------

    def start_file(self, index: int, name: str) -> None:
        """标记开始处理第 index 个（从 1 开始）文件。"""
        self.file_started_at = time.perf_counter()
        self.current = name
        self.status = ""
        self._render()

    def note(self, text: str) -> None:
        """更新当前文件的实时状态，例如「正在描述第 3 张图表」。"""
        self.status = text
        self._render()

    def finish_file(self, name: str, chunk_count: int) -> None:
        """标记一个文件处理完成；非终端下在这里输出一行可读的日志。"""
        self.done += 1
        elapsed = time.perf_counter() - self.file_started_at
        if not self.interactive:
            self.stream.write(
                f"[{self.done}/{self.total}] {name}  {chunk_count} 个片段  "
                f"{format_duration(elapsed)}\n"
            )
            self.stream.flush()
        self.current = ""
        self.status = ""
        self._render()

    def fail_file(self, name: str, reason: str) -> None:
        """标记一个文件处理失败；失败信息在进度条之外单独成行，方便回看。"""
        self.done += 1
        self._clear()
        self.stream.write(f"[{self.done}/{self.total}] 失败：{name} —— {reason}\n")
        self.stream.flush()

    def start_phase(self, name: str) -> None:
        """切换到下一个阶段；此后的进度不再按文件计数。"""
        self.phase = name
        self.current = ""
        self.status = ""
        self._render()

    def close(self) -> None:
        """结束显示，把光标留在进度条下方。"""
        self._clear()

    # ---------------- 渲染 ----------------

    def _line(self, text: str, width: int) -> str:
        """按终端宽度截断一行，避免自动换行破坏已经画好的多行显示。"""
        if len(text) <= width:
            return text
        return text[: max(0, width - 1)] + "…"

    def _bar(self, width: int) -> str:
        """按完成比例画出进度条主体。"""
        filled = int(width * self.done / self.total)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def _lines(self) -> tuple[str, str]:
        """构造要显示的两行：整体进度和当前文件状态。"""
        width = shutil.get_terminal_size(fallback=(100, 24)).columns
        bar_width = max(_MIN_BAR_WIDTH, min(_BAR_WIDTH, width - 44))
        elapsed = time.perf_counter() - self.started_at
        # 已完成的文件数为 0 时无法估算剩余，先不显示，避免出现除零或荒唐的数字。
        eta = ""
        if self.done:
            remaining = elapsed / self.done * (self.total - self.done)
            eta = f"  剩余约 {format_duration(remaining)}"
        head = f"{self._bar(bar_width)} {self.done}/{self.total}  {self.phase}"
        head += f"  已用 {format_duration(elapsed)}{eta}"
        detail = f"  ↳ {self.current}" if self.current else ""
        if self.status:
            detail += f"    {self.status}" if detail else f"  {self.status}"
        return self._line(head, width), self._line(detail, width)

    def _clear(self) -> None:
        """把已画出的进度条擦掉，让后续输出从干净的一行开始。"""
        if not self.interactive or not self._drawn_lines:
            return
        self.stream.write(f"\033[{self._drawn_lines}A")
        for _ in range(self._drawn_lines):
            self.stream.write("\033[2K\n")
        self.stream.write(f"\033[{self._drawn_lines}A")
        self.stream.flush()
        self._drawn_lines = 0

    def _render(self) -> None:
        """重画两行进度；非终端环境下什么都不做。"""
        if not self.interactive:
            return
        first, second = self._lines()
        if self._drawn_lines:
            self.stream.write(f"\033[{self._drawn_lines}A")
        self.stream.write("\033[2K" + first + "\n")
        self.stream.write("\033[2K" + second + "\n")
        self._drawn_lines = 2
        self.stream.flush()
