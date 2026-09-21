"""把会话记录与长期记忆保存成 markdown 文件，供问答时读取。

文件分两种：memory.md 是用户可以手写的长期记忆，sessions/<id>.md 是每个会话一份的问答记录。
两者都是普通 markdown，既能直接打开看，也能被解析回消息列表。
本模块只依赖标准库，不加载模型，也不连接向量库。
"""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunks import atomic_write

MEMORY_FILENAME = "memory.md"
SESSIONS_DIRNAME = "sessions"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# 读长期记忆时的绝对上限，避免误放一个很大的文件后每轮都全量读取。
# 真正的注入上限由 config 的 memory.max_chars 决定，在 context_builder 里截断。
MAX_MEMORY_READ = 200_000

# 会话 id 由服务端生成，形状固定。这里用允许列表而不是黑名单：
# 全角字符、同形字数字、控制字符都不在字符集里，也就拼不出 ../ 或绝对路径。
SESSION_ID = re.compile(r"\A[0-9]{8}-[0-9]{6}-[0-9a-z]{6}\Z")

# 消息边界与来源标题。转义规则见 escape_line：它比这两个正则更宽（所有以 # 开头的行都被转义），
# 因此 BOUNDARY 将来放宽也不会漏转义；真正要守住的不变式是 escape_line 与 unescape_line 互为逆运算。
BOUNDARY = re.compile(r"^ {0,3}##[ \t]*(用户|助手)[ \t]*#*[ \t]*$")
SOURCES = re.compile(r"^ {0,3}###[ \t]*参考来源[ \t]*#*[ \t]*$")

ROLE_LABELS = {"user": "用户", "assistant": "助手"}
LABEL_ROLES = {label: role for role, label in ROLE_LABELS.items()}


def new_session_id(now: float | None = None) -> str:
    """生成会话 id：本地时间到秒，加六位随机后缀，避免同一秒内两次新建重名。"""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return f"{stamp}-{secrets.token_hex(3)}"


def escape_line(line: str) -> str:
    """写入前转义内容行：行首的反斜杠加倍，标题样式的行加一个反斜杠。

    反斜杠本身也要处理：正文里真的出现 `\\## 助手` 时，只按“是否匹配边界”判断会漏转义，
    读回时会被错误地还原成边界样式。
    """
    stripped = line.lstrip(" ")
    if not stripped.startswith(("\\", "#")):
        return line
    offset = len(line) - len(stripped)
    return f"{line[:offset]}\\{line[offset:]}"


def unescape_line(line: str) -> str:
    """还原 escape_line 加上的那一个反斜杠；不是转义形式时原样返回。"""
    stripped = line.lstrip(" ")
    if len(stripped) > 1 and stripped[0] == "\\" and stripped[1] in "\\#":
        offset = len(line) - len(stripped)
        return line[:offset] + stripped[1:]
    return line


def escape_content(text: str) -> str:
    """把整段正文按行转义，保证内容不会被解析成消息边界。

    写入前统一换成 \\n：文件里不出现 \\r，读取时的换行归一化就不会改动正文。
    """
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(escape_line(line) for line in normalized.split("\n"))


def unescape_content(text: str) -> str:
    """把整段正文按行还原，与 escape_content 互为逆操作。"""
    return "\n".join(unescape_line(line) for line in text.split("\n"))


def format_timestamp(milliseconds: int | None) -> str:
    """把毫秒时间戳格式化成会话文件里使用的时间文本，缺省时返回空串。"""
    if not milliseconds:
        return ""
    return time.strftime(TIMESTAMP_FORMAT, time.localtime(milliseconds / 1000))


def parse_timestamp(text: str) -> int | None:
    """解析会话文件里的时间文本；格式不对时返回 None，交给调用方回退到文件时间。"""
    try:
        parsed = time.strptime(text.strip(), TIMESTAMP_FORMAT)
    except (ValueError, AttributeError):
        return None
    return int(time.mktime(parsed) * 1000)


@dataclass
class Turn:
    """会话里的一条消息：角色、正文，以及助手回答对应的参考来源标签。"""

    role: str
    content: str
    sources: list[str] = field(default_factory=list)


@dataclass
class Session:
    """一份完整的会话记录；parse_error 表示文件存在但格式异常。"""

    id: str
    title: str
    created_at_ms: int | None
    updated_at_ms: int
    turns: list[Turn] = field(default_factory=list)
    parse_error: bool = False


@dataclass
class SessionSummary:
    """会话列表用的摘要，不含正文，避免列表接口读取全部记录。"""

    id: str
    title: str
    created_at_ms: int | None
    updated_at_ms: int
    message_count: int
    parse_error: bool = False


def sanitize_label(label) -> str:
    """把来源标签压成单行并去掉反引号，保证它不会撑破 markdown 列表项。"""
    return " ".join(str(label).split()).replace("`", "'")


def session_title(title) -> str:
    """把会话标题压成单行，避免标题里的换行伪造出消息边界。

    标题取自用户问题的前若干字，而提问框允许换行：不处理的话，`# 标题` 之后的
    内容会被 parse_session 解析成一条真实的助手消息，既写进会话记录，也会作为
    历史注入下一轮提示词。str.split() 按所有空白切分，U+2028 这类行分隔符同样会被去掉。
    """
    return " ".join(str(title or "").split())


def render_session(title: str, turns, created_at_ms: int | None = None) -> str:
    """把标题和消息渲染成会话 markdown，消息正文按边界规则转义。"""
    lines = [f"# {session_title(title) or '新对话'}", ""]
    created = format_timestamp(created_at_ms)
    if created:
        lines += [f"- 创建时间：{created}", ""]
    for turn in turns:
        label = ROLE_LABELS.get(turn.role, ROLE_LABELS["user"])
        lines += [f"## {label}", "", escape_content(turn.content), ""]
        # 只记位置标签，不记片段正文：正文是检索到的原文，写进历史后下一轮会被当成
        # 助手说过的话喂回模型，既浪费上下文，也混淆了文档与对话。
        if turn.role == "assistant" and turn.sources:
            lines += ["### 参考来源", ""]
            for index, label_text in enumerate(turn.sources, start=1):
                lines.append(f"{index}. {sanitize_label(label_text)}")
            lines.append("")
    return "\n".join(lines)


def parse_session(text: str, session_id: str, updated_at_ms: int) -> Session:
    """解析会话 markdown，无法识别的行按正文处理，任何输入都不抛异常。

    文件名是会话 id 的唯一权威，文件头里的时间只用于显示；解析不出就回退到文件时间。
    """
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    title = ""
    created_at_ms = None
    turns: list[Turn] = []
    role = None
    buffer: list[str] = []

    def flush():
        """收束当前消息：拆出末尾的参考来源块，其余作为正文。"""
        if role is None:
            return
        body = "\n".join(buffer).strip("\n")
        sources: list[str] = []
        if role == "assistant":
            # 转义规则保证了正文里不会出现未被转义的来源标题，
            # 所以最后一个未转义的「### 参考来源」一定是文件自己写的那个。
            cut = next(
                (index for index in range(len(buffer) - 1, -1, -1) if SOURCES.match(buffer[index])),
                None,
            )
            if cut is not None:
                sources = [
                    re.sub(r"^ {0,3}\d+\.\s*", "", line).strip()
                    for line in buffer[cut + 1 :]
                    if line.strip()
                ]
                body = "\n".join(buffer[:cut]).strip("\n")
        turns.append(Turn(role=role, content=unescape_content(body), sources=sources))

    for line in lines:
        boundary = BOUNDARY.match(line)
        if boundary:
            flush()
            role = LABEL_ROLES[boundary.group(1)]
            buffer = []
            continue
        if role is not None:
            buffer.append(line)
            continue
        if not title and line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("- 创建时间："):
            created_at_ms = parse_timestamp(line[len("- 创建时间：") :])
    flush()
    return Session(
        id=session_id,
        title=title or session_id,
        created_at_ms=created_at_ms,
        updated_at_ms=updated_at_ms,
        turns=turns,
    )


def scan_session(path: Path) -> tuple[str, int] | None:
    """逐行扫描会话文件取出标题和消息条数，避免列表接口把正文全部读进内存。"""
    title = ""
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                line = line.rstrip("\n").rstrip("\r")
                if not title and line.startswith("# "):
                    title = line[2:].strip()
                if BOUNDARY.match(line):
                    count += 1
    except OSError:
        return None
    return title, count


class MemoryStore:
    """读写长期记忆与会话记录，文件都放在同一个记忆目录下。

    append_exchange 的“读-改-写”依赖调用方持有串行锁，本类自身不是线程安全的；
    单次读写用 atomic_write 保证不会读到写了一半的文件。
    """

    def __init__(self, memory_dir, enabled: bool = True):
        """保存记忆目录与启用开关；创建实例不会读写任何文件，也不会建目录。"""
        self.dir = Path(memory_dir)
        self.enabled = bool(enabled)

    @property
    def memory_path(self) -> Path:
        """长期记忆文件的位置。"""
        return self.dir / MEMORY_FILENAME

    @property
    def sessions_dir(self) -> Path:
        """会话记录目录的位置。"""
        return self.dir / SESSIONS_DIRNAME

    def session_path(self, session_id) -> Path:
        """把会话 id 解析成文件路径；形状不符或越出会话目录时抛出 ValueError。"""
        if not isinstance(session_id, str) or not SESSION_ID.match(session_id):
            raise ValueError("会话 id 不合法")
        path = (self.sessions_dir / f"{session_id}.md").resolve()
        # 双保险：即使正则将来被放宽，解析后的路径也必须仍在会话目录内。
        if path.parent != self.sessions_dir.resolve():
            raise ValueError("会话 id 不合法")
        return path

    def read_memory(self) -> str:
        """读取长期记忆全文；未启用或文件不存在时返回空字符串。"""
        if not self.enabled:
            return ""
        try:
            with self.memory_path.open("r", encoding="utf-8", errors="replace") as stream:
                return stream.read(MAX_MEMORY_READ)
        except OSError:
            return ""

    def write_memory(self, text: str) -> None:
        """原子写入长期记忆；未启用时不做任何事。"""
        if not self.enabled:
            return
        atomic_write(self.memory_path, str(text).encode("utf-8"))

    def memory_exists(self) -> bool:
        """报告长期记忆文件是否已经存在，供界面决定是否提示新建。"""
        return self.enabled and self.memory_path.is_file()

    def create_session(self, title: str) -> Session:
        """新建一份会话记录并返回；未启用时返回一份不落盘的内存记录。"""
        started = int(time.time() * 1000)
        session_id = new_session_id()
        session = Session(
            id=session_id,
            # 与落盘内容保持一致：标题统一压成单行，接口返回值就是文件里那一行。
            title=session_title(title) or "新对话",
            created_at_ms=started,
            updated_at_ms=started,
        )
        if self.enabled:
            path = self._reserve_session(session_id)
            atomic_write(path, render_session(session.title, [], started).encode("utf-8"))
        return session

    def _reserve_session(self, session_id: str) -> Path:
        """用 O_EXCL 占位创建会话文件，避免撞名时静默覆盖另一条会话。"""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{session_id}.md"
        # atomic_write 走 os.replace，同名会直接覆盖，所以这里必须先原子地占住文件名。
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(descriptor)
        return path

    def list_sessions(self, limit: int | None = None) -> list[SessionSummary]:
        """按修改时间从新到旧列出会话摘要；坏文件只标记不隐藏，也不让接口失败。"""
        if not self.enabled:
            return []
        directory = self.sessions_dir
        if not directory.is_dir():
            return []
        # 只认 id 形状正确的 .md：atomic_write 的临时文件也落在同一个目录里，
        # 进程崩溃会留下 <id>.md.XXXXXXXX 这样的残留。
        paths = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix == ".md" and SESSION_ID.match(path.stem)
        ]
        summaries = []
        for path in paths:
            try:
                updated_at_ms = path.stat().st_mtime_ns // 1_000_000
            except OSError:
                continue
            scanned = scan_session(path)
            if scanned is None:
                title, count, broken = path.stem, 0, True
            else:
                title, count = scanned
                broken = False
            summaries.append(
                SessionSummary(
                    id=path.stem,
                    title=title or path.stem,
                    created_at_ms=None,
                    updated_at_ms=updated_at_ms,
                    message_count=count,
                    parse_error=broken,
                )
            )
        summaries.sort(key=lambda item: item.updated_at_ms, reverse=True)
        return summaries[:limit] if limit else summaries

    def load_session(self, session_id: str) -> Session | None:
        """读取一份完整会话；id 不合法时抛 ValueError，文件不存在时返回 None。"""
        path = self.session_path(session_id)
        if not self.enabled or not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            updated_at_ms = path.stat().st_mtime_ns // 1_000_000
        except OSError:
            return None
        return parse_session(text, session_id, updated_at_ms)

    def append_exchange(
        self,
        session_id: str,
        question: str,
        answer: str = "",
        sources=(),
    ) -> Session | None:
        """把一轮问答追加到会话记录，返回追加后的会话。

        没有回答时只记下用户那一轮，不写入空的助手段落。
        调用方必须持有串行锁，否则两个请求的“读-改-写”会互相覆盖。
        """
        session = self.load_session(session_id)
        if session is None:
            return None
        session.turns.append(Turn(role="user", content=str(question)))
        if str(answer).strip():
            session.turns.append(
                Turn(
                    role="assistant",
                    content=str(answer),
                    sources=[sanitize_label(label) for label in sources],
                )
            )
        session.updated_at_ms = int(time.time() * 1000)
        self.save_session(session)
        return session

    def save_session(self, session: Session) -> None:
        """把整份会话原子写回文件；未启用时不做任何事。"""
        if not self.enabled:
            return
        text = render_session(session.title, session.turns, session.created_at_ms)
        atomic_write(self.session_path(session.id), text.encode("utf-8"))

    def delete_session(self, session_id: str) -> bool:
        """删除一份会话记录，返回是否真的删掉了文件。"""
        path = self.session_path(session_id)
        if not self.enabled:
            return False
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True
