"""命令行入口：解析参数、创建运行环境并展示流程结果。"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from .bootstrap import Runtime, create_runtime
from .config import load_settings
from .progress import BuildProgress


def print_prompt(prompt: str) -> None:
    """打印实际交给模型的完整提示词，包括检索片段、回答要求和用户问题。"""
    print("=" * 72)
    print("【最终 Prompt】")
    print(prompt)
    print("=" * 72)


def print_token(piece: str) -> None:
    """原样写出模型新生成的文本块，不换行并立即刷新，让答案逐词元显示。"""
    print(piece, end="", flush=True)


def _create_argument_parser() -> argparse.ArgumentParser:
    """集中定义命令行选项，供帮助展示和参数解析共用。"""
    parser = argparse.ArgumentParser(description="能源文档 RAG 问答系统")
    parser.add_argument("--config", default=None, help="配置文件，默认当前目录 config.toml")
    parser.add_argument("--data-root", default=None, help="覆盖文档、模型、索引的数据根目录")
    subcommands = parser.add_subparsers(dest="cmd", required=True)
    build_parser = subcommands.add_parser(
        "build", help="构建文档索引（PDF、TXT、Markdown、DOCX、XLSX）"
    )
    build_parser.add_argument("--max-files", type=int, default=None, help="仅处理前 N 个支持的文档")
    build_parser.add_argument("--incremental", action="store_true", help="增量更新")
    build_parser.add_argument(
        "--only", nargs="+", default=None, help="增量模式下强制重处理指定文件"
    )
    ask_parser = subcommands.add_parser("ask", help="检索并回答")
    ask_parser.add_argument("question", help="用户问题")
    ask_parser.add_argument("--no-rerank", action="store_true", help="跳过重排序")
    ask_parser.add_argument("--no-hybrid", action="store_true", help="只使用向量召回")
    ask_parser.add_argument("--no-print", action="store_true", help="不打印最终 Prompt")
    serve_parser = subcommands.add_parser(
        "serve", help="启动 Web 界面：流式问答、参考来源与索引状态"
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认只允许本机访问")
    serve_parser.add_argument("--port", type=int, default=8000, help="监听端口，默认 8000")
    serve_parser.add_argument(
        "--demo", action="store_true", help="演示模式：不加载模型，回答与来源都是编造的"
    )
    serve_parser.add_argument("--warmup", action="store_true", help="启动后在后台预加载模型")
    serve_parser.add_argument("--front-dir", default=None, help="前端目录，默认使用项目下的 front/")
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """在加载配置和模型前拒绝无效选项组合，错误通过 argparse 展示。"""
    if args.cmd == "build":
        if args.max_files is not None and args.max_files <= 0:
            parser.error("--max-files 必须为正整数")
        if args.only is not None and not args.incremental:
            parser.error("--only 必须与 --incremental 一起使用")
    elif args.cmd == "serve":
        if not 0 < args.port < 65536:
            parser.error("--port 必须在 1 到 65535 之间")
    elif not args.question.strip():
        parser.error("问题不能为空")


def _run_build(runtime: Runtime, args: argparse.Namespace) -> int:
    """执行入库并展示进度与文件统计；有文件失败时返回非零退出码。

    全量重建要逐个文件解析并逐张描述图表，耗时以小时计，因此打开进度条；
    输出不是终端时（如重定向到文件）会自动退化成每个文件一行的日志。
    """
    if not runtime.vision.available:
        logging.warning("未找到视觉模型，跳过图片描述；仍提取正文与表格。")
    result = runtime.ingestion.build(
        max_files=args.max_files,
        incremental=args.incremental,
        only=args.only,
        progress_factory=BuildProgress,
    )
    print(f"成功 {len(result.processed)}，移除 {len(result.removed)}，失败 {len(result.failed)}")
    if result.failed:
        logging.error("失败文件保留旧索引，下次重试：%s", result.failed)
        return 1
    return 0


def _run_ask(runtime: Runtime, args: argparse.Namespace) -> int:
    """转发检索选项，按需打印提示词，并边生成边显示答案与耗时。"""
    _warn_if_index_locked(runtime.settings)
    result = runtime.pipeline.ask(
        args.question,
        with_rerank=not args.no_rerank,
        hybrid=not args.no_hybrid,
        on_prompt=None if args.no_print else print_prompt,
        on_token=print_token,
    )
    if not result.evidence:
        print("未检索到相关文档。")
        return 0
    # 逐词元输出不以换行结尾，这里补一个再打印耗时。
    print()
    print(f"[耗时] 生成 {result.timings['generation']:.2f}s（含首次模型加载）")
    return 0


def _local_milvus_dir(settings):
    """返回本地 Milvus 数据目录；使用服务地址或非 .db 地址时返回 None。"""
    uri = str(settings.milvus.connection_args.get("uri", ""))
    if not uri or "://" in uri or not uri.endswith(".db"):
        return None
    return Path(uri)


def _warn_if_index_locked(settings) -> None:
    """启动前检查本地 Milvus 目录是否已被别的进程占用，提前给出可操作的提示。

    本地 Milvus 同一时刻只能被一个进程打开。撞锁时第三方库会打印一大段英文堆栈，
    看不出该怎么处理，所以这里先试一次非阻塞锁，把话说在前面。
    """
    directory = _local_milvus_dir(settings)
    if directory is None:
        return
    lock_path = directory / "LOCK"
    if not lock_path.exists():
        return
    try:
        import fcntl
    except ImportError:
        # 非 Unix 平台没有 flock，跳过检查，交给实际连接去报错。
        return
    with lock_path.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            logging.warning(
                "%s 已被另一个进程打开；本地 Milvus 同一时刻只能有一个进程使用，"
                "本次的向量检索会失败并降级为只用 BM25。"
                "请关闭其它 serve 或 ask 进程后重试。",
                directory,
            )
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _describe_endpoint(settings) -> None:
    """启动前打印实际使用的数据和索引位置，避免从别的目录启动时静默换了一套索引。"""
    logging.info("数据根目录：%s", settings.data_root)
    logging.info("片段缓存：%s", settings.chunks_path)
    logging.info("构建清单：%s", settings.manifest_path)
    logging.info("Milvus 地址：%s", settings.milvus.connection_args.get("uri"))


def _run_serve(runtime: Runtime, args: argparse.Namespace) -> int:
    """启动 Web 服务，返回退出码。

    FastAPI 和 uvicorn 在这里才导入，未安装 Web 依赖时命令行其他子命令仍可正常使用。
    """
    try:
        import uvicorn
    except ImportError:
        logging.error('未安装 Web 依赖，请先执行 python -m pip install ".[server]"')
        return 1
    from .server.app import create_app
    from .server.service import RagService

    if args.demo:
        from .server.demo import apply_demo

        # 演示模式不加载模型、不读取索引，界面上的标记由服务状态给出。
        apply_demo(runtime)
        logging.warning("演示模式：回答与参考来源都是编造的，仅用于预览界面。")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning(
            "服务将对外开放，/api/build 会改动索引且没有鉴权，请确认只在可信网络中使用。"
        )
    _describe_endpoint(runtime.settings)
    _warn_if_index_locked(runtime.settings)
    service = RagService(runtime, demo=args.demo)
    if args.warmup:
        logging.info("已在后台开始预加载模型，首次提问仍需等待加载完成。")
        service.warmup()
    app = create_app(service, front_dir=args.front_dir)
    print(f"界面已启动：http://{args.host}:{args.port}/   按 Ctrl+C 停止")
    # 单进程运行：多进程会让每个进程各加载一份模型，显存会成倍占用。
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="warning")
    return 0


def _silence_library_output() -> None:
    """压掉第三方库的信息级日志、模型加载进度条和已知兼容性警告。

    问答只需要最终提示词和答案，所以把根日志级别抬到 WARNING，
    再单独调低加载模型、中文分词和向量库相关库的级别。
    项目自身的 logger.warning 属于 WARNING 级，检索降级等真实异常仍会显示。
    """
    logging.getLogger().setLevel(logging.WARNING)
    # 这些库在加载模型和启动 Milvus 时输出信息级内容。
    for name in (
        "sentence_transformers",
        "transformers",
        "huggingface_hub",
        "faiss",
        "milvus_lite",
        "pymilvus",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    # milvus 生成的 protobuf 代码比运行时旧一个大版本，实测不影响功能。
    warnings.filterwarnings(
        "ignore",
        message="Protobuf gencode version .* is exactly one major version older",
        category=UserWarning,
    )
    try:
        import jieba
        from transformers.utils import logging as transformers_logging
    except ImportError:
        # 演示模式等没有模型依赖的环境只需要上面压掉的根日志级别。
        return
    # jieba 在导入时把自己的日志级别设回 DEBUG，所以必须先导入再调低。
    jieba.setLogLevel(logging.WARNING)
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def main(argv: list[str] | None = None) -> int:
    """读取命令参数，创建 Runtime 并执行入库或问答；返回退出码供 main.py 使用。"""
    parser = _create_argument_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.cmd in ("ask", "serve"):
        _silence_library_output()
    # 一次命令只拥有一个运行环境；组件共享模型实例，不使用隐式全局缓存。
    runtime = create_runtime(load_settings(args.config, args.data_root))
    if args.cmd == "build":
        return _run_build(runtime, args)
    if args.cmd == "serve":
        return _run_serve(runtime, args)
    return _run_ask(runtime, args)
