"""验证本地源码的配置路径、组件调用和命令行行为。"""

import importlib.util
import logging
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import cli
from src.config import MemorySettings, load_settings
from src.progress import BuildProgress
from src.schemas import AnswerResult, SearchHit

ROOT = Path(__file__).resolve().parents[2]


def load_script(relative):
    """按文件位置加载工具或评测模块，避免修改 Python 模块搜索路径。"""
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paths_follow_config_file_instead_of_source_or_working_directory(tmp_path, monkeypatch):
    """验证数据路径相对于配置文件解析，不受当前目录或源码位置影响。"""
    config_dir = tmp_path / "配置目录"
    config_dir.mkdir()
    path = config_dir / "config.toml"
    path.write_bytes((ROOT / "config.toml").read_bytes())
    monkeypatch.delenv("ENERGY_RAG_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = load_settings(path)
    # 期望值直接从配置文件读取，调整路径后测试仍然验证同一条规则。
    configured = tomllib.loads(path.read_text(encoding="utf-8"))
    assert settings.data_root == config_dir
    assert settings.doc_dir == config_dir / configured["paths"]["documents"]
    assert settings.embedding.path == config_dir / configured["embedding"]["path"]
    assert settings.milvus.connection_args["uri"] == str(
        config_dir / configured["milvus"]["connection"]["uri"]
    )


def test_explicit_paths_override_environment(tmp_path, monkeypatch):
    """验证显式数据根目录优先于环境变量，环境变量优先于文件配置。"""
    monkeypatch.setenv("ENERGY_RAG_CONFIG", str(ROOT / "config.toml"))
    monkeypatch.setenv("ENERGY_RAG_HOME", str(tmp_path / "env"))
    assert load_settings().data_root == tmp_path / "env"
    assert load_settings(data_root=tmp_path / "explicit").data_root == tmp_path / "explicit"


@pytest.mark.parametrize(
    "before,after",
    [
        ("dense_top_k = 20", "dense_top_k = 0"),
        ("context_top_k = 5", "context_top_k = true"),
        ("top_p = 0.9", "top_p = 1.1"),
        ("temperature = 0.1", "temperature = -1"),
        ("hybrid_enabled = true", 'hybrid_enabled = "true"'),
    ],
)
def test_invalid_configuration_fails_before_model_creation(tmp_path, before, after):
    """验证无效配置在创建任何模型之前被拒绝。"""
    path = tmp_path / "config.toml"
    path.write_text(
        (ROOT / "config.toml").read_text(encoding="utf-8").replace(before, after), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_settings(path)


def test_remote_milvus_uri_is_not_rewritten_as_a_file(tmp_path):
    """验证远程 Milvus 服务地址不会被错误转换成本地路径。"""
    path = tmp_path / "config.toml"
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    path.write_text(
        text.replace('uri = "data/milvus.db"', 'uri = "http://localhost:19530"'), encoding="utf-8"
    )
    assert load_settings(path).milvus.connection_args["uri"] == "http://localhost:19530"


def test_cli_uses_injected_runtime_and_reports_partial_failure(monkeypatch):
    """用模拟 Runtime 检查 CLI 的调用参数，并验证部分文件入库失败时返回非零退出码。"""
    runtime = Mock()
    runtime.ingestion.build.return_value = SimpleNamespace(
        processed=["a.pdf"],
        removed=[],
        failed={"b.pdf": "解析失败"},
    )
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(cli, "create_runtime", factory)
    assert (
        cli.main(
            ["--config", str(ROOT / "config.toml"), "build", "--incremental", "--only", "a.pdf"]
        )
        == 1
    )
    runtime.ingestion.build.assert_called_once_with(
        max_files=None,
        incremental=True,
        only=["a.pdf"],
        progress_factory=BuildProgress,
    )
    factory.assert_called_once()


def test_cli_forwards_query_flags(monkeypatch, capsys):
    """验证 CLI 正确传递混合检索、重排和提示词打印开关，并把词元写进标准输出。"""
    runtime = Mock()

    def fake_ask(question, with_rerank, hybrid, on_prompt, on_token):
        """模拟流水线逐块产出答案，用于确认 CLI 把回调内容显示出来。"""
        on_token("回")
        on_token("答")
        return AnswerResult("回答", [SearchHit(Mock())], timings={"generation": 0.01})

    runtime.pipeline.ask.side_effect = fake_ask
    monkeypatch.setattr(cli, "create_runtime", lambda settings: runtime)
    assert (
        cli.main(
            [
                "--config",
                str(ROOT / "config.toml"),
                "ask",
                "问题",
                "--no-print",
                "--no-rerank",
                "--no-hybrid",
            ]
        )
        == 0
    )
    runtime.pipeline.ask.assert_called_once_with(
        "问题",
        with_rerank=False,
        hybrid=False,
        on_prompt=None,
        on_token=cli.print_token,
    )
    assert "回答" in capsys.readouterr().out


def test_memory_settings_guard_limits_and_directory(tmp_path):
    """验证记忆配置的默认值、注入上限，以及记忆目录不能落在文档目录里。"""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "config.toml"
    text = (root / "config.toml").read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")

    settings = load_settings(path)
    assert settings.memory == MemorySettings()
    assert settings.memory_dir == (tmp_path / "data/memory").resolve()

    # 模型侧不会截断提示词，配置里的上限是唯一防线，超过硬上限直接拒绝。
    path.write_text(text.replace("max_chars = 1000", "max_chars = 99999"), encoding="utf-8")
    with pytest.raises(ValueError):
        load_settings(path)

    # 会话记录是 markdown，落在文档目录里会被下一次入库当成待索引文档。
    path.write_text(
        text.replace('memory = "data/memory"', 'memory = "data/gov_doc/memory"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_settings(path)

    # 旧配置既没有 [memory] 段也没有 paths.memory 时使用默认值。
    path.write_text(
        text.split("[memory]")[0].replace('memory = "data/memory"\n', "")
        + "[prompts]"
        + text.split("[prompts]", 1)[1],
        encoding="utf-8",
    )
    assert load_settings(path).memory == MemorySettings()


def test_cli_warns_when_local_milvus_is_locked(tmp_path, caplog):
    """验证本地 Milvus 被别的进程占用时先给出可操作的提示，服务地址不做这项检查。"""
    import fcntl

    settings = load_settings(ROOT / "config.toml")
    directory = tmp_path / "milvus.db"
    directory.mkdir()
    local = replace(
        settings,
        milvus=replace(settings.milvus, connection_args={"uri": str(directory)}),
    )
    assert cli._local_milvus_dir(local) == directory
    # 远端 Milvus 服务地址不存在本地独占的问题，不做这项检查。
    remote = replace(
        settings,
        milvus=replace(settings.milvus, connection_args={"uri": "http://host:19530"}),
    )
    assert cli._local_milvus_dir(remote) is None

    # 还没建过索引时没有 LOCK 文件，也就不会提示。
    with caplog.at_level(logging.WARNING):
        cli._warn_if_index_locked(local)
    assert caplog.text == ""

    lock_path = directory / "LOCK"
    lock_path.touch()
    with lock_path.open("rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            cli._warn_if_index_locked(local)
        assert "已被另一个进程打开" in caplog.text
        assert "只用 BM25" in caplog.text
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)

    # 锁释放后不再提示。
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        cli._warn_if_index_locked(local)
    assert caplog.text == ""


def test_retrieval_evaluation_explicitly_disables_hybrid():
    """验证纯向量评测明确关闭混合检索，使报告名称与实际行为一致。"""
    module = load_script("tests/evaluation/run_retrieval_eval.py")
    runtime = Mock()
    document = SimpleNamespace(page_content="标注片段", metadata={})
    runtime.pipeline.retrieve.return_value = [SearchHit(document)]
    report = module.run([{"question": "问题", "relevant_chunks": ["标注片段"]}], runtime)
    assert report["n"] == 1
    runtime.pipeline.retrieve.assert_called_once_with("问题", with_rerank=True, hybrid=False)


def test_generation_evaluation_uses_pipeline_evidence():
    """验证生成评测采用问答流程实际返回的答案和证据，并保留题型。"""
    module = load_script("tests/evaluation/run_rag_eval.py")
    runtime = Mock()
    document = SimpleNamespace(page_content="证据", metadata={})
    runtime.pipeline.ask.return_value = AnswerResult("回答", [SearchHit(document)])
    assert module.collect_predictions([{"question": "问题", "type": "table"}], runtime) == [
        {
            "question": "问题",
            "type": "table",
            "contexts": ["证据"],
            "answer": "回答",
            "reference": "",
        }
    ]


def test_rerank_max_length_defaults_and_rejects_non_positive(tmp_path):
    """验证重排词元上限可配置：旧配置缺该字段时用模型上限，非法值提前报错。"""
    configured = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    assert configured["reranker"]["max_length"] == 8192
    legacy = tmp_path / "legacy.toml"
    text = (ROOT / "config.toml").read_text(encoding="utf-8").replace("max_length = 8192\n", "")
    legacy.write_text(text, encoding="utf-8")
    assert load_settings(legacy).rerank_max_length == 8192

    broken = tmp_path / "broken.toml"
    broken.write_text(
        text.replace(
            'path = "models/bge-reranker-v2-m3"',
            'path = "models/bge-reranker-v2-m3"\nmax_length = 0',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reranker.max_length"):
        load_settings(broken)


def test_download_catalog_uses_configured_model_paths(tmp_path):
    """验证下载清单使用配置中的模型标识和解析后的下载目录。"""
    module = load_script("scripts/download_models.py")
    settings = load_settings(ROOT / "config.toml", data_root=tmp_path)
    assert module.model_catalog(settings)["bge-m3"] == (
        settings.model_ids["embedding"],
        tmp_path / "models/bge-m3",
    )
