"""使用真实 PDF、Milvus 和模型测试流程，各测试按自己的文件检查或环境开关决定是否执行。"""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from src.bootstrap import create_runtime
from src.config import load_settings
from src.parsers.pdf import PDFParser


@pytest.fixture
def settings():
    """读取项目配置供集成测试使用，不在准备阶段加载任何模型。"""
    return load_settings(Path(__file__).resolve().parents[2] / "config.toml")


def test_pdf_extraction_has_text(settings):
    """验证真实 PDF 能提取非空正文；没有测试文档时跳过。"""
    pdfs = sorted(settings.doc_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip("未配置测试 PDF")
    runtime = create_runtime(settings)
    parser = PDFParser(runtime.vision, settings.vision)
    docs = parser.parse_text(pdfs[0])
    assert docs and any(doc.page_content.strip() for doc in docs)


def test_build_small_index_and_retrieve(tmp_path, settings):
    """在临时数据库构建单文件索引并查询，避免改动正式索引。"""
    if not settings.embedding.path.exists() or not list(settings.doc_dir.glob("*.pdf")):
        pytest.skip("模型或测试 PDF 不存在")
    # 隔离数据库、缓存和清单，集成测试不会修改正式索引。
    settings = replace(
        settings,
        chunks_path=tmp_path / "chunks.pkl",
        manifest_path=tmp_path / "build_manifest.json",
        milvus=replace(settings.milvus, connection_args={"uri": str(tmp_path / "milvus.db")}),
    )
    runtime = create_runtime(settings)
    result = runtime.ingestion.build(max_files=1)
    assert result.chunks and not result.failed
    assert (tmp_path / "milvus.db").exists()
    assert runtime.pipeline.retrieve("文档的主要内容", with_rerank=False)


@pytest.mark.skipif(os.environ.get("RUN_LLM_TEST") != "1", reason="真实生成需 RUN_LLM_TEST=1")
def test_llm_generates_answer(settings):
    """RUN_LLM_TEST=1 时执行真实问答，需提前准备模型和索引，检查最终答案非空。"""
    runtime = create_runtime(settings)
    result = runtime.pipeline.ask("请概括文档的主要内容。")
    assert result.answer.strip()
