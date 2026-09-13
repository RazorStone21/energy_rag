# 服务器目录与当前项目的对应关系

本次依据之前提供的服务器目录截图、本地旧版代码和当前配置对照，未连接服务器。
截图未展开的数据、文档和模型内部文件无法逐项核对。

## 已补齐的运行目录

```text
data/                          索引、片段缓存和文件哈希清单
gov_doc/                       待入库的原始文档
models/
├── bge-m3/                    嵌入模型权重
├── bge-reranker-v2-m3/         重排模型权重
├── Qwen2.5-VL-3B/             视觉模型权重
└── Qwen3-8B/                  文本生成模型权重
```

这些目录只有空的 `.gitkeep` 占位文件，不包含真实文档或模型。
目录与 config.toml 的默认相对路径一致；使用 --data-root 或 ENERGY_RAG_HOME 时以指定的数据根目录为准。
根目录 models/ 保存权重，src/models/ 保存调用模型的代码，两者用途不同。
`.gitignore` 仍忽略这些运行数据目录，当前占位文件并未改为需要版本管理的文件。

## 原源码的位置

| 服务器原路径 | 当前对应位置 |
| --- | --- |
| main.py | main.py，命令参数处理在 src/cli.py |
| config.py | src/config.py 和 config.toml |
| ingest.py | src/ingestion.py、src/chunker.py、src/parsers/ |
| bm25.py | src/retrieval/bm25.py |
| rag.py | src/pipeline.py、src/retrieval/hybrid.py、src/context_builder.py |
| models.py | src/models/ 下的嵌入、重排和生成模块 |
| multimodal.py | src/parsers/pdf_elements.py |
| download_models.py | scripts/download_models.py |
| export_chunks.py | scripts/export_chunks.py |
| fix_ragas_compat.py | scripts/fix_ragas_compat.py |
| tests/test_pipeline.py | tests/integration/test_pipeline.py，另有 tests/unit/ |
| tests/run_retrieval_eval.py | tests/evaluation/run_retrieval_eval.py |
| tests/run_rag_eval.py | tests/evaluation/run_rag_eval.py |
| tests/eval_questions.json | tests/evaluation/questions.json |
| tests/*_eval_report.json | tests/results/ 下的对应评测报告 |
| README.md、requirements.txt | 保留在项目根目录 |

表中的源码已有对应实现，不额外创建同名空 Python 文件，避免混淆新旧入口。
`.idea/` 和 `__pycache__/` 属于编辑器设置与运行缓存，不是项目功能代码。
开发日记按此前排除要求未补回。

## 需要真实数据或由程序生成的文件

- 原始 PDF、TXT、Markdown、DOCX、XLSX：按实际需要放入 gov_doc/。
- config.json、分词器配置及模型权重：从服务器复制或使用 python -m scripts.download_models 下载。
- data/chunks.pkl、data/build_manifest.json、data/milvus.db：由入库流程生成。

不创建上述文件的空壳：空模型文件不能加载，空缓存或索引会导致读取失败。
视觉模型的可用检查先确认 config.json 存在，避免仅因空目录存在就启用图片描述；
该检查不保证权重或分词器完整，真实加载仍可能报告错误。
