# 包结构与实现说明

## 模块边界

业务代码直接位于 src/。cli.py、scripts/ 和 tests/evaluation/ 是调用者，
通过 load_settings 与 create_runtime 创建自己的运行环境。
bootstrap.py 负责组件组装，其他组件不读取全局配置，也不反向导入调用脚本。

| 层次 | 主要模块 | 边界 |
|---|---|---|
| 配置与入口 | config、cli、bootstrap | 校验配置、创建依赖、展示结果 |
| 流程 | ingestion、pipeline | 决定阶段顺序和失败处理 |
| 数据与契约 | schemas、interfaces | 定义结果与最小接口 |
| 解析与切分 | parsers、chunker | 根据格式解析文档，再转为可检索片段 |
| 检索 | retrieval | 独立双路召回、去重和 RRF |
| 模型适配 | models | 按需加载并调用具体模型 |
| 存储适配 | storage | 执行向量库和文件操作 |
| 上下文 | context_builder | 组织提示词及实际采用的证据 |
| 记忆 | storage/memory | 会话记录与长期记忆的读写，只依赖标准库 |
| Web 接口 | server | 把问答与入库转成事件流，托管 front/ 页面 |

interfaces.py 使用 Protocol 定义 Parser、Chunker、VectorStore、LexicalIndex、
Retriever、Reranker、Generator。实现无需继承共同基类，测试可注入符合接口的替身。
schemas.py 中 DocumentLike 只要求正文和元数据，现有 LangChain Document 可以直接使用。
这次没有迁移持久化文档格式或增加 chunk_id。

## 阅读顺序与关键注释

建议从 bootstrap.py 看依赖组装，再读 pipeline.py / ingestion.py 两条流程。
中文注释集中解释了以下决策：

- schemas.py：每种结果的职责、各阶段分数以及去重身份。
- config.py：配置文件、命令行、环境变量及路径优先级。
- retrieval/hybrid.py：分支内去重、跨分支投票和召回失败降级。
- ingestion.py：完整文件发现、先解析再写入、失败来源的保留。
- storage/chunks.py：临时文件原子替换、中断标记及发布顺序。
- models/reranker.py：单候选标量分数、输出校验与保留原始分数。
- models/local_qwen.py：量化加载和仅解码新增词元。
- chunker.py：表格和图片描述保持完整，不再次语义切分。

变量、类名和第三方 API 名称保留英文，解释性注释与文档字符串使用中文。
覆盖率及静态检查指令保留工具可识别的标记。

## 流程与数据结构

入库：完整发现文件 → 选取新增/变更/--only 文件 → 解析 → 切分与过滤
→ 保留成功结果 → 标记写入开始 → 更新 Milvus → 发布缓存与清单 → 清除标记。

问答：校验输入与索引状态 → 向量/BM25 独立召回 → 分支内去重 → RRF
→ 重排序 → ContextBundle → 生成 → AnswerResult。
pipeline.ask 的 on_prompt 和 on_token 都是可选回调，分别接收完整提示词和生成过程中的文本块；
传入 on_token 时改走流式生成，命令行据此边生成边显示，拼接结果与一次性生成的答案一致。
on_context 在生成前接收 ContextBundle，Web 层据此在答案出现前就展示参考来源。
history 是 (role, text) 列表，只填入提示词的历史段落用于理解指代，不参与检索；
memory 是长期记忆全文，作为背景参考注入，同样不参与检索；
两者为空时提示词与单轮问答一致。检索仍只使用当前问题，不做指代消解。

## 记忆

storage/memory.py 负责把会话记录和长期记忆保存成 markdown，只依赖标准库。
写入复用 chunks.py 的 atomic_write，因此读到的永远是整份文件，不会撞上写了一半的内容。

会话文件的格式必须**可逆**：消息正文里凡是整行呈现边界样式（`## 用户`、`## 助手`、
`### 参考来源`）的行，写入时行首加一个反斜杠，读取时去掉。转义判定比边界判定更宽
（所有以 `#` 开头的行为都会被转义），所以边界正则将来放宽也不会漏转义；
真正要守住的不变式是 escape_line 与 unescape_line 互为逆运算，单元测试按样例表锁定。
模型回答里出现 markdown 标题是常事，没有这条规则会把一条消息切成两条。

读取路径按容错设计：先归一化换行再 split（不能用 splitlines，它还会在 U+2028 处断行）、
按 errors="replace" 解码、每个文件单独捕获异常；解析不了的条目在列表里标记 parse_error，
而不是消失或让接口失败。会话 id 由服务端生成、形状固定并用允许列表校验，
因此不需要靠黑名单去挡 `..`、全角字符或同形字；文件名是 id 的唯一权威。

参考来源只写位置标签。检索到的片段正文一旦进入历史，下一轮会被当成助手说过的话
喂回模型，既占用上下文，又混淆了「文档里的话」和「对话里的话」。

长期记忆按上限从**头部**截取：手写文件的阅读顺序是从上到下，这样文件顶部是稳定的
优先前缀，在末尾追加不会改变模型看到的内容。上限还有硬上限，
因为 local_qwen 的 tokenizer 没有传 truncation，超长提示词不会被自动裁掉。

注入前会对记忆与历史调用 defuse()，把可能冒充分节标题的 `【文档片段】` 等标记换成
半角方括号：旧的回答若被恶意文档片段影响过，可以在下一轮伪造段落。落盘与界面仍显示原话。

SearchHit 携带 DocumentLike 和 dense_score、bm25_score、rrf_score、rerank_score；
None 表示该阶段未产生分数。ContextBundle 中的 evidence 与实际上下文一一对应。
BuildResult 描述成功、删除与失败的文件；增量调用者应检查 failed，不能只看 chunks。
CLI 会在部分失败时返回 1，完整全量解析失败或写入异常直接抛错。

## Web 层

server/ 按是否依赖 Web 框架拆开：events.py 只做 SSE 编码，service.py 把问答和入库
包装成事件流，两者只依赖标准库与项目自身模块；只有 app.py（路由）和 schemas.py
（请求模型）导入 FastAPI，因此服务层可以脱离 Web 框架单测。
cli.py 在 serve 分支内部才导入 FastAPI 和 uvicorn，未安装 Web 依赖时 build 与 ask 不受影响，
python main.py --help 也不需要这些包。

一次问答由生产者线程加队列驱动：on_token 是回调而不是生成器，ask 会一直阻塞到生成结束，
所以回调只能把事件放进队列，由消费生成器逐条转发。这样模型所在的生成器始终留在那个线程，
客户端断开时不会在事件循环线程上执行收尾而卡住整个服务。
事件生成器内部兜底捕获异常并转成 error 事件——响应头在第一个数据块之前就已发送 200，
之后再抛异常对客户端只会表现为连接中断。

问答与入库共用一把 threading.Lock 串行执行，与「不支持并发构建或在线索引切换」一致；
抢不到锁的请求先收到排队事件。客户端断开只停止接收，模型仍会把这一轮生成完，期间锁不释放。

/api/status 只读片段缓存的版本信息与构建清单，不触碰 Milvus 对象和任何模型：
MilvusStore.backend 会顺带加载嵌入模型，状态接口因此不访问它。
每个来源的片段数用以 revision() 为键的缓存计算，与 retrieval/bm25.py 的缓存键策略一致。

MemoryStore 挂在 Runtime 上，与 chunk_store 并列。这既让 apply_demo 能像换 pipeline
一样换掉它，也保证了单元测试不会因为服务层自己构造存储而往仓库的数据目录里写文件。

front/ 是原生 HTML/CSS/JS，没有构建步骤，由 app.py 挂载在 / 与 /static 下。
模型输出经 markdown.js 先转义再转换后渲染；检索到的原文只用 textContent 赋值，
两条渲染规则的边界不同，不能混用。

## 生命周期

create_runtime 每次创建独立实例；同一实例内切分与向量库共用 Embedder，
检索、重排和生成分别按需加载模型。导入包和查看命令帮助不会加载 GPU 模型。

应用应在请求之间复用自己持有的实例。配置快照创建后不会自动变化；
调整参数可用 dataclasses.replace 创建新设置，再构造新 Runtime。
runtime.release_reranker() 释放其持有的重排模型引用，供生成评测腾出显存；
外部持有的引用仍需自行释放。这里不保证并发模型加载安全。

## 配置迁移

| 原配置 | config.toml 对应字段 |
|---|---|
| DOC_DIR | paths.documents |
| CHUNKS_PATH | paths.chunks |
| BUILD_MANIFEST_PATH | paths.manifest |
| EMBED_MODEL_PATH / EMBED_DEVICE | embedding.path / device |
| RERANK_MODEL_PATH | reranker.path |
| LLM_MODEL_PATH | generation.path |
| VLM_MODEL_PATH | vision.path |
| RETRIEVE_K / DENSE_TOP_K | retrieval.dense_top_k |
| BM25_K / BM25_TOP_K | retrieval.bm25_top_k |
| FUSION_TOP_K | retrieval.fusion_top_k |
| RERANK_TOP_K / CONTEXT_TOP_K | retrieval.context_top_k |
| RRF_K / HYBRID_ENABLED | retrieval.rrf_k / hybrid_enabled |
| BREAKPOINT_* 等 | splitting 下各字段 |
| MILVUS_CONNECTION_ARGS | milvus.connection |
| MILVUS_INDEX_PARAMS / SEARCH_PARAMS | milvus.index / search |
| RAG_PROMPT_TEMPLATE / FIGURE_DESC_PROMPT | prompts.rag / figure |

原有默认模型、切分和检索参数已写入 config.toml。
旧版 Python 配置常量和兼容函数已移除，不再存在两套入口或隐式全局 Runtime。
运行数据根目录由配置决定，与代码所在位置无关。

## 调用迁移

| 原用法 | 新用法 |
|---|---|
| rag.ask(...) | runtime.pipeline.ask(...).answer |
| rag.retrieve(...) | runtime.pipeline.retrieve(...)，返回 SearchHit 列表 |
| ingest.build_index(...) | runtime.ingestion.build(...)，返回 BuildResult |
| models.get_llm() | runtime.generator.load() |
| models.get_embedder() | runtime.embedder.load() |
| models.get_reranker() | runtime.reranker.load() |
| models.get_vlm() | runtime.vision.load() |
| 根目录工具脚本 | scripts/ 下同名文件 |
| tests/run_*_eval.py | tests/evaluation/run_*_eval.py |

脚本和测试使用项目中的 src 包，不修改 sys.path，也不依赖可编辑安装。
根目录 main.py 调用 src.cli.main，接收 build/ask 参数并返回命令退出码。
不再注册自定义命令行程序，也不设置包名到源码目录的映射。
工具与评测从项目根目录使用 python -m scripts.脚本名 或 python -m tests.evaluation.脚本名 运行。
这样 Python 能找到同级的 src 包，无需在每个脚本中修改模块搜索路径。
pyproject.toml 是依赖唯一维护位置，requirements.txt 仅引用本地依赖组。

## 索引一致性与限制

沿用既有 Document pickle 和文件哈希清单；新 Milvus 集合启用动态元数据字段，未自动迁移服务器数据。
增量写入前检查旧集合是否支持动态字段，固定字段集合需要先全量重建；检查失败不修改索引或写入标记。
解析失败时全量构建在写入前停止；增量构建保留失败文件旧内容，其余文件独立更新。
--max-files 不影响删除检测；--only 不删除其他来源，且要求目标存在、已有构建清单。

数据库删除/新增与两个本地文件的发布不在同一事务内。
.pending 标记在写入前创建、所有发布完成后清除；中断后阻止查询和增量构建。
修复原因后必须全量重建，不能单独删除标记假定数据正确。
当前流程适用于单写入者离线构建，不支持并发构建或在线索引切换。

save=False 仍写 Milvus，并清除旧本地缓存及清单，避免词法索引与向量索引混用。
需要混合检索和增量构建时使用默认 save=True。

当前已接入 PDF、TXT、Markdown、DOCX 和 XLSX；尚未实现直接图片向量、原图参与回答、上下文词元预算或父子片段。
历史评测采用近似相关性判断，报告保留原样，不能作为本轮效果提升的证据。

## 文档解析器

`bootstrap.py` 创建 `ParserRegistry`，目前注册 `.pdf`、`.txt`、`.md`、`.docx` 与 `.xlsx`。
`IngestionPipeline` 使用注册器提供的后缀集合发现文件，再通过 `parse(path)` 获取 `ParseResult`。
新增格式时，实现对应解析器并在 Runtime 注册即可进入现有切分与存储流程。

PDF 的正文和回退策略仍在 `pdf.py`，表格与图片辅助函数已从 `multimodal.py` 迁到
`pdf_elements.py`，旧文件已移除。单独提取 PDF 正文可直接使用 `PDFParser.parse_text(path)`；
`runtime.parser` 现在是注册器，只提供统一解析入口，不再提供 PDF 专用的 `parse_text` 方法。

TXT 严格读取 UTF-8（含 BOM），正文不主动切段，不添加页码；读取和编码失败通过 errors 返回。
ContextBuilder 仅在存在有效页码时展示页码，TXT 引用显示来源文件名。
Markdown 使用 markdown-it-py 的 CommonMark 规则并启用表格扩展，依赖首次解析时加载。
语法 token 用于判断边界，元素正文取对应原文行，避免改写代码、列表与链接。
顶层标题更新标题路径；顶层表格单独进入 tables；独立代码块进入 texts 并标记 block_kind=code。
Chunker 只对普通正文做语义切分，独立代码块与表格完整保留。嵌套表格和代码仍属于普通正文，
目前不保证它们在后续语义切分中保持完整，也没有超长代码和表格的专门分段策略。
元数据包含 heading_path、line_start、line_end、block_kind；行号为原文块范围，
不是语义切分后每个片段的精确范围。去重、提示词和导出均保留这些位置字段。
语法和元数据实现参考 [markdown-it-py 官方用法](https://markdown-it-py.readthedocs.io/en/latest/using.html)
及 [LangChain Milvus 实现](https://github.com/langchain-ai/langchain-milvus/blob/main/libs/milvus/langchain_milvus/vectorstores/milvus.py)。
增量删除按文件是否仍存在判断，不能把暂时未注册的格式误判成文件删除。

Word 使用 python-docx 读取主体，`iter_inner_content()` 按段落和表格在文档中的顺序遍历。
标题根据大纲级别、内置标题名称及继承样式识别；连续正文在标题或表格边界处分组。
`paragraph_start/end` 表示正文主体段落序号（计入空段落，不计表格内段落）；
`table_index` 表示顶层表格序号；`block_index` 表示原文中段落和表格混合排列的起始位置。
这些位置随语义切分继承，提示词中的段落范围属于原文块，不是切分后片段的精确范围。
ParseResult 仍按 texts/tables 分类，需要还原文档顺序时可按 block_index 排序。
表格使用通用列名并保留全部原始行，合并单元格重复内容；行首尾缺失的网格补空，
单元格中的换行和竖线转义，嵌套表格的文字按内部顺序保留。
不读取页眉页脚、文本框、脚注、图片描述或修订内容，也不还原列表自动编号和 Word 分页。
已启用动态字段的 Milvus 集合无需因增加 DOCX 再次重建；旧固定字段集合仍需先全量重建。
实现参考 [python-docx 文档接口](https://python-docx.readthedocs.io/en/latest/api/document.html)
与 [表格读取说明](https://python-docx.readthedocs.io/en/latest/user/tables.html)。

全量构建仍会用本次成功解析的文档替换整个索引；取消注册某格式不会保留该格式的全量旧索引。

Excel 使用 openpyxl，分别以 data_only=False/True 读取公式和上次保存的缓存。
公式从不执行，缓存缺失明确标注；缓存是否最新不作保证。工作簿始终不被保存或修改。
合并单元格先展开，空行分隔连续区域，裁去区域两侧空列，保留中间空列和原始行号。
每区开头 header_rows 行作为表头，多行表头按列组合；每 rows_per_chunk 行数据组成一片，重复表头。
header_rows=0 使用列字母作为通用表头；只有表头的区域也保留，避免丢失单行说明。
返回 tables，片段保留 sheet_name/sheet_state、row_start/end、column_start/end 和可选 header_row_start/end，
去重、提示词与导出使用这些位置字段。Chunker 不再对表格做语义切分。
ExcelSettings 从 config.toml 的 [excel] 读取，缺少该段的旧配置使用默认值；Runtime 将配置交给解析器。
隐藏数据会读取，图表和图片不解析；数字保留存储值，日期转为 ISO 文本，不还原 Excel 数字显示格式。
表头行数适用于所有数据区域，尚不支持自动识别异构表头或横向独立表格；行数限制不是词元预算。
max_sheet_cells 限制工作表范围的行列乘积，在工作簿正常模式加载后检查；不构成加载阶段的内存上限。
任何工作表失败均写入 errors，即使已有其他工作表片段，入库也拒绝用该文件的部分内容替换旧数据。
已有动态字段集合可直接新增 XLSX；改变解析参数需全量或使用 --incremental --only 文件名 强制重处理。
接口依据 [openpyxl 官方读取说明](https://openpyxl.readthedocs.io/en/stable/tutorial.html)。

## 目录与数据根

代码、模型权重、原始文档和索引数据都位于项目根目录下：
models/ 保存模型权重，gov_doc/ 保存原始文档，data/ 保存索引、片段缓存和哈希清单。
评测产物与生产数据分开存放：tests/evaluation/ 是评测脚本，tests/results/ 是评测报告
与切分方式对比产物，二者都不参与生产入库和问答。
config.toml 中的路径均为相对路径，以该文件所在目录为基准解析，默认无需传入 --data-root。

要把数据放在项目外时，用 --data-root 或 ENERGY_RAG_HOME 覆盖根目录：

    python main.py --help
    python -m pytest tests/unit -q
    ENERGY_RAG_HOME=/path/to/data python -m pytest tests/integration -v
    python main.py --data-root /path/to/data ask "请概括文档的主要内容。" --no-print

真实生成集成测试额外使用 RUN_LLM_TEST=1。当前本地验证以源码启动和无模型测试为主，
OCR、Milvus、GPU 生成及 RAGAS 仍需在服务器原依赖环境运行。
