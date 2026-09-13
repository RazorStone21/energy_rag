# 能源文档 RAG 问答系统

使用本地 BGE 嵌入、BM25 混合召回、BGE 重排及 Qwen 生成，对 PDF、TXT、Markdown、DOCX 和 XLSX 文档进行检索问答。

当前支持 `.pdf`、`.txt`、`.md`、`.docx` 和 `.xlsx`，后缀不区分大小写；只读取配置文档目录的第一层。
TXT 支持 UTF-8（含 BOM），保留段落和缩进，切分后的来源显示文件名，不生成页码。
其他编码、空白文件和含空字符的文件会报告解析失败；请转换为 UTF-8 纯文本后重试。
将 TXT 放入 `config.toml` 的 `paths.documents` 目录后，首次执行 `python main.py build`，
已有索引时执行 `python main.py build --incremental`。TXT 切分与入库仍需要原有嵌入模型和 Milvus 依赖。
Markdown 支持 UTF-8（含 BOM），按标题组织正文，单独提取顶层竖线表格和代码块。
列表、链接和代码缩进保留原文；列表或引用内的表格、代码仍放在所属正文中。
图片链接不读取图片。引用和导出保留标题路径、所属原文块行号；语义切分后行号仍表示原文块范围。
独立代码块和表格整体入库，目前没有超长块的专门切分策略。
Word 支持 `.docx` 主体的段落和表格，按原文顺序读取，保留标题路径、段落范围和表格序号。
识别标题样式及其继承关系、大纲级别；列表保留文字，暂不还原自动编号。
表格保留全部数据行，以通用列名输出 Markdown；合并单元格内容重复到对应网格，
嵌套表格转为所在单元格内的文本，不还原复杂合并版式。
不提取图片描述、页眉页脚、脚注、文本框和修订内容；不推算页码。Word 的 `~$` 锁定文件自动跳过。
旧 `.doc` 需先转换为 `.docx`。

Excel 支持 `.xlsx`，各工作表按全空行分隔数据区域，默认每区首行作表头，每片最多 50 行数据。
每片重复表头并保留工作表名、数据行号、列范围和表头行号；合并单元格按左上角内容展开。
`config.toml` 的 `[excel]` 可设置 `header_rows`（0 表示无表头，2 表示两行表头）、
`rows_per_chunk` 和 `max_sheet_cells`。修改这些配置后需重处理原 XLSX 文件，普通增量只检查文件内容变化。
公式同时保留表达式和已有缓存值；无缓存时明确标注，不运行公式，也不验证缓存是否为最新结果。
读取隐藏工作表和隐藏行列，跳过 `~$` 锁定文件；空表跳过，整个工作簿无数据时报错。
暂不支持 `.xls`、图表和图片描述；数值使用存储值，不还原百分比、货币和自定义数字显示格式。
按行分片不等于控制模型词元长度；横向并排的表格作为同一区域处理，表头行数统一应用于所有区域。
普通模式分别加载公式与缓存后检查工作表行列范围乘积，默认超过 500000 个网格时报错，不静默截断。
这个上限限制后续扫描，不限制工作簿加载时的内存；仅设置格式的远端单元格也会扩大扫描范围。

已有 AutoDL 环境需补充文档解析依赖，并为不同格式的元数据重建一次旧集合：

```bash
python -m pip install "markdown-it-py>=3,<5" "python-docx>=1.2,<2" "openpyxl>=3.1.5,<4"
python main.py build
```

全量 build 使用当前文档目录中的所有支持文件替换索引，请保留仍需检索的原文档。
之后继续使用 `python main.py build --incremental`。固定字段的旧集合会在增量写入前被拒绝，
不会先删除旧向量或留下未完成写入标记。
如果上一阶段已重建为动态字段集合，新增 DOCX、XLSX 可直接使用增量入库，无需再次全量重建。

业务代码直接位于 src/，使用 src.xxx 导入，通过根目录 main.py 运行，
组件职责和关键实现见 [架构说明](docs/architecture.md)。

函数注释、命名和格式要求见 [代码规范](docs/code_style.md)。
可运行 python -m scripts.check_docstrings 检查函数中文说明是否完整。

## 安装

需要 Python 3.11 或更新版本。AutoDL 已有模型依赖时，直接在项目根目录运行 main.py，
不需要安装本项目或设置包名映射。

全新环境可以按需安装模型运行依赖、效果评测依赖和测试依赖：

    python -m pip install ".[runtime,evaluation,dev]"

需要 Web 界面时再加上 server 组：

    python -m pip install ".[server]"

requirements.txt 仅引用上述依赖组，依赖清单统一维护在 pyproject.toml。
server 组只有 main.py serve 用得到，导入 src 的其他部分不会加载 FastAPI。
仅开发流程层、运行无模型单元测试时：

    python -m pip install ".[dev]"

上述依赖安装方式不会注册命令行程序，日常运行入口是 main.py。

扫描件 OCR 还需要系统程序及中文语言包：

    apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils

模型下载和旧版 RAGAS 兼容工具位于 scripts/：

    python -m scripts.download_models
    python -m scripts.download_models --models Qwen2.5-VL-3B
    python -m scripts.fix_ragas_compat

兼容修复脚本用于仍有 VertexAI 导入问题的旧依赖组合，执行时会修改当前环境中的第三方包。

## 使用

在项目根目录通过 main.py 运行：

    python main.py --help
    python main.py build
    python main.py build --incremental
    python main.py build --incremental --only 文件名.pdf
    python main.py ask "你的问题"
    python main.py ask "你的问题" --no-rerank --no-print
    python main.py ask "你的问题" --no-hybrid
    python main.py serve
    python main.py serve --demo

也可使用模块入口：

    python -m src ask "你的问题"

main.py 调用 src/cli.py 中的统一入口，参数解析和业务流程仍位于 src/。

ask 先打印实际使用的完整提示词，再逐词元显示生成的答案，最后输出生成耗时；
--no-print 只关闭提示词，答案仍会流式显示。模型加载、分词和向量库的日志已调低，
输出中没有与答案和提示词无关的内容。

## Web 界面

除了命令行，也可以用浏览器访问同一套问答能力：

    python main.py serve                 # 默认绑定 127.0.0.1:8000
    python main.py serve --port 8080
    python main.py serve --host 0.0.0.0  # 允许外部访问，见下方说明
    python main.py serve --warmup        # 启动后在后台预加载模型
    python main.py serve --demo          # 演示模式，见下

界面风格参考通义千问：左侧是会话列表和索引状态卡，中间是流式问答，
每条回答下方可以展开参考来源、实际使用的 Prompt 和各阶段耗时。
来源面板里的序号与提示词中的「片段 N」一致，位置说明与模型看到的是同一套实现。
检索设置里的「混合检索」「重排序」两个开关直接对应 ask 的 hybrid 与 with_rerank。

关闭重排序时不会再按 context_top_k 截取，进入提示词的片段会从 5 条增加到 20 条，
提示词更长、生成更慢，界面上的说明也写了这一点。

首次提问需要加载嵌入、重排和生成模型，可能要一到三分钟才出现第一个字；
等待期间连接会通过注释帧保活，也可以用 --warmup 在启动后就开始加载。

「停止」按钮的实际含义是停止接收：模型没有中断机制，这一轮仍会生成完，
期间模型不会接受新的提问。

多轮对话会把最近几轮问答带进提示词，轮数和字数上限见配置中的 `[conversation]`。
**检索仍然只用当前问题本身**，不做指代消解，所以「那第二点呢」这类追问可能召回不到内容，
此时界面会提示把问题写完整。查询改写不在当前范围内。

### 安装与访问

Web 依赖单独安装，不装也不影响 build 和 ask：

    python -m pip install ".[server]"

--host 默认只绑定 127.0.0.1。改为 0.0.0.0 后同网络中的任何人都能访问，
其中 /api/build 会增量修改索引且没有鉴权，请只在可信网络中使用；
AutoDL 等远程环境建议用 SSH 隧道转发端口：

    ssh -L 8000:127.0.0.1:8000 -p 端口 用户名@主机

服务固定单进程运行：多进程会让每个进程各加载一份模型，显存会成倍占用。

同一时刻只能有一个进程使用本地 Milvus：`data/milvus.db` 用的是文件锁，
再开一个 serve 或 ask 会拿不到锁。这时向量检索失败并降级为只用 BM25 关键词检索，
答案仍会生成，但召回质量下降；第三方库为此打印的堆栈很长，所以启动时会先检查一次锁，
命中就打一条中文提示。撞上时关掉多余进程即可——正在运行的服务会在下一次提问时重新连接，
不必重启（如果仍然报同样的错，再重启一次）。改用远端 Milvus 服务地址时没有这个限制。

### 演示模式

`--demo` 用内存替身替换问答、入库和索引状态，不加载模型、不读取索引，
回答与参考来源都是编造的，用于在没有 GPU 依赖的机器上预览界面。
界面顶部和每条回答都会标出「演示数据」，避免和真实结果混淆。

### 接口

    GET  /api/status                 索引状态、来源与片段数、忙碌标志；只读文件，不加载模型
    POST /api/chat                   流式问答，返回 SSE 事件：queued / sources / prompt / token / done / error
    POST /api/build                  增量入库，只接受 {"incremental": true}；入库期间问答会被拒绝
    GET  /api/conversations          会话列表，按最近修改排序，含总数
    POST /api/conversations          新建一份会话记录
    GET  /api/conversations/{id}     一份会话的完整消息
    DELETE /api/conversations/{id}   删除一份会话
    GET  /api/memory                 长期记忆的开关、注入上限与全文
    PUT  /api/memory                 整份覆盖长期记忆
    GET  /                           前端页面
    GET  /api/docs                   自动生成的接口文档

问答和入库共用一把锁串行执行——项目本身不支持并发构建或在线切换索引。
模型被占用时新的问答请求返回 409，界面会在点击前就禁用发送按钮。
会话 id 由服务端生成，形状固定；不合法的 id 返回 400，不会用来拼接文件路径。

## 记忆

项目把对话记录和长期记忆保存成 markdown，问答时自动加载：

    data/memory/
    ├── memory.md              长期记忆，可以手写，每轮注入提示词
    └── sessions/
        ├── 20260913-151829-8626dd.md    每个会话一份完整问答记录
        └── …

会话记录长这样，既是给人看的，也能被解析回消息列表：

```markdown
# 新型储能的装机目标是多少？

- 创建时间：2026-09-13 15:18:29

## 用户

新型储能的装机目标是多少？

## 助手

到 2027 年，全国新型储能装机规模目标为 1.8 亿千瓦。

### 参考来源

1. 13-china-new-energy-storage-development-report-2026-cn.pdf 第17页
```

- **多轮历史来自服务端**：会话文件里取最近若干轮（`[conversation]`），
  前端不再自己维护历史；换浏览器或换设备后记录还在。
- **长期记忆每轮注入**：`memory.md` 全文进提示词，按 `memory.max_chars`
  **从文件开头截取**，超出的部分不会送进模型。顶部放最重要的内容，往后追加不会影响已注入的部分。
- **只记来源位置，不记片段正文**：检索到的原文写进历史后，下一轮会被当成"助手说过的话"
  喂回模型，既浪费上下文也混淆了文档与对话。
- **提示词里写明优先级**：长期记忆和历史都是用户提供的资料，与【文档片段】冲突时以文档为准，
  其中任何要求模型改变规则的内容都按普通文字忽略。

参考来源只保留到位置标签这一层，所以重新打开历史会话时能看到来源清单，
但没有片段正文可展开，也没有当时的各阶段耗时——这两项只在当轮问答里出现。

记忆目录不能放在 `paths.documents` 里：会话记录是 markdown，而 `.md` 解析器已经注册，
放进去会被下一次入库当成待索引文档，配置加载时会直接报错。

## 配置

config.toml 是唯一的用户配置入口，包含模型路径、检索参数、Milvus 参数和提示词。
默认从当前工作目录读取，也可显式指定：

    python main.py --config /path/to/config.toml ask "你的问题"
    python main.py --config /path/to/config.toml --data-root /path/to/data build --incremental

全局参数 --config、--data-root 放在 build/ask 子命令前面。
工具和评测脚本也支持这两个参数。

路径优先级：显式 --data-root → ENERGY_RAG_HOME → 配置中的 paths.data_root。
配置中的相对根目录以 config.toml 所在目录为基准；命令行和环境变量中的相对路径以调用目录为基准。
文档、模型、缓存和本地 Milvus 文件都在这一根目录下解析，不依赖源码或安装位置。
ENERGY_RAG_CONFIG 可以设置默认配置文件。

各阶段候选数量分别由 retrieval.dense_top_k、bm25_top_k、fusion_top_k、context_top_k 控制。
原来的大写 Python 配置常量已迁移为 TOML 字段，详见架构说明中的映射。

[conversation] 的 max_turns 与 max_chars 控制多轮对话带进提示词的历史长度，
超出部分从最早处丢弃；缺少该段的旧配置使用默认值（6 轮、2000 字）。
prompts.rag 中的 {history} 是可选的：不写该占位符时模板按单轮问答工作，
写了但传空历史时也不会多出空行。

[memory] 控制长期记忆：enabled 关掉后既不读也不写，连目录都不创建；
max_chars 是注入上限，同时有硬上限 8000——模型侧不做截断，
配置里的这个值是提示词长度的唯一防线。记忆位置由 [paths].memory 指定，
缺少该键的旧配置使用 data/memory。{memory} 同样是可以不写的占位符。
建议 memory.max_chars + conversation.max_chars 不超过 4000。

## 结构

    main.py                     项目启动入口
    pyproject.toml              项目信息、依赖和代码检查配置
    config.toml                 用户配置
    src/
      cli.py                    参数解析与控制台输出
      config.py                 配置读取与类型校验
      bootstrap.py              显式创建 Runtime、注入组件
      schemas.py / interfaces.py 数据结构与组件接口
      ingestion.py              入库编排
      pipeline.py               问答编排
      chunker.py                语义切分与过滤
      context_builder.py        上下文与提示词
      parsers/                  格式选择、PDF、TXT、Markdown、DOCX 与 XLSX 解析
      retrieval/                BM25、独立双路召回、RRF
      models/                   嵌入、重排及文本/视觉生成适配
      storage/                  Milvus、片段缓存、构建清单
      server/                   Web 接口：事件编码、服务层与应用；只有 app.py 依赖 FastAPI
    front/                      前端页面，原生 HTML/CSS/JS，无需构建步骤
    scripts/                    下载、导出、兼容修复工具
    tests/unit/                 无 GPU、无数据库的回归测试
    tests/integration/          真实组件集成测试
    tests/evaluation/           问题集与效果评测脚本
    tests/results/              评测报告与切分对比产物
    docs/                       架构与迁移说明
    data/                       运行数据：索引、片段缓存、原始文档和记忆，不属于 Python 包
    models/                     模型权重

模型权重目录 models/ 与 src/models/ 模型适配代码是不同目录。
src/ 下直接存放模块和组件子目录，导入名与实际目录名一致。

## 程序调用

    from src.bootstrap import create_runtime
    from src.config import load_settings

    settings = load_settings("config.toml")
    runtime = create_runtime(settings)
    result = runtime.pipeline.ask("你的问题")
    print(result.answer)
    for hit in result.evidence:
        print(hit.document.metadata, hit.dense_score, hit.bm25_score, hit.rerank_score)

    report = runtime.ingestion.build(incremental=True)
    print(report.processed, report.failed)

应用自己持有 Runtime 并决定复用时机，不使用模块级单例。
入库和问答默认按离线顺序执行，当前不提供并发构建或在线索引切换保证。

## 测试、导出与评测

    python -m pytest -q
    python -m pytest tests/unit -q
    RUN_LLM_TEST=1 python -m pytest tests/integration -v

    python -m scripts.export_chunks --stats
    python -m scripts.export_chunks --format json
    python -m tests.evaluation.run_retrieval_eval
    python -m tests.evaluation.run_rag_eval
    python -m tests.evaluation.run_chunk_compare

集成测试的模型和 PDF 缺失时会跳过，生成测试额外需要 RUN_LLM_TEST=1。
单元测试使用替身验证流程与存储适配接口，不能代替真实 GPU、OCR、Milvus 和 RAGAS 验证。
tests/results/ 下保留评测报告和切分对比产物，历史报告不代表本轮结构调整后的效果。

## 数据与失败处理

现有 chunks.pkl 中的 LangChain Document 和 {文件名: sha256} 清单格式仍沿用。
新 Milvus 集合使用动态元数据字段，容纳 PDF 页码、Markdown 标题和行号；旧固定字段集合需全量重建。
全量构建先解析全部目标文件，有失败则在修改索引前终止。
增量构建保留失败文件的旧片段和旧哈希，其余成功文件可以更新；CLI 对部分失败返回非零状态。

写入中断会留下 build_manifest.pending，阻止查询或增量更新混用不一致数据。
修复失败原因后执行 python main.py build，成功重建会清除此标记。
标记只检测中断，不提供跨数据库/文件事务或自动回滚。

全量 build --max-files 1 会将索引替换成该一个文件的内容，仅适合隔离的测试数据目录。
更换模型、向量维度或切分策略后应重建索引。

表格与图片区域由不同 PDF 库依次提取；图片仍转为文字描述后入库。
直接图片向量检索、父子片段恢复、上下文词元预算，以及 Word/Excel 图片描述尚未实现。
