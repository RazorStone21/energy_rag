# 能源文档 RAG 问答系统

面向中文能源与政务文档的本地检索增强生成（RAG）系统。文档解析、向量化、检索与生成全部在本机完成，不依赖外部服务，适用于对数据不出域有要求的检索问答场景。

系统提供三种使用方式：命令行、Web 界面与 Python 编程接口。

**技术栈**：BGE-M3（嵌入）/ bge-reranker-v2-m3（重排）/ Qwen3-8B（生成）/ Qwen2.5-VL-3B（图表理解）/ Milvus（向量库）/ jieba + BM25（关键词检索）

## 目录

- [核心特性](#核心特性)
- [处理流程](#处理流程)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [使用](#使用)
- [配置](#配置)
- [文档格式支持](#文档格式支持)
- [记忆](#记忆)
- [编程接口](#编程接口)
- [项目结构](#项目结构)
- [测试与评测](#测试与评测)
- [运行数据与失败处理](#运行数据与失败处理)
- [已知限制](#已知限制)
- [相关文档](#相关文档)

## 核心特性

| 特性 | 说明 |
| --- | --- |
| 混合检索 | 向量召回与 BM25 关键词召回各自独立执行，再以 RRF 融合；单路不可用时自动降级 |
| 语义重排 | bge-reranker-v2-m3 对融合结果精排，默认保留前 5 条进入提示词 |
| 多格式解析 | PDF、TXT、Markdown、DOCX、XLSX，保留页码、标题路径、表格行列等位置元数据 |
| 扫描件与图表 | PDF 按 fast → pypdf → OCR 顺序回退提取正文；图表由视觉模型转为文字描述后入库 |
| 多轮问答 | 会话记录落盘于服务端，最近若干轮注入提示词；追问按需改写为独立问句后再检索 |
| 长期记忆 | `memory.md` 每轮注入提示词，可手工维护，也可在 Web 界面查看与编辑 |
| 增量入库 | 按文件内容哈希识别变更，只重处理新增与修改的文件 |
| 演示模式 | `--demo` 以内存替身替换问答、入库与索引状态，无 GPU 依赖即可预览界面 |

## 处理流程

入库：

```
文档目录（仅第一层，按已注册后缀发现文件）
  → 解析（正文 / 表格 / 图表描述，附带位置元数据）
  → 语义切分与过滤
  → 向量化
  → 写入 Milvus + 片段缓存 + 构建清单
```

问答：

```
问题 → 查询改写（仅多轮追问触发）→ 向量召回 ∥ BM25 召回 → RRF 融合 → 重排
     → 上下文与提示词组装 → Qwen 流式生成 → 答案 + 参考来源 + 各阶段耗时
```

## 环境要求

- **Python** 3.11 或更新版本。
- **GPU**：模型默认加载到 `cuda`（见 `config.toml` 的 `embedding.device`）。
- **模型权重**：默认从项目内 `models/` 的四个子目录加载，可参考 `config.toml` 中的 `model_id` 设定。
- **OCR（可选）**：仅扫描件 PDF 需要，需安装系统程序及中文语言包。

## 安装

依赖清单统一维护在 `pyproject.toml`，按使用场景分组安装；`requirements.txt` 仅引用这些依赖组。
安装不会注册命令行程序，日常运行入口是根目录的 `main.py`。

| 依赖组 | 内容 | 使用场景 |
| --- | --- | --- |
| `runtime` | 模型推理与文档解析依赖 | `build`、`ask`、`serve` |
| `server` | FastAPI、uvicorn | 仅 `main.py serve`；导入 `src` 其他部分不会加载 |
| `evaluation` | RAGAS、datasets、pandas、nltk | 效果评测脚本 |
| `dev` | pytest、ruff 及解析依赖 | 测试与静态检查 |

全新环境可按需安装：

```bash
python -m pip install ".[runtime,evaluation,dev]"   # 运行 + 评测 + 测试
python -m pip install ".[server]"                   # Web 界面
python -m pip install ".[dev]"                      # 仅流程层与无模型单元测试
```

AutoDL 等已预装模型依赖的环境，直接在项目根目录运行 `main.py` 即可，不需要安装本项目，也不需要设置包名映射。

扫描件 OCR 另需安装系统程序：

```bash
apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils
```

模型下载与旧版 RAGAS 兼容修复由 `scripts/` 下的工具完成：

```bash
python -m scripts.download_models                      # 下载全部四个模型
python -m scripts.download_models --models Qwen2.5-VL-3B
python -m scripts.fix_ragas_compat                     # 旧依赖组合的兼容修复
```

> `fix_ragas_compat` 用于仍有 VertexAI 导入问题的旧依赖组合，**会修改当前环境中的第三方包**，仅提供 `ChatVertexAI` 占位类以供导入，不提供实际调用能力。

## 快速开始

```bash
python -m scripts.download_models                        # 1. 准备模型权重
python main.py build                                     # 2. 全量构建索引
python main.py ask "新型储能的装机目标是多少？"            # 3. 命令行问答
python main.py serve                                     # 4. 或启动 Web 界面
```

默认从当前工作目录读取 `config.toml`，文档目录为其中的 `paths.documents`。
已有索引时改用增量更新：`python main.py build --incremental`。

## 使用

### 命令行

在项目根目录通过 `main.py` 运行，也可使用模块入口 `python -m src <子命令>`（`main.py` 调用 `src/cli.py` 中的统一入口，参数解析与业务流程仍位于 `src/`）。

```bash
python main.py --help
python main.py build
python main.py build --incremental
python main.py build --incremental --only 文件名.pdf
python main.py ask "你的问题"
python main.py ask "你的问题" --no-rerank --no-print
python main.py ask "你的问题" --no-hybrid
python main.py serve
python main.py serve --demo
```

| 子命令 | 选项 | 说明 |
| --- | --- | --- |
| `build` | `--incremental` | 只处理新增与变更的文件 |
| | `--only 文件名 …` | 仅增量模式可用，强制重处理指定文件 |
| | `--max-files N` | 只处理前 N 个支持的文档；全量构建会替换索引，需配合 `--allow-partial-index`，见[运行数据与失败处理](#运行数据与失败处理) |
| | `--allow-partial-index` | 确认全量构建只保留前 N 个文件的片段（会删除其余来源的索引） |
| `ask` | `--no-rerank` | 跳过重排序，进入提示词的片段从 5 条增加到 20 条 |
| | `--no-hybrid` | 只使用向量召回，不使用 BM25 |
| | `--no-print` | 不打印最终提示词，答案仍流式显示 |
| `serve` | `--host` / `--port` | 监听地址与端口，默认 `127.0.0.1:8000` |
| | `--warmup` | 启动后在后台预加载模型 |
| | `--demo` | 演示模式，不加载模型，见下文 |

`ask` 先打印实际使用的完整提示词，再逐词元显示生成的答案，最后输出生成耗时。
模型加载、分词和向量库的日志已调低，输出中不含与答案和提示词无关的内容。

### Web 界面

```bash
python main.py serve                 # 默认绑定 127.0.0.1:8000
python main.py serve --port 8080
python main.py serve --host 0.0.0.0  # 允许外部访问，见下方安全说明
python main.py serve --warmup        # 启动后在后台预加载模型
python main.py serve --demo          # 演示模式
```

界面布局参考通义千问：左侧为会话列表与索引状态卡，中间为流式问答区。每条回答下方可展开参考来源、实际使用的 Prompt 及各阶段耗时。来源面板的序号与提示词中的「片段 N」一致，位置说明与模型看到的是同一套实现。

检索设置中的「混合检索」「重排序」两个开关直接对应 `ask` 的 `hybrid` 与 `with_rerank`。
关闭重排序后不再按 `context_top_k` 截取，进入提示词的片段由 5 条增加到 20 条，提示词更长、生成更慢，界面上的说明也写明了这一点。

首次提问需要加载嵌入、重排与生成模型，可能需一到三分钟才出现第一个字；等待期间连接通过注释帧保活，也可用 `--warmup` 在启动时开始加载。

**「停止」按钮的实际含义是停止接收**：模型没有中断机制，这一轮仍会生成完，期间不接受新的提问。
流式生成期间会话列表会被锁定，不能切换或删除会话。

多轮对话会把最近几轮问答带进提示词，轮数与字数上限见配置中的 `[conversation]`。
检索使用的是**结合历史改写后的问句**（见配置中的 `[rewrite]`）：像「那第二点呢」这类含指代的追问，会先结合历史还原成独立问句再检索，而提示词中的【问题】始终保留用户原话。改写是尽力而为的增强——没有历史、功能关闭或改写结果不合格时，一律退回原问题检索。

#### 安全与并发

Web 依赖单独安装，不装也不影响 `build` 和 `ask`：

```bash
python -m pip install ".[server]"
```

- `--host` 默认只绑定 `127.0.0.1`。改为 `0.0.0.0` 后同网络中的任何人均可访问，其中 `/api/build` 会增量修改索引且没有鉴权，请仅在可信网络中使用。AutoDL 等远程环境建议改用 SSH 隧道转发端口：

  ```bash
  ssh -L 8000:127.0.0.1:8000 -p 端口 用户名@主机
  ```

- 服务固定单进程运行：多进程会让每个进程各加载一份模型，显存成倍占用。
- 同一时刻只能有一个进程使用本地 Milvus。`data/milvus.db` 使用文件锁，再开一个 `serve` 或 `ask` 会拿不到锁：向量检索失败并降级为只用 BM25 关键词检索，答案仍会生成，但召回质量下降。第三方库为此打印的堆栈很长，因此启动时会先检查一次锁并给出中文提示。撞上时关掉多余进程即可——正在运行的服务会在下一次提问时重新连接，不必重启（若仍报同样的错，再重启一次）。改用远端 Milvus 服务地址时没有该限制。

#### 演示模式

`--demo` 用内存替身替换问答流程、入库流程、索引状态与记忆存储，不加载模型、不读取索引；回答与参考来源都是编造的，用于在没有 GPU 依赖的机器上预览界面。界面顶部与每条回答都会标出「演示数据」，避免与真实结果混淆。

#### HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/status` | 索引状态、来源与片段数、忙碌标志；只读文件，不加载模型 |
| `POST` | `/api/chat` | 流式问答，请求体含 `question`、`hybrid`、`rerank`、`conversation_id` |
| `POST` | `/api/build` | 增量入库，请求体 `{"incremental": true}`（默认值） |
| `GET` | `/api/conversations` | 会话列表，按最近修改排序，含总数，支持 `limit` 参数 |
| `POST` | `/api/conversations` | 新建一份会话记录 |
| `GET` | `/api/conversations/{id}` | 一份会话的完整消息 |
| `DELETE` | `/api/conversations/{id}` | 删除一份会话 |
| `GET` | `/api/memory` | 长期记忆的开关、注入上限与全文 |
| `PUT` | `/api/memory` | 整份覆盖长期记忆；功能关闭时返回 409 |
| `GET` | `/` | 前端页面（静态资源挂载在 `/static`） |
| `GET` | `/api/docs` | 自动生成的接口文档；OpenAPI 描述位于 `/api/openapi.json` |

`/api/chat` 以 SSE 返回事件流，事件类型为 `queued` / `sources` / `prompt` / `token` / `done` / `error`；`/api/build` 只发 `queued` / `done` / `error`。

问答与入库共用一把锁串行执行——项目本身不支持并发构建或在线索引切换。模型被占用时新的问答请求返回 409，界面会在点击前禁用发送按钮。会话 id 由服务端生成，形状固定；不合法的 id 返回 400，不会被用来拼接文件路径。

## 配置

`config.toml` 是唯一的用户配置入口，包含模型路径、检索参数、Milvus 参数与提示词。
默认从当前工作目录读取，也可显式指定：

```bash
python main.py --config /path/to/config.toml ask "你的问题"
python main.py --config /path/to/config.toml --data-root /path/to/data build --incremental
```

全局参数 `--config`、`--data-root` 放在 `build`/`ask`/`serve` 子命令之前；工具与评测脚本同样支持这两个参数。

**路径解析规则**

- 优先级：显式 `--data-root` → 环境变量 `ENERGY_RAG_HOME` → 配置中的 `paths.data_root`。
- 配置文件中填写的相对根目录以 `config.toml` 所在目录为基准；命令行与环境变量中的相对路径以调用目录为基准。
- 文档、模型、缓存与本地 Milvus 文件都在该根目录下解析，不依赖源码或安装位置。
- 环境变量 `ENERGY_RAG_CONFIG` 可指定默认配置文件。

**主要配置段**

| 配置段 | 关键字段 | 说明 |
| --- | --- | --- |
| `[paths]` | `documents`、`chunks`、`manifest`、`memory` | 文档目录、片段缓存、构建清单与记忆目录 |
| `[embedding]` / `[reranker]` / `[generation]` / `[vision]` | `model_id`、`path` | `model_id` 供下载脚本查找仓库，运行时从 `path` 指定的本地目录加载 |
| `[reranker]` | `max_length` | 单条候选参与重排打分的最大词元数，默认 8192（模型上限）。FlagEmbedding 的默认值是 512，超出的正文对重排不可见，长片段会只按开头排序 |
| `[splitting]` | `threshold_type`、`threshold_amount`、`min_chars` | 语义切分阈值与片段过滤下限 |
| `[retrieval]` | `dense_top_k`、`bm25_top_k`、`fusion_top_k`、`context_top_k`、`rrf_k` | 各阶段候选数量与融合参数 |
| `[milvus]` | `collection`、`connection`、`index`、`search` | 集合名、连接地址与索引/搜索参数 |
| `[excel]` | `header_rows`、`rows_per_chunk`、`max_sheet_cells` | Excel 分片与扫描上限，详见[文档格式支持](#文档格式支持) |
| `[conversation]` | `max_turns`、`max_chars` | 带进提示词的历史长度，超出部分从最早处丢弃 |
| `[rewrite]` | `enabled`、`max_turns`、`max_chars` | 多轮追问的查询改写开关，以及送进改写提示词的历史上限 |
| `[memory]` | `enabled`、`max_chars` | 长期记忆的开关与注入上限 |
| `[prompts]` | `rag`、`rewrite`、`figure` | 问答、改写与图表描述模板 |

说明：

- 缺少 `[conversation]`、`[memory]`、`[rewrite]`、`[excel]` 段的旧配置使用各自的默认值。
- `prompts.rag` 必须包含 `{context}` 与 `{question}`；`{history}` 与 `{memory}` 是可选占位符——不写时模板按单轮问答工作，写了但传入空内容时也不会多出空行。
- `[rewrite]` 默认启用，启用时 `prompts.rewrite` 必须包含 `{question}`，配置加载阶段即校验，不会等到加载模型后才失败。
- `memory.max_chars` 另有硬上限 8000：模型侧不做截断，配置中的该值是提示词长度的唯一防线。建议 `memory.max_chars + conversation.max_chars` 不超过 4000。
- `memory.enabled` 关闭后既不读也不写长期记忆，目录也不会创建。
- 原先的大写 Python 配置常量已迁移为 TOML 字段，对应关系见[架构说明](docs/architecture.md)中的映射表。

## 文档格式支持

支持 `.pdf`、`.txt`、`.md`、`.docx`、`.xlsx`，后缀不区分大小写；**只读取 `paths.documents` 目录的第一层**，不递归子目录。
`~$` 前缀的 Office 锁定文件自动跳过。

> **升级提示**：已有 AutoDL 环境需补充文档解析依赖，并为携带新元数据的格式重建一次旧集合：
>
> ```bash
> python -m pip install "markdown-it-py>=3,<5" "python-docx>=1.2,<2" "openpyxl>=3.1.5,<4"
> python main.py build
> ```
>
> 全量构建会用当前文档目录中的所有支持文件替换索引，请保留仍需检索的原文档；之后继续使用 `python main.py build --incremental`。固定字段的旧集合会在增量写入前被拒绝，不会先删除旧向量，也不会留下未完成写入标记。若上一阶段已重建为动态字段集合，新增 DOCX、XLSX 可直接增量入库，无需再次全量重建。

### PDF

正文按 `fast` → `pypdf` → OCR 顺序尝试提取，取首个非空结果。每页的页眉页脚按元素类别排除；表格与图片区域由不同 PDF 库分别提取，图片转交视觉模型生成文字描述后入库。
正文合并跨页段落，并识别「（一）」这类编号层级以还原标题路径。

### TXT

严格读取 UTF-8（含 BOM），保留段落与缩进，不主动切段、不生成页码，引用时显示来源文件名。
其他编码、空白文件以及含空字符的文件会报告解析失败，请转换为 UTF-8 纯文本后重试。

### Markdown

使用 markdown-it-py 的 CommonMark 规则并启用表格扩展。
按标题组织正文并维护标题路径，顶层竖线表格与独立代码块单独提取并整体入库；列表、链接和代码缩进保留原文；列表或引用内部的表格与代码仍放在所属正文中。
图片链接不读取图片。引用与导出保留标题路径和所属原文块的行号；语义切分后，行号仍表示原文块范围。
独立代码块和表格不参与语义切分，目前也没有针对超长块的专门切分策略。

### Word（DOCX）

使用 python-docx 读取文档主体的段落和表格，按原文顺序遍历，保留标题路径、段落范围与表格序号。
标题依据大纲级别、内置标题名称及样式继承关系识别；列表保留文字，暂不还原自动编号。
表格保留全部数据行，以通用列名输出 Markdown；合并单元格的内容重复到对应网格，嵌套表格转为所在单元格内的文本，不还原复杂合并版式。
不提取图片描述、页眉页脚、脚注、文本框和修订内容，也不推算页码。旧版 `.doc` 需先转换为 `.docx`。

### Excel（XLSX）

各工作表按全空行分隔数据区域，默认每区首行作表头，每片最多 50 行数据；每片重复表头，并保留工作表名、数据行号、列范围与表头行号。合并单元格按左上角内容展开。

`[excel]` 支持以下配置：`header_rows`（0 表示无表头，2 表示两行表头）、`rows_per_chunk`、`max_sheet_cells`。**修改这些配置后需重新处理原 XLSX 文件**，普通增量只检查文件内容是否变化。

其他行为与限制：

- 公式同时保留表达式与已有缓存值；无缓存时明确标注。公式从不执行，也不验证缓存是否为最新结果。
- 读取隐藏工作表与隐藏行列；空表跳过，整个工作簿无数据时报错。
- 暂不支持 `.xls`、图表和图片描述；数值使用存储值，不还原百分比、货币和自定义数字显示格式。
- 按行分片不等于控制模型词元长度；横向并排的表格作为同一区域处理，表头行数统一应用于所有区域。
- 普通模式下分别加载公式与缓存后，检查工作表行列范围的乘积，默认超过 `max_sheet_cells`（500000）时报错而非静默截断。该上限约束的是后续扫描，**不限制工作簿加载时的内存**；仅设置格式的远端单元格也会扩大扫描范围。

## 记忆

系统把对话记录和长期记忆保存为 Markdown，问答时自动加载：

```
data/memory/
├── memory.md              长期记忆，可以手写或在 Web 界面编辑，每轮注入提示词
└── sessions/
    ├── 20260913-151829-8626dd.md    每个会话一份完整问答记录
    └── …
```

会话记录既是给人看的文档，也能被解析回消息列表：

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

设计要点：

- **多轮历史来自服务端**：会话文件里取最近若干轮（`[conversation]`），前端不再自己维护历史；换浏览器或换设备后记录仍在。
- **长期记忆每轮注入**：`memory.md` 全文进入提示词，按 `memory.max_chars` **从文件开头截取**，超出部分不会送进模型。因此应把最重要的内容放在顶部，往后追加不会影响已注入的部分。
- **只记来源位置，不记片段正文**：检索到的原文一旦写进历史，下一轮会被当成「助手说过的话」喂回模型，既浪费上下文，也混淆了文档与对话。
- **提示词中写明优先级**：长期记忆与历史都属于用户提供的资料，与【文档片段】冲突时以文档为准；其中任何要求模型改变规则的内容都按普通文字忽略。

参考来源只保留到位置标签这一层，因此重新打开历史会话时能看到来源清单，但没有片段正文可展开，也没有当时的各阶段耗时——这两项只在当轮问答中出现。

> **注意**：记忆目录不能放在 `paths.documents` 内。会话记录是 Markdown，而 `.md` 解析器已经注册，放进去会被下一次入库当成待索引文档；配置加载阶段会直接报错。

## 编程接口

```python
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
```

`create_runtime` 每次创建独立实例，由应用自己持有并决定复用时机，不使用模块级单例。
同一实例内切分与向量库共用嵌入模型，检索、重排与生成分别按需加载；导入包和查看命令帮助都不会加载 GPU 模型。
实例创建后配置快照不再变化，调整参数可用 `dataclasses.replace` 创建新设置再构造新 Runtime。

入库与问答默认按离线顺序执行，当前不提供并发构建或在线索引切换保证。

## 项目结构

```
main.py                     项目启动入口
pyproject.toml              项目信息、依赖与代码检查配置
config.toml                 用户配置
src/
  cli.py                    参数解析与控制台输出
  config.py                 配置读取与类型校验
  bootstrap.py              显式创建 Runtime、注入组件
  schemas.py / interfaces.py 数据结构与组件接口
  ingestion.py              入库编排
  pipeline.py               问答编排
  chunker.py                语义切分与过滤
  headings.py               标题层级识别
  context_builder.py        上下文与提示词
  progress.py               入库进度显示
  parsers/                  格式注册、PDF、TXT、Markdown、DOCX 与 XLSX 解析
  retrieval/                BM25、独立双路召回、RRF
  models/                   嵌入、重排、查询改写及文本/视觉生成适配
  storage/                  Milvus、片段缓存、构建清单、记忆读写
  server/                   Web 接口：事件编码、服务层与应用；只有 app.py 依赖 FastAPI
front/                      前端页面，原生 HTML/CSS/JS，无需构建步骤
scripts/                    模型下载、片段导出、兼容修复与文档字符串检查
tests/unit/                 无 GPU、无数据库的回归测试
tests/integration/          真实组件集成测试
tests/evaluation/           问题集与效果评测脚本
tests/results/              评测报告与切分对比产物
docs/                       架构、代码规范与服务器对照说明
data/                       运行数据：索引、片段缓存、原始文档与记忆，不纳入版本控制
models/                     模型权重，不纳入版本控制
```

业务代码直接位于 `src/`，使用 `src.xxx` 导入。注意 `models/`（模型权重）与 `src/models/`（模型适配代码）是两个不同的目录。
`data/` 与 `models/` 下的内容体积大或含敏感对话记录，已在 `.gitignore` 中排除，需由使用方自行准备。

## 测试与评测

```bash
python -m pytest -q
python -m pytest tests/unit -q
RUN_LLM_TEST=1 python -m pytest tests/integration -v

python -m scripts.export_chunks --stats
python -m scripts.export_chunks --format json
python -m tests.evaluation.run_retrieval_eval              # 纯向量 vs 向量+重排
python -m tests.evaluation.run_retrieval_eval --hybrid     # 融合 vs 融合+重排（问答实际链路）
python -m tests.evaluation.run_retrieval_eval --hybrid --rrf-k 10   # 对比 RRF 参数
python -m tests.evaluation.run_rag_eval                    # 生成评测：全部题目
python -m tests.evaluation.run_rag_eval --limit 30         # 按题型分层抽 30 题
python -m tests.evaluation.run_rag_eval --only-type table numeric
python -m tests.evaluation.run_rag_eval --limit 60 --resume   # 复用已生成的答案接着跑
python -m tests.evaluation.run_chunk_compare
python -m tests.evaluation.run_overlap_eval
python -m tests.evaluation.run_rewrite_eval
```

- 单元测试使用替身验证流程与存储适配接口，不加载模型、不连数据库。
- 集成测试在模型或 PDF 缺失时跳过；真实生成测试额外需要 `RUN_LLM_TEST=1`。
- 评测脚本经 `create_runtime` 构造运行环境，实际执行需要 GPU、本地模型权重与已构建的索引，产物写入 `tests/results/`。
- **生成评测很慢**（121 题要数小时），所以支持分层抽样与断点续跑：每生成一题就把结果写进报告文件，
  中断只损失当前一题；`--limit` 的结果是更大 `--limit` 的前缀，配合 `--resume` 可以分批推进。
  抽样按题型分层，table、numeric 这类小类不会被随机抽没。
- 单元测试不能代替真实 GPU、OCR、Milvus 与 RAGAS 链路验证。
- 检索与生成评测的参数 A/B 结论见[修复记录](docs/fixes-2026-09-21.md)第三节。
- 代码规范、文档字符串要求与本地检查命令见[代码规范](docs/code_style.md)：

  ```bash
  python -m scripts.check_docstrings    # 检查函数中文说明是否完整
  ```

## 运行数据与失败处理

**索引格式**：`chunks.pkl` 沿用 LangChain Document 与 `{文件名: sha256}` 清单格式；Milvus 集合使用动态元数据字段，以容纳 PDF 页码、Markdown 标题与行号等差异字段，旧固定字段集合需全量重建。
更换模型、向量维度或切分策略后应重建索引。

**构建语义**：

- 全量构建会先解析全部目标文件，任一文件失败即在修改索引前终止。
- 增量构建保留失败文件的旧片段与旧哈希，其余成功文件正常更新；CLI 对部分失败返回非零退出码。
- 写入中断会留下 `build_manifest.pending`，阻止查询或增量更新混用不一致数据。修复失败原因后执行 `python main.py build`，成功重建会清除该标记。该标记只用于检测中断，不提供跨数据库/文件的事务或自动回滚。

**注意**：`build --max-files 1` 会把索引替换成该单个文件的内容，仅适用于隔离的测试数据目录。
因此全量构建配合 `--max-files` 会先被拒绝，确认后加 `--allow-partial-index` 才执行，
执行时会再打一条「其余来源会被删除」的告警。增量构建不受影响：
`build --incremental --max-files N` 只更新选中的文件，其余来源的片段保持不动。

`data/chunks.pkl` 是词法检索（BM25）的唯一来源，缺失时混合检索会静默退化成只用向量。
它丢了只能全量重建：增量构建要求缓存存在，会直接报错而不是重建。

## 已知限制

- 表格与图片区域由不同 PDF 库依次提取；图片转为文字描述后入库，不保留原图。
- 尚未实现：直接图片向量检索、父子片段恢复、上下文词元预算，以及 Word/Excel 的图片描述。
- Markdown 的独立代码块与表格整体入库，超长块没有专门切分策略。
- 不支持旧版 `.doc` 与 `.xls`，需先转换为 `.docx` / `.xlsx`。
- 不支持并发构建或在线索引切换，问答与入库串行执行。
- 查询改写是尽力而为的增强，不做多轮迭代；改写结果不合格或调用失败时退回原问题。

## 相关文档

| 文档 | 内容 |
| --- | --- |
| [架构说明](docs/architecture.md) | 模块边界、流程与数据结构、配置迁移对照表 |
| [代码规范](docs/code_style.md) | 命名、注释、格式约定与本地检查命令 |
| [服务器对照](docs/server_layout.md) | 服务器目录与本项目的文件对应关系 |
| [修复记录](docs/fixes-2026-09-21.md) | 可用性缺陷、评测能力与重排截断的改动说明与 A/B 数据 |
| [前端说明](front/README.md) | 前端文件职责、主题定制与渲染规则 |
