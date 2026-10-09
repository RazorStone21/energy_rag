# PDF 解析层改用 MinerU（2026-09-22）

把 PDF 抽取从 `unstructured` + `pdfplumber` + PyMuPDF 自绘，换成 MinerU 统一出版面块。
动因是 9.21 评测报告定位的头号短板：表格类题目 Context Precision 0.7064（全场最低），
根因在抽取层——双栏正文被当成表格"拍平"后列粘连，词语被拦腰截断、数字丢失。

## 实测结论（MinerU 4.0.3）

| 项 | 结论 |
| --- | --- |
| 调用方式 | 进程内 Python API：`mineru.parse(path, tier=..., page_range=...)`，结果在 `ParseResult.middle_json`；对外稳定接口是 `mineru.render.render_content_list()`，图片字节走 `docvortex.export.materialize_middle()` |
| **不要用 CLI** | `mineru parse` 走 doclib 客户端，只导出 Markdown，且 `--pages` **默认只解析前 10 页** |
| 档位 | 用 **`basic`**：4 页 11 秒、8 页 13 秒、峰值显存 **1.3G**，照样输出 `table`/`chart`/`image` 块（含裁剪图与题注） |
| 为什么不用 `standard` | 它多跑一遍自带 VLM，实测 **4 页 254 秒（63 秒/页）**：`mineru_llama_cpp` 这个 wheel 只有 CPU/Vulkan 后端、没有 CUDA，系统也缺 Vulkan loader。而我们本来就要用自己的 Qwen3-VL-8B 描述图表（口径与现有评测一致） |
| 块类型 | `text` / `paragraph_title` / `doc_title` / `table` / `chart` / `image` / `code` / `list` / `index` / `header` / `page_number` / `page_footnote` |
| bbox | **0-1 归一化浮点**（`extensions.docvortex_layout` 另给每页 `width_pt/height_pt`） |
| `content` | 是 span 列表（`[{"type":"text","content":"…","styles":[]}]`），不是字符串 |
| 题注 | 直接给：`table_caption` / `chart_caption` / `image_caption`，不必再靠几何位置猜 |
| 表格 | `table_body` 是 HTML `<td>` 结构，行列正确 |
| 踩过的坑 | 解析内部用 **spawn** 进程池，调用入口必须是真实文件（`main.py` 满足）；用 `python - <<EOF` 从 stdin 调用会以 `BrokenProcessPool` 崩溃，报错信息指向 `FileNotFoundError: <stdin>` |

### 旧实现的"表格"是什么样

06 号文件第 10 页，旧实现产出的 table 片段（摘录）：

```
| 将绿证合作列为政府交流重点议题, 鼓励社会组织、 研究机构、 | 行业 |
| --- | --- |
| 协会等积极发挥桥梁纽带作用, 多种途径推动中国绿证与国际融 | 合衔 |
```

同一页 MinerU 的 `text` 块：

```
将绿证合作列为政府交流重点议题, 鼓励社会组织、 研究机构、 行业协会等积极发挥桥梁纽带
作用, 多种途径推动中国绿证与国际融合衔接, 加快绿证国际互认进程; …
```

也就是说那两页**本来就没有表格**，是旧的双策略表格检测把双栏版式误判成表格。MinerU 正确
识别了阅读顺序，问题不是"表格解析变好"，而是"不再制造假表格"。

## 设计

### 数据流

```
PDF → MineruEngine.parse_document(path) → list[MineruBlock]
        ↓ blocks_to_parse_result()
ParseResult{texts, tables, figures} → Chunker.split → 索引
        ↓ 图表块
   Qwen3-VL-8B（现有 VisionGenerator.describe_batch，分批 + 整批失败降级逐张）
```

### 模块

| 文件 | 动作 |
| --- | --- |
| `src/parsers/mineru_engine.py` | 新增。MinerU 的唯一边界：懒加载、单文件解析、按 (路径, mtime, size) 复用、`release()` |
| `src/parsers/mineru_pdf.py` | 新增。块→ParseResult：页码 +1、`block_index` 页内序号、`heading_path`、题注、HTML 表格→Markdown（含按行切片）、图表交给视觉模型 |
| `src/parsers/pdf.py` | 重写为 MinerU 单路径；**删除** unstructured 三级回退、pypdf 回退、pdfplumber 表格、PyMuPDF 图表自绘与矢量图推断 |
| `src/parsers/pdf_elements.py` | **删除**；只把仍要用的两块搬进 `mineru_pdf.py`：分批描述（`_collect_descriptions` 的语义）与题注清洗（`normalize_caption` / `caption_prompt`） |
| `src/chunker.py` | `split_texts` 的分派从"有没有 `page`"改成"`page` + `block_kind`"：MinerU 的块逐块切分，不跨页合并、不按句回填页码 |
| `src/config.py` + `config.toml` | 新增 `[mineru]` 段：`tier`、`home`、`model_source`、表格切片参数、图表最小尺寸比例 |
| `src/bootstrap.py` | 装配 `MineruEngine`；新增 `release_mineru()`，入库结束后归还显存 |
| `src/storage/chunks.py` + `src/ingestion.py` | manifest 增加解析器指纹：换解析器后增量构建不再错误地跳过未改文件 |

### metadata 契约

| 块 | metadata |
| --- | --- |
| 正文 | `{source, page(1起), type:"text", block_kind:"section", block_index, bbox, heading_path?, parser:"mineru"}` |
| 表格 | `{… type:"table", block_kind:"table", block_index, bbox, caption?, table_index, row_start, row_end}` |
| 图表 | `{… type:"figure", block_kind:"figure", block_index, bbox, caption?}` |
| 代码/列表 | `block_kind` 设为 `code`/`list`，走 `chunker.WHOLE_UNIT_KINDS` 整块保留 |
| 丢弃 | `header`/`page_number`/`page_footnote`/`index`（目录页）：每页重复或纯导航，进索引只会污染检索 |

`schemas.document_key` 已包含 `page`/`block_index`/`bbox`，三者组合唯一，无需改 schema。

### 显存

| 阶段 | 组成 | 合计 |
| --- | --- | --- |
| 入库 | MinerU 1.3G + 视觉 6G + 嵌入 2.1G | ≈ 9.4G |
| 问答 | 14B 9.25G + 重排 1.1G + 嵌入 2.1G | ≈ 12.5G |

入库前已有 `release_models()`；本次补上**入库后 `release_mineru()`**，避免服务端 warmup
之后再做构建时两边叠加。

### 依赖

- 新增：`mineru[torch]`（依赖预演：新增 39 个包，**覆盖现有包 0 个**，与 transformers 5.16.1 / torch 2.14 / pydantic 2.13.5 全部兼容）。
- 移除：`unstructured`、`unstructured-inference`、`unstructured.pytesseract`、`pdfplumber`（仅 `pdf_elements.py` 在用）。
- **保留 `pymupdf`**：`cnki_crawler/pdf_ingest.py` 仍在用；其余 `pikepdf`/`pdf2image`/`pi-heif` 待确认无引用后再决定。
- MinerU 权重：`MinerU-4_models_torch` 0.89G（layout/公式/OCR/表格），放在 `data/mineru_home`（`MINERU_HOME`），**不下载 standard 档的 2.33G VLM**。

## 验证

1. 单测：新增 MinerU 路径用例（用假 engine，不加载模型）；删除锁定旧实现的用例。
2. 单文件冒烟：`runtime.parser.parse(<真实 PDF>)` → 断言无 errors、metadata 含 `page/block_index/bbox`，且**不含** `page_number`/`header` 块。
3. 全量重入库 47 份 PDF，检查：失败 0；`table` 片段内容为结构化行列（不再出现"…融 | 合衔"这类断词）；`figure` 片段带题注。
4. 重跑检索与生成评测，与 `tests/results/` 旧报告对比——**并注明跨解析版本不可比**（检索评测的命中判定依赖旧切分的文本片段，换解析器后标注与片段的文本对齐关系会变）。

## 冒烟测试与实测数据（2026-09-22）

单文件端到端（`13-china-new-energy-storage-development-report-2026-cn.pdf`，78 页）：

```
耗时 387s  正文 324  表格 0  图表 31  错误 []
正文元数据: {"page": 1, "type": "text", "block_kind": "section", "block_index": 1,
             "parser": "mineru", "bbox": [165.0, 213.0, 835.0, 318.0]}
图表元数据: {"page": 14, "type": "figure", "block_kind": "figure", "block_index": 4,
             "caption": "图1 截至2025年底全球新型储能累计装机规模及各技术路线占比", ...}
```

- 章节路径、页码、页内块序号、题注都对；页眉/页码/目录块被丢弃（正文覆盖 74/78 页）。
- 该文件"表格 0"是**正确**结果：旧索引在它上面的 9 条 table 片段全是误判——第 7—9 页是目录，
  22/29/31/33/42 页 MinerU 只看到正文与图表，没有表。真表格出现在带统计附表的文件
  （如核电报告第 15/43/44 页，HTML 行列正确且带题注）。
- 图表描述现在由 Qwen3-VL-8B 生成，实测约 **10 秒/张**（旧 3B 模型批量下约 1 秒/张）：
  更慢但描述质量明显更高（会给出图例、坐标轴、中心数值与趋势）。全量语料按图数量
  估算要多花约 1 小时，这是本次换代最直接的时间代价。

### 两个实现上的修正

1. **标题只认项目自己的编号规则**。MinerU 的 `text_level` 会把加粗长句也标成
   `paragraph_title`：实测它把「截至2025年底，我国新型储能装机比"十三五"末增长超40倍」
   当成标题，照单全收会让整句正文进入 `heading_path` 与引用标签。现在用
   `headings.heading_level()` 判定，与切分器共用同一口径。
2. **`available()` 必须先设环境变量再 import**。MinerU 在 import 时读 `MINERU_HOME`
   并缓存模型根目录；先 import 再 `setdefault` 会把根目录锁到默认的 `~/.mineru`，
   之后所有就绪检查都报"模型未就绪"。这个顺序问题排查了半小时，报错信息完全指向别处。

## 全量重入库与检索评测（2026-09-22 晚）

**构建**：47 份全部成功、零失败，耗时约 1 小时 30 分（含图表描述的 ~10 秒/张）。
新旧索引对比：

| | 旧（unstructured + pdfplumber） | 新（MinerU） |
| --- | --- | --- |
| 片段总数 | 1625 | 3604 |
| 正文 | 1344 | 3308 |
| 表格 | 117 | 66（其中 31 条是长度 >400 的真数据表，仅 5 条是公文抬头元数据） |
| 图表 | 164 | 230（182 条带题注） |

表格数下降是**预期的**：旧索引里大量"表格"是目录页与双栏正文的误判。新索引里
核电机组统计表这类真表格保持了行列结构（合并单元格按占位对齐），并且自带题注。

**检索评测（`run_retrieval_eval --hybrid`，113 题）**：

| 指标 | 旧（unstructured） | 新（MinerU） |
| --- | --- | --- |
| hybrid_only MRR | 0.7303 | 0.4701 |
| hybrid_rerank MRR | 0.8193 | 0.5165 |
| 标注对齐率 | 119/131 = 90.8% | **52/131 = 39.7%** |

**这两组数字不可比**，原因是尺子坏了而不是检索变差：

1. 检索评测的命中判定是"检索到的片段与 `relevant_chunks` 标注存在子串包含或 3-gram
   Jaccard ≥ 0.6"。标注是从**旧切分**的片段里抄出来的原文，重切之后它们大多不再
   与任何一条新片段逐字重合——对齐率从 91% 掉到 40%。
2. 用最近邻对齐量化：131 条标注里 46% 能在新索引找到高度相似（≥0.6）的片段，
   26% 几乎找不到对应——后者主要是 figure 题（标注是**旧视觉模型**写的描述，换模型
   后措辞完全不同）与旧索引里那些被误判成表格的片段。
3. **文档级命中率 101/103 = 98.1%**（把标注映射回它原本所属的文件，再看新检索是否
   召回同一份文档）。这一指标不受切分方式影响，说明检索与重排本身工作正常；两例
   未召回都是语料中三份名称几乎相同的"十五五"规划（15/16/17 号）互相混淆。

**结论**：新报告是**独立的新基线**，不能与 `docs/evaluation-report-2026-09-21.md` 的
数字横向比较。要得到可对比的片段级指标，需要按新切分重新标注（或做"标注 → 新片段
最近邻对齐"），这是已知的下一步；文档级命中率与拒答率不受此影响，可作为长期跟踪口径。

## 已知遗留

- MinerU 会把数字写成 `6 248` 这种带空格的千分位形式，与旧抽取的 `6248` 不同；评测的确定性数字校验需要兼容。
- `[prompts] figure` 里"不要逐条列出图中的具体数值"的限制仍然保留，是否放宽要用评测决定。
- `tests/evaluation/run_chunk_compare.py` 依赖 unstructured 的元素切分，随旧解析器一起需要改写或停用。
