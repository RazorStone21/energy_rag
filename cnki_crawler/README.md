# 知网能源/电力论文题录爬虫

抓取知网上能源、电力领域中文期刊论文的**题录**（题名、作者、机构、关键词、
学科领域、摘要、被引/下载量），导出为可直接喂给 energy_rag 的格式。

---

## 一、先读这段：合规边界

### 这个工具做什么

- 只抓取知网**对未登录用户公开**的题录页面（`wap.cnki.net`）。
- 请求间隔默认 **2 秒**，单线程，可断点续传，触发风控立即停止。
- 全文只提供**「你自己合法拿到 PDF → 批量解析入库」**的工具链，工具本身不下载任何全文。

### 这个工具不做什么

- ❌ **不解验证码、不绕滑块**。PC 搜索接口被知网用滑块保护，本工具不试图绕过。
- ❌ **不批量下载全文 PDF**。知网全文需机构订阅，其服务协议禁止批量下载。
- ❌ **不用代理池/IP 轮换规避封禁**。
- ❌ **不做「全量爬取」**。知网文献以亿计，全量爬取在国内已有被追究刑事责任的判例
  （涉及非法获取计算机信息系统数据罪）。本工具的目标是**聚焦领域内的合理规模采集**，
  不是把知网搬空。

### 你应该知道的现实

以默认的 2 秒间隔计算：

| 范围 | 请求数 | 耗时 |
|---|---|---|
| 单刊近 1 年（约 24 期） | ~500 | ~17 分钟 |
| 单刊近 10 年 | ~5,000 | ~2.8 小时 |
| 20 本刊近 10 年 | ~100,000 | ~2.3 天 |

**这是一次以天计的礼貌爬取，不是几分钟跑完的脚本。** 调小 `network.delay`
能更快，但会显著提高被封风险，**不建议**。配置文件里 `delay` 有 1.0 秒的硬下限。

---

## 二、安装

无需额外装包，依赖都是 energy_rag 环境里已有的：

```bash
# 用 energy_rag 的 conda 环境
/root/miniconda3/envs/autodl-tmp/bin/python -m pip install requests beautifulsoup4 lxml tqdm pymupdf

# 跑测试
cd /root/autodl-tmp/energy_rag/cnki_crawler
/root/miniconda3/envs/autodl-tmp/bin/python -m pytest tests/ -q
```

从 energy_rag 仓库根目录运行（包在仓库内）：

```bash
cd /root/autodl-tmp/energy_rag
/root/miniconda3/envs/autodl-tmp/bin/python -m cnki_crawler --help
```

---

## 三、快速开始

```bash
PY=/root/miniconda3/envs/autodl-tmp/bin/python
cd /root/autodl-tmp/energy_rag

# 1. 看看配置里有哪些期刊
$PY -m cnki_crawler status

# 2. 冒烟测试：只爬 1 期，确认能跑通（约 45 秒）
$PY -m cnki_crawler crawl --journal DLXT --years 1 --limit 1

# 3. 爬一本刊近 2 年
$PY -m cnki_crawler crawl --journal DLXT --years 2

# 4. 爬配置里全部期刊近 10 年（就是上表说的几天量级）
$PY -m cnki_crawler crawl --all --years 10

# 5. 导出为 energy_rag 可摄入的 Markdown
$PY -m cnki_crawler export --format md --out data/gov_doc
```

**随时 Ctrl-C 都安全**：进度写进 SQLite，重跑同一条命令自动续传。

---

## 四、命令详解

### `crawl` —— 爬取题录

| 参数 | 说明 |
|---|---|
| `--journal CODE` | 期刊代码，可重复指定。已内置的代码见 `config.toml` |
| `--all` | 爬配置里的全部期刊 |
| `--years N` | 爬最近 N 年（覆盖配置） |
| `--limit N` | 最多处理 N 期，用于冒烟测试 |

### `discover` —— 扩展期刊

```bash
$PY -m cnki_crawler discover --probe
```

探测候选期刊代码。**探测结果会校验刊名是否属于能源/电力领域**——实测中
`SDDL` 返回的是《中学政史地》、`YNDL` 返回《云南地理环境研究》，都返回
HTTP 200 但完全无关，不校验就会污染语料库。

### `export` —— 导出

```bash
$PY -m cnki_crawler export --format md    --out data/gov_doc   # 每篇一个 .md
$PY -m cnki_crawler export --format jsonl --out papers.jsonl
$PY -m cnki_crawler export --format csv   --out papers.csv     # Excel 可直接打开
```

### `pdf ingest` —— 自备全文入库

```bash
$PY -m cnki_crawler pdf ingest ~/my_pdfs --out data/gov_doc --dry-run
$PY -m cnki_crawler pdf ingest ~/my_pdfs --out data/gov_doc
```

扫描你**自己合法获得**的 PDF，抽取前两页文本，与已抓题录做标题匹配，
匹配上的按统一命名复制进语料目录。匹配不上的会如实列出，不猜不塞。

### `status` —— 查看进度

显示题录总数、期次完成情况、以及**摘要被截断的比例**。

---

## 五、拿到完整摘要（可选）

默认的 wap 后端**摘要是被知网服务端截断的约 110 字预览**，记录里会标记
`abstract_truncated: true`，导出的 Markdown 里也会写明「并非完整摘要」。

要拿完整摘要，可以配置你自己浏览器的 cookie 走 PC 后端：

1. 浏览器登录知网（或用你有权限的机构网络访问）；
2. 用浏览器扩展（如 **Get cookies.txt**）导出 `cnki.net` 的 cookie，
   存成 `cnki_crawler/data/cookies.txt`；
3. 重跑爬取。日志会显示 `后端：pc+wap`。

**注意事项**：

- 本工具只是把你自己的访问凭证附在请求里，不做任何伪造或绕过。
- ⚠️ **PC 后端的选择器和端点未经真实 cookie 验证**——开发环境的 PC 接口被
  滑块拦截，无法取得有效会话来实测。如果它不工作，会**自动降级到 wap 后端**，
  不会中断爬取，也不会写入错误数据。首次使用请用 `--limit 1` 验证。
- cookie 过期后自动整轮降级到 wap，不会卡住。

---

## 六、数据质量说明

### 能拿到

题名、作者、机构、关键词、学科领域、期刊名、年卷期、被引/下载量、PDF 大小、
公开页可见的摘要前缀。

### 拿不到

- **完整摘要**（除非配 cookie 走 PC 后端）
- **基金、DOI、中图分类号**（wap 页面没有这两个字段，需 PC 后端）
- **全文 PDF**（需机构订阅）

### 摘要截断的诚实标注

这是本工具的一个设计原则：**截断的摘要绝不能被当成完整摘要**。

- 数据库记录里有 `abstract_truncated` 字段
- 导出的 Markdown 里会附上明确说明
- `status` 命令会统计截断比例

---

## 七、对接 energy_rag

energy_rag 的摄入管道有几条硬约束（见 `src/ingestion.py`），导出格式是为它们量身定的：

| 约束 | 本工具的处理 |
|---|---|
| 只读 `data/gov_doc/` 第一层，子目录被忽略 | 导出为平铺的 `.md` 文件 |
| 只认 `.pdf/.txt/.md/.docx/.xlsx`，不认 JSON | 用 Markdown 而非 JSONL 作为主导出格式 |
| `path.name` 是去重键，文件名必须唯一稳定 | 文件名以知网文章 ID 开头，天然唯一 |
| 没有 sidecar 元数据机制 | 题录字段全部写进 Markdown 正文 |

导出后：

```bash
cd /root/autodl-tmp/energy_rag
/root/miniconda3/envs/autodl-tmp/bin/python main.py build --incremental
```

⚠️ **注意**：向 `data/gov_doc/` 添加文档会改变 chunk 集合，而
`tests/evaluation/questions.json` 里的 `relevant_chunks` 是按 chunk 文本精确
匹配的。加语料后需要重新生成评测集，否则评测结果会失真。建议先导出到独立目录
评估，确认无误再并入正式语料。

---

## 八、故障排查

### 日志出现「触发知网风控」

知网识别出异常访问了。工具会立即停止（这是刻意的——继续请求只会更糟）。

**处理**：等几小时再试；确认 `network.delay` 没被调小。

### 大量「期次抓取失败」

可能是该年份的期号已经枚举完（工具连续 3 期取不到就会换下一年），
也可能是个别期次页面结构特殊。用 `--limit 1 -v` 看详细日志。

### PC 后端不工作

先用 `--limit 1 -v` 确认是不是 cookie 失效。如果端点本身有问题
（见第五节的警告），工具会自动降级到 wap，不影响使用，只是摘要仍是截断的。

### 解析报错 `ParseError`

知网改版了。`tests/fixtures/` 里存着实测页面，跑一下
`python -m pytest tests/ -q` 就能定位是哪个页面结构变了。

---

## 九、项目结构

```
cnki_crawler/
├── config.toml        配置：种子期刊表、限速、路径
├── cli.py             命令行入口
├── fetcher.py         HTTP 层：限速、重试、**反爬检测**
├── parsers.py         页面解析（纯函数，可离线测试）
├── store.py           SQLite + JSONL 双写存储
├── discovery.py       期刊发现与刊名校验
├── crawl.py           爬取编排
├── export.py          导出为 Markdown / JSONL / CSV
├── pdf_ingest.py      自备 PDF 入库工具链
├── backends/
│   ├── wap.py         wap 后端（无需登录，摘要截断）
│   └── pc.py          PC 后端（需 cookie，完整摘要）
└── tests/             39 个离线测试 + 实测页面 fixtures
```

### 技术要点

**反爬检测是核心。** 知网触发风控时返回 **HTTP 200 + 验证码页**。如果不加识别，
验证码页会被当成正常页面交给解析器，最终静默写入垃圾数据——这比直接报错危险
得多。所以 `fetcher.py` 里风控检测**优先于状态码判断**，命中即抛 `BlockedError`。

**双后端自动降级。** PC 后端失败（cookie 失效、端点变更）自动降级到 wap，
并在记录的 `backend` 字段里如实标注数据来源。

**双写落盘。** 每条题录同时写 SQLite 和 JSONL。JSONL 是追加写的，进程被
Ctrl-C 或杀掉时已抓数据不丢。
