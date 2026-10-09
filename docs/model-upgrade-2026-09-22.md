# 模型换代记录（2026-09-22）

从 RTX 3080 Ti（12G）换到 RTX 4090（24G）后的模型替换。结论：**显存完全够，
卡住的是数据盘**——50G 且**无法扩容**，因此只能在"少下一点"上做文章。

## 结论

| 项目 | 原来 | 现在 | 磁盘 | 加载后显存 |
|---|---|---|---|---|
| 生成 | Qwen3-8B | **unsloth/Qwen3-14B-bnb-4bit**（预量化 4bit） | 9.93G | 约 9.5G |
| 视觉 | Qwen2.5-VL-3B | **Qwen/Qwen3-VL-8B-Instruct**（官方 bf16，加载时量化） | 17.5G | 约 6G |
| 嵌入 | bge-m3 | 不变 | 2.2G | 2.1G |
| 重排 | bge-reranker-v2-m3 | 不变 | 2.2G | 首次推理后 1.1G |

嵌入与重排不动：bge-m3 与 bge-reranker-v2-m3 已是同尺寸最强，而 9.21 的评测报告把检索
短板定位在 PDF 抽取层（表格列粘连），不在排序算法；换 Qwen3-Embedding-8B 还要全量重建索引。

## 显存基线（换代前实测）

四个组件全部常驻只占 **11G / 24.5G**：生成与视觉本来就是 4bit 加载，余量约 14G。
换 14B 级模型不构成压力，所以决定因素只有磁盘。

**重排的设备需要澄清**：`FlagReranker` 构造完成后权重在 CPU，**首次推理才搬到
`target_devices`（默认解析为 `cuda:0`）**。加载后立刻量显存会得出"重排在 CPU 上跑"的
错误结论（本方案初稿就是这么误判的）。热身后复测：568M 参数全在 `cuda:0`，
20 条长候选重排 0.15s。默认行为已正确，不需要传 `devices=`。

## 小样验证：为什么是"混合"方案

数据盘放不下原始权重（Qwen3-14B bf16 29.6G + Qwen3-VL-8B bf16 17.5G = 47.1G，
删掉两个旧模型后也只有 44.1G），于是先用小模型验证预量化仓库这条路：

- `unsloth/Qwen3-0.6B-bnb-4bit`（0.54G）：**通过**。加载后 196 个 `Linear4bit`，
  聊天模板 + 生成正常，目录自带的量化配置生效。
- `unsloth/Qwen3-VL-2B-Instruct-bnb-4bit`（2.16G）：**失败**。它是"语言侧量化、视觉塔留
  bf16"的混合检查点——语言侧权重存成 `[2097152, 1] U8`（打包量化），视觉塔存成
  `[3072, 1024] BF16`。仓库声明的 `llm_int8_skip_modules` 虽然含 `visual`/`vision_tower`，
  但 transformers 5.16 加载后 `modules_to_not_convert` 是 `None`，视觉塔仍被替换成
  `Linear4bit`/`LinearFP4`，推理时报 `AssertionError`
  （`bitsandbytes/nn/modules.py:498 assert module.weight.shape[1] == 1`）。
  显式指定 `modules_to_not_convert`、以及去掉跳过列表全部量化两种修法都失败。

因此**生成用预量化仓库（省 20G），视觉回到官方 bf16 + 加载时量化**（与换代前加载
Qwen2.5-VL-3B 的方式相同，风险最低）。合计 27.4G，删掉两个旧模型（23.1G）后余约 16.7G。

顺带排除的路线：**AWQ/GPTQ 不可用**。transformers 5 已移除 `AwqConfig`/`GptqConfig`，
环境里也没有 autoawq / gptqmodel / optimum，官方 `Qwen3-14B-AWQ`（9.98G）加载不了。

## 加载预量化检查点的一个坑（已修）

`from_pretrained` 的参数默认值是 `None`，但**显式传 `quantization_config=None` 与不传不是一回事**：
显式传 None 会被 transformers 当成"本次不要量化"，按未量化构建模型，随后去装检查点里打包的
4bit 权重（`weight` 形状 `[N, 1]` 的 uint8 + `.absmax`/`.quant_map` 伴随张量）尺寸对不上，
以 `RuntimeError: You set ignore_mismatched_sizes to False` 加一长串 MISMATCH/UNEXPECTED 报错。
实测四种组合：

| 传参 | 结果 |
| --- | --- |
| 不传 `quantization_config` | 成功，280 个 `Linear4bit` |
| `quantization_config=None` | 失败 |
| `quantization_config=None` + `trust_remote_code=True` | 失败 |
| 不传 + `trust_remote_code=True` | 成功 |

所以 `_quantization_kwargs()` 在目录自带量化配置时返回**空字典**（整键省略），而不是返回 None。
排查依据：加载后 `Linear4bit` 层数应当等于检查点里的量化权重数（14B 是 280）。

## 改了什么

1. `config.toml`：`[generation]`、`[vision]` 的 `model_id` 与 `path`，并写明为什么
   生成用预量化仓库、视觉不能用同类仓库。
2. `scripts/download_models.py`：`MODEL_NAMES` 换成新模型名。
3. `src/models/local_qwen.py`：
   - 新增 `_quantization_config()`：模型目录自带 `quantization_config` 时不再传项目自己的
     `BitsAndBytesConfig`（传了会被忽略，且每次加载都告警）。
   - `load_vlm` 改用 `AutoModelForImageTextToText`，不再写死 `Qwen2_5_VLForConditionalGeneration`
     类名，换视觉模型时不必再改代码。
   - 类与文档字符串改成不写死型号。
4. `README.md`、`docs/server_layout.md` 的模型清单。
5. 换视觉模型后必须**重新入库并重建索引**：图片描述是入库产物，不重跑新模型不生效。

## 已知未做

- `[prompts] figure` 里"不要逐条列出图中的具体数值"的限制保持原样。它是为 3B 模型的读数
  幻觉加的；报告里 figure 题的 context_recall 短板正是图上数据没进描述，换 8B 后值得放宽，
  但要用评测确认收益，不宜凭感觉改。
- `vision.batch_size = 12` 是在 Qwen2.5-VL-3B 上测的，Qwen3-VL-8B 未重测。
- 两套评测（检索、生成）需要重跑，判分走外部 API，全量约 17 元。
