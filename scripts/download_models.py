"""按配置中的模型 ID 下载文件到本地模型目录，默认使用 hf-mirror 下载地址。

用法：
    python -m scripts.download_models                                   # 下载全部 4 个模型
    python -m scripts.download_models --models bge-m3,bge-reranker-v2-m3 # 下载指定模型
"""

import argparse
import os

# 未设置 HF_ENDPOINT 时使用镜像；环境中已有的下载地址优先。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 默认禁用 Xet 下载方式以配合镜像；环境中已设置的值不会被覆盖。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.config import load_settings

MODEL_NAMES = ("bge-m3", "bge-reranker-v2-m3", "Qwen3-8B", "Qwen2.5-VL-3B")


def model_catalog(settings):
    """将便于命令行选择的模型名称映射到配置中的仓库标识和下载路径。"""
    return dict(
        zip(
            MODEL_NAMES,
            [
                (settings.model_ids["embedding"], settings.embedding.path),
                (settings.model_ids["reranker"], settings.reranker_path),
                (settings.model_ids["generation"], settings.generation.path),
                (settings.model_ids["vision"], settings.vision.path),
            ],
        )
    )


# 跳过 onnx/tf 等非必需大文件，节省磁盘与时间
IGNORE_PATTERNS = [
    "*.onnx",
    "*.ot",
    "*.msgpack",
    "*.h5",
    "*.tflite",
    "*.gguf",
    "onnx/**",
    "onnx/*",
    "openvino/**",
    "rust_model.ot",
    # 跳过模型说明中的图片和 macOS 附加文件，它们不参与当前模型推理。
    "imgs/**",
    "imgs/*",
    "*.jpg",
    "*.png",
    "*.webp",
    "**/.DS_Store",
    "*.DS_Store",
]


def _already_downloaded(local_dir) -> bool:
    """按 config.json 和权重文件是否存在，初步判断是否可以跳过下载。

    有 safetensors 索引时检查其中列出的分片；否则只检查是否存在可识别的权重文件。
    不检查文件内容、分词器或其他配置，返回 True 不保证模型一定能加载。
    """
    if not (local_dir / "config.json").exists():
        return False
    index = local_dir / "model.safetensors.index.json"
    if index.exists():
        import json

        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards = sorted(set(weight_map.values()))
        return bool(shards) and all((local_dir / s).exists() for s in shards)
    weights = list(local_dir.glob("*.safetensors")) + list(local_dir.glob("pytorch_model.bin"))
    return bool(weights)


def download(name: str, catalog):
    """检查本地配置和权重是否存在，需要下载时按仓库 ID 保存到配置目录。"""
    from huggingface_hub import snapshot_download

    repo_id, local_dir = catalog[name]
    if _already_downloaded(local_dir):
        print(f"[skip] {name} 已存在于 {local_dir}")
        return
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"[done] {name}")


def main():
    """解析下载选项并依次补齐选中的模型，未知名称会给出提示。"""
    parser = argparse.ArgumentParser(description="下载模型")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(MODEL_NAMES),
        help="逗号分隔的模型名，可选：%s" % ", ".join(MODEL_NAMES),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    catalog = model_catalog(load_settings(args.config, args.data_root))
    for n in args.models.split(","):
        n = n.strip()
        if n not in MODEL_NAMES:
            print(f"[warn] 未知模型 {n}，跳过（可选：{', '.join(MODEL_NAMES)}）")
            continue
        download(n, catalog)
    print("全部完成。")


if __name__ == "__main__":
    main()
