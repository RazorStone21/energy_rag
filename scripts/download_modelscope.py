"""用 ModelScope 下载模型，作为 hf-mirror 通道的补充。

hf-mirror 对 Xet CAS 托管的仓库（如 unsloth 的 bnb-4bit 权重）会回源到
cas-bridge.xethub.hf.co 这个海外 CDN，在国内节点上频繁断连、速度极慢。
ModelScope 上的 Qwen 系列是一手托管、国内 CDN，下载稳定得多。

用法：
    python -m scripts.download_modelscope                 # 下载嵌入 / 重排 / 生成三个模型
    python -m scripts.download_modelscope --models embedding,reranker
"""

import argparse

from modelscope import snapshot_download

from src.config import load_settings

# 命令行名称 -> (模型仓库 ID, 本地目录) 的取值来源
CHOICES = ("embedding", "reranker", "generation", "vision")

# 跳过 onnx/tf 等非必需大文件，与 download_models.py 保持一致
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
    "imgs/**",
    "imgs/*",
    "*.jpg",
    "*.png",
    "*.webp",
    "**/.DS_Store",
    "*.DS_Store",
]


def _catalog(settings):
    return {
        "embedding": (settings.model_ids["embedding"], settings.embedding.path),
        "reranker": (settings.model_ids["reranker"], settings.reranker_path),
        "generation": (settings.model_ids["generation"], settings.generation.path),
        "vision": (settings.model_ids["vision"], settings.vision.path),
    }


def main():
    parser = argparse.ArgumentParser(description="用 ModelScope 下载模型")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(CHOICES),
        help="逗号分隔的模型名，可选：%s" % ", ".join(CHOICES),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()

    settings = load_settings(args.config, args.data_root)
    catalog = _catalog(settings)
    for name in args.models.split(","):
        name = name.strip()
        if name not in CHOICES:
            print(f"[warn] 未知模型 {name}，跳过（可选：{', '.join(CHOICES)}）")
            continue
        repo_id, local_dir = catalog[name]
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[download] {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id,
            local_dir=str(local_dir),
            ignore_file_pattern=IGNORE_PATTERNS,
            max_workers=4,
        )
        print(f"[done] {name}")


if __name__ == "__main__":
    main()
