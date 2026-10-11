"""对新换的模型做最小加载与推理冒烟测试。

在下载完成后运行，逐个验证嵌入 / 重排 / 生成三个模型都能在这张显卡上加载
并产出结果，同时打印每步的耗时与显存占用，便于判断是否留得住余量：

    python -m scripts.smoke_test_models
    python -m scripts.smoke_test_models --skip generation   # 只测嵌入和重排

生成模型最大，放在最后；用 --skip 可以只验证已经下完的那部分。
"""

import argparse
import time

import torch


def _vram_gb():
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return allocated, reserved, total


def _vram_label():
    info = _vram_gb()
    if info is None:
        return "CUDA 不可用"
    allocated, reserved, total = info
    return f"alloc {allocated:.1f}G / reserved {reserved:.1f}G / 总量 {total:.1f}G"


QUERY = "2024 年我国新能源发电量占全社会用电量的比重是多少？"
DOC = "2024 年，我国新能源发电量占全社会用电量的比重持续提升，风电、光伏发电量快速增长。"


def test_embedding(embedder):
    print("[1/3] 嵌入模型", flush=True)
    start = time.time()
    q = embedder.embed_query(QUERY)
    d = embedder.embed_documents([DOC])
    norm = sum(x * x for x in q) ** 0.5
    print(f"  query 维度={len(q)}  文档维度={len(d[0])}  |query|={norm:.4f}  耗时 {time.time()-start:.1f}s")
    print(f"  显存：{_vram_label()}", flush=True)


def test_reranker(reranker):
    print("[2/3] 重排模型", flush=True)
    start = time.time()
    model = reranker.load()
    scores = model.compute_score([[QUERY, DOC]])
    print(f"  score={scores[0]:.4f}  耗时 {time.time()-start:.1f}s")
    print(f"  显存：{_vram_label()}", flush=True)


def test_generation(generator):
    print("[3/3] 生成模型", flush=True)
    start = time.time()
    answer = generator.generate("用一句话回答：中国新能源发电量占全社会用电量的比重在 2024 年大概是多少？")
    print(f"  回答：{answer}")
    print(f"  耗时 {time.time()-start:.1f}s  显存：{_vram_label()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="新模型加载冒烟测试")
    parser.add_argument("--skip", type=str, default="", help="逗号分隔要跳过的部分：embedding/reranker/generation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    from src.config import load_settings
    from src.models.embedder import Embedder
    from src.models.generator import Generator
    from src.models.reranker import Reranker

    settings = load_settings(args.config, args.data_root)
    print(f"设备：{settings.embedding.device}  初始显存：{_vram_label()}", flush=True)

    if "embedding" not in skip:
        test_embedding(Embedder(settings.embedding))
    if "reranker" not in skip:
        test_reranker(Reranker(settings.reranker_path, max_length=settings.rerank_max_length))
    if "generation" not in skip:
        test_generation(Generator(settings.generation))
    print("全部完成。", flush=True)


if __name__ == "__main__":
    main()
