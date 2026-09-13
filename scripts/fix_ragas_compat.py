"""处理部分旧版 RAGAS 导入缺失的 chat_models.vertexai 模块时的报错。

脚本会改写当前环境 langchain_community 包内的 vertexai.py，添加占位类供导入检查使用。
它不提供 VertexAI 模型调用能力，也不保证解决其他 RAGAS 兼容问题。

仅在遇到对应导入错误的旧环境中执行：
    python -m scripts.fix_ragas_compat
"""

import argparse
from importlib import import_module
from pathlib import Path

SHIM = """\"\"\"兼容 shim（由 fix_ragas_compat.py 生成）。

langchain-community 0.4.x 已移除 `chat_models.vertexai`（迁移到 langchain-google-vertexai），
但 ragas 仍 import 它用于 `isinstance` 判断，从不会实例化。这里提供占位类即可。
\"\"\"


class ChatVertexAI:  # pragma: no cover（兼容分支不计入覆盖率）
    \"\"\"占位类：仅用于满足 ragas 的 import 与 isinstance 检查，不提供实际能力。\"\"\"
"""


def main():
    """向当前依赖环境写入兼容文件，并通过导入 RAGAS 验证修复结果。"""
    parser = argparse.ArgumentParser(description="修复旧版 RAGAS 的 VertexAI 导入兼容问题")
    parser.parse_args()
    import langchain_community

    pkg = Path(langchain_community.__file__).resolve().parent
    target = pkg / "chat_models" / "vertexai.py"
    target.write_text(SHIM, encoding="utf-8")
    print(f"[fix] 已写入 shim：{target}")

    # 文件写入不代表问题已解决；实际尝试导入 RAGAS，失败时继续抛出错误。
    import_module("ragas")
    print("[fix] ragas 导入成功，兼容问题已解决")


if __name__ == "__main__":
    main()
