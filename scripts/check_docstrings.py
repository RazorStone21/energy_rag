"""检查业务、工具、评测和测试函数是否都有中文文档字符串。"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE_PATHS = ("main.py", "src", "scripts", "tests")
CHINESE_CHARACTER = re.compile(r"[\u4e00-\u9fff]")


def inspect_functions(path: Path) -> tuple[int, list[str]]:
    """解析单个源文件，返回具名函数数量及缺少中文说明的函数位置。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    total = 0
    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        total += 1
        description = ast.get_docstring(node)
        if not description or not CHINESE_CHARACTER.search(description):
            errors.append(f"{path}:{node.lineno} {node.name} 缺少中文函数说明")
    return total, errors


def main() -> int:
    """检查启动文件和源码目录中的函数说明，发现缺失说明时返回非零退出码。"""
    project_root = Path(__file__).resolve().parents[1]
    total_functions = 0
    errors = []
    for relative_path in SOURCE_PATHS:
        source_path = project_root / relative_path
        if source_path.is_file():
            files = [source_path]
        else:
            files = sorted(source_path.rglob("*.py"))
        for path in files:
            count, file_errors = inspect_functions(path)
            total_functions += count
            errors.extend(file_errors)
    for message in errors:
        print(message)
    print(f"已检查 {total_functions} 个函数，缺少中文说明 {len(errors)} 个。")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
