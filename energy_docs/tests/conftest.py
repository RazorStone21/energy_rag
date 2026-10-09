"""pytest 配置：让测试能导入 energy_docs 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
