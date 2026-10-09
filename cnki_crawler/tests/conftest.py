"""pytest 配置：让测试能导入 cnki_crawler 包。"""

import sys
from pathlib import Path

# 把包所在目录（energy_rag/）加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
