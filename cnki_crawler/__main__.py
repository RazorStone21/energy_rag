"""包入口，支持 `python -m cnki_crawler`。"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
