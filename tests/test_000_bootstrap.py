"""确保标准库 unittest 从 tests 目录发现用例时也能导入 src 布局包。"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

if str(SOURCE_ROOT) not in sys.path:
    # 让 `python -m unittest discover tests -v` 与 pytest 的导入行为保持一致。
    sys.path.insert(0, str(SOURCE_ROOT))
