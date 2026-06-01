"""Root launcher for coursework compatibility."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

from main import main  # noqa: E402


if __name__ in {"__main__", "__mp_main__"}:
    main()
