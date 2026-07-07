"""Pytest configuration — ensure project root and ui/ on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
for p in (str(ROOT), str(UI)):
    if p not in sys.path:
        sys.path.insert(0, p)
