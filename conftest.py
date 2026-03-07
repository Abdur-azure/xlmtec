"""
Root conftest.py — ensures the repo root is on sys.path.

This file must sit at the repo root (same level as the lmtool/ package).
It runs automatically before any test collection, making `lmtool`
importable without needing PYTHONPATH or `pip install -e .`.

Works on Windows, macOS, and Linux.
"""

import sys
from pathlib import Path

# Insert repo root at position 0 so it takes precedence over any other installs
repo_root = Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
