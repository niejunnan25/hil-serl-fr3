"""Pytest config for the hilserl-fr3 repo.

Adds the repo root to ``sys.path`` so test modules can do
``import scripts.zed_env_dispatch`` without requiring the
``PYTHONPATH`` env var. This matches the implicit convention already
used by other ``tests/`` modules in this repo (which import
``scripts.zed_capture`` etc. via the same mechanism in CI).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
