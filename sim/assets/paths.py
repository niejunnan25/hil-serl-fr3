"""sim/assets/paths.py — canonical USD asset path resolver (single source of truth).

All sim-side USD assets ship inside this directory (repo-relative). This module
resolves them WITHOUT any host-specific (fr3-desktop) hardcode and WITHOUT
importing isaaclab or omni, so it is safe to import from any sim-side module
(incl. dev / CI boxes that do not have the fr3-desktop tree).

Override the repo root with the HILSERL_FR3_ROOT env var — this matches the
convention already used by sim/assets/tests/test_files.py.

This is the L1-isolation-clean replacement for the previously divergent USD
path definitions in plug_scene.py, official_fr3_loader.py, and gello_replay.py.
"""
from __future__ import annotations

import os
import pathlib


def _assets_dir() -> pathlib.Path:
    """Resolve the sim/assets directory (env override wins, else module dir)."""
    env = os.environ.get("HILSERL_FR3_ROOT")
    if env:
        return pathlib.Path(env) / "sim" / "assets"
    return pathlib.Path(__file__).resolve().parent


ASSETS_DIR = _assets_dir()

# FR3 robot mesh (the validated repo-local USD; basename fr3.usd).
FR3_USD_PATH = str(ASSETS_DIR / "fr3.usd")
# FR3 mesh patched with finger colliders (built lazily by official_fr3_loader).
FR3_GRIPPER_COLLISION_USD_PATH = str(ASSETS_DIR / "fr3_gripper_collision.usd")
# BULL GN-109K power strip (fixed insertion target) and its tail-cord plug.
STRIP_USD_PATH = str(ASSETS_DIR / "cn_gn109k_strip.usd")
PLUG_USD_PATH = str(ASSETS_DIR / "cn_gn109k_plug.usd")

__all__ = [
    "ASSETS_DIR",
    "FR3_USD_PATH",
    "FR3_GRIPPER_COLLISION_USD_PATH",
    "STRIP_USD_PATH",
    "PLUG_USD_PATH",
]
