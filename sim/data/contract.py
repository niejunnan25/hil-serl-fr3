"""sim/data/contract.py

sim-side single source of truth for cross-module contracts that must
align with the v2.0 real-side byte-for-byte compatible action pipeline.

Design source: docs/superpowers/specs/2026-06-11-sim-build-fork-v2.1-design.md
               .planning/sim-build-fork/DESIGN.md
Reference (real-side): experiments/plug_insertion/config.py::EnvConfig.ACTION_SCALE

This module deliberately does NOT import from experiments/, scripts/,
or the real EnvConfig — that would breach the L1 isolation gate. The
constant values are pinned by hand to the real-side byte layout.
"""
from __future__ import annotations

import numpy as np

# 7D action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
# Real-side value: EnvConfig.ACTION_SCALE in experiments/plug_insertion/config.py
#  - dx,dy,dz: 0.015 m (1.5 cm) per normalized unit
#  - droll,dpitch,dyaw: 0.1 rad per normalized unit
#  - gripper: 1.0 (open/close is already in [-1, 1])
ACTION_SCALE = np.array(
    [0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0],
    dtype=np.float64,
)

# Per-joint-axis index helpers (avoid magic numbers at call sites).
ACTION_POS_IDX = slice(0, 3)      # dx, dy, dz
ACTION_RPY_IDX = slice(3, 6)      # droll, dpitch, dyaw
ACTION_GRIPPER_IDX = 6            # gripper open/close

# Joint-state dimension: 7 arm + 1 gripper = 8D observation state
STATE_DIM = 8
JOINT_DIM = 7
