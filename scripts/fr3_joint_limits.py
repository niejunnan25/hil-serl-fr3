#!/usr/bin/env python3
"""Single source of truth for FR3 joint position limits used by scripts/.

These conservative joint-position clip bounds (radians, 7-DOF) are shared by
every GELLO / teleop / safety script in this directory. They were historically
duplicated verbatim in verify_safety.py, gello_pose_follow.py,
record_hybrid_demos.py, record_gello_demos_serl.py and gello_droid_adapter.py;
this module unifies them so a change happens in exactly one place.

NOTE: these are an intentionally rounded, project-internal envelope, NOT the
exact official Franka FR3 joint limits. Do not "correct" them to the official
values without auditing every np.clip site — they define real-hardware safety
behavior. See the FR3 review doc (Item 21) for the official-limits discrepancy
(e.g. joint 6 lower 0.08 rad is below the official 0.5445 rad minimum).
"""

from __future__ import annotations

import numpy as np

FR3_LOWER_LIMITS = np.array([-2.8, -1.66, -2.8, -2.97, -2.8, 0.08, -2.8])
FR3_UPPER_LIMITS = np.array([2.8, 1.66, 2.8, -0.17, 2.8, 3.65, 2.8])

__all__ = ["FR3_LOWER_LIMITS", "FR3_UPPER_LIMITS"]
