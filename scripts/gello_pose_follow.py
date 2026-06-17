"""gello_pose_follow.py — v2.2.1 B1a

GELLO joint-space follow -> FK -> absolute Cartesian /pose, extracted from the
PROVEN HTTP path in record_gello_demos_serl.py (B-RESEARCH.md). This is the
correct contract for driving franka_server, which exposes ONLY an absolute
`/pose` endpoint ({"arr":[x,y,z,qx,qy,qz,qw]}) — there is NO joint command route.

Pipeline (mirrors record_gello_demos_serl serl:301-331):
    raw GELLO joints (8D: 7 + gripper)
        -> joint target  = q0 + (raw[:7] - raw_gello0) * joint_signs * leader_scale
        -> clip to FR3 joint limits
        -> per-step max_step abort  (max-abs joint delta vs previous target)
        -> cumulative max_total_delta abort  (max-abs joint delta vs q0)
        -> forward_kinematics(target) -> absolute pose [x,y,z,qx,qy,qz,qw]

This module is pure numpy + FK: no network, no real GELLO, no real robot. The
HTTP streaming (read /getstate, FK-bias gate, POST /clearerr + /pose) lives in
p2_t3_e2e_motion_driver.py which consumes this follower.

NOTE: the GelloCartesianDeltaAgent (gello_cartesian_delta_agent.py) is a
DIFFERENT path — it emits a NORMALIZED [-1,1] delta action for the HIL-SERL
env.step (intervention), which must NOT be POSTed to /pose directly. That
confusion was the C2 bug; this module is the raw follow path.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fk_converter import forward_kinematics  # noqa: E402

# Constants mirror record_gello_demos_serl / gello_fr3_desktop_follow (B-RESEARCH).
DEFAULT_JOINT_SIGNS = np.array([1, -1, 1, 1, 1, -1, 1], dtype=float)
DEFAULT_LEADER_SCALE = 0.50
DEFAULT_MAX_STEP = 0.003          # rad, per-step max-abs joint delta
DEFAULT_MAX_TOTAL_DELTA = 0.03    # rad, cumulative max-abs joint delta from q0

from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS  # noqa: E402
FR3_DEFAULT_JOINTS = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])


def joint_target(
    q0: np.ndarray,
    raw_gello0: np.ndarray,
    raw: np.ndarray,
    joint_signs: np.ndarray = DEFAULT_JOINT_SIGNS,
    leader_scale: float = DEFAULT_LEADER_SCALE,
) -> np.ndarray:
    """q0 + (raw[:7] - raw_gello0) * joint_signs * leader_scale, clipped to limits."""
    q0 = np.asarray(q0, dtype=float).flatten()[:7]
    raw_gello0 = np.asarray(raw_gello0, dtype=float).flatten()[:7]
    raw = np.asarray(raw, dtype=float).flatten()[:7]
    raw_delta = raw - raw_gello0
    target = q0 + raw_delta * np.asarray(joint_signs, dtype=float) * float(leader_scale)
    return np.clip(target, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)


class GelloPoseFollower:
    """Stateful GELLO joint-follow -> absolute pose, faithful to the proven
    record_gello_demos_serl path (abort-on-unsafe semantics, not rate-limiting).

    Args:
        q0:           robot's joint positions at start (7,), read from /getstate['q'].
        raw_gello0:   first GELLO reading's 7 joints (the leader anchor).
        joint_signs:  GELLO->FR3 joint sign map.
        leader_scale: GELLO leader scale.
        max_step:     per-step max-abs joint delta (rad); exceed -> abort (ok=False).
        max_total_delta: cumulative max-abs joint delta from q0 (rad); exceed -> abort.
    """

    def __init__(
        self,
        q0: np.ndarray,
        raw_gello0: np.ndarray,
        joint_signs: np.ndarray = DEFAULT_JOINT_SIGNS,
        leader_scale: float = DEFAULT_LEADER_SCALE,
        max_step: float = DEFAULT_MAX_STEP,
        max_total_delta: float = DEFAULT_MAX_TOTAL_DELTA,
    ) -> None:
        self.q0 = np.asarray(q0, dtype=float).flatten()[:7]
        self.raw_gello0 = np.asarray(raw_gello0, dtype=float).flatten()[:7]
        self.joint_signs = np.asarray(joint_signs, dtype=float)
        self.leader_scale = float(leader_scale)
        self.max_step = float(max_step)
        self.max_total_delta = float(max_total_delta)
        # Previous *accepted* target; seeds at q0 so the first step's delta is
        # measured from the robot's actual start pose (first command ~= FK(q0)).
        self.prev_target = self.q0.copy()
        self.step_count = 0

    def step(self, raw: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray], bool, dict]:
        """Compute the next joint target + absolute pose from a GELLO reading.

        Returns (target_joints, abs_pose, ok, info). When ok is False the step
        is unsafe (caller must abort the stream); abs_pose is None in that case
        and prev_target is NOT advanced.
        """
        target = joint_target(
            self.q0, self.raw_gello0, raw, self.joint_signs, self.leader_scale
        )

        step_delta = float(np.abs(target - self.prev_target).max())
        if step_delta > self.max_step:
            return target, None, False, {
                "violation": f"max_step exceeded: {step_delta:.6f} > {self.max_step}",
                "step_delta": step_delta,
                "total_delta": float(np.abs(target - self.q0).max()),
            }

        total_delta = float(np.abs(target - self.q0).max())
        if total_delta > self.max_total_delta:
            return target, None, False, {
                "violation": f"max_total_delta exceeded: {total_delta:.6f} > {self.max_total_delta}",
                "step_delta": step_delta,
                "total_delta": total_delta,
            }

        abs_pose = forward_kinematics(target)
        self.prev_target = target.copy()
        self.step_count += 1
        return target, abs_pose, True, {
            "violation": "",
            "step_delta": step_delta,
            "total_delta": total_delta,
        }
