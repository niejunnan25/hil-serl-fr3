"""Pure-Python FR3 forward kinematics.

DH parameters from franka_description fr3.urdf.xacro / official Franka docs.
Used by Phase 4 paired comparator to derive ee-pose sequences from joint
trajectories without launching sim (batch-friendly, ~50 microseconds per chunk).

Convention: modified DH (a_{i-1}, d_i, α_{i-1}, θ_i)
- a_{i-1}: link length offset (m)
- d_i: link offset along z (m)
- α_{i-1}: link twist (rad)
- θ_i: joint angle (rad), input

Flange (link8) and TCP (hand_tcp) offsets included by default.
"""
from __future__ import annotations

import dataclasses
import numpy as np


# FR3 modified DH parameters (a_{i-1}, d_i, α_{i-1}) for joints 1-7
# Source: franka_description FR3 kinematics.yaml (official)
_DH = np.array([
    # a_{i-1}, d_i,    α_{i-1}
    [0.0,      0.333,  0.0],         # joint 1
    [0.0,      0.0,    -np.pi / 2],  # joint 2
    [0.0,      0.316,  np.pi / 2],   # joint 3
    [0.0825,   0.0,    np.pi / 2],   # joint 4
    [-0.0825,  0.384,  -np.pi / 2],  # joint 5
    [0.0,      0.0,    np.pi / 2],   # joint 6
    [0.088,    0.0,    np.pi / 2],   # joint 7
])

# Flange offset (link8) — fixed transform from joint 7 frame to flange
_FLANGE_D = 0.107   # d8 along z
# TCP offset (Franka Hand) — fixed from flange to fr3_hand_tcp
_TCP_D = 0.1034
# TCP flange Rz rotation (F_T_EE orientation): matches the live/pinocchio-
# validated hybrid_teleop.CorrectFK T_OFFSET_RZ_DEG = -45.0 (deg).
_TCP_RZ = -np.pi / 4


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Modified DH transformation matrix."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct,      -st,      0,      a],
        [st * ca,  ct * ca, -sa,   -sa * d],
        [st * sa,  ct * sa,  ca,    ca * d],
        [0,        0,        0,     1],
    ])


def fk_ee_pose(q: np.ndarray, *, include_tcp: bool = True) -> np.ndarray:
    """Forward kinematics: 7-joint q → 4x4 ee-frame pose matrix.

    Args:
        q: shape (7,) joint angles in rad
        include_tcp: if True, return pose at fr3_hand_tcp (TCP); if False,
            at fr3_link8 (flange).

    Returns:
        4x4 homogeneous transform from base to ee.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"expected (7,), got {q.shape}")

    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _DH[i]
        T = T @ _dh_transform(a, d, alpha, q[i])

    # Flange offset along z
    flange = np.eye(4)
    flange[2, 3] = _FLANGE_D
    T = T @ flange

    if include_tcp:
        tcp = np.eye(4)
        tcp[2, 3] = _TCP_D
        # Franka-hand flange->TCP fixed Rz(-45deg) (F_T_EE orientation),
        # matching hybrid_teleop.CorrectFK T_OFFSET_RZ_DEG = -45.0 (live-validated).
        c, s = np.cos(_TCP_RZ), np.sin(_TCP_RZ)
        tcp[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = T @ tcp

    return T


def fk_position_seq(q_seq: np.ndarray, *, include_tcp: bool = True) -> np.ndarray:
    """Batch FK: (T, 7) joint trajectory → (T, 3) ee-position trajectory."""
    q_arr = np.asarray(q_seq, dtype=np.float64)
    if q_arr.ndim != 2 or q_arr.shape[-1] != 7:
        raise ValueError(f"expected (T, 7), got {q_arr.shape}")
    out = np.zeros((q_arr.shape[0], 3), dtype=np.float64)
    for t in range(q_arr.shape[0]):
        T = fk_ee_pose(q_arr[t], include_tcp=include_tcp)
        out[t] = T[:3, 3]
    return out


__all__ = ["fk_ee_pose", "fk_position_seq"]
