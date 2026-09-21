"""State layout helpers for the plug insertion SERL observation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


PROPRIO_KEYS = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")
FLAT_KEY_ORDER = ("gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel")

FLAT_GRIPPER_INDEX = 0
FLAT_TCP_POSE_START = 4
FLAT_TCP_POSE_Z_INDEX = FLAT_TCP_POSE_START + 2


def flatten_gymnasium_dict_order(keys: Iterable[str]) -> list[str]:
    """Return the key order used by gymnasium.spaces.Dict flattening."""
    return sorted(keys)


def _flat_state_array(state) -> np.ndarray:
    arr = np.asarray(state)
    if arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(-1)


def gripper_position(state) -> float:
    return float(_flat_state_array(state)[FLAT_GRIPPER_INDEX])


def tcp_pose_z(state) -> float:
    return float(_flat_state_array(state)[FLAT_TCP_POSE_Z_INDEX])
