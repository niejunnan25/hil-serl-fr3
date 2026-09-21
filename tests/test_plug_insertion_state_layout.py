from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_live_serl_flat_state_layout_indexes_gripper_and_tcp_pose_z():
    from experiments.plug_insertion.state_layout import (
        FLAT_GRIPPER_INDEX,
        FLAT_TCP_POSE_Z_INDEX,
        flatten_gymnasium_dict_order,
        gripper_position,
        tcp_pose_z,
    )

    keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    assert flatten_gymnasium_dict_order(keys) == [
        "gripper_pose",
        "tcp_force",
        "tcp_pose",
        "tcp_torque",
        "tcp_vel",
    ]

    state = np.arange(19, dtype=np.float32).reshape(1, 19)
    assert FLAT_GRIPPER_INDEX == 0
    assert FLAT_TCP_POSE_Z_INDEX == 6
    assert gripper_position(state) == 0
    assert tcp_pose_z(state) == 6


if __name__ == "__main__":
    test_live_serl_flat_state_layout_indexes_gripper_and_tcp_pose_z()
