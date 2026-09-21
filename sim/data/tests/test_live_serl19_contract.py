"""Live SERL19 state contract for sim replay output.

The real plug_insertion policy observes the post-wrapper flat state:
gripper_pose(1), tcp_force(3), tcp_pose(6), tcp_torque(3), tcp_vel(6).
This test prevents sim replay from reintroducing the legacy 25D
tcp_pose-quaternion / tiled-gripper layout.
"""

import numpy as np


def _demo(n=3):
    return {
        "joint_poses": np.zeros((n, 7), dtype=np.float64),
        "gripper_states": np.linspace(0.25, 0.75, n, dtype=np.float64),
        "timestamps": np.linspace(0.0, 1.0, n, dtype=np.float64),
    }


def test_contract_matches_live_serl19_flat_state_layout():
    from sim.data.contract import STATE_DIMS, STATE_KEY_DIMS, STATE_KEYS_ORDERED

    assert STATE_KEYS_ORDERED == (
        "gripper_pose",
        "tcp_force",
        "tcp_pose",
        "tcp_torque",
        "tcp_vel",
    )
    assert STATE_KEY_DIMS == (1, 3, 6, 3, 6)
    assert STATE_DIMS == 19


def test_capture_observation_emits_live_serl19_flat_state():
    from sim.data import gello_replay
    from sim.kinematics.fr3_fk import fk_ee_pose

    demo = _demo(n=2)
    obs = gello_replay.capture_observation(
        scene=None,
        joint_poses=demo["joint_poses"],
        gripper_states=demo["gripper_states"],
        step_idx=0,
        device="cpu",
        prev_tcp_pose=None,
    )

    state = obs["state"]
    expected_tcp_pos = fk_ee_pose(demo["joint_poses"][0])[:3, 3]

    assert state.dtype == np.float32
    assert state.shape == (19,)
    assert state[0] == np.float32(demo["gripper_states"][0])
    np.testing.assert_allclose(state[1:4], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(state[4:7], expected_tcp_pos, atol=1e-6)
    assert state[7:10].shape == (3,)
    np.testing.assert_allclose(state[10:13], np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(state[13:19], np.zeros(6), atol=1e-7)


def test_replay_pure_fk_outputs_live_serl19_state():
    from sim.data import gello_replay

    transitions = gello_replay.replay_pure_fk(_demo(n=4), max_frames=4)
    assert transitions

    state = transitions[0]["observations"]["state"]
    next_state = transitions[0]["next_observations"]["state"]

    assert state.shape == (19,)
    assert next_state.shape == (19,)


def test_verify_state_keys_order_rejects_legacy_25d_state():
    from sim.scripts.verify_sim_data import verify_state_keys_order

    legacy_state = np.zeros(25, dtype=np.float32)
    assert verify_state_keys_order(legacy_state) is False
