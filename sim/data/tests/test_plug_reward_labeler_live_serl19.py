"""Reward labeler must read live SERL19 state and respect sim base/world frames."""

import numpy as np


def _live_serl19_state(*, tcp_pos_base, tcp_euler=(0.0, 0.0, 0.0), gripper=0.5):
    state = np.zeros(19, dtype=np.float32)
    state[0] = gripper
    state[4:7] = np.asarray(tcp_pos_base, dtype=np.float32)
    state[7:10] = np.asarray(tcp_euler, dtype=np.float32)
    return state


def test_estimate_tcp_from_state_transforms_base_tcp_to_world_socket_frame():
    from sim.data.plug_reward_labeler import estimate_tcp_from_state

    socket_pos_world = np.array([0.12, 0.0, 0.74], dtype=np.float64)
    socket_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tcp_pos_base = np.array([0.12, 0.0, -0.015], dtype=np.float64)

    state = _live_serl19_state(tcp_pos_base=tcp_pos_base)
    tcp_pos_world, tcp_quat = estimate_tcp_from_state(
        state,
        socket_pos_world,
        socket_quat,
    )

    np.testing.assert_allclose(
        tcp_pos_world,
        socket_pos_world + np.array([0.0, 0.0, -0.015]),
        atol=1e-9,
    )
    np.testing.assert_allclose(tcp_quat, [1.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_label_rewards_accepts_base_frame_live_serl19_state_against_world_socket():
    from sim.data.plug_reward_labeler import DEFAULT_SOCKET_POS, label_rewards

    state = _live_serl19_state(tcp_pos_base=np.array([0.12, 0.0, -0.015]))
    transition = {
        "observations": {"state": state},
        "next_observations": {"state": state.copy()},
        "actions": np.zeros(7, dtype=np.float32),
        "rewards": np.float32(0.0),
        "masks": np.float32(1.0),
        "dones": False,
    }

    labeled = label_rewards(
        [transition],
        mode="sparse",
        socket_pos=DEFAULT_SOCKET_POS,
        approach_reward=0.0,
    )

    assert float(labeled[0]["rewards"]) == 1.0
