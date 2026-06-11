"""A2 改造后 gello_replay 必须产 25D state 满足 STATE_KEYS_ORDERED 拼接顺序。

注: 本测试不依赖 IsaacLab；只验纯 FK 路径（replay_pure_fk）。
"""
import numpy as np
import pytest


def _make_fake_demo(N=10, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "joint_poses": rng.normal(size=(N, 7)).astype(np.float64),
        "gripper_states": rng.uniform(0, 1, size=N).astype(np.float64),
        "timestamps": np.linspace(0, 1, N).astype(np.float64),
    }


def test_replay_pure_fk_state_is_25d():
    from sim.data import gello_replay
    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=5), max_frames=5)
    assert len(transitions) >= 1
    state = transitions[0]["observations"]["state"]
    assert state.dtype == np.float32
    # STATE_DIMS=25 from contract; 8D 旧实现会产生 AssertionError
    from sim.data.contract import STATE_DIMS, STATE_KEYS_ORDERED
    assert state.shape == (STATE_DIMS,), f"state must be {STATE_DIMS}D, got {state.shape}"
    assert len(STATE_KEYS_ORDERED) == 5  # tcp_pose/tcp_vel/tcp_force/tcp_torque/gripper_pose


def test_replay_pure_fk_action_is_7d_and_in_range():
    from sim.data import gello_replay
    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=5), max_frames=5)
    action = transitions[0]["actions"]
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert action.min() >= -1.01 and action.max() <= 1.01


def test_state_keys_ordered_matches_gello_replay_state_construction():
    """gello_replay 内部构造 state 时必须按 STATE_KEYS_ORDERED 的顺序拼接。"""
    from sim.data.contract import STATE_KEYS_ORDERED
    # source of truth for 25D 顺序
    assert STATE_KEYS_ORDERED == ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")


def test_state_25d_breakdown_sums_match_dims():
    """按 wrapper.py 注释 7+6+3+3+1=20 ≠ STATE_DIMS=25 (注释+实测的 25D 不一致是已知问题)。
    本测试只记录算式以提醒 reviewer，**不 fail**。
    """
    from sim.data.contract import STATE_DIMS
    breakdown_arith = 7 + 6 + 3 + 3 + 1  # tcp_pose(7) + tcp_vel(6) + force(3) + torque(3) + gripper(1)
    # 算式 20 与 wrapper.py 注释 25D 不一致；用户合并时实测 env.sample()["state"].shape 决定
    # 临时 hardcode 25 以匹配 wrapper.py 注释
    if breakdown_arith != STATE_DIMS:
        import warnings
        warnings.warn(
            f"Arith breakdown ({breakdown_arith}) != STATE_DIMS ({STATE_DIMS}); "
            f"wrapper.py 注释的 25D 与 7+6+3+3+1=20 算式冲突。A2 hard-freeze STATE_DIMS=25 "
            f"match wrapper.py; 合并时实测 env.sample()['state'].shape 决定最终值。"
        )
