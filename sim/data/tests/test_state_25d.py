"""State replay regression tests plus legacy 25D expected-failure history.

The live contract is now SERL19. Do not delete this file silently: it documents
the old A2 hard-freeze and keeps the legacy-only expectations as xfail guards.
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


def test_replay_pure_fk_state_matches_live_contract():
    from sim.data import gello_replay
    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=5), max_frames=5)
    assert len(transitions) >= 1
    state = transitions[0]["observations"]["state"]
    assert state.dtype == np.float32
    from sim.data.contract import STATE_DIMS, STATE_KEYS_ORDERED
    assert state.shape == (STATE_DIMS,), f"state must be {STATE_DIMS}D, got {state.shape}"
    assert len(STATE_KEYS_ORDERED) == 5


def test_replay_pure_fk_action_is_7d_and_in_range():
    from sim.data import gello_replay
    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=5), max_frames=5)
    action = transitions[0]["actions"]
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert action.min() >= -1.01 and action.max() <= 1.01


@pytest.mark.xfail(
    strict=True,
    reason="legacy A2 25D insertion order intentionally superseded by live SERL19",
)
def test_state_keys_ordered_matches_gello_replay_state_construction():
    """gello_replay 内部构造 state 时必须按 STATE_KEYS_ORDERED 的顺序拼接。"""
    from sim.data.contract import STATE_KEYS_ORDERED
    # source of truth for 25D 顺序
    assert STATE_KEYS_ORDERED == ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Arith breakdown 7+6+3+3+1=20 != STATE_DIMS=25 (wrapper.py 注释)。"
        "已知 25D 矛盾：CPU 环境无 isaaclab，无法 env.sample()['state'].shape 实测裁决。"
        "strict xfail：当前算式冲突时记为 xfailed(绿)；若有人把矛盾改成相等而未移除本 marker，"
        "会触发 XPASS->FAILED 强制 reviewer 处理。"
    ),
)
def test_state_25d_breakdown_sums_match_dims():
    """按 wrapper.py 注释 7+6+3+3+1=20 ≠ STATE_DIMS=25 (注释+实测的 25D 不一致是已知问题)。

    硬信号：用 strict xfail 把算术矛盾从 warning 升级为受追踪的 expected failure。
    合并时仍需实测 env.sample()['state'].shape 裁决 20 vs 25，并在裁决后移除本 marker。
    """
    from sim.data.contract import STATE_DIMS
    breakdown_arith = 7 + 6 + 3 + 3 + 1  # tcp_pose(7) + tcp_vel(6) + force(3) + torque(3) + gripper(1)
    assert breakdown_arith == STATE_DIMS, (
        f"Arith breakdown ({breakdown_arith}) != STATE_DIMS ({STATE_DIMS}); "
        f"wrapper.py 注释的 25D 与 7+6+3+3+1=20 算式冲突。A2 hard-freeze STATE_DIMS=25 "
        f"match wrapper.py; 合并时实测 env.sample()['state'].shape 决定最终值。"
    )
