"""A9: FailureScenarioGenerator 必须为 4 类失败各自产出 pkl 文件, 全 reward=0.

复用 sim/data/gello_replay.py 的 replay_pure_fk 骨架产 25D state trajectory,
然后扰动轨迹生成 failure cases。
"""
import os
import pickle
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: 构造一个合成 demo (50 帧, 25D state + 7D action)
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_demo():
    """50 帧合成 demo: 25D gaussian state + 7D random action."""
    rng = np.random.default_rng(0)
    N = 50
    return {
        "joint_poses": rng.normal(size=(N, 7)).astype(np.float64),
        "gripper_states": rng.uniform(0, 1, size=N).astype(np.float64),
        "timestamps": np.linspace(0, 1, N).astype(np.float64),
        "cartesian_deltas": rng.normal(size=(N, 6)).astype(np.float64),
    }


@pytest.fixture
def tmp_pkl_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "failure.pkl")


# ---------------------------------------------------------------------------
# 1) 4 类 generator 全部产出 SERL pkl, schema 与 gello_replay 一致
# ---------------------------------------------------------------------------
def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("failure_class", [
    "mis_alignment", "angle_offset", "insufficient_force", "drop",
])
def test_generator_produces_pkl_with_reward_zero(synthetic_demo, tmp_pkl_path, failure_class):
    """4 类 failure 各自 produce pkl, 全 reward=0."""
    from sim.data.failure_scenario_generator import FailureScenarioGenerator
    gen = FailureScenarioGenerator(seed=42)
    method = getattr(gen, f"gen_{failure_class}")
    transitions = method(synthetic_demo, output_path=tmp_pkl_path)
    # 1) 返回值是 transitions list (not None)
    assert isinstance(transitions, list)
    assert len(transitions) > 0
    # 2) pkl 文件被写出
    assert os.path.exists(tmp_pkl_path)
    # 3) 加载 pkl, 验证 schema + reward
    loaded = _load_pkl(tmp_pkl_path)
    assert isinstance(loaded, list)
    assert len(loaded) == len(transitions)
    for t in loaded:
        assert "observations" in t
        assert "next_observations" in t
        assert "actions" in t
        assert "rewards" in t
        assert "masks" in t
        assert "dones" in t
        # 关键: reward 必须 = 0.0 (failure case)
        assert float(t["rewards"]) == 0.0, (
            f"failure class {failure_class} produced non-zero reward: {t['rewards']}"
        )


# ---------------------------------------------------------------------------
# 2) Schema 满足 contract (25D state, 7D action, transition keys)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("failure_class", [
    "mis_alignment", "angle_offset", "insufficient_force", "drop",
])
def test_generator_pkl_schema_matches_contract(synthetic_demo, tmp_pkl_path, failure_class):
    """4 类 failure pkl 的 state/action shape 必须 match contract."""
    from sim.data.contract import STATE_DIMS, IMAGE_SHAPE
    from sim.data.failure_scenario_generator import FailureScenarioGenerator
    gen = FailureScenarioGenerator(seed=42)
    method = getattr(gen, f"gen_{failure_class}")
    method(synthetic_demo, output_path=tmp_pkl_path)
    loaded = _load_pkl(tmp_pkl_path)
    t = loaded[0]
    # state 25D
    assert t["observations"]["state"].shape == (STATE_DIMS,)
    # action 7D
    assert t["actions"].shape == (7,)
    # 图像 placeholder (3, 128, 128) uint8; failure 阶段不需真实图像
    if "pixels" in t["observations"]:
        pix_key = "pixels" if "pixels" in t["observations"] else "images"
        assert t["observations"][pix_key].shape[-3:] == IMAGE_SHAPE


# ---------------------------------------------------------------------------
# 3) 4 类 generator 互不相同 (扰动策略有差异)
# ---------------------------------------------------------------------------
def test_four_generators_produce_different_trajectories(synthetic_demo):
    """4 类 failure 应该 produce 不同的 perturbed trajectories (扰动策略差异)."""
    from sim.data.failure_scenario_generator import FailureScenarioGenerator
    gen = FailureScenarioGenerator(seed=42)
    trajs = {
        cls: getattr(gen, f"gen_{cls}")(synthetic_demo, output_path=None)
        for cls in ("mis_alignment", "angle_offset", "insufficient_force", "drop")
    }
    # 用 gripper action (action[6]) 区分 insufficient_force vs drop (gripper mutation)
    # drop: 中段 50% 帧 gripper = 0
    # insufficient_force: 最后 10 帧 gripper = 1
    for cls in trajs:
        gripper_actions = np.array([float(t["actions"][6]) for t in trajs[cls]])
        if cls == "drop":
            n_zero = (gripper_actions == 0.0).sum()
            assert n_zero >= len(gripper_actions) * 0.4, (
                f"drop class should have ~50% frames with gripper=0; got {n_zero}/{len(gripper_actions)}"
            )
        elif cls == "insufficient_force":
            n_one_last10 = (gripper_actions[-10:] == 1.0).sum()
            assert n_one_last10 >= 8, (
                f"insufficient_force should have last 10 frames with gripper=1; got {n_one_last10}/10"
            )


# ---------------------------------------------------------------------------
# 4) generate_all 一次性 produce 4 pkl
# ---------------------------------------------------------------------------
def test_generate_all_produces_four_pkls(synthetic_demo):
    """generate_all() 一次性产出 4 个 pkl 文件, 在指定 dir 下."""
    from sim.data.failure_scenario_generator import FailureScenarioGenerator
    with tempfile.TemporaryDirectory() as d:
        gen = FailureScenarioGenerator(seed=42, output_dir=d)
        paths = gen.generate_all(synthetic_demo)
        assert len(paths) == 4
        for cls in ("mis_alignment", "angle_offset", "insufficient_force", "drop"):
            expected = os.path.join(d, f"failure_{cls}.pkl")
            assert expected in paths.values()
            assert os.path.exists(expected)
