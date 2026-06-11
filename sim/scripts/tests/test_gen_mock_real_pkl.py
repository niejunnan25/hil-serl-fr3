"""A11: gen_mock_real_pkl.py 必须产 SERL pkl 满足 3 键 image + 25D state + 80% 正样本.

注意: 这是 MOCK 数据, 仅用于 A10/A12 schema smoke; 不是 real data。
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

from sim.data.contract import IMAGE_SHAPE, STATE_DIMS


@pytest.fixture
def tmp_output():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# 1) 产 pkl, 满足 schema (3 image keys, 25D state, 7D action, transition keys)
# ---------------------------------------------------------------------------
def test_gen_mock_real_pkl_creates_valid_schema(tmp_output):
    from sim.scripts.gen_mock_real_pkl import generate_mock_real_pkl
    output_path = os.path.join(tmp_output, "mock_real.pkl")
    transitions = generate_mock_real_pkl(
        output_path=output_path, num_frames=10, pos_ratio=0.8, seed=0,
    )
    # 文件存在
    assert os.path.exists(output_path)
    # 加载 pkl
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, list)
    assert len(loaded) == 10
    # schema 检查
    t = loaded[0]
    assert "observations" in t
    assert "next_observations" in t
    assert "actions" in t
    assert "rewards" in t
    assert "masks" in t
    assert "dones" in t
    obs = t["observations"]
    # 3 image keys
    assert "side_policy" in obs
    assert "wrist_1" in obs
    assert "side_classifier" in obs
    # 25D state
    assert obs["state"].shape == (STATE_DIMS,)
    # image shape
    for k in ("side_policy", "wrist_1", "side_classifier"):
        assert obs[k].shape == IMAGE_SHAPE
        assert obs[k].dtype == np.uint8


# ---------------------------------------------------------------------------
# 2) 80% 正样本率
# ---------------------------------------------------------------------------
def test_gen_mock_real_pkl_has_80_percent_positive(tmp_output):
    from sim.scripts.gen_mock_real_pkl import generate_mock_real_pkl
    output_path = os.path.join(tmp_output, "mock_real.pkl")
    n = 100
    transitions = generate_mock_real_pkl(
        output_path=output_path, num_frames=n, pos_ratio=0.8, seed=0,
    )
    n_pos = sum(1 for t in transitions if float(t["rewards"]) == 1.0)
    n_neg = n - n_pos
    # 80% 正样本 (allow ±2 抖动)
    assert 75 <= n_pos <= 85, (
        f"pos_ratio=0.8 should yield ~80 positives in 100 frames; got {n_pos}/{n}"
    )


# ---------------------------------------------------------------------------
# 3) state dtype float32, action dtype float32
# ---------------------------------------------------------------------------
def test_gen_mock_real_pkl_state_action_dtypes(tmp_output):
    from sim.scripts.gen_mock_real_pkl import generate_mock_real_pkl
    output_path = os.path.join(tmp_output, "mock_real.pkl")
    transitions = generate_mock_real_pkl(
        output_path=output_path, num_frames=5, pos_ratio=0.8, seed=0,
    )
    t = transitions[0]
    assert t["observations"]["state"].dtype == np.float32
    assert t["actions"].dtype == np.float32
    assert t["actions"].shape == (7,)


# ---------------------------------------------------------------------------
# 4) 通过 A10 verify_sim_data.py schema 验证
# ---------------------------------------------------------------------------
def test_gen_mock_real_pkl_passes_a10_verification(tmp_output):
    """A11 产出的 mock real pkl 必须通过 A10 verify_sim_data.py 的全部 check."""
    from sim.scripts.gen_mock_real_pkl import generate_mock_real_pkl
    from sim.scripts.verify_sim_data import verify_pkl
    output_path = os.path.join(tmp_output, "mock_real.pkl")
    generate_mock_real_pkl(output_path=output_path, num_frames=10, pos_ratio=0.8, seed=0)
    result = verify_pkl(output_path)
    # A10 verify 必须全 PASS (除了 image_keys_complete 之外的 image-related 也都 PASS)
    for check_name, passed in result.items():
        assert passed, (
            f"mock real pkl failed A10 verify on {check_name!r}; got result={result}"
        )
