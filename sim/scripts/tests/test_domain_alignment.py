"""A11: test_domain_alignment.py 必须能 end-to-end 跑通, 打印 report.

⚠️ CRITICAL (codex #6 修复): 本测试 = "脚本能跑通不 crash + 打印报告", NOT accuracy-based。
真正 domain alignment 验证 (KL divergence / 图像分布同分布) = phase6-ready gate (用户批准)。

测试:
  1. 脚本能 import
  2. main() 能跑通 exit 0
  3. report dict 包含 image mean/std + state range overlap
"""
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def sim_pkl_path():
    """A9 风格的 sim pkl (全 reward=0, live state, 3 image keys)."""
    from sim.data.contract import IMAGE_SHAPE, STATE_DIMS
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sim.pkl")
        rng = np.random.default_rng(0)
        N = 10
        transitions = []
        for i in range(N):
            state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
            next_state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
            action = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
            base = rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
            obs = {"state": state, "side_policy": base.copy(),
                   "wrist_1": rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8),
                   "side_classifier": base.copy()}
            next_obs = {**obs, "state": next_state}
            transitions.append({
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": np.float32(0.0),
                "masks": np.float32(0.0 if i == N - 1 else 1.0),
                "dones": (i == N - 1),
            })
        with open(p, "wb") as f:
            pickle.dump(transitions, f)
        yield p


@pytest.fixture
def mock_real_pkl_path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mock_real.pkl")
        from sim.scripts.gen_mock_real_pkl import generate_mock_real_pkl
        generate_mock_real_pkl(output_path=p, num_frames=10, pos_ratio=0.8, seed=0)
        yield p


# ---------------------------------------------------------------------------
# 1) 脚本能 import
# ---------------------------------------------------------------------------
def test_domain_alignment_module_importable():
    from sim.scripts import test_domain_alignment
    assert hasattr(test_domain_alignment, "main")
    assert hasattr(test_domain_alignment, "compute_alignment_report")
    assert hasattr(test_domain_alignment, "print_report")


# ---------------------------------------------------------------------------
# 2) compute_alignment_report 返回 dict 包含 image stats + state range
# ---------------------------------------------------------------------------
def test_compute_alignment_report_returns_dict(sim_pkl_path, mock_real_pkl_path):
    from sim.scripts.test_domain_alignment import compute_alignment_report
    report = compute_alignment_report(sim_pkl_path, mock_real_pkl_path)
    assert isinstance(report, dict)
    # 必须有 image mean/std + state range overlap
    assert "image_mean_sim" in report
    assert "image_mean_real" in report
    assert "image_std_sim" in report
    assert "image_std_real" in report
    assert "state_range_overlap" in report
    # type checks
    assert isinstance(report["image_mean_sim"], (float, np.floating))
    assert isinstance(report["image_mean_real"], (float, np.floating))
    assert isinstance(report["state_range_overlap"], (float, np.floating))
    # state_range_overlap ∈ [0, 1] (intersection / union of ranges)
    assert 0.0 <= float(report["state_range_overlap"]) <= 1.0


# ---------------------------------------------------------------------------
# 3) main() 跑通不 crash, exit 0
# ---------------------------------------------------------------------------
def test_main_runs_end_to_end(sim_pkl_path, mock_real_pkl_path, capsys):
    from sim.scripts.test_domain_alignment import main
    old_argv = sys.argv
    try:
        sys.argv = ["test_domain_alignment.py", "--sim", sim_pkl_path, "--real", mock_real_pkl_path]
        exit_code = main()
        # schema smoke: 跑通就 OK, exit 0
        assert exit_code == 0, f"main() must return 0 (smoke pass); got {exit_code}"
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    # 报告必须打印
    assert "image_mean" in captured.out or "state_range" in captured.out


# ---------------------------------------------------------------------------
# 4) CLI subprocess 跑通
# ---------------------------------------------------------------------------
def test_cli_subprocess_runs(sim_pkl_path, mock_real_pkl_path):
    """Simulating real CLI invocation."""
    result = subprocess.run(
        [sys.executable, "-m", "sim.scripts.test_domain_alignment",
         "--sim", sim_pkl_path, "--real", mock_real_pkl_path],
        cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=30,
    )
    # exit 0 = schema smoke pass
    assert result.returncode == 0, (
        f"CLI must exit 0 (smoke pass); got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # stdout 含 "image_mean" 或 "state_range" 或 "PASS"
    assert ("image_mean" in result.stdout or "state_range" in result.stdout or "PASS" in result.stdout)
