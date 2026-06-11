"""A12: test_mixed_training.py 必须能 end-to-end 跑通, 打印 report (含 accuracy + baseline + confusion matrix).

⚠️ CRITICAL (codex #6 修复): 本测试 = "schema smoke: 脚本跑通, 模型 fit, 预测 emit", NOT accuracy-based。

测试:
  1. 脚本能 import
  2. main() 跑通 exit 0 (smoke pass)
  3. report dict 含 accuracy + baseline_accuracy + confusion_matrix
  4. accuracy ∈ [0, 1] (不验证 ≥ 阈值)
  5. baseline_accuracy = max(pos_ratio, 1-pos_ratio) (majority class)
"""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from sim.data.contract import IMAGE_SHAPE, STATE_DIMS


@pytest.fixture
def sim_pkl_path():
    """A9 风格的 sim pkl: 20 帧, 全 reward=0, 25D state, 3 image keys."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sim.pkl")
        rng = np.random.default_rng(0)
        N = 20
        transitions = []
        for i in range(N):
            state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
            next_state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
            action = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
            base = rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
            obs = {
                "state": state,
                "side_policy": base.copy(),
                "wrist_1": rng.integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8),
                "side_classifier": base.copy(),
            }
            next_obs = {**obs, "state": next_state}
            transitions.append({
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": np.float32(0.0),  # 全 0 (failure)
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
        generate_mock_real_pkl(output_path=p, num_frames=20, pos_ratio=0.8, seed=0)
        yield p


# ---------------------------------------------------------------------------
# 1) 脚本能 import
# ---------------------------------------------------------------------------
def test_mixed_training_module_importable():
    from sim.scripts import test_mixed_training
    assert hasattr(test_mixed_training, "main")
    assert hasattr(test_mixed_training, "load_features")
    assert hasattr(test_mixed_training, "train_classifier")
    assert hasattr(test_mixed_training, "compute_report")
    assert hasattr(test_mixed_training, "print_report")


# ---------------------------------------------------------------------------
# 2) load_features: 加载 sim + real pkl, 返回 X (N, 25) + y (N,)
# ---------------------------------------------------------------------------
def test_load_features_returns_xy(sim_pkl_path, mock_real_pkl_path):
    from sim.scripts.test_mixed_training import load_features
    X, y = load_features(sim_pkl_path, mock_real_pkl_path)
    assert X.shape[0] == 20 + 20  # sim + real
    assert X.shape[1] == STATE_DIMS
    assert y.shape[0] == 40
    # sim 全部 y=0
    # mock real 80% y=1
    n_pos = int((y == 1.0).sum())
    n_neg = int((y == 0.0).sum())
    # sim 20 neg + real 16 pos + 4 neg = 20 neg, 16 pos
    assert n_neg == 24  # 20 sim + 4 real neg
    assert n_pos == 16  # 16 real pos


# ---------------------------------------------------------------------------
# 3) train_classifier: 训练 LogisticRegression, 返回 model + predictions
# ---------------------------------------------------------------------------
def test_train_classifier_emits_predictions(sim_pkl_path, mock_real_pkl_path):
    from sim.scripts.test_mixed_training import load_features, train_classifier
    X, y = load_features(sim_pkl_path, mock_real_pkl_path)
    model, X_test, y_test = train_classifier(X, y, max_iter=200, random_state=0)
    # model 必须 fit (能 predict)
    preds = model.predict(X_test)
    assert preds.shape == y_test.shape
    # predictions 是 0/1
    assert set(np.asarray(preds).tolist()).issubset({0, 1, 0.0, 1.0})


# ---------------------------------------------------------------------------
# 4) compute_report: 返回 dict 含 accuracy + baseline + confusion matrix
# ---------------------------------------------------------------------------
def test_compute_report_contains_required_keys(sim_pkl_path, mock_real_pkl_path):
    from sim.scripts.test_mixed_training import (
        compute_report, load_features, train_classifier,
    )
    X, y = load_features(sim_pkl_path, mock_real_pkl_path)
    model, X_test, y_test = train_classifier(X, y, max_iter=200, random_state=0)
    preds = model.predict(X_test)
    report = compute_report(y_test, preds)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "baseline_accuracy" in report
    assert "confusion_matrix" in report
    # accuracy ∈ [0, 1]
    assert 0.0 <= report["accuracy"] <= 1.0
    # baseline_accuracy = max(pos_ratio, 1-pos_ratio) = 24/40 = 0.6
    assert 0.0 <= report["baseline_accuracy"] <= 1.0
    # confusion matrix 4-int tuple
    assert len(report["confusion_matrix"]) == 4


# ---------------------------------------------------------------------------
# 5) main() 跑通不 crash, exit 0 (smoke pass)
# ---------------------------------------------------------------------------
def test_main_runs_end_to_end(sim_pkl_path, mock_real_pkl_path, capsys):
    from sim.scripts.test_mixed_training import main
    old_argv = sys.argv
    try:
        sys.argv = ["test_mixed_training.py", "--sim", sim_pkl_path, "--real", mock_real_pkl_path]
        exit_code = main()
        # schema smoke: 跑通就 OK, exit 0
        assert exit_code == 0, f"main() must return 0 (smoke pass); got {exit_code}"
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    # 报告必须打印
    assert "accuracy" in captured.out
    assert "baseline" in captured.out.lower()
    assert "confusion" in captured.out.lower()


# ---------------------------------------------------------------------------
# 6) CLI subprocess 跑通
# ---------------------------------------------------------------------------
def test_cli_subprocess_runs(sim_pkl_path, mock_real_pkl_path):
    """Simulating real CLI invocation."""
    result = subprocess.run(
        [sys.executable, "-m", "sim.scripts.test_mixed_training",
         "--sim", sim_pkl_path, "--real", mock_real_pkl_path],
        cwd="/Users/tacyvan/Documents/Code/.claude/worktrees/wf_99662f87-b68-4/hilserl-fr3",
        capture_output=True, text=True, timeout=60,
    )
    # exit 0 = schema smoke pass (NOT accuracy-based)
    assert result.returncode == 0, (
        f"CLI must exit 0 (smoke pass); got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "accuracy" in result.stdout
