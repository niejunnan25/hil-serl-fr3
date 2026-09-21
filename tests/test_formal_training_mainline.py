from __future__ import annotations

import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_training_import_stubs(monkeypatch):
    jax = types.ModuleType("jax")
    jax.random = types.SimpleNamespace(PRNGKey=lambda seed: seed)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", np)

    wrappers = types.ModuleType("franka_env.envs.wrappers")
    wrappers.Quat2EulerWrapper = lambda env: env
    wrappers.MultiCameraBinaryRewardClassifierWrapper = lambda env, reward_func: env
    relative_env = types.ModuleType("franka_env.envs.relative_env")
    relative_env.RelativeFrame = lambda env: env
    franka_env = types.ModuleType("franka_env.envs.franka_env")
    franka_env.DefaultEnvConfig = object
    franka_env.FrankaEnv = object
    rotations = types.ModuleType("franka_env.utils.rotations")
    rotations.euler_2_quat = lambda e: np.array([0.0, 0.0, 0.0, 1.0])
    serl_obs = types.ModuleType("serl_launcher.wrappers.serl_obs_wrappers")
    serl_obs.SERLObsWrapper = lambda env, proprio_keys=None: env
    chunking = types.ModuleType("serl_launcher.wrappers.chunking")
    chunking.ChunkingWrapper = lambda env, obs_horizon=1, act_exec_horizon=None: env

    monkeypatch.setitem(sys.modules, "franka_env.envs.wrappers", wrappers)
    monkeypatch.setitem(sys.modules, "franka_env.envs.relative_env", relative_env)
    monkeypatch.setitem(sys.modules, "franka_env.envs.franka_env", franka_env)
    monkeypatch.setitem(sys.modules, "franka_env.utils.rotations", rotations)
    monkeypatch.setitem(sys.modules, "serl_launcher.wrappers.serl_obs_wrappers", serl_obs)
    monkeypatch.setitem(sys.modules, "serl_launcher.wrappers.chunking", chunking)


def test_random_reset_changes_translation_and_keeps_upright_orientation(monkeypatch):
    _install_training_import_stubs(monkeypatch)
    monkeypatch.setenv("HILSERL_RANDOM_RESET", "1")
    monkeypatch.setenv("HILSERL_RANDOM_XY_RANGE", "0.006")
    monkeypatch.setenv("HILSERL_RANDOM_RZ_RANGE", "0.06")

    from experiments.plug_insertion.config import EnvConfig
    from experiments.plug_insertion.env import sample_reset_pose

    cfg = EnvConfig()
    rng = np.random.default_rng(123)
    samples = np.array([sample_reset_pose(cfg, rng) for _ in range(64)])
    base = np.asarray(cfg.RESET_POSE, dtype=float)

    assert bool(cfg.RANDOM_RESET)
    assert np.max(np.abs(samples[:, 0] - base[0])) <= 0.006 + 1e-9
    assert np.max(np.abs(samples[:, 1] - base[1])) <= 0.006 + 1e-9
    assert np.ptp(samples[:, 0]) > 0
    assert np.ptp(samples[:, 1]) > 0
    np.testing.assert_array_equal(samples[:, 3:], np.repeat(base[None, 3:], 64, axis=0))
    assert np.allclose(samples[:, 2], base[2])


def test_manual_reset_does_not_override_measured_nonconvergence(monkeypatch):
    import pytest
    from test_plug_insertion_reset_safety import _Clock, _make_env

    env, ns = _make_env(_Clock())
    ns["RESET_STRICT"] = True
    ns["MANUAL_RESET"] = True
    env._operator_input = lambda _: pytest.fail("failed reset cannot accept manual override")
    with pytest.raises(RuntimeError, match="not_converged"):
        env._require_reset_settled("clear", (0.05, 0.0))
    assert env.last_reset_info["success"] is False
    assert env.calls == []


def test_current_profile_uses_the_intended_demo_set():
    from hilserl.config import load_config
    config = load_config()
    assert config.demo_dir == "demos/fixed_xyz_front_v2_command20_20260921"
    assert config.image_profile == "insert-front-roi160-v2"
    assert config.replay_buffer_capacity == 20000
    assert config.batch_size == 128
    assert config.action_contract == "fixed-xyz-v1"
    assert len(config.seed_dataset_sha256) == 64
    assert config.max_episode_steps == 190
    assert config.random_reset is True


def test_preflight_rejects_classifier_frame_demo_path(tmp_path):
    bad_demo = tmp_path / "classifier_data" / "plug_insertion_success.pkl"
    bad_demo.parent.mkdir()
    with bad_demo.open("wb") as f:
        pickle.dump([{"rewards": 1.0, "dones": True}], f)

    classifier_ckpt = tmp_path / "classifier_ckpt"
    classifier_ckpt.mkdir()
    (classifier_ckpt / "marker").write_text("ok")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["HILSERL_RANDOM_RESET"] = "1"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/formal_training_preflight.py"),
        "--demo-path",
        str(bad_demo),
        "--checkpoint-path",
        str(tmp_path / "fresh_ckpt"),
        "--classifier-ckpt",
        str(classifier_ckpt),
        "--expect-random-reset",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)

    assert proc.returncode == 2
    assert "classifier frame data" in proc.stdout


def test_terminal_label_overrides_classifier_without_changing_raw_reward():
    from hilserl.episodes import _training_transition
    raw = dict(info={}, observations={}, next_observations={}, actions=np.zeros(7),
               observed_reward=1.0, source_action="policy", global_step=3,
               id="run/capture/ep/3", episode_id="ep", termination_reason="classifier")
    terminal = _training_transition(raw, terminal=True, outcome=0)
    normal = _training_transition(raw)
    assert terminal["rewards"] == 0 and terminal["dones"]
    assert terminal["infos"]["verdict_source"] == "human"
    assert normal["infos"]["manual_success"] is False
    assert raw["observed_reward"] == 1.0
