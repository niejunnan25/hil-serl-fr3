"""Exercise real action/reset code without importing or connecting devices."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hilserl.config import Config
from hilserl.motion_limits import clip_translation_delta, seed_execution_contract
from hilserl import seed_dataset
from test_plug_insertion_reset_safety import _Clock, _make_env
from test_seed_dataset import make_episode

ROOT = Path(__file__).resolve().parents[1]


def actual_config(monkeypatch, z):
    if z is None:
        monkeypatch.delenv("HILSERL_ACTION_MAX_Z_STEP", raising=False)
    else:
        monkeypatch.setenv("HILSERL_ACTION_MAX_Z_STEP", str(z))
    tree = ast.parse((ROOT / "experiments/plug_insertion/config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EnvConfig")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    namespace = dict(os=os)
    exec(compile(ast.Module(body=[init], type_ignores=[]), "<actual-config>", "exec"), namespace)
    value = SimpleNamespace(ACTION_SCALE=np.array([.015, .015, .015, .1, .1, .1, 1.]))
    namespace["__init__"](value)
    return value


@pytest.mark.parametrize("z", [None, .016])
def test_profile_scale_and_historical_environment_are_explicit(monkeypatch, z):
    config = Config(action_contract="fixed-xyz-v1", action_max_z_step=z).validate()
    env = config.environment("actor", run_dir=Path("/tmp/no-device-run"))
    assert env.get("HILSERL_ACTION_MAX_Z_STEP") == (None if z is None else str(z))
    actual = actual_config(monkeypatch, z)
    np.testing.assert_array_equal(actual.ACTION_SCALE[:3], [.015, .015, .015 if z is None else z])
    assert actual.ACTION_MAX_Z_STEP == z


@pytest.mark.parametrize("z", [True, 0., .0079, .0161, float("nan"), float("inf")])
def test_invalid_axis_bounds_rejected_before_device_construction(z):
    with pytest.raises(ValueError, match="Base Z bound"):
        Config(action_contract="fixed-xyz-v1", action_max_z_step=z).validate()


def actual_step(monkeypatch, z):
    source = ROOT / "upstream/hil-serl/serl_robot_infra/franka_env/envs/franka_env.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "FrankaEnv")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in {"step", "_clip_translation_delta", "_action_target_orientation",
                            "_action_target_position"}]
    clock = _Clock()
    clock.time = clock.monotonic
    namespace = dict(np=np, time=clock, Rotation=Rotation)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    base = type("BaseStep", (), {n.name: namespace[n.name] for n in nodes})
    if z is not None:
        reset, _ = _make_env(_Clock())
        overrides = {
            "_clip_translation_delta": type(reset)._clip_translation_delta,
            "_action_target_orientation": type(reset)._action_target_orientation}
        if os.environ.get("HILSERL_POSITION_TARGET_MODE") == "command-relative-v1":
            overrides["_action_target_position"] = type(reset)._action_target_position
        base = type("AxialStep", (base,), overrides)
    env = base()
    env.config = actual_config(monkeypatch, z)
    env.action_scale = env.config.ACTION_SCALE
    env.action_space = SimpleNamespace(low=-np.ones(7), high=np.ones(7))
    env.lock_rotation = True
    env.action_max_step, env.action_max_rot_step = .008, .045
    env.currpos = np.array([.65, -.012, .15, 0., 0., 0., 1.])
    env.resetpos = np.array([.65, -.012, .15, 1., 0., 0., 0.])
    env._command_pose = env.resetpos.copy()
    env.currforce, env.dq = np.zeros(3), np.zeros(7)
    env.safety_force_max, env.safety_dq_max = 45., .35
    env.hz, env.action_dbg_period = 10., 0.
    env.curr_path_length, env.max_episode_length, env.terminate = 0, 100, False
    env._send_gripper_command = lambda p: None
    env.clip_safety_box = lambda p: p
    env._update_currpos = lambda: None
    env._get_obs = lambda: {}
    env.compute_reward = lambda obs: False
    env.sent = []
    def send(p):
        env.sent.append(p.copy())
        env._command_pose = p.copy()
    env._send_pos_command = send
    return env


@pytest.mark.parametrize("sign", [-1., 1.])
def test_real_step_allows_axial_16_mm_and_40_N_spring_term(monkeypatch, sign):
    env = actual_step(monkeypatch, .016)
    env.step(np.array([0., 0., sign, 0., 0., 0., 0.]))
    delta = env.sent[-1][:3] - env.currpos[:3]
    np.testing.assert_allclose(delta, [0., 0., sign*.016], atol=1e-12)
    assert abs(delta[2]) * 2500 == pytest.approx(40)


def test_real_step_keeps_legacy_8_mm_and_lateral_bound(monkeypatch):
    old = actual_step(monkeypatch, None)
    old.step(np.array([0., 0., 1., 0., 0., 0., 0.]))
    assert old.sent[-1][2] - old.currpos[2] == pytest.approx(.008)
    env = actual_step(monkeypatch, .016)
    for action in np.random.default_rng(7).uniform(-1, 1, (100, 3)):
        env.step(np.r_[action, np.zeros(4)])
        delta = env.sent[-1][:3] - env.currpos[:3]
        assert np.linalg.norm(delta[:2]) <= .008 + 1e-12
        assert np.linalg.norm(delta / [.008, .008, .016]) <= 1 + 1e-12


def test_reset_and_final_hold_never_exceed_new_envelope():
    env, _ = _make_env(_Clock(), moving=True)
    env.config.ACTION_MAX_Z_STEP = .016
    env.action_max_step = .008
    deltas = []
    original_send = env._send_pos_command
    def send(pose):
        deltas.append(np.asarray(pose[:3]) - env.currpos[:3])
        original_send(pose)
    env._send_pos_command = send
    env.interpolate_move(np.array([0., 0., .21, 0., 0., 0., 1.]), timeout=8., name="clear")
    env.interpolate_move(np.array([.03, .02, .15, 0., 0., 0., 1.]), timeout=8., name="reset_pose")
    assert max(abs(d[2]) for d in deltas) == pytest.approx(.016)
    assert max(np.linalg.norm(d[:2]) for d in deltas) <= .008 + 1e-12
    assert max(np.linalg.norm(d/[.008, .008, .016]) for d in deltas) <= 1 + 1e-12


def test_nonfinite_action_cannot_reach_pose_sender(monkeypatch):
    env = actual_step(monkeypatch, .016)
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([np.nan, 0., 0., 0., 0., 0., 0.]))
    assert env.sent == []


def test_seed_records_new_mapping_and_refuses_old_new_mix(tmp_path):
    run, _ = make_episode(tmp_path)
    config_path = run / "attempts/config.json"
    config_path.write_text(json.dumps(dict(action_max_step=.008, action_max_z_step=.016)))
    output = tmp_path / "axial-seed"
    manifest = seed_dataset.build_seed_dataset([run], output)
    assert manifest["execution"] == seed_execution_contract(.016)
    assert list(seed_dataset.iter_seed_transitions(output, expected_action_max_z_step=.016))
    with pytest.raises(seed_dataset.SeedDatasetError, match="execution contract"):
        seed_dataset.validate_seed_dataset(output)
    config_path.write_text(json.dumps(dict(action_max_step=.008)))
    old = tmp_path / "old-seed"
    seed_dataset.build_seed_dataset([run], old)
    with pytest.raises(seed_dataset.SeedDatasetError, match="execution contract"):
        seed_dataset.validate_seed_dataset(old, expected_action_max_z_step=.016)


def test_seed_builder_rejects_mixed_motion_mappings(tmp_path):
    run_a, _ = make_episode(tmp_path / "a")
    run_b, _ = make_episode(tmp_path / "b")
    (run_b / "attempts/config.json").write_text(json.dumps(dict(action_max_step=.008, action_max_z_step=.016)))
    with pytest.raises(seed_dataset.SeedDatasetError, match="cannot mix displacement"):
        seed_dataset.build_seed_dataset([run_a, run_b], tmp_path / "mixed")
