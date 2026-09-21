import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hilserl.config import Config
from hilserl.errors import RobotStateUnavailable
from hilserl.motion_limits import advance_position_target
from hilserl import seed_dataset
from test_axial_motion_limits import actual_step
from test_plug_insertion_reset_safety import _Clock, _make_env
from test_seed_dataset import make_episode


def test_xyz_uses_last_command_and_zero_input_cannot_absorb_tracking_drift(monkeypatch):
    monkeypatch.setenv("HILSERL_POSITION_TARGET_MODE", "command-relative-v1")
    env = actual_step(monkeypatch, .016)
    reference = env._command_pose.copy()
    env.currpos[:3] -= [.001, .001, .002]
    env.currpos[3:] = (Rotation.from_euler("xyz", [2,-1,3], degrees=True)
                       * Rotation.from_quat([1,0,0,0])).as_quat()
    env.step(np.array([.1,0,0,0,0,0,0]))
    expected = reference.copy(); expected[0] += .0015
    np.testing.assert_allclose(env.sent[-1], expected, atol=1e-12)
    for _ in range(50):
        env.currpos[:3] = expected[:3] + np.array([-.003,.002,-.001])
        env.step(np.zeros(7))
        np.testing.assert_array_equal(env.sent[-1], expected)


def test_blocked_axis_stops_target_accumulation_and_can_back_off(monkeypatch):
    monkeypatch.setenv("HILSERL_POSITION_TARGET_MODE", "command-relative-v1")
    env = actual_step(monkeypatch, .016)
    origin = env.currpos.copy()
    for _ in range(100):
        env.step(np.array([0,0,1,0,0,0,0]))
        assert env.sent[-1][2]-origin[2] <= .016+1e-12
        np.testing.assert_array_equal(env.sent[-1][3:], [1,0,0,0])
    assert env.sent[-1][2]-origin[2] == pytest.approx(.016)
    env.step(np.array([0,0,-1,0,0,0,0]))
    np.testing.assert_allclose(env.sent[-1][:3], origin[:3], atol=1e-12)


def test_reference_updates_only_after_acknowledged_pose():
    env, _ = _make_env(_Clock())
    env.clearerr_on_pose = False
    env._step_commands = []
    old = np.array([.65,-.012,.15,1,0,0,0])
    env._command_pose = old.copy()
    target = old.copy(); target[0] += .002
    def fail(*args, **kwargs):
        raise RobotStateUnavailable("lost acknowledgement", endpoint="pose", delivery_unknown=True)
    env._robot_post = fail
    with pytest.raises(RobotStateUnavailable):
        type(env)._send_pos_command(env,target)
    np.testing.assert_array_equal(env._command_pose, old)
    env._robot_post = lambda *a, **kw: None
    type(env)._send_pos_command(env,target)
    np.testing.assert_array_equal(env._command_pose, target.astype(np.float32).astype(np.float64))


def test_external_displacement_never_causes_spontaneous_reference_change():
    previous = np.array([0.,0.,.03])
    measured = np.zeros(3)
    np.testing.assert_array_equal(advance_position_target(previous,np.zeros(3),measured,.008,.016),previous)
    np.testing.assert_array_equal(advance_position_target(previous,[0,0,.01],measured,.008,.016),previous)
    np.testing.assert_allclose(advance_position_target(previous,[0,0,-.01],measured,.008,.016),[0,0,.02])


def test_mode_is_explicit_in_runtime_and_old_snapshots_keep_old_mapping():
    old = Config().environment("actor",run_dir=Path("/tmp/not-a-live-run"))
    assert old["HILSERL_POSITION_TARGET_MODE"] == "measured-relative-v1"
    new = Config(action_contract="fixed-xyz-v1",position_target_mode="command-relative-v1").validate()
    assert new.environment("actor",run_dir=Path("/tmp/not-a-live-run"))["HILSERL_POSITION_TARGET_MODE"] == "command-relative-v1"


def test_seed_cannot_be_reused_under_a_different_reference_rule(tmp_path):
    run, _ = make_episode(tmp_path)
    (run/"attempts/config.json").write_text(json.dumps(dict(action_max_step=.008,
        action_max_z_step=.016,position_target_mode="command-relative-v1")))
    output = tmp_path/"seed"
    manifest = seed_dataset.build_seed_dataset([run],output)
    assert manifest["execution"]["position_target_mode"] == "command-relative-v1"
    seed_dataset.validate_seed_dataset(output,expected_action_max_z_step=.016,
                                       expected_position_target_mode="command-relative-v1")
    with pytest.raises(seed_dataset.SeedDatasetError,match="execution contract"):
        seed_dataset.validate_seed_dataset(output,expected_action_max_z_step=.016)
