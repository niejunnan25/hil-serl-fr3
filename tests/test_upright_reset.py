import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from test_axial_motion_limits import actual_step
from test_plug_insertion_reset_safety import _Clock, _make_env

ROOT = Path(__file__).resolve().parents[1]


def test_native_reset_encoding_is_vertical_without_tilt_or_heading_offset():
    native = runpy.run_path(str(ROOT/"upstream/hil-serl/serl_robot_infra/franka_env/utils/rotations.py"))
    cls = next(n for n in ast.parse((ROOT/"experiments/plug_insertion/config.py").read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "EnvConfig")
    value = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "RESET_POSE" for t in n.targets))
    pose = eval(compile(ast.Expression(value), "<actual-reset-pose>", "eval"), {"np": np})
    q = native["euler_2_quat"](pose[3:])
    np.testing.assert_allclose(q, [1,0,0,0], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(q).as_matrix(), np.diag([1,-1,-1]), atol=1e-12)


def test_reset_never_samples_angles_even_with_a_historical_nonzero_range():
    _, ns = _make_env(_Clock())
    cfg = SimpleNamespace(RESET_POSE=np.array([.65,-.012,.15,np.pi,0,0]),
                          RANDOM_RESET=True, RANDOM_XY_RANGE=.006, RANDOM_RZ_RANGE=.06)
    samples = np.array([ns["sample_reset_pose"](cfg, np.random.default_rng(i)) for i in range(100)])
    np.testing.assert_array_equal(samples[:,3:], np.tile(cfg.RESET_POSE[3:], (100,1)))
    assert np.ptp(samples[:,0]) > .005 and np.ptp(samples[:,1]) > .005
    assert np.max(np.abs(samples[:,:2]-cfg.RESET_POSE[:2])) <= .006
    np.testing.assert_array_equal(samples[:,2], np.full(100,.15))


def test_translation_steps_hold_reference_orientation_instead_of_following_drift(monkeypatch):
    env = actual_step(monkeypatch, .016)
    for degrees in ([1,2,3], [-2,1,-4], [5,-3,6]):
        env.currpos[3:] = (Rotation.from_euler("xyz",degrees,degrees=True)
                           * Rotation.from_quat([1,0,0,0])).as_quat()
        env.step(np.array([.2,-.1,.3,0,0,0,0]))
        np.testing.assert_array_equal(env.sent[-1][3:], [1,0,0,0])


def test_withdrawal_keeps_xy_reference_under_small_lateral_tracking_drift():
    env, _ = _make_env(_Clock(), moving=True)
    env.config.ACTION_MAX_Z_STEP = .016
    env.action_max_step = .008
    env._update_currpos = lambda: env.currpos.__setitem__(0, env.currpos[0]+.0003)
    sends = []
    original = env._send_pos_command
    def send(p):
        delta = p[:3]-env.currpos[:3]
        assert np.linalg.norm(delta/[.008,.008,.016]) <= 1+1e-12
        sends.append(p.copy())
        original(p)
    env._send_pos_command = send
    env.interpolate_move(np.array([0,0,.21,0,0,0,1]), timeout=8., name="clear")
    assert len(sends) > 3
    np.testing.assert_allclose(np.array(sends)[:,:2], 0, atol=1e-12)
