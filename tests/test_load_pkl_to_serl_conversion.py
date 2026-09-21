import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "load_pkl_to_serl.py"


def _load_module():
    module_name = "_test_load_pkl_to_serl"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_joints_to_tcp_pose_uses_absolute_fk_pose_not_single_frame_delta():
    module = _load_module()

    def fake_fk(q):
        q = np.asarray(q, dtype=np.float64)
        pose = np.zeros(7, dtype=np.float64)
        pose[:3] = [0.4 + q[0], -0.2 + q[1], 0.7 + q[2]]
        pose[3:7] = [0.9, 0.1, 0.2, 0.3]
        return pose

    module.forward_kinematics = fake_fk
    q = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    tcp_pose = module.joints_to_tcp_pose(q)

    assert tcp_pose.shape == (1, 7)
    np.testing.assert_allclose(tcp_pose[0, :3], [0.5, 0.0, 1.0])
    np.testing.assert_allclose(tcp_pose[0, 3:7], [0.9, 0.1, 0.2, 0.3])


def test_convert_transition_to_serl_preserves_nonzero_tcp_pose_and_quat_order():
    module = _load_module()

    def fake_fk(q):
        q = np.asarray(q, dtype=np.float64)
        pose = np.zeros(7, dtype=np.float64)
        pose[:3] = [0.5 + q[0], 0.6 + q[1], 0.7 + q[2]]
        pose[3:7] = [0.4, 0.5, 0.6, 0.7]
        return pose

    module.forward_kinematics = fake_fk
    transition = {
        "observations": {
            "state": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.25], dtype=np.float32)
        },
        "next_observations": {
            "state": np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.75], dtype=np.float32)
        },
        "actions": np.zeros(7, dtype=np.float32),
        "rewards": np.float32(0.0),
        "masks": np.float32(1.0),
        "dones": False,
    }

    out = module.convert_transition_to_serl(transition)
    state = out["observations"]["state"]
    next_state = out["next_observations"]["state"]

    assert state.shape == (module.SERL_STATE_DIM,)
    np.testing.assert_allclose(state[:7], [0.6, 0.8, 1.0, 0.4, 0.5, 0.6, 0.7])
    np.testing.assert_allclose(next_state[:7], [0.7, 0.9, 1.1, 0.4, 0.5, 0.6, 0.7])
    assert state[-1] == np.float32(0.25)
    assert next_state[-1] == np.float32(0.75)
