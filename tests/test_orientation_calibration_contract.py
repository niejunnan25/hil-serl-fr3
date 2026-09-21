import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIENT_CALIB = REPO_ROOT / "scripts" / "orient_calib.py"
CALIB_ANALYSIS = REPO_ROOT / "scripts" / "calib_analysis.py"


def _install_fake_rotations():
    utils_pkg = types.ModuleType("franka_env.utils")
    rotations = types.ModuleType("franka_env.utils.rotations")

    def euler_2_quat(euler):
        return R.from_euler("xyz", np.asarray(euler, dtype=np.float64)).as_quat()

    def quat_2_euler(quat_xyzw):
        return R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_euler("xyz")

    rotations.euler_2_quat = euler_2_quat
    rotations.quat_2_euler = quat_2_euler
    sys.modules["franka_env.utils"] = utils_pkg
    sys.modules["franka_env.utils.rotations"] = rotations
    return rotations


def _load_module(module_name: str, path: Path):
    sys.modules.pop(module_name, None)
    original_path = list(sys.path)
    sys.path.insert(0, str(REPO_ROOT))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def test_orient_calib_invert_round_trips_in_config_euler_domain():
    _install_fake_rotations()
    orient_calib = _load_module("_test_orient_calib", ORIENT_CALIB)
    target = R.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()

    euler, quat = orient_calib.invert(target)

    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert np.allclose(quat, target, atol=orient_calib.DEFAULT_QUAT_ATOL)
    assert np.allclose(
        orient_calib.euler_2_quat(euler),
        target,
        atol=orient_calib.DEFAULT_QUAT_ATOL,
    )


def test_orient_calib_invert_fails_closed_when_optimizer_does_not_match(monkeypatch):
    _install_fake_rotations()
    orient_calib = _load_module("_test_orient_calib", ORIENT_CALIB)

    failed = Mock()
    failed.x = np.array([10.0, 10.0, 10.0])
    failed.success = False
    failed.cost = 1.0
    failed.optimality = 1.0
    failed.message = "forced failure"
    monkeypatch.setattr(orient_calib, "least_squares", Mock(return_value=failed))

    with pytest.raises(RuntimeError, match="orientation inversion failed"):
        orient_calib.invert(R.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat())


def test_calib_analysis_config_euler_matches_project_euler_2_quat_domain():
    rotations = _install_fake_rotations()
    calib_analysis = _load_module("_test_calib_analysis", CALIB_ANALYSIS)
    q_xyzw = R.from_euler("xyz", [0.25, -0.15, 0.35]).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    config_euler = calib_analysis.wxyz_to_config_euler(q_wxyz)
    round_trip = rotations.euler_2_quat(config_euler)

    assert np.allclose(round_trip, q_xyzw, atol=calib_analysis.CONFIG_EULER_ATOL)


def test_calib_analysis_standard_euler_is_not_labeled_as_config_euler():
    _install_fake_rotations()
    text = CALIB_ANALYSIS.read_text(encoding="utf-8")

    assert "STANDARD scipy euler" in text
    assert "CONFIG euler" in text
    assert "TARGET euler(rad) median=" not in text
    assert "RESET euler(rad)  median=" not in text
