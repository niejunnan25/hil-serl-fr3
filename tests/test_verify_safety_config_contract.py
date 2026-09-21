import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SAFETY = REPO_ROOT / "scripts" / "verify_safety.py"


def _load_verify_safety():
    module_name = "_test_verify_safety_config_contract"
    sys.modules.pop(module_name, None)
    original_path = list(sys.path)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(module_name, VERIFY_SAFETY)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def test_verify_safety_uses_live_env_pose_limits():
    vs = _load_verify_safety()
    constants = vs._load_live_env_config_constants()

    np.testing.assert_allclose(vs.CARTESIAN_SAFETY_BOX_LOW, constants["ABS_POSE_LIMIT_LOW"][:3])
    np.testing.assert_allclose(vs.CARTESIAN_SAFETY_BOX_HIGH, constants["ABS_POSE_LIMIT_HIGH"][:3])


def test_verify_safety_uses_live_env_impedance_params():
    vs = _load_verify_safety()
    constants = vs._load_live_env_config_constants()

    assert vs.COMPLIANCE_PARAM == constants["COMPLIANCE_PARAM"]
    assert vs.PRECISION_PARAM == constants["PRECISION_PARAM"]


def test_verify_safety_dry_run_pose_is_inside_live_box():
    vs = _load_verify_safety()

    result = vs.check_cartesian_safety_box(
        base_url="http://unused.invalid/",
        box_low=vs.CARTESIAN_SAFETY_BOX_LOW,
        box_high=vs.CARTESIAN_SAFETY_BOX_HIGH,
        dry_run=True,
    )

    assert result.status == "pass"


def test_safety_box_integration_fails_on_drift(monkeypatch):
    vs = _load_verify_safety()
    monkeypatch.setattr(vs, "CARTESIAN_SAFETY_BOX_LOW", np.array([0.0, 0.0, 0.0]))

    result = vs.check_safety_box_integration(dry_run=True)

    assert result.status == "fail"


def test_impedance_check_fails_when_live_env_config_cannot_load(monkeypatch):
    vs = _load_verify_safety()
    monkeypatch.setattr(vs, "_LIVE_ENV_CONFIG_ERROR", "EnvConfig parse failed")

    result = vs.check_impedance_params(dry_run=True)

    assert result.status == "fail"
    assert "Cannot load live EnvConfig" in result.message
