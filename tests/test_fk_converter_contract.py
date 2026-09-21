import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FK = REPO_ROOT / "scripts" / "fk_converter.py"


def _load_fk_module():
    module_name = "_test_fk_converter_contract_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, PRIMARY_FK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_fk_converter_duplicate_matches_primary_source():
    primary = (REPO_ROOT / "scripts" / "fk_converter.py").read_text(encoding="utf-8")
    duplicate = (REPO_ROOT / "scripts" / "gello_pipeline" / "fk_converter.py").read_text(
        encoding="utf-8"
    )

    assert duplicate == primary


def test_relative_teleop_remains_fk_free():
    source = (REPO_ROOT / "scripts" / "relative_teleop.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_names = set()
    loaded_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded_names.add(node.id)

    assert "fk_converter" not in imported_names
    assert "forward_kinematics" not in imported_names
    assert "forward_kinematics" not in loaded_names


def test_forward_kinematics_matches_corrected_modified_dh_oracle_at_zero():
    fk = _load_fk_module()
    from sim.kinematics.fr3_fk import fk_ee_pose

    q = np.zeros(7, dtype=np.float64)
    pose = fk.forward_kinematics(q)
    oracle = fk_ee_pose(q, include_tcp=True)

    np.testing.assert_allclose(pose[:3], oracle[:3, 3], atol=1e-9)
    np.testing.assert_allclose(pose[:3], [0.088, 0.0, 0.8226], atol=1e-9)


def test_forward_kinematics_matches_corrected_modified_dh_oracle_off_home():
    fk = _load_fk_module()
    from sim.kinematics.fr3_fk import fk_ee_pose

    q = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.7], dtype=np.float64)
    pose = fk.forward_kinematics(q)
    oracle = fk_ee_pose(q, include_tcp=True)
    oracle_quat = Rotation.from_matrix(oracle[:3, :3]).as_quat()

    np.testing.assert_allclose(pose[:3], oracle[:3, 3], atol=1e-9)
    # Quaternions are sign-ambiguous; compare absolute dot product.
    assert abs(float(np.dot(pose[3:7], oracle_quat))) == np.float64(1.0)


def test_joints_to_cartesian_delta_rotation_channel_is_rotvec_not_euler():
    fk = _load_fk_module()
    q_prev = np.zeros(7, dtype=np.float64)
    q_curr = np.ones(7, dtype=np.float64)
    rotvec = np.array([0.3, 0.4, 0.2], dtype=np.float64)
    quat_prev = Rotation.identity().as_quat()
    quat_curr = Rotation.from_rotvec(rotvec).as_quat()

    def fake_forward_kinematics(q):
        q = np.asarray(q, dtype=np.float64)
        if np.allclose(q, q_prev):
            return np.concatenate([[0.1, 0.2, 0.3], quat_prev])
        if np.allclose(q, q_curr):
            return np.concatenate([[0.4, 0.6, 0.9], quat_curr])
        raise AssertionError(f"unexpected q {q}")

    fk.forward_kinematics = fake_forward_kinematics

    delta = fk.joints_to_cartesian_delta(q_prev, q_curr)

    np.testing.assert_allclose(delta[:3], [0.3, 0.4, 0.6], atol=1e-12)
    np.testing.assert_allclose(delta[3:], rotvec, atol=1e-12)
    assert not np.allclose(delta[3:], Rotation.from_rotvec(rotvec).as_euler("XYZ"))
