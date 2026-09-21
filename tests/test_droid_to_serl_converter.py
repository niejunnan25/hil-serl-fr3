import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATHS = (
    REPO_ROOT / "scripts" / "droid_to_serl_converter.py",
    REPO_ROOT / "scripts" / "gello_pipeline" / "droid_to_serl_converter.py",
)


def _load_converter(path: Path):
    for module_name in ("fk_converter", "normalize_action"):
        sys.modules.pop(module_name, None)

    unique_name = f"_test_{path.parent.name}_{path.stem}"
    sys.modules.pop(unique_name, None)
    original_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(unique_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def _write_demo_npz(path: Path):
    joint_poses = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        ],
        dtype=np.float32,
    )
    gripper_states = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    np.savez(
        path,
        joint_poses=joint_poses,
        gripper_states=gripper_states,
        timestamps=timestamps,
    )
    return joint_poses, gripper_states


def _patch_fk_with_known_deltas(module):
    def fake_trajectory_to_cartesian_deltas(_joint_poses):
        return np.array(
            [
                [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.02, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    module.trajectory_to_cartesian_deltas = fake_trajectory_to_cartesian_deltas


@pytest.mark.parametrize("converter_path", CONVERTER_PATHS)
def test_converter_requires_explicit_legacy_ack(converter_path, tmp_path):
    module = _load_converter(converter_path)
    _patch_fk_with_known_deltas(module)
    npz_path = tmp_path / "demo.npz"
    _write_demo_npz(npz_path)

    with pytest.raises(RuntimeError, match="legacy 8D joint-state"):
        module.convert_droid_to_serl(npz_path, filter_zero=False)


@pytest.mark.parametrize("converter_path", CONVERTER_PATHS)
def test_legacy_converter_actions_are_frame_aligned_and_normalized(converter_path, tmp_path):
    module = _load_converter(converter_path)
    _patch_fk_with_known_deltas(module)
    npz_path = tmp_path / "demo.npz"
    joint_poses, gripper_states = _write_demo_npz(npz_path)

    transitions = module.convert_droid_to_serl(
        npz_path,
        pos_scale=0.05,
        rpy_scale=0.1,
        filter_zero=False,
        allow_legacy_joint_state=True,
    )

    assert len(transitions) == 2
    np.testing.assert_allclose(
        transitions[0]["actions"],
        np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_states[1]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        transitions[1]["actions"],
        np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0, gripper_states[2]], dtype=np.float32),
    )

    np.testing.assert_allclose(transitions[0]["observations"]["state"][:7], joint_poses[0])
    np.testing.assert_allclose(transitions[0]["next_observations"]["state"][:7], joint_poses[1])
    np.testing.assert_allclose(transitions[1]["observations"]["state"][:7], joint_poses[1])
    np.testing.assert_allclose(transitions[1]["next_observations"]["state"][:7], joint_poses[2])


@pytest.mark.parametrize("converter_path", CONVERTER_PATHS)
def test_convert_directory_writes_pickle_when_legacy_acknowledged(converter_path, tmp_path):
    module = _load_converter(converter_path)
    _patch_fk_with_known_deltas(module)
    npz_path = tmp_path / "demo.npz"
    _write_demo_npz(npz_path)
    output_dir = tmp_path / "pkl_out"

    module.convert_directory(
        tmp_path,
        output_dir,
        filter_zero=False,
        allow_legacy_joint_state=True,
    )

    output_path = output_dir / "demo.pkl"
    assert output_path.exists()
    with output_path.open("rb") as f:
        transitions = pickle.load(f)
    assert len(transitions) == 2
