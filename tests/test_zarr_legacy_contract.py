import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERT_PATHS = (
    REPO_ROOT / "scripts" / "convert_to_zarr.py",
    REPO_ROOT / "scripts" / "gello_pipeline" / "convert_to_zarr.py",
)
LOAD_PATHS = (
    REPO_ROOT / "scripts" / "load_zarr_to_serl.py",
    REPO_ROOT / "scripts" / "gello_pipeline" / "load_zarr_to_serl.py",
)


def _load_module(path: Path, name: str):
    for module_name in ("fk_converter", "normalize_action"):
        sys.modules.pop(module_name, None)
    sys.modules.pop(name, None)
    original_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def _write_demo_npz(path: Path):
    joint_poses = np.arange(21, dtype=np.float32).reshape(3, 7) / 100.0
    gripper_states = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    timestamps = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    np.savez(
        path,
        joint_poses=joint_poses,
        gripper_states=gripper_states,
        timestamps=timestamps,
    )
    return joint_poses, gripper_states


def _patch_fk(module):
    def fake_trajectory_to_cartesian_deltas(_joint_poses):
        return np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        )

    module.trajectory_to_cartesian_deltas = fake_trajectory_to_cartesian_deltas


@pytest.mark.parametrize("path", CONVERT_PATHS)
def test_zarr_converter_defaults_match_live_action_scale(path):
    module = _load_module(path, f"_test_{path.parent.name}_convert_to_zarr_defaults")

    assert module.DEFAULT_POS_SCALE == 0.015
    assert module.DEFAULT_RPY_SCALE == 0.1


@pytest.mark.parametrize("path", CONVERT_PATHS)
def test_zarr_converter_requires_explicit_legacy_8d_ack(path, tmp_path):
    module = _load_module(path, f"_test_{path.parent.name}_convert_to_zarr_fail_closed")
    _patch_fk(module)
    npz_path = tmp_path / "demo.npz"
    _write_demo_npz(npz_path)

    with pytest.raises(RuntimeError, match="legacy 8D joint-state"):
        module.convert_npz_to_zarr_data(str(npz_path))


@pytest.mark.parametrize("path", CONVERT_PATHS)
def test_zarr_converter_legacy_ack_preserves_8d_state_and_metadata(path, tmp_path):
    module = _load_module(path, f"_test_{path.parent.name}_convert_to_zarr_legacy")
    _patch_fk(module)
    npz_path = tmp_path / "demo.npz"
    joint_poses, gripper_states = _write_demo_npz(npz_path)

    data = module.convert_npz_to_zarr_data(str(npz_path), allow_legacy_8d=True)

    state = data["observations"]["state"]
    assert state.shape == (3, 8)
    np.testing.assert_allclose(state[:, :7], joint_poses)
    np.testing.assert_allclose(state[:, 7], gripper_states)
    assert data["_meta"]["state_layout"] == "legacy_joint_gripper_8d"
    assert data["_meta"]["allow_legacy_8d"] is True


@pytest.mark.parametrize("path", CONVERT_PATHS)
def test_zarr_verifier_rejects_legacy_8d_by_default(path, tmp_path):
    module = _load_module(path, f"_test_{path.parent.name}_convert_to_zarr_verify")
    _patch_fk(module)
    npz_path = tmp_path / "demo.npz"
    _write_demo_npz(npz_path)
    data = module.convert_npz_to_zarr_data(str(npz_path), allow_legacy_8d=True)
    zarr_path = tmp_path / "demo.zarr"
    module.save_zarr(data, str(zarr_path), overwrite=True)

    assert module.verify_zarr(str(zarr_path)) is False
    assert module.verify_zarr(str(zarr_path), allow_legacy_8d=True) is True


@pytest.mark.parametrize("path", LOAD_PATHS)
def test_zarr_loader_rejects_legacy_8d_by_default(path, tmp_path):
    convert = _load_module(CONVERT_PATHS[0], "_test_convert_to_zarr_for_loader")
    _patch_fk(convert)
    npz_path = tmp_path / "demo.npz"
    _write_demo_npz(npz_path)
    data = convert.convert_npz_to_zarr_data(str(npz_path), allow_legacy_8d=True)
    zarr_path = tmp_path / "demo.zarr"
    convert.save_zarr(data, str(zarr_path), overwrite=True)

    loader = _load_module(path, f"_test_{path.parent.name}_load_zarr")

    assert loader.verify_zarr_for_buffer(str(zarr_path)) is False
    assert loader.verify_zarr_for_buffer(str(zarr_path), allow_legacy_8d=True) is True
    with pytest.raises(RuntimeError, match="legacy 8D joint-state"):
        loader.load_zarr_to_serl_buffer([str(zarr_path)])
