import importlib.util
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = REPO_ROOT / "scripts" / "convert_to_pkl.py"
DUPLICATE = REPO_ROOT / "scripts" / "gello_pipeline" / "convert_to_pkl.py"


def _load_module(path: Path, name: str):
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_demo(path: Path):
    joint_poses = np.arange(28, dtype=np.float64).reshape(4, 7) / 100.0
    gripper_states = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
    timestamps = np.arange(4, dtype=np.float64)
    np.savez(path, joint_poses=joint_poses, gripper_states=gripper_states, timestamps=timestamps)
    return joint_poses, gripper_states


def _patch_deltas_and_normalize(module):
    deltas = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
            [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
            [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
        ],
        dtype=np.float64,
    )

    def fake_deltas(_joint_poses):
        return deltas

    def fake_normalize(cartesian_delta, action_scale, gripper):
        return np.concatenate([np.asarray(cartesian_delta, dtype=np.float32), [np.float32(gripper)]])

    module.trajectory_to_cartesian_deltas = fake_deltas
    module.normalize_action = fake_normalize
    return deltas


def test_primary_converter_action_matches_obs_to_next_obs_delta(tmp_path):
    module = _load_module(PRIMARY, "_test_convert_to_pkl_primary")
    npz = tmp_path / "demo.npz"
    joint_poses, gripper_states = _write_demo(npz)
    deltas = _patch_deltas_and_normalize(module)

    transitions = module.convert_npz_to_pkl_data(str(npz), filter_zero_actions=True)

    assert len(transitions) == 3
    for transition_index, t in enumerate(transitions):
        source_index = transition_index
        target_index = transition_index + 1
        np.testing.assert_allclose(t["observations"]["state"][:7], joint_poses[source_index])
        np.testing.assert_allclose(t["next_observations"]["state"][:7], joint_poses[target_index])
        np.testing.assert_allclose(t["actions"][:6], deltas[target_index])
        assert t["actions"][6] == np.float32(gripper_states[target_index])
    assert transitions[-1]["dones"] is True
    assert transitions[-1]["masks"] == np.float32(0.0)


def test_duplicate_converter_defaults_and_alignment_match_primary(tmp_path):
    primary = _load_module(PRIMARY, "_test_convert_to_pkl_primary_defaults")
    duplicate = _load_module(DUPLICATE, "_test_convert_to_pkl_duplicate")
    assert duplicate.DEFAULT_POS_SCALE == primary.DEFAULT_POS_SCALE == 0.015
    assert duplicate.DEFAULT_RPY_SCALE == primary.DEFAULT_RPY_SCALE == 0.1

    npz = tmp_path / "demo.npz"
    _write_demo(npz)
    deltas = _patch_deltas_and_normalize(duplicate)

    transitions = duplicate.convert_npz_to_pkl_data(str(npz), filter_zero_actions=True)

    assert len(transitions) == 3
    for transition_index, t in enumerate(transitions):
        target_index = transition_index + 1
        np.testing.assert_allclose(t["actions"][:6], deltas[target_index])
