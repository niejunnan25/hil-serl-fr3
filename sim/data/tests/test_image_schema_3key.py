"""P4 (sim image schema -> 3-key, for sim-to-real).

DECISION (user): sim producers MUST emit the SAME 3 image keys as real SERL pkl
so a sim-trained policy transfers to real:
  side_policy, wrist_1, side_classifier   (each (3,128,128) uint8)

per contract:
  VALID_PKL_IMAGE_KEYS = ("side_policy", "wrist_1", "side_classifier")
  IMAGE_KEY_ALIAS_MAP  = {"side_classifier": "side_policy"}

This test asserts BOTH real producers (gello_replay.replay_pure_fk AND
FailureScenarioGenerator) emit observations that pass
verify_sim_data.verify_image_keys_complete (the 3-key check), and that the
written pkl passes the full verify_pkl image_keys_complete gate.

The single "pixels" key (pre-P4) must FAIL this; the 3-key emission must PASS.
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

from sim.data.contract import (
    IMAGE_DTYPE,
    IMAGE_KEY_ALIAS_MAP,
    IMAGE_SHAPE,
    VALID_PKL_IMAGE_KEYS,
)
from sim.scripts.verify_sim_data import verify_image_keys_complete, verify_pkl


def _make_fake_demo(N=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "joint_poses": rng.normal(size=(N, 7)).astype(np.float64),
        "gripper_states": rng.uniform(0, 1, size=N).astype(np.float64),
        "timestamps": np.linspace(0, 1, N).astype(np.float64),
    }


def _assert_obs_3key(obs: dict) -> None:
    """Every observation dict must carry exactly the 3 contract image keys,
    each (3,128,128) uint8, and no leftover single 'pixels' key."""
    assert "pixels" not in obs, "pre-P4 single 'pixels' key must be gone"
    assert verify_image_keys_complete(obs), (
        f"obs image keys {set(obs.keys()) - {'state'}} "
        f"!= {set(VALID_PKL_IMAGE_KEYS)}"
    )
    for k in VALID_PKL_IMAGE_KEYS:
        assert obs[k].shape == IMAGE_SHAPE, f"{k} shape {obs[k].shape} != {IMAGE_SHAPE}"
        assert obs[k].dtype == np.dtype(IMAGE_DTYPE), f"{k} dtype {obs[k].dtype}"


# ---------------------------------------------------------------------------
# 1) gello_replay.replay_pure_fk: each transition obs/next_obs has 3 image keys
# ---------------------------------------------------------------------------
def test_gello_replay_emits_3_image_keys():
    from sim.data import gello_replay

    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=8), max_frames=8)
    assert len(transitions) >= 1
    for t in transitions:
        _assert_obs_3key(t["observations"])
        _assert_obs_3key(t["next_observations"])


def test_gello_replay_side_classifier_is_alias_of_side_policy():
    """Per IMAGE_KEY_ALIAS_MAP, side_classifier must be byte-identical to side_policy."""
    from sim.data import gello_replay

    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=8), max_frames=8)
    alias_key, source_key = next(iter(IMAGE_KEY_ALIAS_MAP.items()))
    for t in transitions:
        obs = t["observations"]
        np.testing.assert_array_equal(obs[alias_key], obs[source_key])


def test_gello_replay_written_pkl_passes_verify_image_keys_complete():
    from sim.data import gello_replay

    transitions = gello_replay.replay_pure_fk(_make_fake_demo(N=8), max_frames=8)
    with tempfile.TemporaryDirectory() as d:
        pkl_path = os.path.join(d, "gello_sim.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(transitions, f)
        result = verify_pkl(pkl_path)
    assert result["image_keys_complete"] is True, result
    assert result["image_shape_correct"] is True, result
    assert result["image_dtype_correct"] is True, result


# ---------------------------------------------------------------------------
# 2) FailureScenarioGenerator: each class pkl has 3 image keys
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure_class",
    ["mis_alignment", "angle_offset", "insufficient_force", "drop"],
)
def test_failure_generator_emits_3_image_keys(failure_class):
    from sim.data.failure_scenario_generator import FailureScenarioGenerator

    gen = FailureScenarioGenerator(seed=42)
    method = getattr(gen, f"gen_{failure_class}")
    with tempfile.TemporaryDirectory() as d:
        pkl_path = os.path.join(d, f"failure_{failure_class}.pkl")
        transitions = method(_make_fake_demo(N=12), output_path=pkl_path)
        assert len(transitions) >= 1
        for t in transitions:
            _assert_obs_3key(t["observations"])
            _assert_obs_3key(t["next_observations"])
        result = verify_pkl(pkl_path)
    assert result["image_keys_complete"] is True, result
    assert result["image_shape_correct"] is True, result
    assert result["image_dtype_correct"] is True, result


def test_failure_generator_side_classifier_is_alias_of_side_policy():
    from sim.data.failure_scenario_generator import FailureScenarioGenerator

    gen = FailureScenarioGenerator(seed=42)
    transitions = gen.gen_mis_alignment(_make_fake_demo(N=12), output_path=None)
    alias_key, source_key = next(iter(IMAGE_KEY_ALIAS_MAP.items()))
    for t in transitions:
        obs = t["observations"]
        np.testing.assert_array_equal(obs[alias_key], obs[source_key])
