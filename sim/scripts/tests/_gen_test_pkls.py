"""Helper: 构造测试用 pkl, valid + 各种 missing 场景."""
import os
import pickle
import tempfile

import numpy as np

from sim.data.contract import (
    ACTION_SCALE, IMAGE_SHAPE, STATE_DIMS, TRANSITION_KEYS,
)


def _make_observation(state: np.ndarray, image_keys=("side_policy", "wrist_1", "side_classifier")) -> dict:
    """Construct obs dict with 3 image keys (default) or subset (for negative tests)."""
    obs = {"state": state.astype(np.float32)}
    for k in image_keys:
        obs[k] = np.random.default_rng(0).integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
    return obs


def make_valid_pkl(path: str, N: int = 5, seed: int = 0) -> list[dict]:
    """Construct a valid SERL pkl with 3 image keys + ordered state."""
    rng = np.random.default_rng(seed)
    transitions = []
    for i in range(N):
        state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        next_state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        action = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
        obs = _make_observation(state, image_keys=("side_policy", "wrist_1", "side_classifier"))
        next_obs = _make_observation(next_state, image_keys=("side_policy", "wrist_1", "side_classifier"))
        transitions.append({
            "observations": obs,
            "next_observations": next_obs,
            "actions": action,
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": False,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(transitions, f)
    return transitions


def make_missing_classifier_pkl(path: str, N: int = 5, seed: int = 0) -> list[dict]:
    """Construct a pkl MISSING side_classifier (for negative test)."""
    rng = np.random.default_rng(seed)
    transitions = []
    for i in range(N):
        state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        next_state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        action = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
        # ONLY side_policy + wrist_1, NO side_classifier
        obs = _make_observation(state, image_keys=("side_policy", "wrist_1"))
        next_obs = _make_observation(next_state, image_keys=("side_policy", "wrist_1"))
        transitions.append({
            "observations": obs,
            "next_observations": next_obs,
            "actions": action,
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": False,
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(transitions, f)
    return transitions
