"""Admission and real-transition counts for the fixed XYZ replay contract.

Native image replay allocates support slots as well as real transitions. Its
length cannot establish either the online warmup threshold or an update ratio.
"""
from __future__ import annotations

from threading import Lock
import numpy as np
from hilserl.image_profile import FULL_FRAME, get_image_profile, observation_image_schema


def validate_transition(transition, image_profile=FULL_FRAME, *, reward_spec=None):
    """Validate before mutation; success labels must describe real terminals."""
    if not isinstance(transition, dict):
        raise ValueError("Replay transition must be a dict")
    info = transition.get("infos", {})
    if not isinstance(info, dict) or info.get("action_contract") != "fixed-xyz-v1":
        raise ValueError("Replay action_contract is not fixed-xyz-v1")
    profile = get_image_profile(image_profile)
    if info.get("image_profile", FULL_FRAME) != profile["name"]:
        raise ValueError("Replay image profile differs from the model")
    raw_id = info.get("raw_transition_id")
    if not isinstance(raw_id, str) or not raw_id.rpartition("/")[0] or not raw_id.rpartition("/")[2].isdecimal():
        raise ValueError("Replay needs a real raw_transition_id")
    action = np.asarray(transition.get("actions"))
    if action.shape != (3,) or action.dtype.kind != "f" or not np.isfinite(action).all() or np.any(np.abs(action) > 1.00001):
        raise ValueError("Replay actions must be finite normalized XYZ")
    for field in ("observations", "next_observations"):
        obs = transition.get(field)
        if not isinstance(obs, dict):
            raise ValueError(f"Missing {field}")
        state = np.asarray(obs.get("state"))
        if state.shape != (1, 19) or state.dtype.kind != "f" or not np.isfinite(state).all():
            raise ValueError(f"{field}.state must be finite (1,19)")
        for key in ("wrist_1", "side_policy"):
            image = np.asarray(obs.get(key))
            expected_shape = tuple(observation_image_schema(profile)[key]["shape"])
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(f"Invalid {field}.{key}")
    values = {key: np.asarray(transition.get(key)) for key in ("rewards", "dones", "masks")}
    if any(v.shape != () or v.dtype.kind not in "bif" or not np.isfinite(v) for v in values.values()):
        raise ValueError("Reward, done and mask must be finite scalars")
    reward, done, mask = (float(values[k]) for k in ("rewards", "dones", "masks"))
    if done not in (0, 1) or mask != 1 - done:
        raise ValueError("Replay reward/terminal/mask contract violated")
    if reward_spec is None:
        if reward not in (0, 1) or reward and not done or "reward" in info:
            raise ValueError("Replay reward/terminal/mask contract violated")
    else:
        from hilserl.reward_provider import validate_reward
        validate_reward(transition, reward_spec)
    if done and info.get("verdict_source") != "human":
        raise ValueError("Terminal replay needs a human verdict")
    if info.get("manual_success_credit") or float(transition.get("grasp_penalty", 0)) != 0:
        raise ValueError("Synthetic success credit or gripper penalty is forbidden")
    return raw_id


class ContractReplayStore:
    """Preserve the native sampler while validating and counting real inserts."""

    def __init__(self, store, *, image_profile=FULL_FRAME, reward_spec=None):
        self.store = store
        self.image_profile = get_image_profile(image_profile)["name"]
        self.reward_spec = reward_spec
        self._ids = set()
        self._admission_lock = Lock()

    @property
    def transition_count(self):
        with self._admission_lock:
            return len(self._ids)

    def existing_ids(self, ids):
        with self._admission_lock:
            return self._ids.intersection(ids)

    def insert(self, transition):
        raw_id = validate_transition(transition, self.image_profile, reward_spec=self.reward_spec)
        with self._admission_lock:
            if raw_id in self._ids:
                raise ValueError(f"Duplicate raw transition: {raw_id}")
            self.store.insert(transition)
            self._ids.add(raw_id)

    def batch_insert(self, transitions):
        for transition in transitions:
            self.insert(transition)

    def __len__(self):
        return len(self.store)

    def __getattr__(self, name):
        return getattr(self.store, name)
