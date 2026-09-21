"""Versioned model actions, independent of robot I/O and environment wrappers.

Both contracts use the existing normalized body-frame translation and device
scaling. The fixed contract removes unused axes from the model; it does not
change the robot's displacement, force, velocity or joint limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


LEGACY = "legacy-hybrid-7d-v1"
FIXED_XYZ = "fixed-xyz-v1"
ACTION_BOUND_TOLERANCE = 1e-5


def _action(value, size, *, label):
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (size,) or not np.all(np.isfinite(action)):
        raise ValueError(f"{label} must be a finite ({size},) action, got {action.shape}")
    return action.copy()


@dataclass(frozen=True)
class ActionContract:
    name: str
    learning_dim: int
    setup_mode: str

    @property
    def fixed_xyz(self):
        return self.name == FIXED_XYZ

    def to_device_action(self, learning_action):
        """Expand a model action to the audited seven-axis body-frame request."""
        action = _action(learning_action, self.learning_dim, label=self.name)
        if not self.fixed_xyz:
            return action
        if np.any(np.abs(action) > 1.0 + ACTION_BOUND_TOLERANCE):
            raise ValueError("fixed-xyz-v1 model action must be within [-1, 1]")
        return np.concatenate((action, np.zeros(4, dtype=np.float32)))

    def project_device_action(self, device_action):
        """Apply task locks explicitly; retain the input separately for auditing."""
        action = _action(device_action, 7, label="device action")
        action[3:6] = 0
        if self.fixed_xyz:
            action[6] = 0
        return action

    def to_learning_action(self, device_action):
        """Validate a selected action before admitting it to model replay.

        Projection is deliberately separate: a stale or incompatible seven-axis
        replay item must not silently enter the new three-axis dataset.
        """
        action = _action(device_action, 7, label="selected device action")
        if not self.fixed_xyz:
            return action
        if np.any(action[3:] != 0):
            raise ValueError("fixed-xyz-v1 selected action contains unlocked rotation or gripper")
        if np.any(np.abs(action[:3]) > 1.0 + ACTION_BOUND_TOLERANCE):
            raise ValueError("fixed-xyz-v1 selected body action must be within [-1, 1]")
        return action[:3].copy()


_CONTRACTS = {
    LEGACY: ActionContract(LEGACY, 7, "single-arm-learned-gripper"),
    FIXED_XYZ: ActionContract(FIXED_XYZ, 3, "single-arm-fixed-gripper"),
}


def resolve_action_contract(name=None):
    """Missing version means the historical contract, never an implicit upgrade."""
    if isinstance(name, ActionContract):
        if _CONTRACTS.get(name.name) != name:
            raise ValueError(f"unsupported action contract: {name!r}")
        return name
    try:
        return _CONTRACTS[LEGACY if name is None else name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported action contract: {name!r}") from exc


def action_contract_from_env():
    return resolve_action_contract(os.environ.get("HILSERL_ACTION_CONTRACT", LEGACY))
