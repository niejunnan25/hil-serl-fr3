"""Translation limits in the robot base frame, shared by action and reset."""
from __future__ import annotations

import numpy as np


def clip_translation_delta(delta, xy_limit, z_limit=None):
    """Project radially onto the XY/Z ellipsoid without changing direction."""
    value = np.asarray(delta, dtype=float)
    limits = np.array([xy_limit, xy_limit, xy_limit if z_limit is None else z_limit], dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("translation delta must contain three finite values")
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("translation limits must be finite and positive")
    ratio = float(np.linalg.norm(value / limits))
    return value.copy() if ratio <= 1.0 else value / ratio


def advance_position_target(previous, delta, measured, xy_limit, z_limit=None):
    """Accumulate commanded displacement without integrating blocked tracking error.

    Measurements bound further advance; they never replace the reference.
    Zero input always holds the previous target. An external displacement
    outside the tracking envelope allows only commands that reduce that error.
    """
    previous, delta, measured = [np.asarray(v, dtype=float) for v in (previous, delta, measured)]
    limits = np.array([xy_limit, xy_limit, xy_limit if z_limit is None else z_limit], dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (previous, delta, measured)):
        raise ValueError("position target inputs must contain three finite values")
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("tracking limits must be finite and positive")
    e, d = (previous-measured)/limits, delta/limits
    candidate = previous+delta
    before, after = float(e@e), float((e+d)@(e+d))
    if after <= 1.0:
        return candidate
    a = float(d@d)
    if a == 0:
        return previous.copy()
    if before > 1.0:
        return candidate if after < before else previous.copy()
    b = 2*float(e@d)
    fraction = np.clip((-b+np.sqrt(max(0., b*b-4*a*(before-1))))/(2*a), 0., 1.)
    return previous+fraction*delta


def seed_execution_contract(z_limit=None, position_target_mode="measured-relative-v1"):
    """Describe the physical action mapping; old snapshots retain their mapping."""
    result = dict(rotation_locked=True, gripper_locked=True, action_max_step_m=.008)
    if position_target_mode not in {"measured-relative-v1", "command-relative-v1"}:
        raise ValueError("unknown position target mode")
    if position_target_mode == "command-relative-v1":
        result["position_target_mode"] = position_target_mode
    if z_limit is not None:
        if type(z_limit) not in (int, float) or not .008 <= z_limit <= .016:
            raise ValueError("supported base Z displacement must be between 8 and 16 mm")
        result.update(action_max_z_step_m=float(z_limit),
                      translation_limit="base_xyz_ellipsoid_v1",
                      action_scale_m=[.015, .015, float(z_limit)])
    return result
