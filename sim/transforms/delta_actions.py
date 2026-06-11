"""DeltaActions transform (per spec §0.4 Q decisions, handoff-confirmed convention).

Mask = (True,)*7 + (False,) means: first 7 dims (joint targets) are
encoded as delta from base; last dim (gripper) is raw.

Reference implementation for unit testing. Real integration uses OpenPI's
DeltaActions in droid/openpi_client; this mirror exists so Phase 1 can
verify our understanding of the math without import side effects.
"""
from __future__ import annotations

import numpy as np


DEFAULT_MASK: tuple[bool, ...] = (True, True, True, True, True, True, True, False)


def apply_delta(
    action: np.ndarray,
    base: np.ndarray | None = None,
    mask: tuple[bool, ...] = DEFAULT_MASK,
) -> np.ndarray:
    """Convert absolute action to delta-action for masked dims.

    action: shape (T, 8). base: shape (8,) — the absolute reference (typically
    action[0] or the robot's q at t=0). If None, uses action[0].
    For dims where mask[i] is True, output[:, i] = action[:, i] - base[i].
    For dims where mask[i] is False, output[:, i] = action[:, i] (passthrough).
    """
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != len(mask):
        raise ValueError(
            f"expected (T, {len(mask)}), got {arr.shape}"
        )
    if base is None:
        base = arr[0].copy()
    base_arr = np.asarray(base, dtype=np.float64)
    if base_arr.shape != (len(mask),):
        raise ValueError(
            f"base shape {base_arr.shape} != ({len(mask)},)"
        )
    out = arr.copy()
    for i, is_delta in enumerate(mask):
        if is_delta:
            out[:, i] = arr[:, i] - base_arr[i]
    return out


def invert_delta(
    delta_action: np.ndarray,
    base: np.ndarray,
    mask: tuple[bool, ...] = DEFAULT_MASK,
) -> np.ndarray:
    """Reconstruct absolute action from delta + base (inverse of apply_delta)."""
    arr = np.asarray(delta_action, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != len(mask):
        raise ValueError(f"expected (T, {len(mask)}), got {arr.shape}")
    base_arr = np.asarray(base, dtype=np.float64)
    out = arr.copy()
    for i, is_delta in enumerate(mask):
        if is_delta:
            out[:, i] = arr[:, i] + base_arr[i]
    return out


__all__ = ["DEFAULT_MASK", "apply_delta", "invert_delta"]
