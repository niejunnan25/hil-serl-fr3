from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np
import requests


TARGET_BACKEND = "fr3_hil_serl_plug_bridge"
NO_MOTION_STEP_REJECTED = "NO_MOTION_STEP_REJECTED"
DEFAULT_STALE_SECONDS = 0.5
DEFAULT_MAX_FUTURE_SECONDS = 0.05


@dataclass(frozen=True)
class ActionValidation:
    valid: bool
    reason: str
    original_shape: list[int]
    target_backend: str
    timestamp: float | None
    stale_after_seconds: float = DEFAULT_STALE_SECONDS
    max_future_seconds: float = DEFAULT_MAX_FUTURE_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActionRejected(RuntimeError):
    def __init__(self, validation: ActionValidation):
        self.validation = validation
        super().__init__(validation.reason)


def _shape_of(action: Any) -> list[int]:
    try:
        return list(np.asarray(action).shape)
    except Exception:
        return []


def _parse_timestamp(timestamp: Any) -> tuple[float | None, str | None]:
    if timestamp is None:
        return None, None
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None, "REJECT_BAD_TIMESTAMP"
    if not np.isfinite(value):
        return None, "REJECT_BAD_TIMESTAMP"
    return value, None


def validate_action_7d(
    action: Any,
    timestamp: Any = None,
    now: float | None = None,
    stale_after_seconds: float = DEFAULT_STALE_SECONDS,
    max_future_seconds: float = DEFAULT_MAX_FUTURE_SECONDS,
) -> ActionValidation:
    original_shape = _shape_of(action)
    arr = np.asarray(action)
    timestamp_value, timestamp_reason = _parse_timestamp(timestamp)

    if timestamp_reason is not None:
        return ActionValidation(
            valid=False,
            reason=timestamp_reason,
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )

    if arr.shape == (8,):
        return ActionValidation(
            valid=False,
            reason="REJECT_OLD_OPENPI_DROID_8D_ACTION",
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )
    if arr.shape != (7,):
        return ActionValidation(
            valid=False,
            reason="REJECT_ACTION_SHAPE",
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )
    if not np.issubdtype(arr.dtype, np.number):
        return ActionValidation(
            valid=False,
            reason="REJECT_NON_NUMERIC_ACTION",
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )
    if not np.all(np.isfinite(arr)):
        return ActionValidation(
            valid=False,
            reason="REJECT_NON_FINITE_ACTION",
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )
    if np.any(arr < -1.0) or np.any(arr > 1.0):
        return ActionValidation(
            valid=False,
            reason="REJECT_ACTION_OUT_OF_RANGE",
            original_shape=original_shape,
            target_backend=TARGET_BACKEND,
            timestamp=timestamp_value,
            stale_after_seconds=stale_after_seconds,
            max_future_seconds=max_future_seconds,
        )
    if timestamp_value is not None:
        current = time.time() if now is None else now
        current, current_reason = _parse_timestamp(current)
        if current_reason is not None or current is None:
            return ActionValidation(
                valid=False,
                reason="REJECT_BAD_TIMESTAMP",
                original_shape=original_shape,
                target_backend=TARGET_BACKEND,
                timestamp=timestamp_value,
                stale_after_seconds=stale_after_seconds,
                max_future_seconds=max_future_seconds,
            )
        if timestamp_value - current > max_future_seconds:
            return ActionValidation(
                valid=False,
                reason="REJECT_FUTURE_ACTION",
                original_shape=original_shape,
                target_backend=TARGET_BACKEND,
                timestamp=timestamp_value,
                stale_after_seconds=stale_after_seconds,
                max_future_seconds=max_future_seconds,
            )
        if current - timestamp_value > stale_after_seconds:
            return ActionValidation(
                valid=False,
                reason="REJECT_STALE_ACTION",
                original_shape=original_shape,
                target_backend=TARGET_BACKEND,
                timestamp=timestamp_value,
                stale_after_seconds=stale_after_seconds,
                max_future_seconds=max_future_seconds,
            )

    return ActionValidation(
        valid=True,
        reason="ACCEPT_7D_ACTION",
        original_shape=original_shape,
        target_backend=TARGET_BACKEND,
        timestamp=timestamp_value,
        stale_after_seconds=stale_after_seconds,
        max_future_seconds=max_future_seconds,
    )


class PlugFr3Bridge:
    def __init__(self, base_url: str = "http://127.0.0.1:5017/", no_motion: bool = True):
        self.base_url = base_url.rstrip("/") + "/"
        self.no_motion = no_motion

    def connect_state_only(self) -> dict[str, Any]:
        response = requests.get(self.base_url + "healthz", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
        if payload.get("ok") is not True:
            raise RuntimeError(f"BRIDGE_HEALTH_NOT_OK {payload}")
        return {"connected": True, "mode": payload.get("mode", "unknown")}

    def get_state(self) -> dict[str, Any]:
        response = requests.post(self.base_url + "getstate", json={}, timeout=2.0)
        response.raise_for_status()
        state = response.json()
        if len(state.get("pose", [])) != 7:
            raise RuntimeError(f"BRIDGE_BAD_POSE_SHAPE {state.get('pose')}")
        return state

    def validate_action_7d(self, action: Any, timestamp: Any = None) -> ActionValidation:
        return validate_action_7d(action, timestamp=timestamp)

    def step_7d(self, action: Any, timestamp: Any = None) -> dict[str, Any]:
        validation = self.validate_action_7d(action, timestamp=timestamp)
        if not validation.valid:
            raise ActionRejected(validation)
        if self.no_motion:
            rejected = ActionValidation(
                valid=False,
                reason=NO_MOTION_STEP_REJECTED,
                original_shape=validation.original_shape,
                target_backend=validation.target_backend,
                timestamp=validation.timestamp,
                stale_after_seconds=validation.stale_after_seconds,
                max_future_seconds=validation.max_future_seconds,
            )
            raise ActionRejected(rejected)
        raise RuntimeError("LIVE_STEP_NOT_ENABLED")

    def get_safety_state(self) -> dict[str, Any]:
        return {
            "no_motion": self.no_motion,
            "backend": TARGET_BACKEND,
            "step_enabled": False,
        }

    def stop(self) -> dict[str, Any]:
        return {"stopped": True, "no_motion": self.no_motion}
