from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any

import numpy as np

from fr3_hil_bridge.plug_bridge import validate_action_7d


ACTION_SCHEMA_VERSION = "xbox_7d_v1"
DEFAULT_FIXTURE_AXIS_BASE = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class XboxTeleopConfig:
    """Bounded Xbox-to-HIL-SERL action config.

    Values are normalized action magnitudes, not direct physical FR3 commands.
    The realtime bridge is responsible for converting accepted normalized
    actions to physical limits.
    """

    action_schema_version: str = ACTION_SCHEMA_VERSION
    stick_deadzone: float = 0.15
    trigger_deadzone: float = 0.07
    input_stale_ms: float = 75.0
    fixture_axis_base: tuple[float, float, float] = DEFAULT_FIXTURE_AXIS_BASE
    coarse_trans_norm: float = 0.45
    coarse_rot_norm: float = 0.35
    fine_trans_norm: float = 0.12
    fine_rot_norm: float = 0.08
    insert_axis_norm: float = 0.08
    retreat_axis_norm: float = 0.18
    insert_lateral_norm: float = 0.04
    insert_yaw_norm: float = 0.03
    require_deadman: bool = True
    deadman_button: str = "RB"
    gripper_close_button: str = "A"
    gripper_open_button: str = "B"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fixture_axis_base"] = list(self.fixture_axis_base)
        return data


@dataclass(frozen=True)
class XboxControllerState:
    connected: bool
    axes: dict[str, float] = field(default_factory=dict)
    buttons: dict[str, bool] = field(default_factory=dict)
    hats: dict[str, tuple[int, int]] = field(default_factory=dict)
    mode: str | None = None
    seq: int = 0
    t_mono_ns: int = 0
    device_name: str = "mock"
    guid: str = "mock"
    instance_id: int = -1
    backend: str = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "axes": dict(self.axes),
            "buttons": dict(self.buttons),
            "hats": {k: list(v) for k, v in self.hats.items()},
            "mode": self.mode,
            "seq": self.seq,
            "t_mono_ns": self.t_mono_ns,
            "device_name": self.device_name,
            "guid": self.guid,
            "instance_id": self.instance_id,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class ActionMappingResult:
    action: np.ndarray
    mode: str
    enabled: bool
    reject_code: str | None
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.astype(float).tolist(),
            "mode": self.mode,
            "enabled": self.enabled,
            "reject_code": self.reject_code,
            "info": self.info,
        }


def _clip_unit(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _axis(state: XboxControllerState, name: str) -> float:
    return _clip_unit(float(state.axes.get(name, 0.0)))


def _button(state: XboxControllerState, name: str) -> bool:
    return bool(state.buttons.get(name, False))


def _trigger_value(raw: float, deadzone: float) -> float:
    raw = _clip_unit(raw)
    # Pygame/SDL backends differ: triggers may be [0, 1] or [-1, 1].
    value = (raw + 1.0) * 0.5 if raw < 0.0 else raw
    value = float(np.clip(value, 0.0, 1.0))
    if value <= deadzone:
        return 0.0
    return float(np.clip((value - deadzone) / (1.0 - deadzone), 0.0, 1.0))


def _deadzone(value: float, deadzone: float) -> float:
    value = _clip_unit(value)
    if abs(value) <= deadzone:
        return 0.0
    scaled = (abs(value) - deadzone) / (1.0 - deadzone)
    return float(np.sign(value) * np.clip(scaled, 0.0, 1.0))


def _mode_from_state(state: XboxControllerState) -> str:
    if state.mode:
        return state.mode
    hat = state.hats.get("DPAD") or state.hats.get("0") or (0, 0)
    if tuple(hat) == (0, 1):
        return "coarse"
    if tuple(hat) == (0, -1):
        return "fine"
    if tuple(hat) == (1, 0):
        return "rotation"
    if tuple(hat) == (-1, 0):
        return "insert_axis"
    return "coarse"


def _zero_result(state: XboxControllerState, mode: str, reason: str, cfg: XboxTeleopConfig) -> ActionMappingResult:
    return ActionMappingResult(
        action=np.zeros(7, dtype=np.float32),
        mode=mode,
        enabled=False,
        reject_code=reason,
        info={
            "teleop_enabled": False,
            "reject_code": reason,
            "controller_seq": state.seq,
            "controller_t_mono_ns": state.t_mono_ns,
            "mapping_version": cfg.action_schema_version,
            "raw_axes": dict(state.axes),
            "raw_buttons": dict(state.buttons),
        },
    )


def _axis_unit(axis: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("BAD_FIXTURE_AXIS_BASE")
    return arr / norm


def map_xbox_to_action(
    state: XboxControllerState,
    cfg: XboxTeleopConfig | None = None,
    now_ns: int | None = None,
) -> ActionMappingResult:
    cfg = cfg or XboxTeleopConfig()
    mode = _mode_from_state(state)
    now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)

    if not state.connected:
        return _zero_result(state, mode, "CONTROLLER_DISCONNECTED", cfg)
    if state.t_mono_ns and (now_ns - state.t_mono_ns) > int(cfg.input_stale_ms * 1_000_000):
        return _zero_result(state, mode, "CONTROLLER_INPUT_STALE", cfg)
    deadman = _button(state, cfg.deadman_button)
    if cfg.require_deadman and not deadman:
        return _zero_result(state, mode, "DEADMAN_RELEASED", cfg)

    lx = _deadzone(_axis(state, "LEFTX"), cfg.stick_deadzone)
    ly = _deadzone(_axis(state, "LEFTY"), cfg.stick_deadzone)
    rx = _deadzone(_axis(state, "RIGHTX"), cfg.stick_deadzone)
    ry = _deadzone(_axis(state, "RIGHTY"), cfg.stick_deadzone)
    lt = _trigger_value(_axis(state, "LT"), cfg.trigger_deadzone)
    rt = _trigger_value(_axis(state, "RT"), cfg.trigger_deadzone)

    action = np.zeros(7, dtype=np.float32)

    if mode == "coarse":
        action[0] = -ly * cfg.coarse_trans_norm
        action[1] = lx * cfg.coarse_trans_norm
        action[2] = (lt - rt) * cfg.coarse_trans_norm
        action[5] = rx * cfg.coarse_rot_norm
    elif mode == "fine":
        action[0] = -ly * cfg.fine_trans_norm
        action[1] = lx * cfg.fine_trans_norm
        action[2] = (lt - rt) * cfg.fine_trans_norm
        action[5] = rx * cfg.fine_rot_norm
    elif mode == "rotation":
        action[3] = lx * cfg.coarse_rot_norm
        action[4] = -ly * cfg.coarse_rot_norm
        action[5] = rx * cfg.coarse_rot_norm
    elif mode == "insert_axis":
        axis = _axis_unit(cfg.fixture_axis_base)
        lateral = np.array([-ly, lx, 0.0], dtype=np.float32) * cfg.insert_lateral_norm
        axial = axis * (rt * cfg.insert_axis_norm) - axis * (lt * cfg.retreat_axis_norm)
        action[:3] = lateral + axial
        if _button(state, "RS"):
            action[5] = rx * cfg.insert_yaw_norm
    else:
        return _zero_result(state, mode, "UNKNOWN_TELEOP_MODE", cfg)

    if _button(state, cfg.gripper_close_button) and not _button(state, cfg.gripper_open_button):
        action[6] = 1.0
    elif _button(state, cfg.gripper_open_button) and not _button(state, cfg.gripper_close_button):
        action[6] = -1.0

    preclip = action.copy()
    action = np.clip(action, -1.0, 1.0).astype(np.float32)
    validation = validate_action_7d(action, timestamp=None)
    if not validation.valid:
        return _zero_result(state, mode, validation.reason, cfg)

    return ActionMappingResult(
        action=action,
        mode=mode,
        enabled=True,
        reject_code=None,
        info={
            "teleop_enabled": True,
            "mode": mode,
            "deadman": deadman,
            "controller_seq": state.seq,
            "controller_t_mono_ns": state.t_mono_ns,
            "actor_t_mono_ns": now_ns,
            "mapping_version": cfg.action_schema_version,
            "fixture_axis_base": list(cfg.fixture_axis_base),
            "raw_axes": dict(state.axes),
            "raw_buttons": dict(state.buttons),
            "action_preclip": preclip.astype(float).tolist(),
            "executed_action_norm": action.astype(float).tolist(),
        },
    )


def validate_mapping_result(result: ActionMappingResult) -> dict[str, Any]:
    validation = validate_action_7d(result.action)
    return {
        "ok": validation.valid,
        "reason": validation.reason,
        "action_shape": list(np.asarray(result.action).shape),
        "action_min": float(np.min(result.action)),
        "action_max": float(np.max(result.action)),
        "mode": result.mode,
        "enabled": result.enabled,
        "reject_code": result.reject_code,
        "target_backend": validation.target_backend,
        "action_schema_version": ACTION_SCHEMA_VERSION,
    }
