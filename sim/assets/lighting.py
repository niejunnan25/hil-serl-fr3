"""Scene lighting helpers for sim camera rendering.

Without lights, RGB output is all-zero (Phase 0 smoke observation). This module
provides illumination tuned to match real ZED frame mean intensities so the
existing 8010 policy sees sim images close to its training distribution.

Reference values sampled from
``/home/robot/droid/record/20260413_140912/frames/`` step 0:
* external mean intensity ≈ 136 / 255
* wrist mean intensity    ≈ 106 / 255
* table dark patch median ≈ (55, 57, 50) ≈ 0.22
* plate patch median       ≈ (152, 153, 107) ≈ 0.60

Earlier defaults (DomeLight 1500 / DistantLight 2000) yielded sim external mean
≈ 110-120, slightly darker than real ≈ 136. Bumping to 2200/2400 OVERSHOT
(sim mean ≈ 150, dome flooded the background to near-white). The pair below
(900 / 1600) brings sim mean closer to real while keeping the dark table dark.
"""
from __future__ import annotations

import os
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

_CAPTURE_PROFILE_LABEL_ENV = "_FR3_CAPTURE_PROFILE_LABEL"
_CAPTURE_DOME_INTENSITY_ENV = "_FR3_CAPTURE_DOME_INTENSITY"
_CAPTURE_DOME_COLOR_RGB_ENV = "_FR3_CAPTURE_DOME_COLOR_RGB"
_CAPTURE_SUN_INTENSITY_ENV = "_FR3_CAPTURE_SUN_INTENSITY"
_CAPTURE_SUN_COLOR_RGB_ENV = "_FR3_CAPTURE_SUN_COLOR_RGB"
_CAPTURE_SUN_ANGLE_DEG_ENV = "_FR3_CAPTURE_SUN_ANGLE_DEG"

_DOME_DEFAULT_COLOR = (0.9, 0.9, 0.9)
_SUN_DEFAULT_COLOR = (1.0, 1.0, 0.95)


def _capture_env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
) -> tuple[float, str]:
    raw = os.environ.get(name)
    if raw is None:
        return float(default), "default"
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value, "env"


def _capture_env_rgb(
    name: str,
    default: tuple[float, float, float],
) -> tuple[tuple[float, float, float], str]:
    raw = os.environ.get(name)
    if raw is None:
        return tuple(float(v) for v in default), "default"
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{name} must contain only floats, got {raw!r}") from exc
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} values must be in [0.0, 1.0], got {raw!r}")
    return values, "env"


def _capture_profile_entry(
    name: str,
    value: float | tuple[float, float, float],
    source: str,
) -> dict[str, Any]:
    if isinstance(value, tuple):
        json_value: float | list[float] = [float(v) for v in value]
    else:
        json_value = float(value)
    return {"env": name, "value": json_value, "source": source}


def capture_lighting_profile_from_env() -> dict[str, Any]:
    """Return resolved capture-only lighting values for report metadata."""
    dome_intensity, dome_intensity_source = _capture_env_float(
        _CAPTURE_DOME_INTENSITY_ENV,
        1000.0,  # Phase3: brighter dome for uniform overhead
        min_value=0.0,
    )
    dome_color, dome_color_source = _capture_env_rgb(
        _CAPTURE_DOME_COLOR_RGB_ENV,
        _DOME_DEFAULT_COLOR,
    )
    sun_intensity, sun_intensity_source = _capture_env_float(
        _CAPTURE_SUN_INTENSITY_ENV,
        1400.0,  # Phase3: soften sun for less harsh directional
        min_value=0.0,
    )
    sun_color, sun_color_source = _capture_env_rgb(
        _CAPTURE_SUN_COLOR_RGB_ENV,
        _SUN_DEFAULT_COLOR,
    )
    sun_angle_deg, sun_angle_source = _capture_env_float(
        _CAPTURE_SUN_ANGLE_DEG_ENV,
        0.5,
        min_value=0.0,
    )
    return {
        "label_env": _CAPTURE_PROFILE_LABEL_ENV,
        "dome_intensity": _capture_profile_entry(
            _CAPTURE_DOME_INTENSITY_ENV,
            dome_intensity,
            dome_intensity_source,
        ),
        "dome_color_rgb": _capture_profile_entry(
            _CAPTURE_DOME_COLOR_RGB_ENV,
            dome_color,
            dome_color_source,
        ),
        "sun_intensity": _capture_profile_entry(
            _CAPTURE_SUN_INTENSITY_ENV,
            sun_intensity,
            sun_intensity_source,
        ),
        "sun_color_rgb": _capture_profile_entry(
            _CAPTURE_SUN_COLOR_RGB_ENV,
            sun_color,
            sun_color_source,
        ),
        "sun_angle_deg": _capture_profile_entry(
            _CAPTURE_SUN_ANGLE_DEG_ENV,
            sun_angle_deg,
            sun_angle_source,
        ),
    }


def build_dome_light(prim_path: str = "/World/DomeLight",
                     intensity: float = 900.0) -> AssetBaseCfg:
    """Ambient dome light — fills shadows, gives base illumination."""
    resolved_intensity, _ = _capture_env_float(
        _CAPTURE_DOME_INTENSITY_ENV,
        intensity,
        min_value=0.0,
    )
    resolved_color, _ = _capture_env_rgb(_CAPTURE_DOME_COLOR_RGB_ENV, _DOME_DEFAULT_COLOR)
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.DomeLightCfg(
            intensity=resolved_intensity,
            color=resolved_color,
        ),
    )


def build_distant_light(prim_path: str = "/World/SunLight",
                        intensity: float = 1400.0,  # Phase3: soften sun for less harsh directional
                        angle_deg: float = 0.5) -> AssetBaseCfg:
    """Directional / sun-like light for scene definition."""
    resolved_intensity, _ = _capture_env_float(
        _CAPTURE_SUN_INTENSITY_ENV,
        intensity,
        min_value=0.0,
    )
    resolved_color, _ = _capture_env_rgb(_CAPTURE_SUN_COLOR_RGB_ENV, _SUN_DEFAULT_COLOR)
    resolved_angle_deg, _ = _capture_env_float(
        _CAPTURE_SUN_ANGLE_DEG_ENV,
        angle_deg,
        min_value=0.0,
    )
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.DistantLightCfg(
            intensity=resolved_intensity,
            color=resolved_color,
            angle=resolved_angle_deg,
        ),
    )


__all__ = [
    "build_dome_light",
    "build_distant_light",
    "capture_lighting_profile_from_env",
]
