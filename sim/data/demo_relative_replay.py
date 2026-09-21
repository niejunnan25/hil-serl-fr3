"""Demo-relative cartesian replay plans for 8014 reverse-vegetable sim.

Raw 8014 demos were collected in the real scene frame. Directly replaying their
absolute joint positions in the current IsaacLab scene can miss the object when
the plate/object layout differs by a few centimeters. This module is deliberately
Isaac-free and no-motion: it preserves the human-demo grasp/lift/release deltas
but anchors the secure-grasp pose to the current sim object pose.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from droid.sim.data.grasp_priors import (
    DemoCartesianProbe,
    DemoGraspPriors,
    plan_from_demo_priors,
)


@dataclasses.dataclass(frozen=True)
class DemoRelativeReplayPlan:
    """Cartesian waypoints from demo priors anchored to a current object pose."""

    object: str
    object_initial_pos: tuple[float, float, float]
    source_secure_grasp_xyz: tuple[float, float, float]
    preclose_xyz: tuple[float, float, float]
    secure_grasp_xyz: tuple[float, float, float]
    lift_xyz: tuple[float, float, float]
    lift_xyz_offset: tuple[float, float, float]
    release_xyz: tuple[float, float, float]
    release_xyz_offset: tuple[float, float, float]
    secure_orientation_3: tuple[float, float, float]
    close_hold_steps: int
    close_hold_s: float
    secure_gripper_width_m: float

    def as_waypoints(self) -> list[dict[str, Any]]:
        return [
            {"name": "preclose", "xyz": self.preclose_xyz, "gripper": "open"},
            {"name": "secure_grasp", "xyz": self.secure_grasp_xyz, "gripper": "closed"},
            {"name": "lift", "xyz": self.lift_xyz, "gripper": "closed"},
            {"name": "release", "xyz": self.release_xyz, "gripper": "open"},
        ]


def _vec3(values: tuple[float, float, float] | list[float], *, key: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{key} must be a 3-vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def _add_vec3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub_vec3(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _open_approach_path(
    *,
    preclose_xyz: tuple[float, float, float],
    secure_xyz: tuple[float, float, float],
    approach_steps: int,
    preapproach_z_offset: float = 0.0,
    approach_via_above_secure: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the open-gripper approach, optionally starting above preclose.

    When ``approach_via_above_secure`` is True, the preclose-to-secure leg is
    split into a lateral move at preclose_z to ``(secure_x, secure_y,
    preclose_z)`` followed by a vertical descent to ``secure_xyz``. This avoids
    diagonal scraping of the object during the open-gripper phase when the
    human-demo preclose and secure poses differ substantially in XY.
    """
    approach_steps = max(0, int(approach_steps))
    preapproach_z_offset = float(preapproach_z_offset)
    if approach_steps <= 0:
        return np.zeros((0, 3), dtype=np.float64), []

    has_preapproach = abs(preapproach_z_offset) > 1.0e-9

    if not approach_via_above_secure:
        if not has_preapproach:
            return (
                _linspace_xyz(preclose_xyz, secure_xyz, approach_steps, endpoint=False),
                [],
            )
        preapproach_xyz = (
            float(preclose_xyz[0]),
            float(preclose_xyz[1]),
            float(preclose_xyz[2] + preapproach_z_offset),
        )
        descent_steps = max(1, approach_steps // 2)
        secure_steps = max(0, approach_steps - descent_steps)
        descent = _linspace_xyz(preapproach_xyz, preclose_xyz, descent_steps, endpoint=False)
        to_secure = _linspace_xyz(preclose_xyz, secure_xyz, secure_steps, endpoint=False)
        return (
            np.vstack([part for part in (descent, to_secure) if len(part)]),
            [{"name": "preapproach", "xyz": preapproach_xyz, "gripper": "open"}],
        )

    above_secure_xyz = (
        float(secure_xyz[0]),
        float(secure_xyz[1]),
        float(preclose_xyz[2]),
    )
    if not has_preapproach:
        lateral_steps = max(1, approach_steps // 2)
        vertical_steps = max(0, approach_steps - lateral_steps)
        lateral = _linspace_xyz(preclose_xyz, above_secure_xyz, lateral_steps, endpoint=False)
        vertical = _linspace_xyz(above_secure_xyz, secure_xyz, vertical_steps, endpoint=False)
        return (
            np.vstack([part for part in (lateral, vertical) if len(part)]),
            [{"name": "above_secure", "xyz": above_secure_xyz, "gripper": "open"}],
        )

    preapproach_xyz = (
        float(preclose_xyz[0]),
        float(preclose_xyz[1]),
        float(preclose_xyz[2] + preapproach_z_offset),
    )
    descent_steps = max(1, approach_steps // 3)
    lateral_steps = max(1, approach_steps // 3)
    vertical_steps = max(0, approach_steps - descent_steps - lateral_steps)
    descent = _linspace_xyz(preapproach_xyz, preclose_xyz, descent_steps, endpoint=False)
    lateral = _linspace_xyz(preclose_xyz, above_secure_xyz, lateral_steps, endpoint=False)
    vertical = _linspace_xyz(above_secure_xyz, secure_xyz, vertical_steps, endpoint=False)
    return (
        np.vstack([part for part in (descent, lateral, vertical) if len(part)]),
        [
            {"name": "preapproach", "xyz": preapproach_xyz, "gripper": "open"},
            {"name": "above_secure", "xyz": above_secure_xyz, "gripper": "open"},
        ],
    )


def plan_relative_replay_from_demo_priors(
    priors: DemoGraspPriors,
    object_name: str,
    *,
    object_initial_pos: tuple[float, float, float] | list[float],
    secure_xy_offset: tuple[float, float] = (0.0, 0.0),
    secure_z: float | None = None,
    lift_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    release_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
) -> DemoRelativeReplayPlan:
    """Anchor a demo-prior grasp plan to the current object position.

    The source human-demo plan contributes relative geometry:
    ``preclose-secure``, ``lift-secure``, and ``release-secure``. The current
    sim scene contributes the target object's XY. Z defaults to the demo-prior
    secure height because the current table/plate frame is still the FR3 base
    frame; callers can override it when calibrating a different prop height.
    """
    source = plan_from_demo_priors(priors, object_name)
    object_pos = _vec3(object_initial_pos, key="object_initial_pos")
    lift_offset = _vec3(lift_xyz_offset, key="lift_xyz_offset")
    release_offset = _vec3(release_xyz_offset, key="release_xyz_offset")
    secure_xyz = (
        object_pos[0] + float(secure_xy_offset[0]),
        object_pos[1] + float(secure_xy_offset[1]),
        float(source.secure_grasp_xyz[2] if secure_z is None else secure_z),
    )

    preclose_delta = _sub_vec3(source.preclose_xyz, source.secure_grasp_xyz)
    lift_delta = _sub_vec3(source.lift_xyz, source.secure_grasp_xyz)
    release_delta = _sub_vec3(source.release_xyz, source.secure_grasp_xyz)
    # Sim-only env-var hook (default no-op): replace median release_delta with
    # per-demo release_delta when both per-demo secure and release xyz are supplied.
    # Used by physics-fidelity sweeps to test whether per-demo priors break the
    # 14-15/60 generalization ceiling observed under median priors.
    import os as _os, json as _json
    _per_demo_secure = _os.environ.get("_FR3_PER_DEMO_SECURE_XYZ")
    _per_demo_release = _os.environ.get("_FR3_PER_DEMO_RELEASE_XYZ")
    if _per_demo_secure and _per_demo_release:
        try:
            _sec = _json.loads(_per_demo_secure)
            _rel = _json.loads(_per_demo_release)
            if isinstance(_sec, list) and len(_sec) == 3 and isinstance(_rel, list) and len(_rel) == 3:
                release_delta = (
                    float(_rel[0]) - float(_sec[0]),
                    float(_rel[1]) - float(_sec[1]),
                    float(_rel[2]) - float(_sec[2]),
                )
        except (ValueError, TypeError):
            pass
    # Per-demo lift_delta hook (default no-op): replace median lift_delta with
    # per-demo lift_xyz - secure_xyz when both env vars set.
    _per_demo_lift = _os.environ.get("_FR3_PER_DEMO_LIFT_XYZ")
    if _per_demo_secure and _per_demo_lift:
        try:
            _sec_l = _json.loads(_per_demo_secure)
            _lift = _json.loads(_per_demo_lift)
            if isinstance(_sec_l, list) and len(_sec_l) == 3 and isinstance(_lift, list) and len(_lift) == 3:
                lift_delta = (
                    float(_lift[0]) - float(_sec_l[0]),
                    float(_lift[1]) - float(_sec_l[1]),
                    float(_lift[2]) - float(_sec_l[2]),
                )
        except (ValueError, TypeError):
            pass
    # Per-demo preclose_delta hook (default no-op): replace median preclose_delta
    # with per-demo preclose_xyz - secure_xyz when both env vars set.
    _per_demo_preclose = _os.environ.get("_FR3_PER_DEMO_PRECLOSE_XYZ")
    if _per_demo_secure and _per_demo_preclose:
        try:
            _sec_p = _json.loads(_per_demo_secure)
            _pre = _json.loads(_per_demo_preclose)
            if isinstance(_sec_p, list) and len(_sec_p) == 3 and isinstance(_pre, list) and len(_pre) == 3:
                preclose_delta = (
                    float(_pre[0]) - float(_sec_p[0]),
                    float(_pre[1]) - float(_sec_p[1]),
                    float(_pre[2]) - float(_sec_p[2]),
                )
        except (ValueError, TypeError):
            pass
    # Lift-transport blending hook (default no-op): blend a fraction of the
    # release XY delta into the lift target. This creates a diagonal lift
    # trajectory that more closely matches human demo motion patterns, which
    # may improve object contact retention during the lift phase.
    _lift_xy_blend = _os.environ.get("_FR3_LIFT_XY_FROM_RELEASE_FRACTION")
    if _lift_xy_blend:
        try:
            _blend_frac = float(_lift_xy_blend)
            if 0.0 < _blend_frac <= 1.0:
                lift_delta = (
                    lift_delta[0] + _blend_frac * release_delta[0],
                    lift_delta[1] + _blend_frac * release_delta[1],
                    lift_delta[2],
                )
        except (ValueError, TypeError):
            pass
    # Minimum lift Z delta hook (default no-op): ensure the Z component of the
    # lift delta is at least the given value. Useful when per-demo lift data
    # yields very small Z deltas that cause the robot to barely move up.
    _lift_min_z = _os.environ.get("_FR3_LIFT_MIN_Z_DELTA")
    if _lift_min_z:
        try:
            _min_z = float(_lift_min_z)
            if lift_delta[2] < _min_z:
                lift_delta = (lift_delta[0], lift_delta[1], _min_z)
        except (ValueError, TypeError):
            pass

    return DemoRelativeReplayPlan(
        object=source.object,
        object_initial_pos=object_pos,
        source_secure_grasp_xyz=source.secure_grasp_xyz,
        preclose_xyz=_add_vec3(secure_xyz, preclose_delta),
        secure_grasp_xyz=secure_xyz,
        lift_xyz=_add_vec3(_add_vec3(secure_xyz, lift_delta), lift_offset),
        lift_xyz_offset=lift_offset,
        release_xyz=_add_vec3(_add_vec3(secure_xyz, release_delta), release_offset),
        release_xyz_offset=release_offset,
        secure_orientation_3=source.secure_orientation_3,
        close_hold_steps=source.close_hold_steps,
        close_hold_s=source.close_hold_s,
        secure_gripper_width_m=source.secure_gripper_width_m,
    )


def _linspace_xyz(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    steps: int,
    *,
    endpoint: bool = True,
) -> np.ndarray:
    if steps <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.linspace(
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
        num=steps,
        endpoint=endpoint,
    )


def _closed_relative_path(
    plan: DemoRelativeReplayPlan,
    steps: int,
    *,
    lift_phase_fraction: float = 0.5,
) -> np.ndarray:
    if steps <= 1:
        return _linspace_xyz(plan.secure_grasp_xyz, plan.release_xyz, max(steps, 1))
    fraction = min(1.0, max(0.0, float(lift_phase_fraction)))
    first_steps = max(1, min(int(steps), int(np.floor(steps * fraction))))
    second_steps = max(0, steps - first_steps)
    first = _linspace_xyz(plan.secure_grasp_xyz, plan.lift_xyz, first_steps)
    second = (
        _linspace_xyz(plan.lift_xyz, plan.release_xyz, second_steps + 1)[1:]
        if second_steps else np.zeros((0, 3), dtype=np.float64)
    )
    return np.vstack([first, second])


def _phase_linspace_indices(
    start: int,
    end: int,
    steps: int,
    *,
    endpoint: bool = True,
) -> np.ndarray:
    if steps <= 0:
        return np.zeros((0,), dtype=np.int64)
    lo = int(min(start, end))
    hi = int(max(start, end))
    values = np.linspace(float(start), float(end), num=steps, endpoint=endpoint)
    return np.clip(np.rint(values).astype(np.int64), lo, hi)


def demo_orientation_indices_for_probe(
    action_chunk: np.ndarray,
    *,
    raw_start_rel_idx: int,
    raw_secure_rel_idx: int,
    raw_release_rel_idx: int,
    raw_end_rel_idx: int,
) -> np.ndarray:
    """Map a demo-relative probe back to raw-demo orientation sample indices.

    Position targets are re-anchored to the current sim object, but the gripper
    wrist should still follow the human demo's approach/carry/release attitude.
    This helper assigns each open/closed/open probe step to a raw demo joint
    sample so the Isaac runner can derive per-step FK quaternions.
    """
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 8:
        raise ValueError(f"expected action_chunk shape (T, >=8), got {actions.shape}")
    raw_end_rel_idx = max(0, int(raw_end_rel_idx))
    raw_start_rel_idx = int(np.clip(raw_start_rel_idx, 0, raw_end_rel_idx))
    raw_secure_rel_idx = int(np.clip(raw_secure_rel_idx, 0, raw_end_rel_idx))
    raw_release_rel_idx = int(np.clip(raw_release_rel_idx, raw_secure_rel_idx, raw_end_rel_idx))
    close_mask = actions[:, 7] > 0.20
    closed_indices = np.flatnonzero(close_mask)
    if len(closed_indices) == 0:
        return _phase_linspace_indices(
            raw_start_rel_idx,
            raw_end_rel_idx,
            actions.shape[0],
        )

    close_start = int(closed_indices[0])
    close_end_exclusive = int(closed_indices[-1]) + 1
    approach_steps = close_start
    closed_steps = close_end_exclusive - close_start
    release_steps = actions.shape[0] - close_end_exclusive

    parts = [
        _phase_linspace_indices(
            raw_start_rel_idx,
            raw_secure_rel_idx,
            approach_steps,
            endpoint=False,
        ),
        _phase_linspace_indices(
            raw_secure_rel_idx,
            raw_release_rel_idx,
            closed_steps,
        ),
        _phase_linspace_indices(
            raw_release_rel_idx,
            raw_end_rel_idx,
            release_steps,
        ),
    ]
    return np.concatenate([part for part in parts if len(part)])


def cartesian_probe_from_relative_plan(
    plan: DemoRelativeReplayPlan,
    *,
    approach_steps: int = 20,
    preapproach_z_offset: float = 0.0,
    approach_via_above_secure: bool = False,
    lift_phase_fraction: float = 0.5,
    close_hold_steps: int | None = None,
    release_hold_steps: int = 0,
    release_retreat_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    release_steps: int = 20,
) -> DemoCartesianProbe:
    """Create a no-motion checker probe from a demo-relative plan."""
    hold_steps = int(close_hold_steps if close_hold_steps is not None else plan.close_hold_steps)
    hold_steps = max(1, hold_steps)
    approach_steps = max(0, int(approach_steps))
    release_hold_steps = max(0, int(release_hold_steps))
    release_retreat_offset = _vec3(
        release_retreat_xyz_offset,
        key="release_retreat_xyz_offset",
    )
    release_steps = max(0, int(release_steps))
    release_retreat_xyz = _add_vec3(plan.release_xyz, release_retreat_offset)

    approach, preapproach_waypoints = _open_approach_path(
        preclose_xyz=plan.preclose_xyz,
        secure_xyz=plan.secure_grasp_xyz,
        approach_steps=approach_steps,
        preapproach_z_offset=preapproach_z_offset,
        approach_via_above_secure=approach_via_above_secure,
    )
    closed = _closed_relative_path(
        plan,
        hold_steps,
        lift_phase_fraction=lift_phase_fraction,
    )
    release_hold = _linspace_xyz(plan.release_xyz, plan.release_xyz, release_hold_steps)
    release = _linspace_xyz(plan.release_xyz, release_retreat_xyz, release_steps)
    ee_pos_seq = np.vstack([part for part in (approach, closed, release_hold, release) if len(part)])

    action_chunk = np.zeros((ee_pos_seq.shape[0], 8), dtype=np.float64)
    close_start = approach.shape[0]
    close_end = close_start + closed.shape[0] + release_hold.shape[0]
    action_chunk[close_start:close_end, 7] = 1.0

    from droid.sim.safety.feasibility_checker import check_reverse_grasp_trajectory

    report = check_reverse_grasp_trajectory(
        [action_chunk],
        ee_pos_seqs=[ee_pos_seq],
        min_total_close_steps=min(hold_steps + release_hold_steps, 170),
    )
    return DemoCartesianProbe(
        object=plan.object,
        waypoints=(
            [*preapproach_waypoints, *plan.as_waypoints()]
            if not any(release_retreat_offset)
            else [
                *preapproach_waypoints,
                *plan.as_waypoints(),
                {
                    "name": "release_retreat",
                    "xyz": release_retreat_xyz,
                    "gripper": "open",
                },
            ]
        ),
        action_chunk=action_chunk,
        ee_pos_seq=ee_pos_seq,
        violations=tuple(str(v) for v in report.violations),
    )


def cartesian_probe_from_raw_relative_sequence(
    *,
    object_name: str,
    raw_action_chunk,
    raw_ee_pos_seq,
    raw_secure_xyz: tuple[float, float, float] | list[float],
    target_secure_xyz: tuple[float, float, float] | list[float],
    raw_secure_rel_idx: int,
    raw_release_rel_idx: int,
    release_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
) -> DemoCartesianProbe:
    """Anchor a dense raw-demo cartesian path to the current sim grasp pose."""
    action_chunk = np.asarray(raw_action_chunk, dtype=np.float64).copy()
    if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
        raise ValueError(f"expected raw_action_chunk shape (T, 8), got {action_chunk.shape}")
    raw_xyz = np.asarray(raw_ee_pos_seq, dtype=np.float64)
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3:
        raise ValueError(f"expected raw_ee_pos_seq shape (T, 3), got {raw_xyz.shape}")
    if raw_xyz.shape[0] != action_chunk.shape[0]:
        raise ValueError(
            "raw_action_chunk and raw_ee_pos_seq must have the same number of frames"
        )

    raw_secure = np.asarray(_vec3(raw_secure_xyz, key="raw_secure_xyz"), dtype=np.float64)
    target_secure = np.asarray(_vec3(target_secure_xyz, key="target_secure_xyz"), dtype=np.float64)
    release_offset = np.asarray(_vec3(release_xyz_offset, key="release_xyz_offset"), dtype=np.float64)
    anchored = raw_xyz + (target_secure - raw_secure)

    secure_idx = int(np.clip(raw_secure_rel_idx, 0, raw_xyz.shape[0] - 1))
    release_idx = int(np.clip(raw_release_rel_idx, secure_idx, raw_xyz.shape[0] - 1))
    if np.any(release_offset):
        weights = np.zeros((raw_xyz.shape[0], 1), dtype=np.float64)
        if release_idx > secure_idx:
            weights[secure_idx:release_idx + 1, 0] = np.linspace(
                0.0,
                1.0,
                num=release_idx - secure_idx + 1,
            )
        weights[release_idx + 1:, 0] = 1.0
        anchored = anchored + weights * release_offset

    from droid.sim.safety.feasibility_checker import check_reverse_grasp_trajectory

    report = check_reverse_grasp_trajectory(
        [action_chunk],
        ee_pos_seqs=[anchored],
    )
    return DemoCartesianProbe(
        object=object_name,
        waypoints=[
            {
                "name": "raw_relative_start",
                "xyz": tuple(float(x) for x in anchored[0]),
                "gripper": "open",
            },
            {
                "name": "raw_relative_secure",
                "xyz": tuple(float(x) for x in anchored[secure_idx]),
                "gripper": "closed",
            },
            {
                "name": "raw_relative_release",
                "xyz": tuple(float(x) for x in anchored[release_idx]),
                "gripper": "open",
            },
            {
                "name": "raw_relative_end",
                "xyz": tuple(float(x) for x in anchored[-1]),
                "gripper": "open",
            },
        ],
        action_chunk=action_chunk,
        ee_pos_seq=anchored,
        violations=tuple(str(v) for v in report.violations),
    )


def _first_closed_index(action_chunk: np.ndarray, *, threshold: float = 0.20) -> int:
    closed = np.flatnonzero(action_chunk[:, 7] > threshold)
    if len(closed) == 0:
        raise ValueError("raw_action_chunk has no closed-gripper phase")
    return int(closed[0])


def cartesian_probe_from_raw_phase_profile(
    *,
    object_name: str,
    raw_action_chunk,
    raw_ee_pos_seq,
    target_preclose_xyz: tuple[float, float, float] | list[float],
    target_secure_xyz: tuple[float, float, float] | list[float],
    raw_secure_rel_idx: int,
    raw_release_rel_idx: int,
    release_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    approach_steps: int = 20,
    preapproach_z_offset: float = 0.0,
    approach_via_above_secure: bool = False,
    close_hold_steps: int | None = None,
    pre_secure_close_lead_steps: int = 0,
    pre_lift_close_dwell_steps: int = 0,
    release_hold_steps: int = 0,
    release_retreat_xyz_offset: tuple[float, float, float] | list[float] = (0.0, 0.0, 0.0),
    release_steps: int = 20,
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    raw_phase_lift_xy_scale: float | None = None,
) -> DemoCartesianProbe:
    """Anchor the raw demo's closed-phase shape instead of the dense path.

    Human demos usually lift while already translating away from the plate; the
    max-lift point is not always vertically above the secure grasp point. This
    helper keeps that secure->max-lift->release phase profile while still
    anchoring the secure grasp to the current sim object.
    """
    raw_actions = np.asarray(raw_action_chunk, dtype=np.float64).copy()
    if raw_actions.ndim != 2 or raw_actions.shape[1] != 8:
        raise ValueError(f"expected raw_action_chunk shape (T, 8), got {raw_actions.shape}")
    raw_xyz = np.asarray(raw_ee_pos_seq, dtype=np.float64)
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3:
        raise ValueError(f"expected raw_ee_pos_seq shape (T, 3), got {raw_xyz.shape}")
    if raw_xyz.shape[0] != raw_actions.shape[0]:
        raise ValueError(
            "raw_action_chunk and raw_ee_pos_seq must have the same number of frames"
        )

    secure_idx = int(np.clip(raw_secure_rel_idx, 0, raw_xyz.shape[0] - 1))
    close_idx = min(_first_closed_index(raw_actions), secure_idx)
    release_idx = int(np.clip(raw_release_rel_idx, secure_idx, raw_xyz.shape[0] - 1))
    raw_secure = raw_xyz[secure_idx]
    closed_raw = raw_xyz[close_idx:release_idx + 1]
    closed_delta = closed_raw - raw_secure
    max_lift_local_idx = int(np.argmax(closed_delta[:, 2]))
    max_lift_delta = closed_delta[max_lift_local_idx].copy()
    release_delta = (raw_xyz[release_idx] - raw_secure).copy()
    release_scale = np.asarray([float(xy_scale), float(xy_scale), float(z_scale)])
    lift_xy_scale = float(xy_scale) if raw_phase_lift_xy_scale is None else float(raw_phase_lift_xy_scale)
    max_lift_delta[:2] *= lift_xy_scale
    max_lift_delta[2] *= float(z_scale)
    release_delta *= release_scale
    release_delta += np.asarray(_vec3(release_xyz_offset, key="release_xyz_offset"))

    target_preclose = _vec3(target_preclose_xyz, key="target_preclose_xyz")
    target_secure = _vec3(target_secure_xyz, key="target_secure_xyz")
    target_max_lift = tuple(float(x) for x in np.asarray(target_secure) + max_lift_delta)
    target_release = tuple(float(x) for x in np.asarray(target_secure) + release_delta)

    hold_steps = int(close_hold_steps if close_hold_steps is not None else release_idx - close_idx + 1)
    hold_steps = max(1, hold_steps)
    pre_secure_close_lead_steps = max(0, int(pre_secure_close_lead_steps))
    pre_lift_close_dwell_steps = max(0, int(pre_lift_close_dwell_steps))
    approach_steps = max(0, int(approach_steps))
    release_hold_steps = max(0, int(release_hold_steps))
    release_retreat_offset = np.asarray(
        _vec3(release_retreat_xyz_offset, key="release_retreat_xyz_offset"),
        dtype=np.float64,
    )
    release_steps = max(0, int(release_steps))
    max_lift_fraction = (
        max_lift_local_idx / max(1, closed_delta.shape[0] - 1)
        if closed_delta.shape[0] > 1 else 0.5
    )
    first_steps = max(1, min(hold_steps, int(round(hold_steps * max_lift_fraction)) + 1))
    second_steps = max(0, hold_steps - first_steps)

    approach, preapproach_waypoints = _open_approach_path(
        preclose_xyz=target_preclose,
        secure_xyz=target_secure,
        approach_steps=approach_steps,
        preapproach_z_offset=preapproach_z_offset,
        approach_via_above_secure=approach_via_above_secure,
    )
    pre_lift_close_dwell = _linspace_xyz(
        target_secure,
        target_secure,
        pre_lift_close_dwell_steps,
    )
    first = _linspace_xyz(target_secure, target_max_lift, first_steps)
    second = (
        _linspace_xyz(target_max_lift, target_release, second_steps + 1)[1:]
        if second_steps else np.zeros((0, 3), dtype=np.float64)
    )
    release_hold = _linspace_xyz(target_release, target_release, release_hold_steps)
    target_release_retreat = tuple(float(x) for x in np.asarray(target_release) + release_retreat_offset)
    release = _linspace_xyz(target_release, target_release_retreat, release_steps)
    ee_pos_seq = np.vstack([
        part
        for part in (
            approach,
            pre_lift_close_dwell,
            first,
            second,
            release_hold,
            release,
        )
        if len(part)
    ])

    action_chunk = np.zeros((ee_pos_seq.shape[0], 8), dtype=np.float64)
    close_start = max(0, approach.shape[0] - min(pre_secure_close_lead_steps, approach.shape[0]))
    close_end = (
        approach.shape[0]
        + pre_lift_close_dwell.shape[0]
        + first.shape[0]
        + second.shape[0]
        + release_hold.shape[0]
    )
    action_chunk[close_start:close_end, 7] = 1.0

    from droid.sim.safety.feasibility_checker import check_reverse_grasp_trajectory

    report = check_reverse_grasp_trajectory(
        [action_chunk],
        ee_pos_seqs=[ee_pos_seq],
        min_total_close_steps=min(
            min(pre_secure_close_lead_steps, approach.shape[0])
            + pre_lift_close_dwell_steps
            + hold_steps
            + release_hold_steps,
            170,
        ),
    )
    return DemoCartesianProbe(
        object=object_name,
        waypoints=[
            *[
                {
                    "name": waypoint["name"],
                    "xyz": tuple(float(x) for x in waypoint["xyz"]),
                    "gripper": waypoint["gripper"],
                }
                for waypoint in preapproach_waypoints
            ],
            {
                "name": "raw_phase_preclose",
                "xyz": tuple(float(x) for x in target_preclose),
                "gripper": "open",
            },
            *(
                [
                    {
                        "name": "raw_phase_pre_secure_close_lead",
                        "xyz": tuple(float(x) for x in target_preclose),
                        "gripper": "closed",
                        "steps": int(min(pre_secure_close_lead_steps, approach.shape[0])),
                    }
                ]
                if pre_secure_close_lead_steps > 0 else []
            ),
            {
                "name": "raw_phase_secure",
                "xyz": tuple(float(x) for x in target_secure),
                "gripper": "closed",
            },
            *(
                [
                    {
                        "name": "raw_phase_pre_lift_close_dwell",
                        "xyz": tuple(float(x) for x in target_secure),
                        "gripper": "closed",
                        "steps": int(pre_lift_close_dwell_steps),
                    }
                ]
                if pre_lift_close_dwell_steps > 0 else []
            ),
            {
                "name": "raw_phase_max_lift",
                "xyz": target_max_lift,
                "gripper": "closed",
            },
            {
                "name": "raw_phase_release",
                "xyz": target_release,
                "gripper": "open",
            },
            *(
                [
                    {
                        "name": "raw_phase_release_retreat",
                        "xyz": target_release_retreat,
                        "gripper": "open",
                    }
                ]
                if np.any(release_retreat_offset) else []
            ),
        ],
        action_chunk=action_chunk,
        ee_pos_seq=ee_pos_seq,
        violations=tuple(str(v) for v in report.violations),
    )


__all__ = [
    "DemoRelativeReplayPlan",
    "cartesian_probe_from_relative_plan",
    "cartesian_probe_from_raw_phase_profile",
    "cartesian_probe_from_raw_relative_sequence",
    "demo_orientation_indices_for_probe",
    "plan_relative_replay_from_demo_priors",
]
