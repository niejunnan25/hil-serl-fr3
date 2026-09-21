"""Feasibility checker for action chunks against real-robot hard limits.

Per spec v2 §4.3 Phase 1 + §2.5 (limits抄自 franka_hardware_left.yaml).
The point is a static check on (q_seq, gripper_seq) BEFORE sim execution:
if checker says fail, real robot would also fail — that's the value of
the no-motion safety filter.

Used by:
  - bug_injection_suite (Phase 1 task 1.6): inject known faults, assert
    checker catches all 14
  - phase1_replay_lerobot.py: gate per-episode replay
  - shadow_executor (Phase 2): gate policy action chunks before sim step
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Sequence

import numpy as np


class ViolationCategory(str, enum.Enum):
    JOINT_LIMIT = "joint_limit"
    CART_LIMIT = "cart_limit"
    JOINT_VEL = "joint_vel"
    STEP_DELTA = "step_delta"
    GRIPPER_RANGE = "gripper_range"
    GRIPPER_PHASE = "gripper_phase"
    CART_Z = "cart_z"
    CHUNK_SHAPE = "chunk_shape"


@dataclasses.dataclass(frozen=True)
class FeasibilityViolation:
    category: ViolationCategory
    detail: str
    timestep: int = -1   # -1 = chunk-level, not specific step

    def __str__(self) -> str:
        loc = f"t={self.timestep}" if self.timestep >= 0 else "chunk"
        return f"[{self.category.value} {loc}] {self.detail}"


@dataclasses.dataclass
class FeasibilityReport:
    violations: list[FeasibilityViolation]

    @property
    def pass_(self) -> bool:
        return not self.violations

    def categories(self) -> set[ViolationCategory]:
        return {v.category for v in self.violations}

    def __str__(self) -> str:
        if self.pass_:
            return "FeasibilityReport: PASS"
        return "FeasibilityReport: FAIL\n  " + "\n  ".join(str(v) for v in self.violations)


JOINT_POS_UPPER = np.array([2.80, 1.66, 2.80, -0.17, 2.80, 3.65, 2.80], dtype=np.float64)
JOINT_POS_LOWER = np.array([-2.80, -1.66, -2.80, -2.97, -2.80, 0.5445, -2.80], dtype=np.float64)
JOINT_VEL_LIMIT = np.array([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51], dtype=np.float64)
CART_POS_UPPER = np.array([1.0, 0.4, 1.0], dtype=np.float64)
CART_POS_LOWER = np.array([0.1, -0.4, -0.05], dtype=np.float64)
DEFAULT_CONTROL_HZ = 15.0

# Step delta threshold: a single 15Hz control step at max joint vel ≈ 2.5 rad/s
# would move ~0.17 rad. Allow generous 0.5 rad/step to avoid false positives.
DEFAULT_MAX_STEP_DELTA = 0.5  # rad per control step

# Z-floor for safety (spec §3.4 / handoff). Phase 0 plate sits at z≈0.06.
# The earlier grasp-specific 0.12 m floor was too strict for the user's
# successful 8014 human reverse demos: closed-grasp z_min spans down to about
# 0.092 m, with p10 around 0.122 m. Keep the sim/no-motion grasp floor just
# below the observed successful range instead of rejecting real demonstrations.
MIN_CART_Z_DEFAULT = -0.05  # soft: matches CART_POS_LOWER[2]
MIN_CART_Z_GRASP = 0.09     # used when checking grasp trajectories specifically

# Gripper command convention (handoff verified): 0=open 1=close, threshold>0.20
GRIPPER_RANGE = (0.0, 1.0)
# Tolerance absorbs float-noise from policy outputs (~1e-3 negative is
# typical and harmless: real robot uses >0.20 close threshold). Catches
# real bugs like action_x10 (gripper=10) or gripper_inverted out of range.
GRIPPER_RANGE_TOLERANCE = 0.1
GRIPPER_CLOSE_THRESHOLD = 0.20


def check_joint_limits(q: np.ndarray) -> list[FeasibilityViolation]:
    """q: shape (T, 7) or (7,). Returns list of violations (empty = OK)."""
    q_arr = np.atleast_2d(np.asarray(q, dtype=np.float64))
    if q_arr.shape[-1] != 7:
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected last dim 7 (joints), got shape {q_arr.shape}",
        )]
    out: list[FeasibilityViolation] = []
    for t in range(q_arr.shape[0]):
        for j in range(7):
            v = q_arr[t, j]
            if v > JOINT_POS_UPPER[j] or v < JOINT_POS_LOWER[j]:
                out.append(FeasibilityViolation(
                    ViolationCategory.JOINT_LIMIT,
                    f"joint{j+1}={v:.4f} out of [{JOINT_POS_LOWER[j]:.4f}, {JOINT_POS_UPPER[j]:.4f}]",
                    timestep=t,
                ))
    return out


def check_step_delta(
    q_seq: np.ndarray,
    max_delta: float = DEFAULT_MAX_STEP_DELTA,
) -> list[FeasibilityViolation]:
    """q_seq: shape (T, 7). Checks |q[t+1] - q[t]| < max_delta per joint."""
    q_arr = np.asarray(q_seq, dtype=np.float64)
    if q_arr.ndim != 2 or q_arr.shape[-1] != 7:
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected (T, 7), got {q_arr.shape}",
        )]
    out: list[FeasibilityViolation] = []
    for t in range(1, q_arr.shape[0]):
        delta = q_arr[t] - q_arr[t - 1]
        for j in range(7):
            if abs(delta[j]) > max_delta:
                out.append(FeasibilityViolation(
                    ViolationCategory.STEP_DELTA,
                    f"joint{j+1} delta={delta[j]:.4f} > {max_delta}",
                    timestep=t,
                ))
    return out


def check_joint_velocity(
    q_seq: np.ndarray,
    *,
    control_hz: float = DEFAULT_CONTROL_HZ,
    vel_limit: np.ndarray = JOINT_VEL_LIMIT,
) -> list[FeasibilityViolation]:
    """q_seq: shape (T, 7). Checks joint velocity against FR3 per-joint limits."""
    q_arr = np.asarray(q_seq, dtype=np.float64)
    if q_arr.ndim != 2 or q_arr.shape[-1] != 7:
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected (T, 7), got {q_arr.shape}",
        )]
    if control_hz <= 0:
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"control_hz must be positive, got {control_hz}",
        )]
    limits = np.asarray(vel_limit, dtype=np.float64).reshape(-1)
    if limits.shape != (7,):
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected vel_limit shape (7,), got {limits.shape}",
        )]

    out: list[FeasibilityViolation] = []
    for t in range(1, q_arr.shape[0]):
        velocity = (q_arr[t] - q_arr[t - 1]) * control_hz
        for j in range(7):
            if abs(velocity[j]) > limits[j]:
                out.append(FeasibilityViolation(
                    ViolationCategory.JOINT_VEL,
                    f"joint{j+1} velocity={velocity[j]:.4f} > {limits[j]:.4f} rad/s",
                    timestep=t,
                ))
    return out


def check_cart_limits(
    ee_pos: np.ndarray,
    z_floor: float = MIN_CART_Z_DEFAULT,
) -> list[FeasibilityViolation]:
    """ee_pos: shape (T, 3) or (3,). Workspace box check + z floor."""
    arr = np.atleast_2d(np.asarray(ee_pos, dtype=np.float64))
    if arr.shape[-1] != 3:
        return [FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected last dim 3 (xyz), got shape {arr.shape}",
        )]
    out: list[FeasibilityViolation] = []
    for t in range(arr.shape[0]):
        for axis, name in enumerate("xyz"):
            v = arr[t, axis]
            if v > CART_POS_UPPER[axis] or v < CART_POS_LOWER[axis]:
                out.append(FeasibilityViolation(
                    ViolationCategory.CART_LIMIT,
                    f"{name}={v:.4f} out of [{CART_POS_LOWER[axis]:.4f}, {CART_POS_UPPER[axis]:.4f}]",
                    timestep=t,
                ))
        if arr[t, 2] < z_floor:
            out.append(FeasibilityViolation(
                ViolationCategory.CART_Z,
                f"z={arr[t, 2]:.4f} below floor {z_floor}",
                timestep=t,
            ))
    return out


def check_gripper_command(gripper_seq: np.ndarray) -> list[FeasibilityViolation]:
    """gripper_seq: shape (T,) or (T, 1). Range check."""
    arr = np.asarray(gripper_seq, dtype=np.float64).reshape(-1)
    out: list[FeasibilityViolation] = []
    for t, v in enumerate(arr):
        if v < GRIPPER_RANGE[0] - GRIPPER_RANGE_TOLERANCE or v > GRIPPER_RANGE[1] + GRIPPER_RANGE_TOLERANCE:
            out.append(FeasibilityViolation(
                ViolationCategory.GRIPPER_RANGE,
                f"gripper={v:.4f} out of [{GRIPPER_RANGE[0]}, {GRIPPER_RANGE[1]}]",
                timestep=t,
            ))
    return out


def check_gripper_phase(
    gripper_seq: np.ndarray,
    require_close_then_open: bool = False,
    min_close_steps: int = 0,
) -> list[FeasibilityViolation]:
    """Check gripper command phase plausibility.

    require_close_then_open: True for grasp-then-place tasks; closes must
        appear before opens. min_close_steps: how long close must be held.
    """
    arr = np.asarray(gripper_seq, dtype=np.float64).reshape(-1)
    is_closed = arr > GRIPPER_CLOSE_THRESHOLD
    out: list[FeasibilityViolation] = []
    if require_close_then_open:
        # find first close + first open
        close_idx = np.where(is_closed)[0]
        open_idx = np.where(~is_closed)[0]
        if len(close_idx) == 0:
            out.append(FeasibilityViolation(
                ViolationCategory.GRIPPER_PHASE,
                "task requires close-then-open but no close phase observed",
            ))
        elif min_close_steps > 0:
            # first close run length
            close_run = 0
            in_close = False
            max_run = 0
            for v in is_closed:
                if v:
                    close_run += 1
                    in_close = True
                    max_run = max(max_run, close_run)
                else:
                    close_run = 0
            if max_run < min_close_steps:
                out.append(FeasibilityViolation(
                    ViolationCategory.GRIPPER_PHASE,
                    f"longest close run={max_run} < min_close_steps={min_close_steps}",
                ))
    return out


def check_action_chunk(
    action_chunk: np.ndarray,
    *,
    ee_pos_seq: np.ndarray | None = None,
    z_floor: float = MIN_CART_Z_DEFAULT,
    max_step_delta: float = DEFAULT_MAX_STEP_DELTA,
    control_hz: float = DEFAULT_CONTROL_HZ,
    require_close_then_open: bool = False,
    min_close_steps: int = 0,
) -> FeasibilityReport:
    """Top-level check: action_chunk shape (T, 8) = 7 joint + 1 gripper.

    Optionally ee_pos_seq (T, 3) to check cart limits (computed by sim
    after FK; not in action chunk itself).
    """
    arr = np.asarray(action_chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[-1] != 8:
        return FeasibilityReport(violations=[FeasibilityViolation(
            ViolationCategory.CHUNK_SHAPE,
            f"expected (T, 8), got {arr.shape}",
        )])

    q_seq = arr[:, :7]
    gripper_seq = arr[:, 7]
    violations: list[FeasibilityViolation] = []
    violations += check_joint_limits(q_seq)
    violations += check_joint_velocity(q_seq, control_hz=control_hz)
    violations += check_step_delta(q_seq, max_delta=max_step_delta)
    violations += check_gripper_command(gripper_seq)
    violations += check_gripper_phase(
        gripper_seq,
        require_close_then_open=require_close_then_open,
        min_close_steps=min_close_steps,
    )
    if ee_pos_seq is not None:
        violations += check_cart_limits(ee_pos_seq, z_floor=z_floor)
    return FeasibilityReport(violations=violations)


def check_reverse_grasp_trajectory(
    action_chunks: list[np.ndarray],
    *,
    min_total_close_steps: int = 170,
    z_floor: float = MIN_CART_Z_GRASP,
    ee_pos_seqs: list[np.ndarray] | None = None,
    plate_xy: tuple[float, float] | None = None,
    min_distance_from_plate: float = 0.05,
) -> FeasibilityReport:
    """Phase 2-B reverse-specific aggregate check across multiple chunks of one episode.

    Per spec §4.4 task 2.7, a reverse (plate→table) trajectory must:
      - have total gripper-close ≥ min_total_close_steps across the episode
        (default 170, calibrated from successful 8014 human reverse demos;
        callers can still override to the older stricter 200-step spec gate)
      - never dip below z_floor (default 0.09 m, calibrated from successful
        8014 human reverse demos)
      - move the grasped object away during the closed-carry phase by at least
        min_distance_from_plate. Direction is measured from the first closed
        ee xy to the last closed ee xy, not from episode start/end, because real
        DROID demos start and finish at home-like poses.

    action_chunks: list of (T, 8) chunks (one per saved-obs sample, in order).
    ee_pos_seqs: optional list of (T, 3) ee positions for the same chunks
        (computed by sim FK when sim is launched; can be None for action-only check).
    """
    violations: list[FeasibilityViolation] = []

    # Aggregate close steps
    total_close = 0
    for ch in action_chunks:
        gripper = np.asarray(ch, dtype=np.float64)[:, 7]
        total_close += int(np.sum(gripper > GRIPPER_CLOSE_THRESHOLD))
    if total_close < min_total_close_steps:
        violations.append(FeasibilityViolation(
            ViolationCategory.GRIPPER_PHASE,
            f"total close steps {total_close} < min {min_total_close_steps} "
            f"(spec §4.4 task 2.7: reverse needs sustained close phase before release)",
        ))

    # z-floor check (only when ee_pos_seqs supplied)
    if ee_pos_seqs is not None:
        for chunk_idx, seq in enumerate(ee_pos_seqs):
            arr = np.asarray(seq, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[-1] != 3:
                continue
            below = np.where(arr[:, 2] < z_floor)[0]
            for t in below:
                violations.append(FeasibilityViolation(
                    ViolationCategory.CART_Z,
                    f"chunk {chunk_idx} t={int(t)}: z={arr[int(t), 2]:.4f} below floor {z_floor}",
                ))

        # Direction check: use the closed-carry phase, not episode start/end.
        # DROID demonstrations typically approach from a high home pose, close
        # near the plate, carry while closed, release, and return. Comparing the
        # first and last frames of the whole episode rejects valid demos.
        flat_positions: list[np.ndarray] = []
        flat_gripper: list[np.ndarray] = []
        for chunk, seq in zip(action_chunks, ee_pos_seqs):
            chunk_arr = np.asarray(chunk, dtype=np.float64)
            seq_arr = np.asarray(seq, dtype=np.float64)
            if (
                chunk_arr.ndim != 2 or chunk_arr.shape[-1] < 8
                or seq_arr.ndim != 2 or seq_arr.shape[-1] != 3
            ):
                continue
            n = min(chunk_arr.shape[0], seq_arr.shape[0])
            if n <= 0:
                continue
            flat_gripper.append(chunk_arr[:n, 7])
            flat_positions.append(seq_arr[:n, :3])
        if flat_positions:
            gripper = np.concatenate(flat_gripper)
            positions = np.concatenate(flat_positions)
            closed = np.where(gripper > GRIPPER_CLOSE_THRESHOLD)[0]
            if len(closed) > 0:
                start_xy = positions[int(closed[0]), :2]
                end_xy = positions[int(closed[-1]), :2]
                carry_distance = float(np.linalg.norm(end_xy - start_xy))
                if carry_distance < min_distance_from_plate:
                    detail = (
                        f"ee did not move away during closed carry: "
                        f"carry_distance={carry_distance:.3f}m "
                        f"(need ≥{min_distance_from_plate}m)"
                    )
                    if plate_xy is not None:
                        plate = np.array(plate_xy, dtype=np.float64)
                        d_start = np.linalg.norm(start_xy - plate)
                        d_end = np.linalg.norm(end_xy - plate)
                        detail += f"; d_start={d_start:.3f}m d_end={d_end:.3f}m"
                    violations.append(FeasibilityViolation(
                        ViolationCategory.CART_LIMIT,
                        detail,
                    ))

    return FeasibilityReport(violations=violations)


__all__ = [
    "ViolationCategory",
    "FeasibilityViolation",
    "FeasibilityReport",
    "JOINT_POS_UPPER", "JOINT_POS_LOWER", "JOINT_VEL_LIMIT",
    "CART_POS_UPPER", "CART_POS_LOWER",
    "DEFAULT_CONTROL_HZ", "DEFAULT_MAX_STEP_DELTA",
    "MIN_CART_Z_DEFAULT", "MIN_CART_Z_GRASP",
    "GRIPPER_CLOSE_THRESHOLD",
    "check_joint_limits",
    "check_joint_velocity",
    "check_step_delta",
    "check_cart_limits",
    "check_gripper_command",
    "check_gripper_phase",
    "check_action_chunk",
    "check_reverse_grasp_trajectory",
]
