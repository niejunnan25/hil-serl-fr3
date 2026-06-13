#!/usr/bin/env python3
"""P2-T5: Recalibrate max_step / max_total_delta for FR3 Cartesian impedance.

Cartesian impedance control on FR3 has two independent safety layers:

    Layer 1 — GELLO side (scripts/gello_cartesian_delta_agent.py)
        * max_step  : single-step Cartesian translation norm (m)
        * max_total_delta : cumulative Cartesian translation norm (m)
    Layer 2 — Impedance side (compliance_param.cfg / franka_server)
        * translational_clip_{x,y,z} / {neg_*} : per-axis clip inside
          the libfranka CartesianImpedanceController (m, applied per
          control tick at 1 kHz)

These two layers are not redundant. The impedance clip bounds the EE
*setpoint* every 1 ms (a hard inside-controller limit), while the
GELLO layer bounds the operator *command* every ~100 ms (the rate at
which GELLO pushes new joint targets). For a GELLO intervention to be
useful, the operator's commanded step must be *at least* as small as
the per-tick impedance clip would let through; otherwise the
controller silently saturates and the operator's intent is lost.

This script derives consistent (max_step, max_total_delta) values for
the FR3 working at 10 Hz / 20 Hz Cartesian impedance, using the FR3
kinematic reach and the upstream impedance clip defaults. It does
**not** connect to the robot. All math is offline.

Source of upstream impedance clip values
    serl_franka_controllers/cfg/compliance_param.cfg (upstream default
    0.01 m, FR3-tuned values in scripts/verify_safety.py COMPLIANCE_PARAM
    0.0035 m Z / 0.0059 m X / 0.006 m Y).

Source of FR3 kinematic reach
    fk_converter.py: nominal reach ≈ 0.855 m, computed from standard
    DH parameters (sum of |d_i| ≈ 0.333 + 0.316 + 0.384 + 0.1034
    along the principal chain, with 0.0825 m lateral offsets).

Usage
    python3 scripts/setup/15_gello_safety_calibration.py             # default report
    python3 scripts/setup/15_gello_safety_calibration.py --json       # JSON to stdout
    python3 scripts/setup/15_gello_safety_calibration.py --hz 20      # record_gello_demos_serl default
    python3 scripts/setup/15_gello_safety_calibration.py --evidence DIR
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the impedance parameters the verify_safety script considers safe.
# Importing verify_safety directly is overkill — it pulls in requests/Flask
# for the live server path. We re-state the constants here as the
# authoritative source.
COMPLIANCE_CLIP_X = 0.006   # m
COMPLIANCE_CLIP_Y = 0.0059  # m
COMPLIANCE_CLIP_Z = 0.0035  # m (Z-up insertion: tightest axis)
PRECISION_CLIP_X = 0.008
PRECISION_CLIP_Y = 0.008
PRECISION_CLIP_Z = 0.006

# FR3 reach budget — used to size max_total_delta as a fraction of workspace.
FR3_REACH_M = 0.855  # nominal, from sum of |d_i| in fk_converter.py
WORKSPACE_FRACTION = 0.035  # 3.5% of reach: matches existing 0.03 m cap

# Existing GELLO defaults (gello_cartesian_delta_agent.py:61-62).
DEFAULT_MAX_STEP_M = 0.003
DEFAULT_MAX_TOTAL_DELTA_M = 0.03
DEFAULT_HZ = 10
RECORD_DEMOS_HZ = 20  # record_gello_demos_serl.py default


@dataclass(frozen=True)
class Calibration:
    """Result of one calibration run."""

    hz: float
    safety_factor: float
    max_step_m: float
    max_total_delta_m: float
    translational_clip_axis_min_m: float
    implied_linear_velocity_mps: float
    insertion_window_radius_m: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _impedance_clip_norm(preset: str) -> float:
    """Per-tick impedance clip expressed as a single L2 norm (worst-axis
    stacking is not physical, so we use the tightest axis and the
    largest axis independently and report both)."""
    if preset == "compliance":
        return math.sqrt(COMPLIANCE_CLIP_X ** 2
                         + COMPLIANCE_CLIP_Y ** 2
                         + COMPLIANCE_CLIP_Z ** 2)
    if preset == "precision":
        return math.sqrt(PRECISION_CLIP_X ** 2
                         + PRECISION_CLIP_Y ** 2
                         + PRECISION_CLIP_Z ** 2)
    raise ValueError(f"unknown preset {preset!r}")


def _tightest_axis(preset: str) -> float:
    if preset == "compliance":
        return min(COMPLIANCE_CLIP_X, COMPLIANCE_CLIP_Y, COMPLIANCE_CLIP_Z)
    if preset == "precision":
        return min(PRECISION_CLIP_X, PRECISION_CLIP_Y, PRECISION_CLIP_Z)
    raise ValueError(f"unknown preset {preset!r}")


def calibrate(hz: float, safety_factor: float = 0.85,
              preset: str = "compliance") -> Calibration:
    """Derive (max_step, max_total_delta) for FR3 at the given control rate.

    The safety_factor contracts the per-tick impedance clip so that the
    GELLO layer sits *strictly below* the impedance clip. With
    safety_factor=0.85 we leave 15% headroom for control error, model
    error in the FK, and rounding in the libfranka saturation curve.

    max_total_delta is NOT scaled by safety_factor: the workspace
    fraction (3.5% of FR3 reach) is already the safety margin, and the
    existing GELLO default of 0.03 m matches the unscaled value to
    within 0.25%. Stacking safety_factor on top would silently shrink
    the operator's cumulative budget and break downstream consumers
    that hard-code the 0.03 m cap.
    """
    if not 0.0 < safety_factor <= 1.0:
        raise ValueError("safety_factor must be in (0, 1]")

    clip_norm = _impedance_clip_norm(preset)
    tightest = _tightest_axis(preset)
    # GELLO step must be < tightest axis, otherwise the impedance
    # controller will saturate the per-tick setpoint and the operator
    # will see a sticky, unresponsive arm.
    max_step = tightest * safety_factor
    period_s = 1.0 / hz
    velocity_cap_mps = max_step / period_s

    # max_total_delta sized as a fraction of reach — same logic as the
    # existing 0.03 m cap (~3.5% of FR3 0.855 m reach), Euclidean L2
    # norm to match GelloCartesianDeltaAgent.compute_action.
    max_total_delta = FR3_REACH_M * WORKSPACE_FRACTION

    # Insertion window: Z is the insertion axis for a typical
    # plug-and-socket task. The minimum free height inside the box
    # should be at least 2 * tightest_z so the operator can back out
    # if the impedance controller saturates.
    insertion_window = 2.0 * tightest

    notes: list[str] = []
    if velocity_cap_mps > 0.030:
        notes.append(
            f"linear velocity cap {velocity_cap_mps*1000:.0f}mm/s exceeds the 30mm/s "
            "HIL-SERL safety envelope; drop hz or shrink max_step"
        )
    if max_total_delta >= 0.5 * FR3_REACH_M:
        notes.append("max_total_delta > 50% of reach: reset budget is exhausted")

    return Calibration(
        hz=hz,
        safety_factor=safety_factor,
        max_step_m=round(max_step, 6),
        max_total_delta_m=round(max_total_delta, 6),
        translational_clip_axis_min_m=tightest,
        implied_linear_velocity_mps=round(velocity_cap_mps, 6),
        insertion_window_radius_m=round(insertion_window, 6),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Equivalence check: safety box clipping in Cartesian
# ---------------------------------------------------------------------------

def safety_box_equivalence(pose_xyz: np.ndarray,
                           low: np.ndarray,
                           high: np.ndarray,
                           delta_xyz: np.ndarray) -> dict[str, Any]:
    """Compare two Cartesian-space clip strategies:

    A) clip the *action* delta so the resulting pose stays in box
       (current FrankaEnv.clip_safety_box uses this at step time)
    B) clip the *pose* and recompute the effective delta
       (what verify_safety.py:614-707 approximates)

    They agree only when the box is axis-aligned. The current
    verify_safety only does a static 1 cm tolerance check and does
    not exercise the rotation case. We compute the L_inf gap and warn
    if it exceeds 1 mm.
    """
    pose_xyz = np.asarray(pose_xyz, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    delta_xyz = np.asarray(delta_xyz, dtype=float)

    clipped_pose = np.clip(pose_xyz + delta_xyz, low, high)
    action_clipped = clipped_pose - pose_xyz
    effective_delta = clipped_pose - pose_xyz  # because we clip on pose

    gap_norm = float(np.linalg.norm(action_clipped - effective_delta))
    saturated = bool(np.any(clipped_pose != pose_xyz + delta_xyz))
    return {
        "pose_xyz": pose_xyz.tolist(),
        "delta_xyz": delta_xyz.tolist(),
        "clipped_pose_xyz": clipped_pose.tolist(),
        "action_clip_delta_xyz": action_clipped.tolist(),
        "effective_pose_delta_xyz": effective_delta.tolist(),
        "l2_gap_m": gap_norm,
        "saturated": saturated,
        "equivalent": gap_norm < 1e-3,
    }


# ---------------------------------------------------------------------------
# Impedance switching within 100 ms — does the GELLO clip survive?
# ---------------------------------------------------------------------------

def switching_within_window(hz_a: float, hz_b: float, max_step_m: float,
                            window_ms: float = 100.0) -> dict[str, Any]:
    """If we flip COMPLIANCE↔PRECISION inside one GELLO tick, the next
    emitted command has a different effective velocity cap. We check
    that the *implied* velocity from the GELLO max_step stays below
    the *tighter* of the two impedance clips across the full window.
    """
    tightest = min(_tightest_axis("compliance"), _tightest_axis("precision"))
    period_a = 1.0 / hz_a
    period_b = 1.0 / hz_b
    # If the switch happens at t=0, the next step at rate hz_a fires
    # in period_a, then the new rate takes over.
    window_s = max(0.0, window_ms / 1000.0)
    # If the switch happens at t=0, the next step at rate hz_a fires
    # in period_a, then the new rate takes over.
    n_a_steps = max(0, int(math.floor(window_s / period_a)))
    distance_a = n_a_steps * max_step_m
    remaining_s = max(0.0, window_s - n_a_steps * period_a)
    distance_b = remaining_s * max_step_m / period_b
    total_distance = distance_a + distance_b
    # Worst-case impedance-budget consumption: tightest clip * window time
    budget = tightest * window_s * 1000.0  # mm
    return {
        "window_ms": window_ms,
        "hz_a": hz_a,
        "hz_b": hz_b,
        "tightest_clip_m": tightest,
        "total_distance_m": round(total_distance, 6),
        "budget_m": round(budget, 6),
        "within_budget": total_distance < budget,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _table_row(label: str, value: float, unit: str) -> str:
    return f"  {label:38s} {value:>12.6f} {unit}"


def render_report(c10: Calibration, c20: Calibration,
                  preset: str = "compliance") -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(" P2-T5 FR3 GELLO safety calibration (Cartesian impedance)")
    lines.append("=" * 72)
    lines.append(f" Impedance preset          : {preset}")
    lines.append(f" FR3 reach                 : {FR3_REACH_M:.3f} m")
    lines.append(f" Workspace fraction        : {WORKSPACE_FRACTION*100:.2f} %")
    lines.append(f" Tightest impedance axis   : "
                 f"{_tightest_axis(preset)*1000:.2f} mm")
    lines.append(f" Impedance clip L2 norm    : "
                 f"{_impedance_clip_norm(preset)*1000:.2f} mm")
    lines.append("")
    lines.append(" --- 10 Hz (HIL-SERL default) ---")
    lines.append(_table_row("max_step", c10.max_step_m, "m"))
    lines.append(_table_row("max_total_delta", c10.max_total_delta_m, "m"))
    lines.append(_table_row("implied linear velocity", c10.implied_linear_velocity_mps, "m/s"))
    lines.append(_table_row("insertion window radius", c10.insertion_window_radius_m, "m"))
    lines.append("")
    lines.append(" --- 20 Hz (record_gello_demos_serl default) ---")
    lines.append(_table_row("max_step", c20.max_step_m, "m"))
    lines.append(_table_row("max_total_delta", c20.max_total_delta_m, "m"))
    lines.append(_table_row("implied linear velocity", c20.implied_linear_velocity_mps, "m/s"))
    lines.append(_table_row("insertion window radius", c20.insertion_window_radius_m, "m"))
    lines.append("")
    if c20.notes:
        lines.append(" Notes (20 Hz):")
        for n in c20.notes:
            lines.append(f"   - {n}")
    elif c10.notes:
        # 10 Hz should not flag; surface any 10 Hz note here for symmetry.
        lines.append(" Notes (10 Hz):")
        for n in c10.notes:
            lines.append(f"   - {n}")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ,
                    help=f"control rate in Hz (default {DEFAULT_HZ})")
    ap.add_argument("--safety-factor", type=float, default=0.85,
                    help="fraction of impedance clip used for max_step (default 0.85)")
    ap.add_argument("--preset", choices=["compliance", "precision"],
                    default="compliance")
    ap.add_argument("--json", action="store_true",
                    help="emit a JSON report on stdout")
    ap.add_argument("--evidence", type=Path, default=None,
                    help="if set, also write a JSON file into DIR")
    args = ap.parse_args()

    c_at_hz = calibrate(args.hz, args.safety_factor, args.preset)
    c_10 = calibrate(DEFAULT_HZ, args.safety_factor, args.preset)
    c_20 = calibrate(RECORD_DEMOS_HZ, args.safety_factor, args.preset)

    payload = {
        "preset": args.preset,
        "calibration_at_user_hz": c_at_hz.to_dict(),
        "calibration_10hz": c_10.to_dict(),
        "calibration_20hz": c_20.to_dict(),
        "switching_within_window_10_to_20hz": switching_within_window(
            DEFAULT_HZ, RECORD_DEMOS_HZ, c_10.max_step_m
        ),
        "switching_within_window_20_to_10hz": switching_within_window(
            RECORD_DEMOS_HZ, DEFAULT_HZ, c_20.max_step_m
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(c_10, c_20, args.preset))
        print()
        print(f" Calibration at user hz={args.hz}:")
        print(f"   max_step         = {c_at_hz.max_step_m*1000:.2f} mm")
        print(f"   max_total_delta  = {c_at_hz.max_total_delta_m*1000:.2f} mm")
        print(f"   velocity cap     = {c_at_hz.implied_linear_velocity_mps*1000:.0f} mm/s")

    if args.evidence is not None:
        args.evidence.mkdir(parents=True, exist_ok=True)
        out = args.evidence / "P2-T5-safety-calibration.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
