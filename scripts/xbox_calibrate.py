"""xbox_calibrate.py — v2.2.1 B2a

Xbox controller deadzone / axis-range / trigger-threshold calibration harness.

Pure estimators (unit-tested without hardware) + a thin hub sampler + an
interactive CLI. The CLI produces REAL numbers only once a handle is plugged
into fr3-desktop-ts (B2b live); this module exists so that the moment the
handle is connected, calibration is one command — no on-site coding.

Outputs a JSON calibration profile consumed (B4) to set XboxIntervention's
deadzone / scale and the RT/LT gripper thresholds, instead of the current
hardcoded defaults.

Estimators are pure functions over lists of teleop_hub.XboxState; the sampler
polls TeleopDeviceHub. No real motion, no robot — controller input only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from teleop_hub import TeleopDeviceHub, get_hub  # noqa: E402

STICK_AXES = ("left_x", "left_y", "right_x", "right_y", "dpad_x", "dpad_y")

DEFAULT_DEADZONE_MARGIN = 1.5   # deadzone = rest_noise_max * margin
DEFAULT_DEADZONE_FLOOR = 0.05   # never below this (matches XboxIntervention default)
DEFAULT_DEADZONE_CAP = 0.30     # never above this (a stuck/biased stick guard)
DEFAULT_TRIGGER_MARGIN = 1.5


class DeviceUnavailable(RuntimeError):
    """Raised when the hub reports no live Xbox device (e.g. unplugged)."""


# ---------------------------------------------------------------------------
# Pure estimators
# ---------------------------------------------------------------------------
def estimate_deadzone(
    rest_samples,
    margin: float = DEFAULT_DEADZONE_MARGIN,
    floor: float = DEFAULT_DEADZONE_FLOOR,
    cap: float = DEFAULT_DEADZONE_CAP,
) -> float:
    """Deadzone from at-rest noise: max |stick axis| over the rest capture,
    scaled by ``margin``, clamped to [floor, cap]."""
    worst = 0.0
    for s in rest_samples:
        for ax in ("left_x", "left_y", "right_x", "right_y"):
            worst = max(worst, abs(float(getattr(s, ax))))
    return float(min(cap, max(floor, worst * margin)))


def estimate_axis_ranges(samples) -> dict:
    """Per-axis (min, max) over a full-deflection capture."""
    ranges = {}
    for ax in STICK_AXES:
        vals = [float(getattr(s, ax)) for s in samples]
        if vals:
            ranges[ax] = (min(vals), max(vals))
        else:
            ranges[ax] = (0.0, 0.0)
    return ranges


def estimate_trigger_thresholds(
    rest_samples,
    pressed_samples,
    margin: float = DEFAULT_TRIGGER_MARGIN,
) -> tuple[float, float]:
    """RT/LT trigger thresholds: above the at-rest noise floor and below the
    pressed level, so a resting trigger never registers but a real press does.
    Threshold = midpoint(rest_max*margin, pressed_min) clamped to [0.05, 0.5]."""

    def thr(field: str) -> float:
        rest_max = max((float(getattr(s, field)) for s in rest_samples), default=0.0)
        pressed_min = min(
            (float(getattr(s, field)) for s in pressed_samples), default=1.0
        )
        floor = max(0.05, rest_max * margin)
        # midpoint between the noise floor and the weakest real press
        mid = (floor + pressed_min) / 2.0 if pressed_min > floor else floor
        return float(min(0.5, max(0.05, mid)))

    return thr("rt"), thr("lt")


def build_calibration(rest_samples, range_samples) -> dict:
    """Assemble a JSON-serialisable calibration profile."""
    rt_thr, lt_thr = estimate_trigger_thresholds(rest_samples, range_samples)
    return {
        "deadzone": estimate_deadzone(rest_samples),
        "axis_ranges": {ax: list(rng) for ax, rng in estimate_axis_ranges(range_samples).items()},
        "rt_threshold": rt_thr,
        "lt_threshold": lt_thr,
        "n_rest": len(rest_samples),
        "n_range": len(range_samples),
    }


# ---------------------------------------------------------------------------
# Hub sampler
# ---------------------------------------------------------------------------
def collect_samples(hub: TeleopDeviceHub, n: int, sleep_s: float = 0.02) -> list:
    """Poll the hub ``n`` times into a list of XboxState. Raises
    DeviceUnavailable if the hub has no live device (so the CLI fails loudly
    rather than calibrating against a zeroed phantom stream)."""
    if not hub.available:
        raise DeviceUnavailable(
            "no live Xbox device on the hub (handle not plugged in / pygame missing)"
        )
    out = []
    for _ in range(int(n)):
        out.append(hub.poll())
        if sleep_s:
            time.sleep(sleep_s)
    return out


# ---------------------------------------------------------------------------
# CLI (interactive; needs a real handle)
# ---------------------------------------------------------------------------
def _prompt(msg: str) -> None:  # pragma: no cover - interactive
    try:
        input(msg)
    except EOFError:
        pass


def main(argv: Optional[list] = None) -> int:  # pragma: no cover - interactive/hardware
    p = argparse.ArgumentParser(description="Xbox deadzone/range/trigger calibration (B2a)")
    p.add_argument("--out", default="xbox_calibration.json")
    p.add_argument("--rest-samples", type=int, default=100)
    p.add_argument("--range-samples", type=int, default=200)
    p.add_argument("--hz", type=float, default=50.0)
    args = p.parse_args(argv)

    hub = get_hub()
    if not hub.available:
        print(
            "[ERR] no Xbox device detected on fr3-desktop-ts (no /dev/input/js* "
            "or pygame missing). Plug the handle in and retry.",
            file=sys.stderr,
        )
        return 2

    dt = 1.0 / args.hz
    print("[1/2] Leave both sticks + triggers at REST, then press Enter.")
    _prompt("")
    rest = collect_samples(hub, args.rest_samples, sleep_s=dt)
    print("[2/2] Sweep both sticks to full range + fully press RT and LT, then press Enter.")
    _prompt("")
    rng = collect_samples(hub, args.range_samples, sleep_s=dt)

    cal = build_calibration(rest, rng)
    with open(args.out, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"Calibration written: {args.out}")
    print(json.dumps(cal, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
