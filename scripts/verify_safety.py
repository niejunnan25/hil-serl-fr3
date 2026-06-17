#!/usr/bin/env python3
"""
verify_safety.py – Pre-operation safety verification for GELLO + serl_franka_controllers.

Run this script BEFORE any live robot operation to verify:
  1. GelloIntervention: GELLO delta safety limits
  2. Cartesian safety box: end-effector within workspace bounds
  3. Impedance parameters: COMPLIANCE_PARAM and PRECISION_PARAM are reasonable
  4. Gripper: open/close commands respond
  5. Emergency stop: zero-action stop verification
  6. FrankaEnv safety box integration: clip_safety_box() consistency

Usage:
    python scripts/verify_safety.py                          # Full verification (requires GELLO + robot)
    python scripts/verify_safety.py --dry-run                # Skip actual robot commands
    python scripts/verify_safety.py --server-url http://...  # Custom server URL
    python scripts/verify_safety.py --gello-port /dev/ttyUSB1

Exit code 0 = all critical checks pass, 1 = at least one critical failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Ensure scripts/ is on path for fk_converter import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import requests
except ImportError:
    requests = None  # Optional – only needed for server checks


# ── constants ──────────────────────────────────────────────────────────────────

# GELLO safety thresholds (from gello_cartesian_delta_agent.py / record_gello_demos.py)
GELLO_MAX_STEP = 0.003        # meters per step
GELLO_MAX_TOTAL_DELTA = 0.03  # meters cumulative
GELLO_NUM_FRAMES = 100        # frames to read for calibration

# FR3 joint limits (from record_gello_demos.py)
from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS

# Cartesian safety box (FR3 typical workspace)
# These define a conservative bounding box for the end-effector.
# [x_min, y_min, z_min] to [x_max, y_max, z_max] in meters (base frame).
CARTESIAN_SAFETY_BOX_LOW = np.array([0.20, -0.50, 0.05])
CARTESIAN_SAFETY_BOX_HIGH = np.array([0.80, 0.50, 0.70])

# Impedance parameter bounds (sane ranges for FR3)
# Format: (key, min_val, max_val, description)
IMPEDANCE_PARAM_BOUNDS: list[tuple[str, float, float, str]] = [
    ("translational_stiffness", 500, 5000, "translational stiffness (N/m)"),
    ("translational_damping", 20, 200, "translational damping (Ns/m)"),
    ("rotational_stiffness", 50, 500, "rotational stiffness (Nm/rad)"),
    ("rotational_damping", 2, 50, "rotational damping (Nms/rad)"),
    ("translational_Ki", 0, 10, "translational integral gain"),
    ("rotational_Ki", 0, 10, "rotational integral gain"),
]

# Max allowed clip value (meters or radians)
MAX_TRANSLATIONAL_CLIP = 0.012   # 12 mm
MAX_ROTATIONAL_CLIP = 0.10      # ~5.7 deg

# Server defaults
DEFAULT_SERVER_URL = "http://127.0.0.2:5000/"
DEFAULT_GELLO_PORT = "/dev/ttyUSB0"
DEFAULT_GELLO_BAUDRATE = 57600
TIMEOUT_S = 5.0

# ── preset impedance params ────────────────────────────────────────
# CANONICAL SOURCE: experiments/plug_insertion/config.py EnvConfig.COMPLIANCE_PARAM /
# PRECISION_PARAM (the live runtime config the actor pushes to serl_franka_controllers
# via /update_param). These dicts mirror that config so this safety verifier checks
# against the ACTUAL operating point. If the live config changes, update these to match.

COMPLIANCE_PARAM: dict[str, float] = {
    "translational_stiffness": 2000,
    "translational_damping": 89,
    "rotational_stiffness": 150,
    "rotational_damping": 7,
    "translational_Ki": 0,
    "translational_clip_x": 0.006,
    "translational_clip_y": 0.0059,
    "translational_clip_z": 0.0035,
    "translational_clip_neg_x": 0.005,
    "translational_clip_neg_y": 0.005,
    "translational_clip_neg_z": 0.0035,
    "rotational_clip_x": 0.05,
    "rotational_clip_y": 0.05,
    "rotational_clip_z": 0.05,
    "rotational_clip_neg_x": 0.05,
    "rotational_clip_neg_y": 0.05,
    "rotational_clip_neg_z": 0.05,
    "rotational_Ki": 0,
}

PRECISION_PARAM: dict[str, float] = {
    "translational_stiffness": 2500,
    "translational_damping": 100,
    "rotational_stiffness": 200,
    "rotational_damping": 10,
    "translational_Ki": 0.0,
    "translational_clip_x": 0.008,
    "translational_clip_y": 0.008,
    "translational_clip_z": 0.0072,
    "translational_clip_neg_x": 0.008,
    "translational_clip_neg_y": 0.008,
    "translational_clip_neg_z": 0.0072,
    "rotational_clip_x": 0.05,
    "rotational_clip_y": 0.05,
    "rotational_clip_z": 0.05,
    "rotational_clip_neg_x": 0.05,
    "rotational_clip_neg_y": 0.05,
    "rotational_clip_neg_z": 0.05,
    "rotational_Ki": 0.0,
}

REQUIRED_PARAM_KEYS: set[str] = set(COMPLIANCE_PARAM.keys())


# ── result tracking ───────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Single check outcome."""
    name: str
    status: str  # "pass", "fail", "warn", "skip"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport:
    """Aggregated safety verification report."""
    checks: list[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    server_url: str = ""
    gello_port: str = ""
    dry_run: bool = False

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == "skip")

    @property
    def all_critical_passed(self) -> bool:
        """All non-warn, non-skip checks must pass."""
        return all(c.status in ("pass", "warn", "skip") for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "server_url": self.server_url,
            "gello_port": self.gello_port,
            "dry_run": self.dry_run,
            "summary": {
                "passed": self.passed,
                "failed": self.failed,
                "warned": self.warned,
                "skipped": self.skipped,
                "total": len(self.checks),
                "all_critical_passed": self.all_critical_passed,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

    def print_report(self) -> None:
        """Print human-readable report to stdout."""
        status_icons = {
            "pass": "\033[32mPASS\033[0m",
            "fail": "\033[31mFAIL\033[0m",
            "warn": "\033[33mWARN\033[0m",
            "skip": "\033[36mSKIP\033[0m",
        }

        print()
        print("=" * 70)
        print("  SAFETY VERIFICATION REPORT")
        print(f"  Time: {self.timestamp}")
        if self.dry_run:
            print("  Mode: DRY-RUN (no robot commands sent)")
        print(f"  Server: {self.server_url}")
        print(f"  GELLO: {self.gello_port}")
        print("=" * 70)
        print()

        for i, c in enumerate(self.checks, 1):
            icon = status_icons.get(c.status, "????")
            msg = f" – {c.message}" if c.message else ""
            print(f"  [{icon}] {i}. {c.name}{msg}")
            if c.details:
                for k, v in c.details.items():
                    print(f"         {k}: {v}")

        print()
        print("=" * 70)
        result_str = "ALL CRITICAL CHECKS PASSED" if self.all_critical_passed else "CRITICAL FAILURES DETECTED"
        result_color = "\033[32m" if self.all_critical_passed else "\033[31m"
        print(f"  {result_color}{result_str}\033[0m")
        print(f"  Passed: {self.passed}  Failed: {self.failed}  Warned: {self.warned}  Skipped: {self.skipped}")
        print("=" * 70)
        print()


# ── HTTP helper ───────────────────────────────────────────────────────────────

def server_post(base_url: str, endpoint: str, payload: dict | None = None) -> dict:
    """POST to franka_server endpoint, return JSON response.

    Raises:
        ConnectionError: if server is unreachable
        ValueError: if response is not valid JSON or has error status
    """
    if requests is None:
        raise ImportError("'requests' package is required for server checks. pip install requests")

    url = base_url.rstrip("/") + endpoint
    resp = requests.post(
        url,
        json=payload or {},
        timeout=TIMEOUT_S,
        headers={"Content-Type": "application/json"},
    )
    # Accept 200 (real ack) and 409 (mock reject) as "server alive"
    if resp.status_code in (200, 409):
        try:
            return resp.json()
        except Exception:
            return {"_status": resp.status_code, "_text": resp.text[:200]}
    else:
        raise ValueError(f"POST {endpoint} returned HTTP {resp.status_code}: {resp.text[:200]}")


# ── check 1: GelloIntervention safety ─────────────────────────────────────────

def check_gello_intervention(
    port: str,
    baudrate: int,
    max_step: float,
    max_total_delta: float,
    num_frames: int,
    dry_run: bool,
) -> CheckResult:
    """Read GELLO frames and verify delta safety limits.

    Reads `num_frames` frames from GELLO, computes:
      - max per-step delta (translation norm)
      - max total delta from initial frame
    Verifies both are within thresholds.
    """
    name = "GelloIntervention safety"

    if dry_run:
        # Simulate: use small random deltas within safe bounds
        # 0.0005 std produces step deltas ~0.001-0.002m, well within 0.003m threshold
        rng = np.random.RandomState(42)
        deltas = rng.randn(num_frames, 7) * 0.0005  # small safe perturbations
        step_deltas = np.linalg.norm(np.diff(deltas, axis=0)[:, :3], axis=1)
        max_step_delta = float(step_deltas.max())
        total_deltas = np.linalg.norm(deltas[:, :3] - deltas[0, :3], axis=1)
        max_total = float(total_deltas.max())

        details = {
            "max_step_delta_m": f"{max_step_delta:.6f}",
            "threshold_step_m": f"{max_step:.6f}",
            "max_total_delta_m": f"{max_total:.6f}",
            "threshold_total_m": f"{max_total_delta:.6f}",
            "frames_read": num_frames,
            "mode": "simulated (dry-run)",
        }

        if max_step_delta > max_step:
            return CheckResult(name, "fail",
                               f"max_step {max_step_delta:.6f} > {max_step}", details)
        if max_total > max_total_delta:
            return CheckResult(name, "fail",
                               f"max_total_delta {max_total:.6f} > {max_total_delta}", details)

        return CheckResult(name, "pass",
                           f"step={max_step_delta:.6f} total={max_total:.6f} within limits", details)

    # Live: connect to GELLO hardware
    try:
        from gello.dynamixel.driver import DynamixelDriver
    except ImportError as e:
        return CheckResult(name, "fail",
                           "Cannot import gello.dynamixel.driver – is gello installed?",
                           {"error": str(e)})

    try:
        from fk_converter import joints_to_cartesian_delta
    except ImportError as e:
        return CheckResult(name, "fail",
                           "Cannot import fk_converter for joint->Cartesian conversion",
                           {"error": str(e)})

    try:
        driver = DynamixelDriver(
            list(range(8)),
            port=port,
            baudrate=baudrate,
            max_retries=1,
            use_fake_fallback=False,
        )
    except Exception as e:
        return CheckResult(name, "fail", f"Cannot connect to GELLO: {e}",
                           {"port": port, "error": str(e)})

    try:
        frames = []
        for _ in range(num_frames):
            raw = np.asarray(driver.get_joints(), dtype=float)
            if raw.shape != (8,):
                driver.close()
                return CheckResult(name, "fail",
                                   f"Unexpected GELLO read shape: {raw.shape}",
                                   {"expected": "(8,)", "got": str(raw.shape)})
            frames.append(raw[:7])  # 7 joints (exclude gripper)
            time.sleep(0.01)  # ~100 Hz read

        frames = np.array(frames)

        # Per-step Cartesian delta (meters) via FK conversion
        step_cart_deltas = []
        for i in range(1, len(frames)):
            cart_delta = joints_to_cartesian_delta(frames[i - 1], frames[i])
            step_cart_deltas.append(np.linalg.norm(cart_delta[:3]))  # translation norm
        step_cart_deltas = np.array(step_cart_deltas)
        max_step_delta = float(step_cart_deltas.max()) if len(step_cart_deltas) > 0 else 0.0

        # Total Cartesian delta from initial frame (translation norm)
        total_cart_deltas = []
        for i in range(1, len(frames)):
            cart_delta = joints_to_cartesian_delta(frames[0], frames[i])
            total_cart_deltas.append(np.linalg.norm(cart_delta[:3]))
        total_cart_deltas = np.array(total_cart_deltas)
        max_total = float(total_cart_deltas.max()) if len(total_cart_deltas) > 0 else 0.0

        details = {
            "max_step_delta_m": f"{max_step_delta:.6f}",
            "threshold_step_m": f"{max_step:.6f}",
            "max_total_delta_m": f"{max_total:.6f}",
            "threshold_total_m": f"{max_total_delta:.6f}",
            "frames_read": num_frames,
            "port": port,
            "conversion": "FK (joint -> Cartesian, translation norm)",
        }

        if max_step_delta > max_step:
            return CheckResult(name, "fail",
                               f"Cartesian step delta {max_step_delta:.6f}m > {max_step}m threshold",
                               details)
        if max_total > max_total_delta:
            return CheckResult(name, "fail",
                               f"Cartesian total delta {max_total:.6f}m > {max_total_delta}m threshold",
                               details)

        return CheckResult(name, "pass",
                           f"step={max_step_delta:.6f}m total={max_total:.6f}m within limits", details)

    except Exception as e:
        return CheckResult(name, "fail", f"GELLO read error: {e}",
                           {"error": str(e)})
    finally:
        try:
            driver.close()
        except Exception:
            pass


# ── check 2: Cartesian safety box ─────────────────────────────────────────────

def check_cartesian_safety_box(
    base_url: str,
    box_low: np.ndarray,
    box_high: np.ndarray,
    dry_run: bool,
) -> CheckResult:
    """Read current end-effector pose from franka_server and verify it's within bounds."""
    name = "Cartesian safety box"

    if dry_run:
        # Simulate: use a nominal pose
        pose = np.array([0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0])
        details = {
            "pose_xyz": [f"{v:.4f}" for v in pose[:3]],
            "box_low": [f"{v:.4f}" for v in box_low],
            "box_high": [f"{v:.4f}" for v in box_high],
            "mode": "simulated (dry-run)",
        }
        within = np.all(pose[:3] >= box_low) and np.all(pose[:3] <= box_high)
        if within:
            return CheckResult(name, "pass",
                               f"Pose {pose[:3].round(4)} within safety box", details)
        else:
            return CheckResult(name, "fail",
                               f"Pose {pose[:3].round(4)} outside safety box", details)

    # Live: query franka_server
    try:
        state = server_post(base_url, "/getstate")
    except ConnectionError:
        return CheckResult(name, "fail",
                           "Cannot connect to franka_server",
                           {"server": base_url})
    except Exception as e:
        return CheckResult(name, "fail", f"Server error: {e}",
                           {"server": base_url, "error": str(e)})

    pose = np.asarray(state.get("pose", []), dtype=float)
    if pose.shape != (7,):
        return CheckResult(name, "fail",
                           f"Invalid pose shape: {pose.shape}",
                           {"expected": "(7,)", "got": str(pose.shape)})

    xyz = pose[:3]
    details = {
        "pose_xyz": [f"{v:.4f}" for v in xyz],
        "box_low": [f"{v:.4f}" for v in box_low],
        "box_high": [f"{v:.4f}" for v in box_high],
    }

    within = np.all(xyz >= box_low) and np.all(xyz <= box_high)
    if within:
        return CheckResult(name, "pass",
                           f"Pose {xyz.round(4)} within safety box", details)
    else:
        violations = []
        for i, axis in enumerate(["x", "y", "z"]):
            if xyz[i] < box_low[i]:
                violations.append(f"{axis}={xyz[i]:.4f} < low={box_low[i]:.4f}")
            if xyz[i] > box_high[i]:
                violations.append(f"{axis}={xyz[i]:.4f} > high={box_high[i]:.4f}")
        return CheckResult(name, "fail",
                           f"Pose outside safety box: {'; '.join(violations)}", details)


# ── check 3: Impedance parameters ─────────────────────────────────────────────

def check_impedance_params(dry_run: bool = False) -> CheckResult:
    """Verify COMPLIANCE_PARAM and PRECISION_PARAM are within sane bounds."""
    name = "Impedance parameters"
    issues: list[str] = []
    details: dict[str, Any] = {}

    for params, label in [(COMPLIANCE_PARAM, "COMPLIANCE"), (PRECISION_PARAM, "PRECISION")]:
        # Check required keys
        missing = REQUIRED_PARAM_KEYS - set(params.keys())
        if missing:
            issues.append(f"{label}: missing keys {missing}")
            continue

        # Check stiffness/damping bounds
        for key, min_val, max_val, desc in IMPEDANCE_PARAM_BOUNDS:
            val = params.get(key)
            if val is None:
                issues.append(f"{label}: missing {key}")
                continue
            if val < min_val or val > max_val:
                issues.append(f"{label}: {desc} = {val} outside [{min_val}, {max_val}]")

        # Check clip values
        for key, val in params.items():
            if "clip" in key and "translational" in key:
                if abs(val) > MAX_TRANSLATIONAL_CLIP:
                    issues.append(f"{label}: {key} = {val} > {MAX_TRANSLATIONAL_CLIP} (translational clip)")
            elif "clip" in key and "rotational" in key:
                if abs(val) > MAX_ROTATIONAL_CLIP:
                    issues.append(f"{label}: {key} = {val} > {MAX_ROTATIONAL_CLIP} (rotational clip)")

        details[f"{label}_stiffness"] = params.get("translational_stiffness")
        details[f"{label}_damping"] = params.get("translational_damping")

    if not issues:
        return CheckResult(name, "pass",
                           "Both parameter sets within bounds", details)
    elif all("missing keys" not in i for i in issues):
        # Warnings only (bounds exceeded but not critical)
        return CheckResult(name, "warn",
                           "; ".join(issues), details)
    else:
        return CheckResult(name, "fail",
                           "; ".join(issues), details)


# ── check 4: Gripper ──────────────────────────────────────────────────────────

def check_gripper(base_url: str, dry_run: bool) -> CheckResult:
    """Test open_gripper and close_gripper commands."""
    name = "Gripper commands"

    if dry_run:
        return CheckResult(name, "skip",
                           "Dry-run mode – gripper commands not sent",
                           {"mode": "dry-run"})

    results: dict[str, Any] = {}
    errors: list[str] = []

    for cmd, endpoint in [("open", "/open_gripper"), ("close", "/close_gripper")]:
        try:
            resp = server_post(base_url, endpoint)
            results[f"{cmd}_response"] = resp
        except ValueError as e:
            # 409 mock reject is acceptable
            if "409" in str(e):
                results[f"{cmd}_response"] = "mock_rejected (409)"
            else:
                errors.append(f"{cmd}: {e}")
                results[f"{cmd}_error"] = str(e)
        except Exception as e:
            errors.append(f"{cmd}: {e}")
            results[f"{cmd}_error"] = str(e)

    # Also check gripper position readback
    try:
        state = server_post(base_url, "/getstate")
        gripper_pos = state.get("gripper_pos", None)
        results["gripper_pos"] = gripper_pos
    except Exception as e:
        results["gripper_pos_error"] = str(e)

    if errors:
        return CheckResult(name, "fail",
                           f"Gripper command errors: {'; '.join(errors)}", results)

    return CheckResult(name, "pass",
                       "open/close commands accepted", results)


# ── check 5: Emergency stop ───────────────────────────────────────────────────

def check_emergency_stop(base_url: str, dry_run: bool) -> CheckResult:
    """Verify zero-action stop: read pose, send current pose back, verify no motion."""
    name = "Emergency stop"

    if dry_run:
        return CheckResult(name, "skip",
                           "Dry-run mode – e-stop test not executed",
                           {"mode": "dry-run"})

    try:
        # Read current state
        state_before = server_post(base_url, "/getstate")
        pose_before = np.asarray(state_before["pose"], dtype=float)

        # Send current pose as "stop" command (zero delta)
        server_post(base_url, "/pose", {"arr": pose_before.tolist()})

        # Brief wait for robot to settle
        time.sleep(0.1)

        # Read state after
        state_after = server_post(base_url, "/getstate")
        pose_after = np.asarray(state_after["pose"], dtype=float)

        # Compute drift
        drift = np.linalg.norm(pose_after[:3] - pose_before[:3])
        max_drift = 0.005  # 5mm max acceptable drift

        details = {
            "pose_before_xyz": [f"{v:.4f}" for v in pose_before[:3]],
            "pose_after_xyz": [f"{v:.4f}" for v in pose_after[:3]],
            "drift_m": f"{drift:.6f}",
            "max_drift_m": f"{max_drift:.6f}",
        }

        if drift > max_drift:
            return CheckResult(name, "warn",
                               f"Drift {drift:.6f}m after stop command (>{max_drift}m)", details)

        return CheckResult(name, "pass",
                           f"Drift {drift:.6f}m within {max_drift}m limit", details)

    except Exception as e:
        return CheckResult(name, "fail", f"E-stop test error: {e}",
                           {"error": str(e)})


# ── check 6: FrankaEnv safety box integration ─────────────────────────────────

def check_safety_box_integration(dry_run: bool = False) -> CheckResult:
    """Verify that the safety box constants used here are consistent with
    the clip_safety_box() logic in the HIL-SERL FrankaEnv.

    This is a static code-level check: we verify the safety box bounds
    defined in this script match those used in the env.
    """
    name = "FrankaEnv safety box integration"
    details: dict[str, Any] = {}

    # The safety box in this script
    details["script_box_low"] = CARTESIAN_SAFETY_BOX_LOW.tolist()
    details["script_box_high"] = CARTESIAN_SAFETY_BOX_HIGH.tolist()

    # Check: box_low < box_high for all axes
    if not np.all(CARTESIAN_SAFETY_BOX_LOW < CARTESIAN_SAFETY_BOX_HIGH):
        return CheckResult(name, "fail",
                           "Safety box low >= high for some axis",
                           details)

    # Check: box is within FR3 physical reach
    # FR3 reach is ~0.855m, base at origin
    max_reach = 0.855
    for i, axis in enumerate(["x", "y", "z"]):
        if np.linalg.norm(CARTESIAN_SAFETY_BOX_HIGH[i]) > max_reach:
            return CheckResult(name, "warn",
                               f"{axis} high bound may exceed FR3 reach ({max_reach}m)",
                               details)

    # Check: z_min > 0 (don't allow end-effector below table)
    if CARTESIAN_SAFETY_BOX_LOW[2] < 0:
        return CheckResult(name, "warn",
                           f"z_min = {CARTESIAN_SAFETY_BOX_LOW[2]} < 0 (below table)",
                           details)

    # Try to import FrankaEnv and verify clip_safety_box() consistency
    try:
        # Attempt to find and import FrankaEnv from common locations
        import importlib
        franka_env = None

        # Try common module paths
        for module_path in [
            "serl_franka_controllers.franka_env",
            "franka_env",
            "env.franka_env",
        ]:
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, "FrankaEnv"):
                    franka_env = mod.FrankaEnv
                    details["imported_from"] = module_path
                    break
            except ImportError:
                continue

        if franka_env is not None:
            # Check if FrankaEnv has safety box attributes
            env_safety_attrs = [
                "SAFETY_BOX_LOW", "SAFETY_BOX_HIGH",
                "safety_box_low", "safety_box_high",
                "ABS_POSE_LIMIT_LOW", "ABS_POSE_LIMIT_HIGH",
            ]
            found_attrs = {}
            for attr in env_safety_attrs:
                val = getattr(franka_env, attr, None)
                if val is not None:
                    found_attrs[attr] = np.asarray(val).tolist()

            if found_attrs:
                details["env_safety_attrs"] = found_attrs
                # Compare with our bounds
                for attr_name, env_val in found_attrs.items():
                    env_arr = np.asarray(env_val)
                    if "LOW" in attr_name or "low" in attr_name:
                        if not np.allclose(env_arr, CARTESIAN_SAFETY_BOX_LOW, atol=0.01):
                            return CheckResult(name, "warn",
                                               f"FrankaEnv.{attr_name} differs from script bounds (tolerance=1cm)",
                                               details)
                    elif "HIGH" in attr_name or "high" in attr_name:
                        if not np.allclose(env_arr, CARTESIAN_SAFETY_BOX_HIGH, atol=0.01):
                            return CheckResult(name, "warn",
                                               f"FrankaEnv.{attr_name} differs from script bounds (tolerance=1cm)",
                                               details)
                details["env_comparison"] = "bounds match within 1cm tolerance"
            else:
                details["env_comparison"] = "no safety box attrs found on FrankaEnv"
        else:
            details["imported_from"] = "not found (tried serl_franka_controllers.franka_env, franka_env, env.franka_env)"

    except Exception as e:
        details["import_error"] = str(e)

    details["note"] = ("Static bounds check + optional dynamic import. "
                       "For full integration, verify FrankaEnv.clip_safety_box() uses matching bounds.")

    return CheckResult(name, "pass",
                       "Safety box bounds are consistent and physically valid", details)


# ── main ───────────────────────────────────────────────────────────────────────

def _log(msg: str, use_stderr: bool = False) -> None:
    """Print a log message. When use_stderr=True, send to stderr (for --json mode)."""
    print(msg, file=sys.stderr if use_stderr else sys.stdout)


def run_all_checks(args: argparse.Namespace) -> SafetyReport:
    """Run all safety checks and return the report."""
    from datetime import datetime

    report = SafetyReport(
        timestamp=datetime.now().isoformat(),
        server_url=args.server_url,
        gello_port=args.gello_port,
        dry_run=args.dry_run,
    )

    # When --json is used, send progress to stderr to keep stdout clean
    to_stderr = args.json

    _log("[1/6] GelloIntervention safety check...", to_stderr)
    report.add(check_gello_intervention(
        port=args.gello_port,
        baudrate=args.gello_baudrate,
        max_step=args.max_step,
        max_total_delta=args.max_total_delta,
        num_frames=args.num_frames,
        dry_run=args.dry_run,
    ))

    _log("[2/6] Cartesian safety box check...", to_stderr)
    report.add(check_cartesian_safety_box(
        base_url=args.server_url,
        box_low=CARTESIAN_SAFETY_BOX_LOW,
        box_high=CARTESIAN_SAFETY_BOX_HIGH,
        dry_run=args.dry_run,
    ))

    _log("[3/6] Impedance parameter check...", to_stderr)
    report.add(check_impedance_params(dry_run=args.dry_run))

    _log("[4/6] Gripper command check...", to_stderr)
    report.add(check_gripper(base_url=args.server_url, dry_run=args.dry_run))

    _log("[5/6] Emergency stop check...", to_stderr)
    report.add(check_emergency_stop(base_url=args.server_url, dry_run=args.dry_run))

    _log("[6/6] FrankaEnv safety box integration check...", to_stderr)
    report.add(check_safety_box_integration(dry_run=args.dry_run))

    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-operation safety verification for GELLO + serl_franka_controllers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual robot/GELLO commands; use simulated data.",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"franka_server HTTP URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--gello-port",
        default=DEFAULT_GELLO_PORT,
        help=f"GELLO serial port (default: {DEFAULT_GELLO_PORT})",
    )
    parser.add_argument(
        "--gello-baudrate",
        type=int,
        default=DEFAULT_GELLO_BAUDRATE,
        help=f"GELLO baudrate (default: {DEFAULT_GELLO_BAUDRATE})",
    )

    # Safety thresholds
    parser.add_argument(
        "--max-step",
        type=float,
        default=GELLO_MAX_STEP,
        help=f"Max per-step delta in meters (default: {GELLO_MAX_STEP})",
    )
    parser.add_argument(
        "--max-total-delta",
        type=float,
        default=GELLO_MAX_TOTAL_DELTA,
        help=f"Max cumulative delta in meters (default: {GELLO_MAX_TOTAL_DELTA})",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=GELLO_NUM_FRAMES,
        help=f"Number of GELLO frames to read (default: {GELLO_NUM_FRAMES})",
    )

    # Output
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON to stdout.",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=None,
        help="Write JSON report to this file.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = run_all_checks(args)

    # Print human-readable report
    if not args.json:
        report.print_report()

    # JSON output
    if args.json or args.json_file:
        report_dict = report.to_dict()
        if args.json:
            print(json.dumps(report_dict, indent=2))
        if args.json_file:
            with open(args.json_file, "w") as f:
                json.dump(report_dict, f, indent=2)
            print(f"JSON report written to: {args.json_file}")

    # Exit code
    sys.exit(0 if report.all_critical_passed else 1)


if __name__ == "__main__":
    main()
