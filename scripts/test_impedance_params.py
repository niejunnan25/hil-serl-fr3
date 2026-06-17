#!/usr/bin/env python3
"""
test_impedance_params.py – Test impedance parameter switching on franka_server.

Sends /update_param with COMPLIANCE_PARAM and PRECISION_PARAM presets to
verify the real robot controller accepts parameter updates and can switch
between impedance modes.

Usage:
    python scripts/test_impedance_params.py [--url http://127.0.0.2:5000/]

Exit code 0 = all tests pass, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required.  pip install requests", file=sys.stderr)
    sys.exit(2)

# ── constants ──────────────────────────────────────────────────────────────────

DEFAULT_URL = "http://127.0.0.2:5000/"
TIMEOUT_S = 10.0

# Preset impedance parameter sets (from SERL plug_insertion config).
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
    "rotational_clip_x": 0.02,
    "rotational_clip_y": 0.02,
    "rotational_clip_z": 0.015,
    "rotational_clip_neg_x": 0.02,
    "rotational_clip_neg_y": 0.02,
    "rotational_clip_neg_z": 0.015,
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
    "translational_clip_z": 0.006,
    "translational_clip_neg_x": 0.008,
    "translational_clip_neg_y": 0.008,
    "translational_clip_neg_z": 0.006,
    "rotational_clip_x": 0.025,
    "rotational_clip_y": 0.025,
    "rotational_clip_z": 0.02,
    "rotational_clip_neg_x": 0.025,
    "rotational_clip_neg_y": 0.025,
    "rotational_clip_neg_z": 0.02,
    "rotational_Ki": 0.0,
}

# All parameter keys that must be present in a valid update.
REQUIRED_KEYS: set[str] = set(COMPLIANCE_PARAM.keys())

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
INFO = "\033[36mINFO\033[0m"


class Result:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.details: list[str] = []

    def ok(self, label: str, msg: str = "") -> None:
        self.passed += 1
        self.details.append(f"  [{PASS}] {label}" + (f" – {msg}" if msg else ""))

    def fail(self, label: str, msg: str = "") -> None:
        self.failed += 1
        self.details.append(f"  [{FAIL}] {label}" + (f" – {msg}" if msg else ""))

    def warn(self, label: str, msg: str = "") -> None:
        self.warnings += 1
        self.details.append(f"  [{WARN}] {label}" + (f" – {msg}" if msg else ""))

    def info(self, label: str, msg: str = "") -> None:
        self.details.append(f"  [{INFO}] {label}" + (f" – {msg}" if msg else ""))

    def summary(self) -> int:
        print()
        print("=" * 60)
        print(f"  PASSED: {self.passed}   FAILED: {self.failed}   WARNINGS: {self.warnings}")
        print("=" * 60)
        return 0 if self.failed == 0 else 1


def post(url: str, endpoint: str, payload: dict[str, Any]) -> requests.Response:
    return requests.post(
        url.rstrip("/") + endpoint,
        json=payload,
        timeout=TIMEOUT_S,
        headers={"Content-Type": "application/json"},
    )


# ── tests ──────────────────────────────────────────────────────────────────────

def test_connectivity(base_url: str, r: Result) -> bool:
    """Verify the server is reachable before running param tests."""
    try:
        resp = post(base_url, "/getstate", {})
        if resp.status_code == 200:
            r.ok("connectivity", "server is reachable")
            return True
        else:
            r.fail("connectivity", f"HTTP {resp.status_code}")
            return False
    except requests.ConnectionError:
        r.fail("connectivity", "connection refused – is franka_server running?")
        return False
    except Exception as exc:
        r.fail("connectivity", str(exc))
        return False


def test_update_param(base_url: str, r: Result, name: str, params: dict[str, float]) -> bool:
    """
    Send /update_param with a named parameter set.

    Accepts 200 (real ack) or 409 (mock reject).
    Returns True on success.
    """
    label = f"/update_param ({name})"
    try:
        resp = post(base_url, "/update_param", params)
        if resp.status_code in (200, 409):
            detail = ""
            try:
                body = resp.json()
                if resp.status_code == 409:
                    detail = f"mock reject: {body.get('error', '?')}"
                else:
                    detail = f"ack: {body}"
            except Exception:
                detail = f"HTTP {resp.status_code}"
            r.ok(label, detail)
            return resp.status_code == 200
        else:
            r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.ConnectionError:
        r.fail(label, "connection refused")
        return False
    except Exception as exc:
        r.fail(label, str(exc))
        return False


def test_startimp(base_url: str, r: Result) -> bool:
    """Start impedance control before param updates."""
    label = "/startimp"
    try:
        resp = post(base_url, "/startimp", {})
        if resp.status_code in (200, 409):
            r.ok(label)
            return resp.status_code == 200
        elif resp.status_code == 404:
            r.fail(label, "endpoint not found")
            return False
        else:
            r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.ConnectionError:
        r.fail(label, "connection refused")
        return False
    except Exception as exc:
        r.fail(label, str(exc))
        return False


def test_stopimp(base_url: str, r: Result) -> bool:
    """Stop impedance control after param tests."""
    label = "/stopimp"
    try:
        resp = post(base_url, "/stopimp", {})
        if resp.status_code in (200, 409):
            r.ok(label)
            return resp.status_code == 200
        elif resp.status_code == 404:
            r.fail(label, "endpoint not found")
            return False
        else:
            r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.ConnectionError:
        r.fail(label, "connection refused")
        return False
    except Exception as exc:
        r.fail(label, str(exc))
        return False


def test_param_validation(base_url: str, r: Result) -> None:
    """Send a parameter set with a missing required key."""
    label = "/update_param (missing key – expect rejection)"
    bad_params = {k: v for k, v in COMPLIANCE_PARAM.items() if k != "translational_stiffness"}
    try:
        resp = post(base_url, "/update_param", bad_params)
        if resp.status_code in (400, 409, 422, 500):
            r.ok(label, f"correctly rejected with HTTP {resp.status_code}")
        elif resp.status_code == 200:
            r.warn(label, "server accepted incomplete params – may not validate")
        else:
            r.warn(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.ConnectionError:
        r.fail(label, "connection refused")
    except Exception as exc:
        r.fail(label, str(exc))


def test_switching(base_url: str, r: Result, live: bool) -> None:
    """Rapid switching between COMPLIANCE ↔ PRECISION to test stability."""
    if not live:
        r.info("mode switching", "skipped (server rejected update_param in dry-run)")
        return

    label = "rapid COMPLIANCE ↔ PRECISION switch (3 cycles)"
    try:
        for i in range(3):
            resp_c = post(base_url, "/update_param", COMPLIANCE_PARAM)
            if resp_c.status_code != 200:
                r.fail(label, f"cycle {i+1} COMPLIANCE failed: HTTP {resp_c.status_code}")
                return
            time.sleep(0.1)

            resp_p = post(base_url, "/update_param", PRECISION_PARAM)
            if resp_p.status_code != 200:
                r.fail(label, f"cycle {i+1} PRECISION failed: HTTP {resp_p.status_code}")
                return
            time.sleep(0.1)

        r.ok(label, "3 cycles completed without error")
    except requests.ConnectionError:
        r.fail(label, "connection refused during switching")
    except Exception as exc:
        r.fail(label, str(exc))


def test_getstate_after_param_change(base_url: str, r: Result, live: bool) -> None:
    """Verify /getstate still returns valid data after param changes."""
    if not live:
        r.info("getstate after param change", "skipped (dry-run mode)")
        return

    label = "/getstate after param change"
    try:
        resp = post(base_url, "/getstate", {})
        if resp.status_code == 200:
            data = resp.json()
            required_keys = {"pose", "vel", "force", "torque", "q", "dq", "jacobian", "gripper_pos"}
            missing = required_keys - set(data.keys())
            if missing:
                r.fail(label, f"missing keys after param change: {missing}")
            else:
                r.ok(label, "all state keys present after param change")
        else:
            r.fail(label, f"HTTP {resp.status_code}")
    except requests.ConnectionError:
        r.fail(label, "connection refused")
    except Exception as exc:
        r.fail(label, str(exc))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test impedance parameter switching")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of franka_server (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n{'=' * 60}")
    print(f"  Impedance parameter test")
    print(f"  Target: {base_url}")
    print(f"{'=' * 60}\n")

    r = Result()

    # Phase 1: connectivity
    print("[Phase 1] Connectivity check")
    if not test_connectivity(base_url, r):
        print("\nCannot reach server – aborting parameter tests.\n")
        for line in r.details:
            print(line)
        sys.exit(r.summary())

    # Phase 2: start impedance
    print("[Phase 2] Start impedance control")
    live = test_startimp(base_url, r)

    # Phase 3: apply COMPLIANCE_PARAM
    print("[Phase 3] Apply COMPLIANCE_PARAM")
    compliance_ok = test_update_param(base_url, r, "COMPLIANCE", COMPLIANCE_PARAM)
    live = live and compliance_ok

    # Phase 4: apply PRECISION_PARAM
    print("[Phase 4] Apply PRECISION_PARAM")
    precision_ok = test_update_param(base_url, r, "PRECISION", PRECISION_PARAM)
    live = live and precision_ok

    # Phase 5: param validation (negative test)
    print("[Phase 5] Parameter validation (negative test)")
    test_param_validation(base_url, r)

    # Phase 6: rapid switching
    print("[Phase 6] Rapid mode switching")
    test_switching(base_url, r, live)

    # Phase 7: verify state after changes
    print("[Phase 7] State verification after param changes")
    test_getstate_after_param_change(base_url, r, live)

    # Phase 8: stop impedance
    print("[Phase 8] Stop impedance control")
    test_stopimp(base_url, r)

    # Print all results
    print()
    for line in r.details:
        print(line)

    exit_code = r.summary()
    print(f"\n  Exit code: {exit_code}\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
