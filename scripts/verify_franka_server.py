#!/usr/bin/env python3
"""
verify_franka_server.py – Smoke-test every franka_server endpoint.

Connects to the SERL franka_server (Flask HTTP bridge) and verifies:
  • Read endpoints return correct JSON keys / types
  • Command endpoints are reachable and accept payloads
  • Health check responds

Usage:
    python scripts/verify_franka_server.py [--url http://127.0.0.2:5000/]

Exit code 0 = all tests pass, 1 = at least one failure.
"""
from __future__ import annotations

import argparse
import json
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
TIMEOUT_S = 5.0

# Expected key → expected type (list or scalar) for each read endpoint.
# /getstate returns everything; the others return one key each.
READ_ENDPOINTS: dict[str, dict[str, type | tuple[type, ...]]] = {
    "/getstate": {
        "pose": list,       # len 7 (xyz + quaternion)
        "vel": list,        # len 6
        "force": list,      # len 3
        "torque": list,     # len 3
        "q": list,          # len 7
        "dq": list,         # len 7
        "jacobian": list,   # 6×7 matrix
        "gripper_pos": (int, float),
    },
    "/getpos": {"pose": list},
    "/getvel": {"vel": list},
    "/getforce": {"force": list},
    "/gettorque": {"torque": list},
    "/getq": {"q": list},
    "/getdq": {"dq": list},
    "/getjacobian": {"jacobian": list},
    "/get_gripper": {"gripper": (int, float)},
}

# Command endpoints – we send a minimal / safe payload and just check the
# server doesn't 404 or 500.  On the real robot these return 200 + ack JSON;
# on the mock server they return 409 (NO_MOTION_MOCK_REJECTED) which is also
# a valid "server is alive" signal.
COMMAND_ENDPOINTS: dict[str, dict[str, Any]] = {
    "/pose":           {"pose": [0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]},
    "/open_gripper":   {},
    "/close_gripper":  {},
    "/jointreset":     {},
    "/activate_gripper": {},
    "/reset_gripper":  {},
    "/move_gripper":   {"gripper_pos": 0.04},
    "/startimp":       {},
    "/stopimp":        {},
    "/clearerr":       {},
    "/set_load":       {"load": 0.0},
    "/update_param": {
        "translational_stiffness": 2500,
        "translational_damping": 100,
        "rotational_stiffness": 200,
        "rotational_damping": 10,
        "translational_Ki": 0,
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
        "rotational_Ki": 0,
    },
}

# ── helpers ────────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"
SKIP = "\033[36mSKIP\033[0m"


class Result:
    """Accumulate pass/fail counts."""

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

    def skip(self, label: str, msg: str = "") -> None:
        self.details.append(f"  [{SKIP}] {label}" + (f" – {msg}" if msg else ""))

    def summary(self) -> int:
        print()
        print("=" * 60)
        print(f"  PASSED: {self.passed}   FAILED: {self.failed}   WARNINGS: {self.warnings}")
        print("=" * 60)
        return 0 if self.failed == 0 else 1


def post(url: str, endpoint: str, payload: dict | None = None) -> requests.Response:
    """POST to endpoint, return Response (may raise on connection error)."""
    return requests.post(
        url.rstrip("/") + endpoint,
        json=payload or {},
        timeout=TIMEOUT_S,
        headers={"Content-Type": "application/json"},
    )


def get(url: str, endpoint: str) -> requests.Response:
    return requests.get(url.rstrip("/") + endpoint, timeout=TIMEOUT_S)


# ── tests ──────────────────────────────────────────────────────────────────────

def test_healthz(base_url: str, r: Result) -> None:
    """GET /healthz – basic liveness probe."""
    label = "GET /healthz"
    try:
        resp = get(base_url, "/healthz")
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok") is True:
                r.ok(label, f"ok={data['ok']}, mode={data.get('mode', '?')}")
            else:
                r.fail(label, f"'ok' not True: {data}")
        elif resp.status_code == 404:
            # Some builds don't expose /healthz – treat as warning, not failure
            r.warn(label, "endpoint not found (404) – server may not expose /healthz")
        else:
            r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
    except requests.ConnectionError:
        r.fail(label, "connection refused – is the server running?")
    except Exception as exc:
        r.fail(label, str(exc))


def test_read_endpoints(base_url: str, r: Result) -> None:
    """Validate every read endpoint returns the expected JSON keys & types."""
    for endpoint, expected_keys in READ_ENDPOINTS.items():
        label = f"POST {endpoint}"
        try:
            resp = post(base_url, endpoint)
            if resp.status_code != 200:
                r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()

            # Check all expected keys exist
            missing = [k for k in expected_keys if k not in data]
            if missing:
                r.fail(label, f"missing keys: {missing}")
                continue

            # Type checks
            type_ok = True
            for key, expected_type in expected_keys.items():
                val = data[key]
                if expected_type is list:
                    if not isinstance(val, list):
                        r.fail(label, f"'{key}' expected list, got {type(val).__name__}")
                        type_ok = False
                        break
                elif isinstance(expected_type, tuple):
                    if not isinstance(val, expected_type):
                        r.fail(label, f"'{key}' expected {expected_type}, got {type(val).__name__}")
                        type_ok = False
                        break
            if not type_ok:
                continue

            # Length checks for known-size vectors
            if endpoint in ("/getstate", "/getpos") and "pose" in data:
                if len(data["pose"]) != 7:
                    r.fail(label, f"pose length {len(data['pose'])}, expected 7")
                    continue
            if endpoint in ("/getstate", "/getq", "/getdq") and "q" in data:
                if len(data["q"]) != 7:
                    r.fail(label, f"q length {len(data['q'])}, expected 7")
                    continue
            if endpoint == "/getjacobian":
                j = data["jacobian"]
                if not (len(j) == 6 and all(len(row) == 7 for row in j)):
                    j_rows = len(j)
                    j_cols = len(j[0]) if j else "?"
                    r.fail(label, f"jacobian shape {j_rows}x{j_cols}, expected 6x7")
                    continue

            r.ok(label, f"keys={list(data.keys())}")
        except requests.ConnectionError:
            r.fail(label, "connection refused")
        except json.JSONDecodeError:
            r.fail(label, "response is not valid JSON")
        except Exception as exc:
            r.fail(label, str(exc))


def test_command_endpoints(base_url: str, r: Result) -> None:
    """
    Send each command endpoint with its payload.

    Accepts:
      • 200 (real robot ack)
      • 409 (mock server no-motion reject – still proves the endpoint exists)
    Rejects:
      • 404 (endpoint not registered)
      • 500 (server error)
    """
    for endpoint, payload in COMMAND_ENDPOINTS.items():
        label = f"POST {endpoint}"
        try:
            resp = post(base_url, endpoint, payload)
            if resp.status_code in (200, 409):
                # 409 = mock reject, still means the endpoint is registered
                detail = ""
                try:
                    body = resp.json()
                    if resp.status_code == 409:
                        detail = f"(mock reject: {body.get('error', '?')})"
                    else:
                        detail = f"ack={body}"
                except Exception:
                    pass
                r.ok(label, detail)
            elif resp.status_code == 404:
                r.fail(label, "endpoint not found (404)")
            else:
                r.fail(label, f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.ConnectionError:
            r.fail(label, "connection refused")
        except Exception as exc:
            r.fail(label, str(exc))


def test_unknown_endpoint(base_url: str, r: Result) -> None:
    """Verify the server rejects an unknown endpoint with 404."""
    label = "POST /nonexistent_endpoint (expect 404)"
    try:
        resp = post(base_url, "/nonexistent_endpoint")
        if resp.status_code == 404:
            r.ok(label, "correctly returned 404")
        else:
            r.warn(label, f"expected 404, got HTTP {resp.status_code}")
    except requests.ConnectionError:
        r.fail(label, "connection refused")
    except Exception as exc:
        r.fail(label, str(exc))


def test_latency(base_url: str, r: Result) -> None:
    """Round-trip latency check on /getstate."""
    label = "latency /getstate"
    samples = 5
    times_ms: list[float] = []
    try:
        for _ in range(samples):
            t0 = time.perf_counter()
            resp = post(base_url, "/getstate")
            elapsed = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                r.fail(label, f"HTTP {resp.status_code} during latency test")
                return
            times_ms.append(elapsed)
        avg = sum(times_ms) / len(times_ms)
        p95 = sorted(times_ms)[int(len(times_ms) * 0.9)]
        if avg < 50:
            r.ok(label, f"avg={avg:.1f}ms  p95≈{p95:.1f}ms")
        elif avg < 200:
            r.warn(label, f"avg={avg:.1f}ms  p95≈{p95:.1f}ms (high but acceptable)")
        else:
            r.fail(label, f"avg={avg:.1f}ms  p95≈{p95:.1f}ms (too slow)")
    except requests.ConnectionError:
        r.fail(label, "connection refused")
    except Exception as exc:
        r.fail(label, str(exc))


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify franka_server endpoints")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Base URL of franka_server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--skip-commands",
        action="store_true",
        help="Skip command endpoints (read-only verification)",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip latency benchmark",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"\n{'=' * 60}")
    print(f"  franka_server endpoint verification")
    print(f"  Target: {base_url}")
    print(f"{'=' * 60}\n")

    r = Result()

    # Phase 1: health check
    print("[Phase 1] Health check")
    test_healthz(base_url, r)

    # Phase 2: read endpoints
    print("[Phase 2] Read endpoints")
    test_read_endpoints(base_url, r)

    # Phase 3: command endpoints (dry-run)
    if not args.skip_commands:
        print("[Phase 3] Command endpoints (dry-run)")
        test_command_endpoints(base_url, r)
    else:
        r.skip("command endpoints", "skipped by --skip-commands")

    # Phase 4: error handling
    print("[Phase 4] Error handling")
    test_unknown_endpoint(base_url, r)

    # Phase 5: latency
    if not args.skip_latency:
        print("[Phase 5] Latency")
        test_latency(base_url, r)
    else:
        r.skip("latency", "skipped by --skip-latency")

    # Print all results
    print()
    for line in r.details:
        print(line)

    exit_code = r.summary()
    print(f"\n  Exit code: {exit_code}\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
