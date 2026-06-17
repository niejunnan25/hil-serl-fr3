"""ITEM 24: e-stop / emergency-stop verification test for verify_safety.py.

verify_safety.check_emergency_stop() exercises the "zero-action stop"
safety path: it reads the current TCP pose via POST /getstate, commands
that exact pose back via POST /pose (a zero-delta "hold" command), waits
briefly, re-reads the pose, and asserts the end-effector did not drift
more than 5 mm. On the real robot this proves the impedance controller
holds position when handed a no-motion command (the software analogue of
an emergency stop / safety hold).

This module unit-tests that path WITHOUT a real robot by spinning up an
in-process HTTP server that speaks the franka_server JSON contract
(POST /getstate, POST /pose with an {"arr": [...]} envelope) -- mirroring
the in-process HTTP mock in tests/test_gello_e2e_scaffolding.py and the
FakeFrankaServer stub in tests/test_record_gello_demos_serl.py. The
verify_franka_server.py probe is tested with --skip-commands; here the
e-stop check is intrinsically a command path, so we instead point it at
a mock server that honors the hold command and returns a non-drifting
pose.

Coverage:
  - dry_run=True            -> "skip" (no robot command issued)
  - stop-honoring server    -> "pass" (drift 0, /getstate + /pose hit,
                               /pose uses the 7D arr envelope)
  - drifting server         -> "warn" (drift exceeds 5 mm limit)
  - unreachable server      -> "fail" (connection error surfaced)
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

_spec = importlib.util.spec_from_file_location(
    "verify_safety", SCRIPTS / "verify_safety.py"
)
assert _spec and _spec.loader
vs = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = vs
_spec.loader.exec_module(vs)

requests = vs.requests

pytestmark = pytest.mark.skipif(
    requests is None,
    reason="'requests' not installed; verify_safety server checks unavailable",
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_handler(poses: list, recorded: list):
    """Build a BaseHTTPRequestHandler that:
      - on POST /getstate, returns the next pose from `poses` (clamped to
        the last entry once exhausted), so a 1-entry list models a robot
        that holds perfectly and a 2-entry list models drift;
      - on POST /pose, accepts the {"arr": [...]} hold command;
      - records (path, body) for every request.
    """
    state = {"i": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a, **k):  # silence stderr spam
            pass

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length).decode("utf-8", "ignore") if length else ""
            recorded.append((self.path, raw))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if self.path == "/getstate":
                idx = min(state["i"], len(poses) - 1)
                state["i"] += 1
                payload = json.dumps(
                    {
                        "pose": poses[idx],
                        "vel": [0.0] * 6,
                        "force": [0.0] * 3,
                        "torque": [0.0] * 3,
                        "q": [0.0] * 7,
                        "dq": [0.0] * 7,
                        "jacobian": [[0.0] * 7 for _ in range(6)],
                        "gripper_pos": 0.04,
                    }
                )
            elif self.path == "/pose":
                body = json.loads(raw or "{}")
                arr = body.get("arr")
                assert arr is not None and len(arr) == 7, f"bad /pose body: {raw!r}"
                payload = json.dumps({"ok": True})
            else:
                payload = json.dumps({"error": "no route"})
            self.wfile.write(payload.encode())

    return H


def _serve(poses: list, recorded: list) -> Iterator[str]:
    port = _free_port()
    server = http.server.HTTPServer(
        ("127.0.0.1", port), _make_handler(poses, recorded)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        thread.join(timeout=2)


@pytest.fixture
def mock_server(request) -> Iterator[tuple[str, list]]:
    """Yield (base_url, recorded_requests). The pose sequence is taken from
    the indirect param; default is a single non-drifting pose."""
    poses = getattr(request, "param", None) or [[0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]]
    recorded: list[tuple[str, str]] = []
    gen = _serve(poses, recorded)
    url = next(gen)
    try:
        yield url, recorded
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


class TestEmergencyStopCheck:
    def test_dry_run_skips_without_touching_server(self):
        # dry_run must NOT contact any server and must report "skip".
        result = vs.check_emergency_stop(base_url="http://127.0.0.1:1/", dry_run=True)
        assert result.status == "skip"
        assert result.details.get("mode") == "dry-run"

    def test_hold_command_passes_against_non_drifting_server(self, mock_server):
        url, recorded = mock_server
        result = vs.check_emergency_stop(base_url=url, dry_run=False)
        assert result.status == "pass", result
        # The check must read state, then command the same pose back.
        paths = [p for p, _ in recorded]
        assert "/getstate" in paths
        assert "/pose" in paths
        # The hold command must use the 7D {"arr": [...]} envelope.
        pose_bodies = [json.loads(b) for p, b in recorded if p == "/pose"]
        assert pose_bodies, "no /pose command was issued"
        assert "arr" in pose_bodies[0] and len(pose_bodies[0]["arr"]) == 7
        # Reported drift is within the 5 mm acceptance band.
        assert float(result.details["drift_m"]) <= float(result.details["max_drift_m"])

    @pytest.mark.parametrize(
        "mock_server",
        [[[0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0], [0.45, 0.0, 0.32, 0.0, 0.0, 0.0, 1.0]]],
        indirect=True,
    )
    def test_excessive_drift_warns(self, mock_server):
        # 2 cm of z-drift between the two /getstate reads exceeds the 5 mm
        # limit, so the check must downgrade to "warn" (not crash, not pass).
        url, _ = mock_server
        result = vs.check_emergency_stop(base_url=url, dry_run=False)
        assert result.status == "warn", result
        assert float(result.details["drift_m"]) > float(result.details["max_drift_m"])

    def test_unreachable_server_fails_gracefully(self):
        # Port 1 is not listening; the connection error must surface as a
        # "fail" CheckResult rather than an unhandled exception.
        result = vs.check_emergency_stop(base_url="http://127.0.0.1:1/", dry_run=False)
        assert result.status == "fail"
        assert "error" in result.details
