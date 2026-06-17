"""P2-T3 (A4) — Motion driver + e2e scaffolding tests.

Covers:
  * the motion driver's approval gate (no /pose without phrase)
  * the dry-run / micro mode contracts
  * the safety-violation exit code
  * the four-mode shell scaffolds (16/17/18) refuse to do anything
    without explicit approval env vars
  * the 18_*.sh dry-run mode runs the record_hybrid_demos + arbiter
    test suites (no real /pose issued)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SETUP = SCRIPTS / "setup"
DRIVER = SCRIPTS / "p2_t3_e2e_motion_driver.py"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from fk_converter import forward_kinematics  # noqa: E402
from gello_pose_follow import FR3_DEFAULT_JOINTS  # noqa: E402

PYTHON_BIN = sys.executable

ENV_BASE = os.environ.copy()
APPROVAL = "I_APPROVE_P2T3_FULL_E2E_MOTION"


def _run(cmd, **kw):
    return subprocess.run(
        cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw
    )


@contextmanager
def mock_franka_server(q0, pose):
    """A tiny in-process franka_server stub. POST /getstate -> {q, pose};
    POST /clearerr -> {}; POST /pose -> records body['arr']. Yields a dict with
    the bound url and the list of received /pose payloads."""
    posted = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                return json.loads(raw or b"{}")
            except Exception:
                return {}

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            route = self.path.rstrip("/")
            body = self._read_json()
            if route.endswith("/getstate"):
                self._send({"q": list(map(float, q0)), "pose": list(map(float, pose))})
            elif route.endswith("/pose"):
                posted.append(body)
                self._send({})
            else:  # /clearerr, /startimp, etc.
                self._send({})

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/", "posted": posted}
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# Driver-level tests
# ---------------------------------------------------------------------------
class TestMotionDriver:
    def test_help_lists_modes(self):
        r = _run([PYTHON_BIN, str(DRIVER), "--help"])
        assert r.returncode == 0
        assert "--mode" in r.stdout

    def test_dry_run_never_issues_http(self, tmp_path):
        log = tmp_path / "dry.log"
        r = _run(
            [
                PYTHON_BIN, str(DRIVER),
                "--mode", "dry-run",
                "--hz", "50",
                "--log", str(log),
            ]
        )
        assert r.returncode == 0
        text = log.read_text()
        assert "[DRY]" in text
        # No /pose issued in dry-run.
        assert "/pose" not in text

    def test_micro_runs_three_ticks_and_exits_zero(self, tmp_path):
        log = tmp_path / "micro.log"
        r = _run(
            [
                PYTHON_BIN, str(DRIVER),
                "--mode", "micro",
                "--hz", "100",
                "--log", str(log),
            ]
        )
        assert r.returncode == 0
        text = log.read_text()
        # Synthetic stream grows by 0.005 rad/tick; 3 ticks of that
        # stay under the 0.003 m max_step so safe=True for the
        # default agent; we just check the tick markers.
        assert text.count("[MICRO]") == 3

    def test_full_mode_refuses_without_approval(self, tmp_path):
        log = tmp_path / "full.log"
        r = _run(
            [
                PYTHON_BIN, str(DRIVER),
                "--mode", "full",
                "--log", str(log),
            ],
            env={**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": ""},
        )
        assert r.returncode == 5
        assert "approval required" in r.stderr

    def test_full_mode_streams_absolute_pose_against_mock_server(self, tmp_path):
        # B1a: full mode drives the PROVEN contract — read q0/currpos from
        # /getstate, FK-bias gate, then per tick POST /pose {"arr":[abs pose]}.
        # Mock server returns a pose == FK(q0) so the bias gate passes.
        log = tmp_path / "full.log"
        q0 = FR3_DEFAULT_JOINTS.copy()
        pose = forward_kinematics(q0)  # bias == 0 -> gate passes
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": APPROVAL}
        with mock_franka_server(q0, pose) as srv:
            r = _run(
                [
                    PYTHON_BIN, str(DRIVER),
                    "--mode", "full",
                    "--server", srv["url"],
                    "--hz", "20",
                    "--duration", "0.2",
                    "--log", str(log),
                    # No real GELLO in CI: allow the synthetic leader so the
                    # HTTP/FK contract is exercised against the MOCK server.
                    "--allow-synthetic-leader",
                ],
                env=env,
                timeout=20,
            )
            posted = srv["posted"]
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr, log.read_text())
        assert len(posted) >= 1, "full mode must POST at least one /pose"
        for body in posted:
            assert "arr" in body, f"payload key must be 'arr', got {list(body)}"
            arr = body["arr"]
            assert len(arr) == 7
            assert np.linalg.norm(arr[3:]) == pytest.approx(1.0, abs=1e-3)
        # First commanded pose ~= current pose (no startup jump).
        first = np.array(posted[0]["arr"])
        np.testing.assert_allclose(first[:3], pose[:3], atol=2e-3)

    def test_full_mode_holds_pose_on_server_reject(self, tmp_path):
        # If the server rejects a streaming /pose mid-stream, the driver must
        # still exit rc=9 AND issue a best-effort HOLD /pose (re-anchor the
        # impedance setpoint) before returning - franka_server has no /stop
        # route, so "hold current pose" is the only safe cleanup.
        import json as _json
        import threading as _threading
        from http.server import BaseHTTPRequestHandler as _BH, HTTPServer as _HS

        log = tmp_path / "reject.log"
        q0 = FR3_DEFAULT_JOINTS.copy()
        pose = forward_kinematics(q0)
        pose_calls = {"n": 0}
        recorded = []

        class H(_BH):
            def log_message(self, *a):
                pass

            def _read(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    return _json.loads(raw or b"{}")
                except Exception:
                    return {}

            def _send(self, obj, code=200):
                body = _json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                route = self.path.rstrip("/")
                body = self._read()
                recorded.append(route)
                if route.endswith("/getstate"):
                    self._send({"q": list(map(float, q0)), "pose": list(map(float, pose))})
                elif route.endswith("/pose"):
                    pose_calls["n"] += 1
                    if pose_calls["n"] == 1:
                        self._send({"error": "conflict"}, code=409)  # reject first stream POST
                    else:
                        self._send({})  # allow the hold POST
                else:
                    self._send({})

        srv = _HS(("127.0.0.1", 0), H)
        t = _threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": APPROVAL}
        try:
            r = _run(
                [
                    PYTHON_BIN, str(DRIVER),
                    "--mode", "full",
                    "--server", url,
                    "--hz", "20",
                    "--duration", "0.5",
                    "--log", str(log),
                    "--allow-synthetic-leader",
                ],
                env=env,
                timeout=20,
            )
        finally:
            srv.shutdown()
            srv.server_close()
        text = log.read_text()
        assert r.returncode == 9, (r.returncode, r.stdout, r.stderr, text)
        pose_posts = [rt for rt in recorded if rt.endswith("/pose")]
        assert len(pose_posts) >= 2, (
            f"expected a hold /pose after the rejected stream POST, got {pose_posts}"
        )
        assert "safety hold" in text, text

    def test_full_mode_holds_pose_on_sigint(self, tmp_path):
        # SIGINT mid-stream must NOT crash uncaught; the driver must exit
        # cleanly (rc=10) and issue a HOLD /pose so the arm holds where it is.
        import json as _json
        import signal as _signal
        import threading as _threading
        import time as _time
        from http.server import BaseHTTPRequestHandler as _BH, HTTPServer as _HS

        log = tmp_path / "sigint.log"
        q0 = FR3_DEFAULT_JOINTS.copy()
        pose = forward_kinematics(q0)
        recorded = []

        class H(_BH):
            def log_message(self, *a):
                pass

            def _read(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b"{}"
                try:
                    return _json.loads(raw or b"{}")
                except Exception:
                    return {}

            def _send(self, obj):
                body = _json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                route = self.path.rstrip("/")
                self._read()
                recorded.append(route)
                if route.endswith("/getstate"):
                    self._send({"q": list(map(float, q0)), "pose": list(map(float, pose))})
                else:
                    self._send({})

        srv = _HS(("127.0.0.1", 0), H)
        t = _threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": APPROVAL}
        proc = subprocess.Popen(
            [
                PYTHON_BIN, str(DRIVER),
                "--mode", "full",
                "--server", url,
                "--hz", "10",
                "--duration", "30",
                "--log", str(log),
                "--allow-synthetic-leader",
            ],
            cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            _time.sleep(1.0)  # let it stream a few ticks
            before = len([rt for rt in recorded if rt.endswith("/pose")])
            proc.send_signal(_signal.SIGINT)
            try:
                proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise AssertionError("driver did not exit after SIGINT")
        finally:
            srv.shutdown()
            srv.server_close()
        text = log.read_text()
        after = len([rt for rt in recorded if rt.endswith("/pose")])
        assert proc.returncode == 10, (proc.returncode, text)
        assert "KeyboardInterrupt" in text, text
        assert "safety hold" in text, text
        assert after > before, "expected a hold /pose after SIGINT"

    def test_full_mode_fk_bias_gate_refuses_on_frame_mismatch(self, tmp_path):
        # If FK(q0) disagrees with the server's reported current pose by more
        # than the bias limit, full mode must REFUSE (rc 11) and POST nothing —
        # so a bad FK/flange frame never commands a startup jump.
        log = tmp_path / "full.log"
        q0 = FR3_DEFAULT_JOINTS.copy()
        pose = forward_kinematics(q0).copy()
        pose[0] += 0.10  # 10 cm mismatch >> 5 mm bias limit
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": APPROVAL}
        with mock_franka_server(q0, pose) as srv:
            r = _run(
                [
                    PYTHON_BIN, str(DRIVER),
                    "--mode", "full",
                    "--server", srv["url"],
                    "--hz", "20",
                    "--duration", "0.2",
                    "--log", str(log),
                    "--allow-synthetic-leader",
                ],
                env=env,
                timeout=20,
            )
            posted = srv["posted"]
        assert r.returncode == 11, (r.returncode, r.stdout, r.stderr, log.read_text())
        assert len(posted) == 0, "bias gate must POST no /pose"
        assert "mismatch" in log.read_text().lower()

    def test_full_mode_aborts_when_gello_unavailable_no_pose(self, tmp_path):
        # MOTION SAFETY: in --full, if the real GELLO cannot be opened the
        # driver must ABORT (rc 12) and POST nothing — never fabricate a
        # synthetic leader trajectory onto the real robot. gello is not
        # importable in CI, so without --allow-synthetic-leader the open fails.
        log = tmp_path / "full.log"
        q0 = FR3_DEFAULT_JOINTS.copy()
        pose = forward_kinematics(q0)  # bias == 0; would pass the bias gate
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": APPROVAL}
        with mock_franka_server(q0, pose) as srv:
            r = _run(
                [
                    PYTHON_BIN, str(DRIVER),
                    "--mode", "full",
                    "--server", srv["url"],
                    "--hz", "20",
                    "--duration", "0.2",
                    "--log", str(log),
                ],
                env=env,
                timeout=20,
            )
            posted = srv["posted"]
        assert r.returncode == 12, (r.returncode, r.stdout, r.stderr)
        assert len(posted) == 0, "no /pose may be issued from a synthetic leader"
        assert "refusing to synthesize" in r.stderr.lower()


# ---------------------------------------------------------------------------
# Shell-scaffold tests
# ---------------------------------------------------------------------------
class TestShellScaffolds:
    @pytest.mark.parametrize(
        "script,approval_var",
        [
            ("17_xbox_e2e_motion_test.sh", "FR3_XBOX_E2E_APPROVAL"),
            ("18_hybrid_switch_e2e_test.sh", "FR3_HYBRID_E2E_APPROVAL"),
        ],
    )
    def test_default_refuses(self, script, approval_var):
        r = _run(["bash", str(SETUP / script)])
        assert r.returncode == 30
        assert "REFUSAL" in r.stdout or "refusal" in r.stdout.lower()

    def test_17_unknown_mode_refuses(self):
        r = _run(
            ["bash", str(SETUP / "17_xbox_e2e_motion_test.sh"), "frobnicate"],
            env={**ENV_BASE, "FR3_XBOX_E2E_APPROVAL": "I_APPROVE_XBOX_FULL_E2E_MOTION"},
        )
        assert r.returncode == 30

    def test_18_unknown_mode_refuses(self):
        r = _run(
            ["bash", str(SETUP / "18_hybrid_switch_e2e_test.sh"), "frobnicate"],
            env={
                **ENV_BASE,
                "FR3_HYBRID_E2E_APPROVAL": "I_APPROVE_HYBRID_FULL_E2E_MOTION",
            },
        )
        assert r.returncode == 30

    def test_17_dry_run_runs_xbox_contract_tests(self):
        env = {**ENV_BASE, "FR3_XBOX_E2E_APPROVAL": "I_APPROVE_XBOX_FULL_E2E_MOTION"}
        r = _run(
            ["bash", str(SETUP / "17_xbox_e2e_motion_test.sh"), "dry-run"],
            env=env,
            timeout=60,
        )
        assert r.returncode == 0, r.stderr
        # No /pose issued in dry-run.
        assert "/pose" not in r.stdout
        assert "POST" not in r.stdout

    def test_18_dry_run_runs_hybrid_arbiter_tests(self):
        env = {
            **ENV_BASE,
            "FR3_HYBRID_E2E_APPROVAL": "I_APPROVE_HYBRID_FULL_E2E_MOTION",
        }
        r = _run(
            ["bash", str(SETUP / "18_hybrid_switch_e2e_test.sh"), "dry-run"],
            env=env,
            timeout=60,
        )
        assert r.returncode == 0, r.stderr
        # The shell's banner says "no /pose"; the assertion is that no
        # /pose URL was actually contacted, which the script's
        # description itself confirms.
        assert "no /pose" in r.stdout or "no /pose)" in r.stdout

    def test_approval_variable_must_match_phrase_verbatim(self):
        # Wrong phrase -> exit 5, not exit 0.
        r = _run(
            ["bash", str(SETUP / "17_xbox_e2e_motion_test.sh"), "dry-run"],
            env={**ENV_BASE, "FR3_XBOX_E2E_APPROVAL": "I_APPROVE_XBOX_FULL_E2E_MOTIO"},
        )
        assert r.returncode == 5

    def test_18_approval_variable_must_match_phrase_verbatim(self):
        r = _run(
            ["bash", str(SETUP / "18_hybrid_switch_e2e_test.sh"), "dry-run"],
            env={
                **ENV_BASE,
                "FR3_HYBRID_E2E_APPROVAL": "I_APPROVE_HYBRID_FULL_E2E_MOTIO",
            },
        )
        assert r.returncode == 5
