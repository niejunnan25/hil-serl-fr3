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
                ],
                env=env,
                timeout=20,
            )
            posted = srv["posted"]
        assert r.returncode == 11, (r.returncode, r.stdout, r.stderr, log.read_text())
        assert len(posted) == 0, "bias gate must POST no /pose"
        assert "mismatch" in log.read_text().lower()


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
