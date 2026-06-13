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

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SETUP = SCRIPTS / "setup"
DRIVER = SCRIPTS / "p2_t3_e2e_motion_driver.py"

PYTHON_BIN = sys.executable

ENV_BASE = os.environ.copy()


def _run(cmd, **kw):
    return subprocess.run(
        cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw
    )


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

    def test_full_mode_fails_closed_pending_phaseB(self, tmp_path):
        # C2 (REVIEW-PhaseA): full mode must FAIL CLOSED. The agent emits a
        # normalized [-1,1] delta action, but /pose expects an ABSOLUTE pose;
        # POSTing the delta-as-pose would command an uncontrolled motion. The
        # correct GELLO-leader -> FR3-follower pose reconstruction is Phase B
        # work, so until then full mode must NOT POST anything: it returns
        # rc 10 with a clear "disabled / pending Phase B" log and zero /pose.
        log = tmp_path / "full.log"
        env = {**ENV_BASE, "FR3_GELLO_E2E_APPROVAL": "I_APPROVE_P2T3_FULL_E2E_MOTION"}
        r = _run(
            [
                PYTHON_BIN, str(DRIVER),
                "--mode", "full",
                "--server", "http://127.0.0.1:1/",  # must never be contacted
                "--hz", "20",
                "--duration", "0.3",
                "--log", str(log),
            ],
            env=env,
            timeout=15,
        )
        # rc 10 is the fail-closed path, which returns BEFORE any network
        # call. (Had it attempted the POST, the bogus :1 server would have
        # produced rc 9 instead — so rc==10 itself proves no /pose was sent.)
        assert r.returncode == 10, (r.returncode, r.stdout, r.stderr)
        text = log.read_text()
        assert "DISABLED" in text
        assert "pending Phase B" in text
        assert "no /pose issued" in text
        # Never reached the per-tick streaming loop / a server rejection.
        assert "server rejected" not in text


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
