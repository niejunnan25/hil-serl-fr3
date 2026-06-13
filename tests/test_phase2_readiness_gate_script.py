from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup" / "17_phase2_readiness_gate.sh"

EXPECTED_PYTEST_TARGETS = (
    "tests/test_gello_intervention_contract.py",
    "tests/test_gello_cartesian_delta_agent.py",
    "tests/test_gello_safety_calibration.py",
    "tests/test_record_gello_demos_serl.py",
    "tests/test_gello_e2e_scaffolding.py",
)

EXPECTED_SUITE_LABELS = (
    "gello_intervention_contract",
    "gello_safety_calibration",
    "gello_demos_serl",
    "gello_e2e_scaffolding",
)


def test_phase2_readiness_gate_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_phase2_readiness_gate_invokes_all_four_pytest_suites() -> None:
    text = SCRIPT.read_text()

    for target in EXPECTED_PYTEST_TARGETS:
        assert target in text, f"missing pytest target in script: {target}"

    for label in EXPECTED_SUITE_LABELS:
        assert label in text, f"missing suite label in script: {label}"


def test_phase2_readiness_gate_does_not_call_motion_or_live_servers() -> None:
    text = SCRIPT.read_text()

    forbidden = (
        "franka_server.py",
        "echo_robot_state",
        "motion_with_control",
        "communication_test",
        "fr3-zed-camera-verify",
    )
    for token in forbidden:
        assert token not in text, f"phase2 gate must not reference live token: {token}"


def test_phase2_readiness_gate_motion_approval_is_opt_in() -> None:
    text = SCRIPT.read_text()

    assert "FR3_GELLO_E2E_APPROVAL" in text
    assert "RUN_MOTION_APPROVAL" in text
    # default skip
    assert 'RUN_MOTION_APPROVAL:-0' in text or 'RUN_MOTION_APPROVAL="${RUN_MOTION_APPROVAL:-0}"' in text


def test_phase2_readiness_gate_exits_zero_when_all_suites_skipped(tmp_path: Path) -> None:
    if shutil.which("python") is None:
        return

    evidence_dir = tmp_path / "evidence"

    env = os.environ.copy()
    env.update(
        {
            "RUN_GELLO_CONTRACT": "0",
            "RUN_GELLO_SAFETY": "0",
            "RUN_GELLO_DEMOS": "0",
            "RUN_GELLO_E2E_SCAFFOLD": "0",
            "RUN_MOTION_APPROVAL": "0",
            "EVIDENCE_DIR": str(evidence_dir),
            "PYTEST": "false",  # should not be invoked when all suites are skipped
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "FR3 Phase 2 readiness gate passed" in result.stdout
    assert (evidence_dir / "summary.csv").exists()


def test_phase2_readiness_gate_refuses_motion_without_approval() -> None:
    if shutil.which("python") is None:
        return

    env = os.environ.copy()
    env.update(
        {
            "RUN_GELLO_CONTRACT": "0",
            "RUN_GELLO_SAFETY": "0",
            "RUN_GELLO_DEMOS": "0",
            "RUN_GELLO_E2E_SCAFFOLD": "0",
            "RUN_MOTION_APPROVAL": "1",
            "FR3_GELLO_E2E_APPROVAL": "0",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "FR3_GELLO_E2E_APPROVAL" in result.stderr or "approval" in result.stderr.lower()


def test_phase2_readiness_gate_fails_when_pytest_target_missing(tmp_path: Path) -> None:
    if shutil.which("python") is None:
        return

    # A standalone python script that mimics a failing pytest run: prints
    # a "1 failed" line and exits 3. The gate script invokes PYTEST with a
    # path argument, so this script must consume sys.argv[1] gracefully.
    failing_py = tmp_path / "failing_pytest.py"
    failing_py.write_text(
        "import sys\n"
        "sys.stderr.write('1 failed in pytest_fake\\n')\n"
        "sys.exit(3)\n"
    )

    env = os.environ.copy()
    env.update(
        {
            "RUN_GELLO_CONTRACT": "1",
            "RUN_GELLO_SAFETY": "0",
            "RUN_GELLO_DEMOS": "0",
            "RUN_GELLO_E2E_SCAFFOLD": "0",
            "RUN_MOTION_APPROVAL": "0",
            "EVIDENCE_DIR": str(tmp_path / "evidence"),
            "PYTEST": f"{shutil.which('python')} {failing_py}",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    assert "gello_intervention_contract" in (result.stdout + result.stderr)
    assert (tmp_path / "evidence" / "summary.csv").exists()
