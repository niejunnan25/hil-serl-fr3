#!/usr/bin/env bash
# FR3 Phase 2 readiness gate.
# Runs the four pytest suites that cover P2 sub-tasks and reports
# pass/fail counts as evidence. Motion approval is gated behind
# FR3_GELLO_E2E_APPROVAL=1 because it requires real robot motion.
set -euo pipefail

# Repository root resolution. Priority:
#  1. ${REPO_ROOT} env var (caller override)
#  2. ${SCRIPT_DIR}/../..  (laptop layout: scripts/setup/*.sh)
#  3. ${SCRIPT_DIR}/..      (desktop layout: /home/robot/*.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_ROOT:-}" ]]; then
    ROOT="${REPO_ROOT}"
elif [[ -f "${SCRIPT_DIR}/../../.planning/STATE.md" ]]; then
    ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

RUN_GELLO_CONTRACT="${RUN_GELLO_CONTRACT:-1}"
RUN_GELLO_SAFETY="${RUN_GELLO_SAFETY:-1}"
RUN_GELLO_DEMOS="${RUN_GELLO_DEMOS:-1}"
RUN_GELLO_E2E_SCAFFOLD="${RUN_GELLO_E2E_SCAFFOLD:-1}"
RUN_MOTION_APPROVAL="${RUN_MOTION_APPROVAL:-0}"

EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/fr3-phase2-readiness}"
# mkdir only if the path looks sane (avoid /home/robot/... running as user
# other than the desktop runtime owner).
if ! mkdir -p "$EVIDENCE_DIR" 2>/dev/null; then
    EVIDENCE_DIR="/tmp/fr3-phase2-readiness"
    mkdir -p "$EVIDENCE_DIR"
fi
mkdir -p "$EVIDENCE_DIR"

# Desktop runtime: prefer the isaaclab env python (has gym + gello
# deps). Fall back to system python3. Override via env var PYTEST=...
PYTHON_BIN="${PYTHON_BIN:-/home/robot/miniconda3/envs/isaaclab/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="/home/robot/miniconda3/envs/hilserl-fr3/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
# Need real $ROOT expanded now. Run pytest with PYTHONPATH including
# both the project root (for top-level scripts/imports) and the scripts
# directory (for legacy module-style imports). Override via env var.
PYTEST="${PYTEST:-${PYTHON_BIN} -m pytest}"

# Run pytest from the project root so test_path globs resolve.
cd "$ROOT"

log() { printf '[INFO] %s\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
fail() { printf '[ERR] %s\n' "$*" >&2; }

run_pytest_suite() {
    local name="$1"
    # test_path is intentionally a single string here — we feed it to
    # `python -m pytest <test_path>` so multi-file suites must be passed as
    # a single space-separated list and we split it back to multiple args.
    local test_path="$2"
    local log_file="$EVIDENCE_DIR/${name}.log"
    local rc=0

    log "running $name suite: $test_path"
    # Disable -e around the pytest invocation so a non-zero exit does not
    # abort the gate; we want to record the failure and continue.
    set +e
    # shellcheck disable=SC2086
    PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}" $PYTEST $test_path -q --tb=short >"$log_file" 2>&1
    rc=$?
    set -e

    # pytest summary line example: "=== 3 passed, 1 failed in 0.42s ==="
    # tolerate leading "===" and trailing "in <duration>".
    local summary_line
    summary_line="$(grep -E 'passed|failed' "$log_file" | grep -E 'in [0-9]+(\.[0-9]+)?s' | tail -n1 || true)"
    if [[ -z "$summary_line" ]]; then
        summary_line="$(grep -E 'passed|failed' "$log_file" | tail -n1 || true)"
    fi
    local passed
    local failed
    passed="$(echo "$summary_line" | grep -oE '[0-9]+ passed' | awk '{print $1}' | tail -n1 || true)"
    failed="$(echo "$summary_line" | grep -oE '[0-9]+ failed' | awk '{print $1}' | tail -n1 || true)"
    : "${passed:=0}"
    : "${failed:=0}"

    if (( rc == 0 )); then
        ok "$name passed (passed=$passed failed=$failed) -> $log_file"
        echo "PASS,$name,$passed,$failed,$log_file" >>"$EVIDENCE_DIR/summary.csv"
        return 0
    fi

    fail "$name failed (passed=$passed failed=$failed) rc=$rc -> $log_file"
    echo "FAIL,$name,$passed,$failed,$log_file" >>"$EVIDENCE_DIR/summary.csv"
    return 1
}

main() {
    local failures=0
    local total_pass=0
    local total_fail=0

    log "FR3 Phase 2 readiness gate"
    log "This gate runs pytest suites only — no real robot motion, no GELLO hardware required."

    : >"$EVIDENCE_DIR/summary.csv"
    echo "status,suite,passed,failed,log" >>"$EVIDENCE_DIR/summary.csv"

    if [[ "$RUN_GELLO_CONTRACT" == "1" ]]; then
        if ! run_pytest_suite "gello_intervention_contract" \
                "tests/test_gello_intervention_contract.py tests/test_gello_cartesian_delta_agent.py"; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_GELLO_CONTRACT=0; skipping GELLO contract suites"
    fi

    if [[ "$RUN_GELLO_SAFETY" == "1" ]]; then
        if ! run_pytest_suite "gello_safety_calibration" \
                "tests/test_gello_safety_calibration.py"; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_GELLO_SAFETY=0; skipping GELLO safety calibration suite"
    fi

    if [[ "$RUN_GELLO_DEMOS" == "1" ]]; then
        if ! run_pytest_suite "gello_demos_serl" \
                "tests/test_record_gello_demos_serl.py"; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_GELLO_DEMOS=0; skipping GELLO demo recording suite"
    fi

    if [[ "$RUN_GELLO_E2E_SCAFFOLD" == "1" ]]; then
        if ! run_pytest_suite "gello_e2e_scaffolding" \
                "tests/test_gello_e2e_scaffolding.py"; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_GELLO_E2E_SCAFFOLD=0; skipping E2E scaffolding suite"
    fi

    if [[ "$RUN_MOTION_APPROVAL" == "1" ]]; then
        if [[ "${FR3_GELLO_E2E_APPROVAL:-0}" != "1" ]]; then
            fail "FR3_GELLO_E2E_APPROVAL is not set; refusing to run P2-T3 motion approval gate"
            failures=$((failures + 1))
        else
            log "FR3_GELLO_E2E_APPROVAL=1; P2-T3 motion approval gate would run here (not implemented in this script)"
        fi
    else
        log "RUN_MOTION_APPROVAL=0; skipping P2-T3 motion approval (set FR3_GELLO_E2E_APPROVAL=1 to enable)"
    fi

    if [[ -s "$EVIDENCE_DIR/summary.csv" ]]; then
        total_pass="$(awk -F, 'NR>1 && $1=="PASS" {p+=$3} END {print p+0}' "$EVIDENCE_DIR/summary.csv")"
        total_fail="$(awk -F, 'NR>1 && $1=="FAIL" {f+=$4} END {print f+0}' "$EVIDENCE_DIR/summary.csv")"
    fi

    log "summary: passed=$total_pass failed=$total_fail gate_failures=$failures"
    log "evidence: $EVIDENCE_DIR"

    if (( failures == 0 )); then
        ok "FR3 Phase 2 readiness gate passed"
        exit 0
    fi

    fail "FR3 Phase 2 readiness gate failed with $failures failed suite(s)"
    exit 1
}

main "$@"
