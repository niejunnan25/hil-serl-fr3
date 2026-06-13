#!/usr/bin/env bash
# =============================================================================
# 19_phase_a_readiness_gate.sh — v2.2.1 Phase A readiness gate
#
# Composes the 4 GELLO Phase 2 suites + the 5 new Phase A suites
# (teleop_hub, xbox_intervention, teleop_arbiter, record_hybrid_demos,
# motion_driver) plus the 17/18 e2e shell scaffolds (dry-run mode).
#
# Default behavior: no FR3 motion, no GELLO hardware, no live learner,
# no ZED. Every sub-gate is pure-mock. RC=0 means every test file ran
# to completion and every assertion passed.
#
# Sub-gates
# ---------
#   test_gello_intervention_contract
#   test_gello_cartesian_delta_agent
#   test_gello_safety_calibration
#   test_record_gello_demos_serl
#   test_gello_e2e_scaffolding                 (4 P2 GELLO suites)
#   test_teleop_hub                            (A1)
#   test_xbox_intervention_contract            (A1)
#   test_teleop_arbiter                        (A2)
#   test_gello_intervention_a2_contract        (A2)
#   test_record_hybrid_demos                   (A3)
#   test_p2t3_motion_driver                    (A4 — driver + 16/17/18 shells)
#
# Approval variables
# ------------------
#   None of the Phase A tests issue motion. Approval gating lives in
#   16/17/18_*.sh's "motion" mode; the gate here only exercises
#   "preflight" and "dry-run" modes.
#
# Exit codes
# ----------
#   0  all sub-gates green
#   1  at least one sub-gate failed
#   2  usage error
#
# Environment overrides
#   PYTEST              command to invoke pytest (default: ${PYTHON} -m pytest)
#   EVIDENCE_DIR        per-suite log directory (default: /tmp/fr3-phase-a-readiness)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/robot/miniconda3/envs/isaaclab/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="/home/robot/miniconda3/envs/hilserl-fr3/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
PYTEST="${PYTEST:-${PYTHON_BIN} -m pytest}"

EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/fr3-phase-a-readiness}"
mkdir -p "${EVIDENCE_DIR}"

# 4 GELLO Phase 2 suites (baseline, carried into Phase A)
GELLO_SUITES=(
    "tests/test_gello_intervention_contract.py"
    "tests/test_gello_cartesian_delta_agent.py"
    "tests/test_gello_safety_calibration.py"
    "tests/test_record_gello_demos_serl.py"
    "tests/test_gello_e2e_scaffolding.py"
)

# 5 new Phase A suites
A_SUITES=(
    "tests/test_teleop_hub.py"
    "tests/test_xbox_intervention_contract.py"
    "tests/test_teleop_arbiter.py"
    "tests/test_gello_intervention_a2_contract.py"
    "tests/test_record_hybrid_demos.py"
    "tests/test_p2t3_motion_driver.py"
)

SHELL_SCAFFOLDS=(
    "scripts/setup/17_xbox_e2e_motion_test.sh"
    "scripts/setup/18_hybrid_switch_e2e_test.sh"
)

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { printf '%b[INFO]%b %s\n' "${CYAN}" "${NC}" "$*"; }
ok()  { printf '%b[OK]%b   %s\n' "${GREEN}" "${NC}" "$*"; }
fail() { printf '%b[FAIL]%b %s\n' "${RED}" "${NC}" "$*"; }

usage() {
    cat <<USAGE
Phase A readiness gate (v2.2.1).

Usage:
  $0                run all sub-gates
  $0 --help         this message

Environment:
  PYTEST             pytest command (default: python -m pytest)
  EVIDENCE_DIR       per-suite log dir (default: /tmp/fr3-phase-a-readiness)
USAGE
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            -h|--help) usage; exit 0 ;;
            *) printf 'unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
        esac
        shift
    done
}

run_pytest_suite() {
    local rel="$1"
    local log_file="${EVIDENCE_DIR}/$(basename "${rel}" .py).log"
    log "running ${rel}"
    if (cd "${ROOT}" && ${PYTEST} "${rel}" -q) >"${log_file}" 2>&1; then
        ok "${rel} passed"
        return 0
    fi
    fail "${rel} failed (see ${log_file})"
    return 1
}

run_shell_scaffold() {
    local rel="$1"
    local name
    name="$(basename "${rel}" .sh)"
    local log_file="${EVIDENCE_DIR}/${name}.log"
    log "running ${rel} (dry-run)"
    # The dry-run modes of 17/18_*.sh do not issue motion; they
    # invoke the test suites directly.
    local env_var=""
    if [[ "${name}" == "17_xbox_e2e_motion_test" ]]; then
        env_var="FR3_XBOX_E2E_APPROVAL=I_APPROVE_XBOX_FULL_E2E_MOTION"
    elif [[ "${name}" == "18_hybrid_switch_e2e_test" ]]; then
        env_var="FR3_HYBRID_E2E_APPROVAL=I_APPROVE_HYBRID_FULL_E2E_MOTION"
    fi
    if env "${env_var}" /usr/bin/env bash "${ROOT}/${rel}" dry-run \
            >"${log_file}" 2>&1; then
        ok "${rel} dry-run passed"
        return 0
    fi
    fail "${rel} dry-run failed (see ${log_file})"
    return 1
}

main() {
    parse_args "$@"

    log "FR3 Phase A readiness gate (v2.2.1)"
    log "Default safety: no FR3 motion, no GELLO hardware, no live learner, no ZED."

    local failures=0
    local total_passed=0

    for suite in "${GELLO_SUITES[@]}" "${A_SUITES[@]}"; do
        if run_pytest_suite "${suite}"; then
            : # pass
        else
            failures=$((failures + 1))
        fi
    done

    for shell in "${SHELL_SCAFFOLDS[@]}"; do
        if ! run_shell_scaffold "${shell}"; then
            failures=$((failures + 1))
        fi
    done

    # Aggregate pass/fail counts from the per-suite logs.
    local pass=0
    for log_file in "${EVIDENCE_DIR}"/*.log; do
        [[ -f "${log_file}" ]] || continue
        # pytest -q output ends with "X passed" or similar; tolerate
        # the gating summary style.
        local p
        p="$(grep -oE '[0-9]+ passed' "${log_file}" | head -1 || true)"
        if [[ -n "${p}" ]]; then
            pass=$((pass + ${p% passed}))
        fi
    done

    echo
    log "summary: passed=${pass} failed=${failures}"
    log "evidence: ${EVIDENCE_DIR}"

    if [[ "${failures}" -eq 0 ]]; then
        ok "FR3 Phase A readiness gate passed"
        exit 0
    fi
    fail "FR3 Phase A readiness gate failed (${failures} sub-gate(s))"
    exit 1
}

main "$@"
