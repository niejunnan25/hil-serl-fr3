#!/usr/bin/env bash
# =============================================================================
# 18_hybrid_switch_e2e_test.sh — Hybrid GELLO->Xbox switch e2e scaffold (A4)
#
# Mirrors 17_xbox_e2e_motion_test.sh but for the GELLO->Xbox switch
# (record_hybrid_demos logic in real motion). Four modes, approval
# gated behind FR3_HYBRID_E2E_APPROVAL. Default REFUSAL (exit 30).
#
# Approval phrase:  FR3_HYBRID_E2E_APPROVAL=I_APPROVE_HYBRID_FULL_E2E_MOTION
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="${ROOT}/scripts"
LOG_DIR="${LOG_DIR:-${ROOT}/artifacts/logs}"
FRANKA_SERVER_URL="${FRANKA_SERVER_URL:-http://127.0.0.1:5000/}"
HZ="${HZ:-10}"
DURATION="${DURATION:-5}"

PYTHON_BIN="${PYTHON_BIN:-/home/robot/miniconda3/envs/isaaclab/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="/home/robot/miniconda3/envs/hilserl-fr3/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

APPROVAL_PHRASE="I_APPROVE_HYBRID_FULL_E2E_MOTION"
APPROVAL_VAR="FR3_HYBRID_E2E_APPROVAL"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

refuse_default() {
    cat <<USAGE
Hybrid switch e2e scaffold (GELLO_FOLLOW --RB--> XBOX_DELTA -> /pose)

Default behavior: REFUSAL (exit 30).

Usage:
  $0 preflight                host checks only
  $0 dry-run                  record_hybrid_demos mock + arbiter truth table
  $0 franka-server-probe      read-only HTTP probe
  $0 motion                   FULL hybrid e2e (requires approval)

Approval (motion only):
  export ${APPROVAL_VAR}="${APPROVAL_PHRASE}"

Environment:
  FRANKA_SERVER_URL=${FRANKA_SERVER_URL}
  HZ=${HZ}
  DURATION=${DURATION}
  LOG_DIR=${LOG_DIR}

Exit codes: 0 ok, 2 usage, 3 preflight, 4 missing binary,
            5 approval missing, 6 server unreachable, 7 dry-run fail,
            8 safety violation, 9 server reject, 30 default refuse,
            31 motion aborted by safety, 99 internal.
USAGE
    exit 30
}

require_approval() {
    local mode="$1"
    if [[ "${!APPROVAL_VAR:-}" != "${APPROVAL_PHRASE}" ]]; then
        err "Missing approval for mode '${mode}'."
        info "Set: ${APPROVAL_VAR}=${APPROVAL_PHRASE}"
        exit 5
    fi
    log "Approval phrase present for mode '${mode}'"
}

preflight() {
    require_approval "preflight"
    echo "=============================================="
    echo " Hybrid e2e — preflight"
    echo "=============================================="
    for f in record_hybrid_demos.py teleop_arbiter.py gello_intervention.py \
             xbox_intervention.py verify_franka_server.py; do
        if [[ ! -f "${SCRIPTS_DIR}/${f}" ]]; then
            err "Missing required script: ${f}"
            exit 4
        fi
    done
    log "Required scripts present"
}

dry_run() {
    require_approval "dry-run"
    echo "=============================================="
    echo " Hybrid e2e — dry-run"
    echo "=============================================="

    info "Running record_hybrid_demos + teleop_arbiter test suites (mock)"
    set +e
    "${PYTHON_BIN}" -m pytest \
        "${ROOT}/tests/test_record_hybrid_demos.py" \
        "${ROOT}/tests/test_teleop_arbiter.py" \
        -q 2>&1 | tail -5
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        err "Hybrid dry-run failed (rc=${rc})"
        exit 7
    fi
    log "Hybrid dry-run passed (mock-only, no real device / no /pose)"
}

franka_server_probe() {
    require_approval "franka-server-probe"
    echo "=============================================="
    echo " Hybrid e2e — franka_server probe"
    echo "=============================================="

    info "Probing ${FRANKA_SERVER_URL} (read-only)"
    if [[ ! -f "${SCRIPTS_DIR}/verify_franka_server.py" ]]; then
        err "verify_franka_server.py not found"
        exit 4
    fi
    set +e
    python3 "${SCRIPTS_DIR}/verify_franka_server.py" \
        --url "${FRANKA_SERVER_URL}" --skip-commands
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        err "franka_server probe failed (rc=${rc})"
        exit 6
    fi
    log "franka_server probe passed (no /pose hit)"
}

motion() {
    require_approval "motion"
    echo "=============================================="
    echo " Hybrid e2e — FULL STREAM"
    echo "  (Phase A: scaffold only; --full is NOT exercised in A)"
    echo "=============================================="
    warn "Operator must be present + E-stop armed."
    mkdir -p "${LOG_DIR}"
    local ts log_file
    ts="$(date +%Y%m%dT%H%M%S)"
    log_file="${LOG_DIR}/hybrid-e2e-motion-${ts}.log"
    info "Log: ${log_file}"
    info "Phase A does not run --full motion. Wire record_hybrid_demos +"
    info "TeleopArbiter into a live bridge in Phase B; see A4 review notes."
    log "Scaffold accepted; no /pose issued (Phase A boundary)."
}

main() {
    local cmd="${1:-}"
    if [[ -z "${cmd}" ]]; then
        refuse_default
    fi
    case "${cmd}" in
        preflight)              preflight ;;
        dry-run)                dry_run ;;
        franka-server-probe)    franka_server_probe ;;
        motion)                 motion ;;
        -h|--help|help)         refuse_default ;;
        *)
            err "Unknown mode: ${cmd}"
            refuse_default
            ;;
    esac
}

main "$@"
