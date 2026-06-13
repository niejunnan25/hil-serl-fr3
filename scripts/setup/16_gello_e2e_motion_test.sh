#!/usr/bin/env bash
# =============================================================================
# 16_gello_e2e_motion_test.sh — P2-T3 protected end-to-end motion scaffold
#
# P2-T3: GELLO -> GelloCartesianDeltaAgent -> franka_server (HTTP) -> FR3
#
# This script is a *gated* scaffold for the live end-to-end motion test.
# It is split into four modes, all of which require explicit operator
# approval via the FR3_GELLO_E2E_APPROVAL env var. The default behavior
# is REFUSAL: exit 30 with usage.
#
#   Modes
#   -----
#   1. preflight (no FR3, no GELLO)
#         Verify host tools, libfranka wrappers, residual-process check,
#         approval guard logic, franka_server reachability (read-only).
#         Does NOT connect to GELLO, does NOT send motion commands.
#         Exit 0 on success.
#
#   2. dry-run-gello (no FR3, no GELLO hardware)
#         Import + dry-run GelloCartesianDeltaAgent (synthetic joints),
#         confirm FK -> delta -> normalize pipeline produces a 7D
#         action in [-1, 1] and zero /log/ on a no-motion call.
#         Does NOT touch /dev/ttyUSB0.
#         Exit 0 on success.
#
#   3. franka-server-probe (read-only, no motion)
#         GET /getstate / /getpos / /healthz, latency sample. 200 or
#         409 (mock no-motion reject) are acceptable; 5xx is not.
#         Does NOT call /pose or any motion endpoint.
#         Exit 0 on success.
#
#   4. motion (FULL E2E — APPROVED, currently FAIL-CLOSED)
#         GELLO read -> FK -> Cartesian delta -> normalize -> (would) POST /pose.
#         Invokes the driver with --mode full. The driver currently
#         FAILS CLOSED (rc 10, no /pose issued): the normalized delta ->
#         absolute /pose conversion is pending Phase B server-contract
#         verification (REVIEW-PhaseA C2). Until then this mode exits 32
#         without sending any motion command.
#         Requires FR3_GELLO_E2E_APPROVAL set to the approval phrase.
#         Logs to artifacts/logs/p2-t3-e2e-motion-<ts>.log.
#
# Approval phrase:  FR3_GELLO_E2E_APPROVAL=I_APPROVE_P2T3_FULL_E2E_MOTION
#
# Exit codes
#   0  success (per mode)
#   2  usage / argument error
#   3  host preflight failure (kernel, wrappers, residual procs)
#   4  missing binary / Python import
#   5  approval env var missing or wrong
#   6  franka_server unreachable / unhealthy
#   7  GELLO dry-run failed
#   8  safety violation (max_step / max_total_delta breach, etc.)
#   9  motion command rejected by franka_server
#   30 default refusal (no mode argument)
#   31 motion stream aborted by safety guard
#   99 unexpected internal error
#
# Environment overrides (all optional):
#   ROBOT_IP            (default 172.16.0.2)
#   FRANKA_SERVER_URL   (default http://127.0.0.1:5000/)
#   GELLO_PORT          (default /dev/ttyUSB0)
#   HZ                  (default 10)
#   DURATION            (default 5 seconds)
#   LOG_DIR             (default ${ROOT}/artifacts/logs)
# =============================================================================
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="${ROOT}/scripts"
TESTS_DIR="${ROOT}/tests"
LOG_DIR="${LOG_DIR:-${ROOT}/artifacts/logs}"
ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
FRANKA_SERVER_URL="${FRANKA_SERVER_URL:-http://127.0.0.1:5000/}"
GELLO_PORT="${GELLO_PORT:-/dev/ttyUSB0}"
HZ="${HZ:-10}"
DURATION="${DURATION:-5}"

# Default Python: prefer the conda env that has gym + gello + numpy
# (currently isaaclab). Override via env var PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-/home/robot/miniconda3/envs/isaaclab/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="/home/robot/miniconda3/envs/hilserl-fr3/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

APPROVAL_PHRASE="I_APPROVE_P2T3_FULL_E2E_MOTION"
APPROVAL_VAR="FR3_GELLO_E2E_APPROVAL"

# ─── Coloring ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# ─── Default refusal (no mode argument) ──────────────────────────────────────
refuse_default() {
    cat <<USAGE
P2-T3 end-to-end motion scaffold (GELLO -> FRANKA_SERVER -> FR3)

Default behavior: REFUSAL (exit 30). This is intentional: motion to the
real FR3 is a hard human-gated action and must be approved explicitly.

Usage:
  $0 preflight               host checks only, no FR3 / GELLO
  $0 dry-run-gello           synthetic joints through the agent, no /dev/ttyUSB0
  $0 franka-server-probe     read-only HTTP probe (no /pose), no GELLO
  $0 motion                  FULL E2E motion stream (requires approval)

Approval (only required for 'motion'):
  export ${APPROVAL_VAR}="${APPROVAL_PHRASE}"

Environment:
  ROBOT_IP=${ROBOT_IP}
  FRANKA_SERVER_URL=${FRANKA_SERVER_URL}
  GELLO_PORT=${GELLO_PORT}
  HZ=${HZ}
  DURATION=${DURATION}
  LOG_DIR=${LOG_DIR}

Exit codes: see top of script.
USAGE
    exit 30
}

# ─── Approval guard (all four modes are gated) ───────────────────────────────
require_approval() {
    local mode="$1"
    if [[ "${!APPROVAL_VAR:-}" != "${APPROVAL_PHRASE}" ]]; then
        err "Missing explicit approval for mode '${mode}'."
        info "Set: ${APPROVAL_VAR}=${APPROVAL_PHRASE}"
        info "(see HANDOFF-PHASE1.md P2-T3 — default refusal is exit 30)"
        exit 5
    fi
    log "Approval phrase present for mode '${mode}'"
}

# ─── Preflight (mode 1) ─────────────────────────────────────────────────────
preflight() {
    require_approval "preflight"

    echo "=============================================="
    echo " P2-T3 e2e motion — preflight"
    echo "=============================================="

    # 1. Host must be the laptop RT runtime
    local kernel realtime
    kernel="$(uname -a)"
    realtime="$(cat /sys/kernel/realtime 2>/dev/null || echo missing)"
    info "Kernel: ${kernel}"
    info "Realtime flag: ${realtime}"
    if [[ "${kernel}" != *"PREEMPT_RT"* ]]; then
        err "Expected PREEMPT_RT kernel (5.9.1-rt20)"
        exit 3
    fi
    if [[ "${realtime}" != "1" ]]; then
        err "Expected /sys/kernel/realtime == 1"
        exit 3
    fi
    log "PREEMPT_RT runtime verified"

    # 2. libfranka wrappers must block direct motion binaries
    for cmd in communication_test echo_robot_state; do
        local path
        path="$(command -v "${cmd}" || true)"
        if [[ "${path}" != "/usr/local/bin/${cmd}" ]]; then
            err "${cmd} must resolve to /usr/local/bin wrapper (got: ${path:-missing})"
            exit 3
        fi
        set +e
        "${cmd}" --help >/dev/null 2>/tmp/p2t3-wrapper-test.err
        local rc=$?
        set -e
        if [[ "${rc}" -ne 126 ]] || ! grep -q "BLOCKED: '${cmd}'" /tmp/p2t3-wrapper-test.err; then
            err "${cmd} wrapper is not blocking (rc=${rc})"
            exit 3
        fi
    done
    log "libfranka wrappers block direct motion binaries"

    # 3. No residual control / motion processes
    local pattern='communication_test|echo_robot_state|joint_|cartesian_|franka_server|roslaunch|controller_manager|moveit|serl|gello'
    local matches
    matches="$(pgrep -af "${pattern}" 2>/dev/null | grep -v "pgrep -af" | grep -v "$0" || true)"
    if [[ -n "${matches}" ]]; then
        err "Residual control-related processes found:"
        echo "${matches}"
        exit 3
    fi
    log "No residual control-related processes"

    # 4. Required scripts present
    for f in gello_cartesian_delta_agent.py fk_converter.py normalize_action.py \
             verify_franka_server.py gello_intervention.py; do
        if [[ ! -f "${SCRIPTS_DIR}/${f}" ]]; then
            err "Missing required script: ${f}"
            exit 4
        fi
    done
    log "Required scripts present"

    # 5. Robot reachable (best-effort, warns if not)
    if ping -c 1 -W 2 "${ROBOT_IP}" &>/dev/null; then
        log "Robot reachable at ${ROBOT_IP}"
    else
        warn "Robot not reachable at ${ROBOT_IP} — server may fail to connect"
    fi

    log "Preflight passed. No FR3 connection was opened."
}

# ─── Dry-run GELLO agent (mode 2) ───────────────────────────────────────────
dry_run_gello() {
    require_approval "dry-run-gello"

    echo "=============================================="
    echo " P2-T3 e2e motion — dry-run GELLO agent"
    echo "=============================================="

    # We must NOT touch GELLO hardware. The agent's --dry-run path
    # uses a synthetic joint trajectory (see gello_cartesian_delta_agent.py).
    info "Running agent --dry-run (no GELLO hardware open)"
    set +e
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/gello_cartesian_delta_agent.py" \
        --dry-run --duration 0.5 --hz 20
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        err "Agent --dry-run exited with ${rc}"
        exit 7
    fi
    log "GELLO agent dry-run passed (no /dev/ttyUSB0 opened)"
}

# ─── franka_server read-only probe (mode 3) ─────────────────────────────────
franka_server_probe() {
    require_approval "franka-server-probe"

    echo "=============================================="
    echo " P2-T3 e2e motion — franka_server probe"
    echo "=============================================="

    info "Probing ${FRANKA_SERVER_URL} (read-only)"
    if [[ ! -f "${SCRIPTS_DIR}/verify_franka_server.py" ]]; then
        err "verify_franka_server.py not found"
        exit 4
    fi

    set +e
    python3 "${SCRIPTS_DIR}/verify_franka_server.py" \
        --url "${FRANKA_SERVER_URL}" \
        --skip-commands
    local rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
        err "franka_server probe failed (rc=${rc})"
        exit 6
    fi
    log "franka_server probe passed (no /pose or motion endpoint hit)"
}

# ─── FULL E2E motion (mode 4) — APPROVED ────────────────────────────────────
motion() {
    require_approval "motion"

    echo "=============================================="
    echo " P2-T3 e2e motion — FULL STREAM (FAIL-CLOSED)"
    echo "  GELLO -> FK -> delta -> normalize -> (would) POST /pose"
    echo "  NOTE: driver --mode full is fail-closed pending Phase B"
    echo "        (REVIEW-PhaseA C2); no /pose will be issued (exit 32)."
    echo "  ROBOT=${ROBOT_IP}  SERVER=${FRANKA_SERVER_URL}"
    echo "  HZ=${HZ}  DURATION=${DURATION}s"
    echo "=============================================="

    mkdir -p "${LOG_DIR}"
    local ts log_file
    ts="$(date +%Y%m%dT%H%M%S)"
    log_file="${LOG_DIR}/p2-t3-e2e-motion-${ts}.log"
    info "Log: ${log_file}"

    # 1. Re-verify the server is alive before we send anything
    if ! curl -sS -o /dev/null -w "%{http_code}" \
            --connect-timeout 3 --max-time 5 \
            "${FRANKA_SERVER_URL%/}/healthz" | grep -qE '^(200|404)$'; then
        err "franka_server not reachable at ${FRANKA_SERVER_URL}/healthz"
        exit 6
    fi

    # 2. Run the GELLO motion stream. The Python driver:
    #      - opens GELLO at ${GELLO_PORT}
    #      - reads joints at ${HZ}
    #      - feeds them through GelloCartesianDeltaAgent
    #      - POSTs the resulting 7D action to ${FRANKA_SERVER_URL}/pose
    #      - aborts on safety violation (max_step / max_total_delta)
    #    The driver writes a one-line summary per tick to ${log_file}
    #    and exits 0 on graceful completion, 8 on safety violation.
    local driver="${SCRIPTS_DIR}/p2_t3_e2e_motion_driver.py"
    if [[ ! -f "${driver}" ]]; then
        err "Motion driver not found: ${driver}"
        err "Run 'python3 scripts/p2_t3_e2e_motion_driver.py --emit-driver' to scaffold it,"
        err "or implement it as the GELLO->server bridge (see HANDOFF-PHASE1.md P2-T3)."
        exit 4
    fi

    set +e
    "${PYTHON_BIN}" "${driver}" \
        --mode full \
        --server "${FRANKA_SERVER_URL}" \
        --robot-ip "${ROBOT_IP}" \
        --gello-port "${GELLO_PORT}" \
        --hz "${HZ}" \
        --duration "${DURATION}" \
        --log "${log_file}"
    local rc=$?
    set -e

    case "${rc}" in
        0)
            log "Motion stream completed cleanly. Inspect ${log_file}."
            ;;
        8)
            err "Safety violation aborted the stream (see ${log_file})."
            exit 31
            ;;
        9)
            err "franka_server rejected a /pose command (see ${log_file})."
            exit 9
            ;;
        10)
            err "FULL E2E disabled: GELLO->follower /pose conversion pending"
            err "Phase B server-contract verification (REVIEW-PhaseA C2);"
            err "no /pose issued. Implement + verify the pose reconstruction"
            err "in Phase B (B1) before this mode can stream. See ${log_file}."
            exit 32
            ;;
        *)
            err "Motion stream exited with unexpected rc=${rc} (see ${log_file})."
            exit "${rc}"
            ;;
    esac
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    local cmd="${1:-}"
    if [[ -z "${cmd}" ]]; then
        refuse_default
    fi
    case "${cmd}" in
        preflight)              preflight ;;
        dry-run-gello)          dry_run_gello ;;
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
