#!/usr/bin/env bash
# FR3 Phase 2 readiness gate.
# Mirrors the Phase 1 readiness gate shape but composes the Phase 2
# pytest suites plus an optional no-motion control preflight and an
# optional zktitan connectivity probe.
#
# By default this gate is no-motion, no-GELLO, no-FR3-FCI, and no live
# learner. All pytest suites are pure-mock and never instantiate
# DynamixelDriver, franka_server, or pyzed.sl.Camera.
#
# Sub-gates
# ---------
#   control no-motion  : /home/robot/fr3-phase1-control-no-motion-verify
#                        (Phase 1 control runtime contract — required for
#                         any Phase 2 motion later)
#   pytest suites      : scripts/setup/17_phase2_readiness_gate.sh
#                        (4 P2 sub-task pytest suites)
#   zktitan probe      : opt-in via ZKTITAN_PROBE=1; uses the fr3-desktop-ts
#                        SSH jump host (no Pulse VPN assumption baked in)
#   motion approval    : opt-in via FR3_GELLO_E2E_APPROVAL=1; only then
#                        will the script refuse with a clear error
#
# Skipping sub-gates
# ------------------
#   RUN_NO_MOTION=0   skip control no-motion
#   RUN_PYTEST=0      skip the 4-suite pytest composition
#   RUN_ZKTITAN=0     skip zktitan probe (default 1 — local ping only)
#   RUN_MOTION=0      skip motion-approval check (default 0)
#
# Exit codes
# ----------
#   0  all enabled sub-gates passed
#   2  usage / argument error
#   3  control no-motion gate failed
#   4  pytest suite sub-gate failed
#   5  zktitan probe failed (or env contract missing)
#   6  motion approval sub-gate triggered (refusal)
#   7  unexpected internal error
set -euo pipefail

NO_MOTION_GATE="${NO_MOTION_GATE:-/home/robot/fr3-phase1-control-no-motion-verify}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTEST_GATE_DEFAULT="$SCRIPT_DIR/17_phase2_readiness_gate.sh"
if [[ -x "$PYTEST_GATE_DEFAULT" ]]; then
    PYTEST_GATE="${PYTEST_GATE:-$PYTEST_GATE_DEFAULT}"
elif [[ -f "$PYTEST_GATE_DEFAULT" ]]; then
    PYTEST_GATE="${PYTEST_GATE:-$PYTEST_GATE_DEFAULT}"
else
    PYTEST_GATE="${PYTEST_GATE:-$PYTEST_GATE_DEFAULT}"
fi
ZKTITAN_PROBE="${ZKTITAN_PROBE:-$SCRIPT_DIR/19_zktitan_connectivity_probe.sh}"

JUMP_HOST="${JUMP_HOST:-fr3-desktop-ts}"
ZKTITAN_HOST="${ZKTITAN_HOST:-162.105.195.74}"
ZKTITAN_USER="${ZKTITAN_USER:-njn}"
ZKTITAN_SSH_CONNECT_TIMEOUT="${ZKTITAN_SSH_CONNECT_TIMEOUT:-8}"

RUN_NO_MOTION="${RUN_NO_MOTION:-1}"
RUN_PYTEST="${RUN_PYTEST:-1}"
RUN_ZKTITAN="${RUN_ZKTITAN:-1}"
RUN_MOTION="${RUN_MOTION:-0}"

log() {
    printf '[INFO] %s\n' "$*"
}

ok() {
    printf '[OK] %s\n' "$*"
}

fail() {
    printf '[ERR] %s\n' "$*" >&2
}

run_executable_gate() {
    local name="$1"
    local cmd="$2"

    if [[ ! -x "$cmd" ]]; then
        # Allow non-executable shell scripts to be sourced with bash.
        if [[ -f "$cmd" ]]; then
            log "running $name gate via /usr/bin/env bash: $cmd"
            if /usr/bin/env bash "$cmd"; then
                ok "$name gate passed"
                return 0
            fi
            fail "$name gate failed"
            return 1
        fi
        fail "$name gate missing or not executable: $cmd"
        return 1
    fi

    log "running $name gate: $cmd"
    if "$cmd"; then
        ok "$name gate passed"
        return 0
    fi

    fail "$name gate failed"
    return 1
}

run_pytest_gate() {
    local cmd="$PYTEST_GATE"
    if [[ ! -f "$cmd" ]]; then
        fail "phase2 pytest gate script missing: $cmd"
        return 1
    fi

    log "running phase2 pytest sub-gate: $cmd"
    EVIDENCE_DIR="${EVIDENCE_DIR:-/tmp/fr3-phase2-readiness}" \
        /usr/bin/env bash "$cmd"
}

run_zktitan_probe() {
    local probe="$ZKTITAN_PROBE"
    if [[ -x "$probe" || -f "$probe" ]]; then
        log "running zktitan probe (delegated): $probe"
        JUMP_HOST="$JUMP_HOST" \
        ZKTITAN_HOST="$ZKTITAN_HOST" \
        ZKTITAN_USER="$ZKTITAN_USER" \
        ZKTITAN_SSH_CONNECT_TIMEOUT="$ZKTITAN_SSH_CONNECT_TIMEOUT" \
        CHECK_NETWORK="${CHECK_NETWORK:-1}" \
            run_executable_gate "zktitan" "$probe" || return 1
        return 0
    fi

    # Fallback inline probe: mac → desktop (fr3-desktop-ts) → zktitan.
    # This is the same path documented in fzt-connection.md.
    log "running inline zktitan connectivity probe: $JUMP_HOST -> $ZKTITAN_HOST"

    if ! command -v ssh >/dev/null 2>&1; then
        fail "ssh not available; cannot run inline zktitan probe"
        return 1
    fi

    if ! ssh -o BatchMode=yes -o ConnectTimeout="$ZKTITAN_SSH_CONNECT_TIMEOUT" \
            "$JUMP_HOST" true 2>/dev/null; then
        fail "jump host '$JUMP_HOST' not reachable via SSH (BatchMode, ${ZKTITAN_SSH_CONNECT_TIMEOUT}s)"
        return 1
    fi
    ok "jump host '$JUMP_HOST' reachable"

    local probe_output
    if ! probe_output="$(ssh -o BatchMode=yes -o ConnectTimeout="$ZKTITAN_SSH_CONNECT_TIMEOUT" \
            "$JUMP_HOST" \
            "ssh -o BatchMode=yes -o ConnectTimeout=$ZKTITAN_SSH_CONNECT_TIMEOUT $ZKTITAN_HOST \
             'hostname; whoami; date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; \
              nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null; \
              df -h /nvme/$ZKTITAN_USER 2>/dev/null | tail -2'" 2>&1)"; then
        printf '%s\n' "$probe_output" >&2
        fail "zktitan SSH probe via $JUMP_HOST -> $ZKTITAN_HOST failed"
        return 1
    fi

    printf '%s\n' "$probe_output"
    # The zktitan SSH banner prints 'zktitan' (or whatever $HOSTNAME returns)
    # before the queried info, so we accept either the configured IP or
    # the configured hostname 'zktitan' (set up via /etc/hosts on
    # fr3-desktop-ts).
    if ! printf '%s' "$probe_output" | grep -Eq "^${ZKTITAN_HOST}\$|^zktitan\$"; then
        fail "zktitan hostname response did not match '$ZKTITAN_HOST' or 'zktitan'"
        return 1
    fi

    ok "zktitan probe passed (hostname=$ZKTITAN_HOST user=$ZKTITAN_USER)"
}

check_motion_approval() {
    # Default refusal: any motion path must be explicitly approved.
    if [[ "$RUN_MOTION" == "1" ]]; then
        if [[ "${FR3_GELLO_E2E_APPROVAL:-0}" != "1" ]]; then
            fail "RUN_MOTION=1 but FR3_GELLO_E2E_APPROVAL is not 1; refusing to allow motion approval sub-gate"
            return 1
        fi
        log "FR3_GELLO_E2E_APPROVAL=1 acknowledged; motion approval sub-gate is symbolic only on this script"
        return 0
    fi

    log "RUN_MOTION=0; skipping P2-T3 motion approval sub-gate"
    return 0
}

usage() {
    cat <<USAGE
FR3 Phase 2 readiness gate

Sub-gates (all default ON unless overridden):
  --skip-no-motion        do not run /home/robot/fr3-phase1-control-no-motion-verify
  --skip-pytest           do not run scripts/setup/17_phase2_readiness_gate.sh
  --skip-zktitan          do not run the zktitan connectivity probe
  --run-motion            also enforce FR3_GELLO_E2E_APPROVAL=1 (default off)

Environment:
  NO_MOTION_GATE          path to control no-motion verifier
  PYTEST_GATE             path to 17_phase2_readiness_gate.sh
  ZKTITAN_PROBE           path to dedicated zktitan probe (optional)
  JUMP_HOST               SSH jump host (default: fr3-desktop-ts)
  ZKTITAN_HOST            zktitan IP (default: 162.105.195.74)
  ZKTITAN_USER            zktitan user (default: njn)
  EVIDENCE_DIR            pytest evidence dir (default: /tmp/fr3-phase2-readiness)
  CHECK_NETWORK           forwarded to actor-learner preflight / zktitan probe
USAGE
}

parse_args() {
    while (( $# > 0 )); do
        case "$1" in
            --skip-no-motion)
                RUN_NO_MOTION=0
                shift
                ;;
            --skip-pytest)
                RUN_PYTEST=0
                shift
                ;;
            --skip-zktitan)
                RUN_ZKTITAN=0
                shift
                ;;
            --run-motion)
                RUN_MOTION=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                fail "unknown argument: $1"
                usage >&2
                exit 2
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    local failures=0

    log "FR3 Phase 2 readiness gate"
    log "Default safety: no FR3 motion, no GELLO hardware, no live learner, no ZED frame grab."

    if [[ "$RUN_NO_MOTION" == "1" ]]; then
        if ! run_executable_gate "control no-motion" "$NO_MOTION_GATE"; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_NO_MOTION=0; skipping control no-motion gate"
    fi

    if [[ "$RUN_PYTEST" == "1" ]]; then
        if ! run_pytest_gate; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_PYTEST=0; skipping phase2 pytest suite sub-gate"
    fi

    if [[ "$RUN_ZKTITAN" == "1" ]]; then
        if ! run_zktitan_probe; then
            failures=$((failures + 1))
        fi
    else
        log "RUN_ZKTITAN=0; skipping zktitan connectivity probe"
    fi

    if ! check_motion_approval; then
        failures=$((failures + 1))
    fi

    if (( failures == 0 )); then
        ok "FR3 Phase 2 readiness gate passed"
        exit 0
    fi

    fail "FR3 Phase 2 readiness gate failed with $failures failed sub-gate(s)"
    exit 1
}

main "$@"
