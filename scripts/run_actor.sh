#!/usr/bin/env bash
# =============================================================================
# run_actor.sh — Launch HIL-SERL Actor (robot side)
#
# Actor runs on fr3-desktop-ts, connects to the robot via franka_server,
# collects data, and sends transitions to the Learner via agentlace.
#
# Default mode:  Training actor with GelloIntervention (human-in-the-loop)
# Eval mode:     Inference-only evaluation of a checkpoint
#
# Usage:
#   bash scripts/run_actor.sh                          # training mode
#   bash scripts/run_actor.sh --eval_checkpoint_step 50000
#   bash scripts/run_actor.sh --eval_checkpoint_step 50000 --eval_n_trajs 20
#
# Target host: fr3-desktop-ts (Ubuntu 22.04)
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"
FRANKA_PORT="${SERL_FRANKA_PORT:-5000}"
EXPERIMENT="plug_insertion"

# agentlace network (learner address — must match run_learner.sh)
LEARNER_IP="${LEARNER_IP:-10.192.4.249}"     # zktitan
LEARNER_PORT="${LEARNER_PORT:-50051}"

# ─── Parse arguments ─────────────────────────────────────────────────────────
EVAL_MODE=false
EVAL_CHECKPOINT_STEP=""
EVAL_N_TRAJS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_checkpoint_step)
            EVAL_MODE=true
            EVAL_CHECKPOINT_STEP="$2"
            shift 2
            ;;
        --eval_n_trajs)
            EVAL_N_TRAJS="$2"
            shift 2
            ;;
        --learner_ip)
            LEARNER_IP="$2"
            shift 2
            ;;
        --learner_port)
            LEARNER_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eval_checkpoint_step N   Run evaluation at checkpoint step N"
            echo "  --eval_n_trajs N           Number of evaluation trajectories (default: 10)"
            echo "  --learner_ip IP            Learner IP address (default: ${LEARNER_IP})"
            echo "  --learner_port PORT        Learner port (default: ${LEARNER_PORT})"
            echo ""
            echo "Environment variables:"
            echo "  SERL_FR3_ROOT              Project root (default: /home/robot/serl_projects/hil-serl-fr3)"
            echo "  SERL_FRANKA_PORT           franka_server port (default: 5000)"
            echo "  LEARNER_IP                 Learner machine IP"
            echo "  LEARNER_PORT               Learner gRPC port"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✗]${NC} $1"; }
info()  { echo -e "${CYAN}[i]${NC} $1"; }

echo "=============================================="
echo " HIL-SERL Actor — ${EXPERIMENT}"
echo "=============================================="
echo ""

# ─── 1. Activate conda environment ──────────────────────────────────────────
info "Activating conda env: ${CONDA_ENV}..."
if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    log "Conda env activated: $(python --version 2>&1)"
else
    warn "conda not found — using system Python"
fi

# ─── 2. Set PYTHONPATH ──────────────────────────────────────────────────────
export PYTHONPATH="${SERL_FR3_ROOT}:${SERL_FR3_ROOT}/experiments:${PYTHONPATH:-}"
info "PYTHONPATH set to include ${SERL_FR3_ROOT}"

# ─── 3. Set environment variables ───────────────────────────────────────────
export SERL_FRANKA_PORT="${FRANKA_PORT}"
export SERL_FR3_NO_MOTION=0
info "SERL_FRANKA_PORT=${SERL_FRANKA_PORT}"
info "SERL_FR3_NO_MOTION=${SERL_FR3_NO_MOTION}"

# agentlace network config
export LEARNER_IP="${LEARNER_IP}"
export LEARNER_PORT="${LEARNER_PORT}"
info "Learner endpoint: ${LEARNER_IP}:${LEARNER_PORT}"

# ─── 4. Safety preamble ─────────────────────────────────────────────────────
echo ""
echo "── Safety checks ──"

# Check franka_server is running
FRANKA_HEALTH="http://127.0.0.1:${SERL_FRANKA_PORT}/getstate"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 3 --max-time 5 \
    "$FRANKA_HEALTH" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    log "franka_server is healthy (port ${SERL_FRANKA_PORT})"
else
    err "franka_server not responding at ${FRANKA_HEALTH} (HTTP ${HTTP_CODE})"
    err "Start it first: bash scripts/setup/03_start_franka_server.sh start"
    exit 1
fi

# Check GELLO connected (only in training mode)
if [ "$EVAL_MODE" = false ]; then
    info "Checking GELLO device..."
    if [ -e /dev/ttyUSB0 ] || [ -e /dev/ttyACM0 ]; then
        log "GELLO serial device found"
    else
        warn "No GELLO serial device detected — GelloIntervention may fail at runtime"
        warn "Ensure GELLO (Dynamixel) is connected via USB"
    fi
fi

# Check E-stop awareness
warn "Ensure E-stop is reachable before starting!"
echo ""

# ─── 5. Launch Actor ────────────────────────────────────────────────────────
cd "${SERL_FR3_ROOT}"

if [ "$EVAL_MODE" = true ]; then
    # ── Evaluation mode ──
    EVAL_N_TRAJS="${EVAL_N_TRAJS:-10}"
    info "Running EVALUATION mode"
    info "  checkpoint_step: ${EVAL_CHECKPOINT_STEP}"
    info "  num_trajectories: ${EVAL_N_TRAJS}"
    echo ""

    python -m experiments.${EXPERIMENT}.run_actor \
        --eval_mode \
        --eval_checkpoint_step "${EVAL_CHECKPOINT_STEP}" \
        --eval_n_trajs "${EVAL_N_TRAJS}"
else
    # ── Training mode (default) ──
    info "Running TRAINING actor with GelloIntervention"
    info "  Config: experiments.${EXPERIMENT}.config.TrainConfig"
    echo ""

    python -m experiments.${EXPERIMENT}.run_actor
fi
