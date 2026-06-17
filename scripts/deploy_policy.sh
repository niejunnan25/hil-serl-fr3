#!/usr/bin/env bash
# =============================================================================
# deploy_policy.sh — Deploy a trained policy for inference-only evaluation
#
# Loads a trained checkpoint and runs the policy without human intervention
# (no GelloIntervention) for a specified number of episodes. Reports
# success rate at the end.
#
# Usage:
#   bash scripts/deploy_policy.sh --checkpoint_path checkpoints/50000
#   bash scripts/deploy_policy.sh --checkpoint_path checkpoints/50000 --num_episodes 20
#
# Target host: fr3-desktop-ts (robot side)
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"
FRANKA_PORT="${SERL_FRANKA_PORT:-5000}"
EXPERIMENT="plug_insertion"

# ─── Parse arguments ─────────────────────────────────────────────────────────
CHECKPOINT_PATH=""
NUM_EPISODES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --checkpoint_path PATH   Path to trained checkpoint"
            echo ""
            echo "Options:"
            echo "  --num_episodes N         Number of evaluation episodes (default: 10)"
            echo ""
            echo "Environment variables:"
            echo "  SERL_FR3_ROOT            Project root"
            echo "  SERL_FRANKA_PORT         franka_server port (default: 5000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required args
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: --checkpoint_path is required"
    echo "Usage: $0 --checkpoint_path <path> [--num_episodes N]"
    exit 1
fi

NUM_EPISODES="${NUM_EPISODES:-10}"

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
echo " HIL-SERL Policy Deployment — ${EXPERIMENT}"
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

# ─── 3. Set environment variables ───────────────────────────────────────────
export SERL_FRANKA_PORT="${FRANKA_PORT}"
export SERL_FR3_NO_MOTION=0
info "SERL_FRANKA_PORT=${SERL_FRANKA_PORT}"

# ─── 4. Safety checks ───────────────────────────────────────────────────────
echo ""
echo "── Safety checks ──"

# Check franka_server
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

# Check checkpoint exists
if [ -d "$CHECKPOINT_PATH" ] || [ -f "$CHECKPOINT_PATH" ]; then
    log "Checkpoint found: ${CHECKPOINT_PATH}"
else
    err "Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

warn "Ensure E-stop is reachable before starting!"
echo ""

# ─── 5. Run deployment ──────────────────────────────────────────────────────
cd "${SERL_FR3_ROOT}"

info "Deploying policy (inference-only, no GelloIntervention)"
info "  Checkpoint: ${CHECKPOINT_PATH}"
info "  Episodes:   ${NUM_EPISODES}"
echo ""

# Run evaluation — uses the actor script in eval mode
python -m experiments.${EXPERIMENT}.run_actor \
    --eval_mode \
    --eval_checkpoint_path "${CHECKPOINT_PATH}" \
    --eval_n_trajs "${NUM_EPISODES}"
