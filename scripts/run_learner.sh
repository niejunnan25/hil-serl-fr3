#!/usr/bin/env bash
# =============================================================================
# run_learner.sh — Launch HIL-SERL Learner (GPU side)
#
# Learner runs on zktitan (GPU machine), receives transitions from Actor
# via agentlace, runs SAC updates, and serves updated policy weights.
#
# Usage:
#   bash scripts/run_learner.sh                          # fresh training
#   bash scripts/run_learner.sh --resume_from 50000      # resume from checkpoint
#
# Target host: zktitan (GPU machine, remote)
# Local usage: also works on fr3-desktop-ts for debugging (no GPU required for fake_env)
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"
EXPERIMENT="plug_insertion"

# agentlace network config
LEARNER_IP="${LEARNER_IP:-0.0.0.0}"          # listen on all interfaces
LEARNER_PORT="${LEARNER_PORT:-50051}"
ACTOR_IP="${ACTOR_IP:-10.192.4.136}"         # fr3-desktop-ts

# ─── Parse arguments ─────────────────────────────────────────────────────────
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume_from)
            RESUME_FROM="$2"
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
        --actor_ip)
            ACTOR_IP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --resume_from N      Resume training from checkpoint step N"
            echo "  --learner_ip IP      Learner listen address (default: 0.0.0.0)"
            echo "  --learner_port PORT  Learner gRPC port (default: 50051)"
            echo "  --actor_ip IP        Actor machine IP (default: fr3-desktop-ts)"
            echo ""
            echo "Environment variables:"
            echo "  SERL_FR3_ROOT        Project root"
            echo "  LEARNER_IP           Learner listen address"
            echo "  LEARNER_PORT         Learner gRPC port"
            echo "  ACTOR_IP             Actor machine IP"
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
echo " HIL-SERL Learner — ${EXPERIMENT}"
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
# agentlace network config
export LEARNER_IP="${LEARNER_IP}"
export LEARNER_PORT="${LEARNER_PORT}"
export ACTOR_IP="${ACTOR_IP}"
info "Learner listening on: ${LEARNER_IP}:${LEARNER_PORT}"
info "Actor expected at:    ${ACTOR_IP}"

# JAX configuration
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
info "JAX preallocate=false, mem_fraction=0.9"

# ─── 4. Verify GPU ──────────────────────────────────────────────────────────
echo ""
echo "── GPU check ──"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: ${GPU_NAME} (${GPU_MEM})"
else
    warn "nvidia-smi not found — GPU may not be available"
fi

# Verify JAX sees GPU
python -c "
import jax
devices = jax.devices()
gpu_devices = [d for d in devices if 'gpu' in str(d.platform).lower()]
if gpu_devices:
    print(f'[OK] JAX GPU: {gpu_devices[0]}')
else:
    print(f'[WARN] No GPU in JAX — devices: {devices}')
" 2>/dev/null || warn "Could not verify JAX GPU"

# ─── 5. Launch Learner ──────────────────────────────────────────────────────
echo ""
cd "${SERL_FR3_ROOT}"

if [ -n "$RESUME_FROM" ]; then
    info "Resuming training from checkpoint step ${RESUME_FROM}"
    echo ""

    python -m experiments.${EXPERIMENT}.run_learner \
        --resume_from "${RESUME_FROM}"
else
    info "Starting fresh SAC learner training"
    info "  Config: experiments.${EXPERIMENT}.config.TrainConfig"
    echo ""

    python -m experiments.${EXPERIMENT}.run_learner
fi
