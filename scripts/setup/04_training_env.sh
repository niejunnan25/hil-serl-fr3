#!/usr/bin/env bash
# =============================================================================
# 04_training_env.sh — Setup training environment for HIL-SERL
#
# Installs SERL dependencies (serl_launcher, agentlace), configures
# PYTHONPATH, and verifies JAX/GPU and network connectivity.
#
# Run this on both:
#   - fr3-desktop-ts (actor side — may not have GPU)
#   - zktitan        (learner side — must have GPU)
#
# Usage:
#   bash scripts/setup/04_training_env.sh
#   bash scripts/setup/04_training_env.sh --skip-install   # only verify
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"
ZKTITAN_IP="${ZKTITAN_IP:-10.192.4.249}"
FR3_DESKTOP_IP="${FR3_DESKTOP_IP:-10.192.4.136}"

SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-install    Skip package installation, only verify"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS="${GREEN}[OK]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"
INFO="${CYAN}[INFO]${NC}"

ERRORS=0
WARNINGS=0

log_pass() { echo -e "  ${PASS} $1"; }
log_fail() { echo -e "  ${FAIL} $1"; ERRORS=$((ERRORS + 1)); }
log_warn() { echo -e "  ${WARN} $1"; WARNINGS=$((WARNINGS + 1)); }
log_info() { echo -e "  ${INFO} $1"; }

echo "=============================================="
echo " HIL-SERL Training Environment Setup"
echo "=============================================="
echo ""

# ─── 1. Activate conda environment ──────────────────────────────────────────
echo "── conda environment ──"

if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    log_pass "Conda env activated: ${CONDA_ENV} ($(python --version 2>&1))"
else
    log_fail "conda not found"
    exit 1
fi

echo ""

# ─── 2. Install SERL dependencies ───────────────────────────────────────────
echo "── SERL packages ──"

if [ "$SKIP_INSTALL" = true ]; then
    log_info "Skipping installation (--skip-install)"
else
    # Check for SERL source directories
    SERL_LAUNCHER=""
    for candidate in \
        "${SERL_FR3_ROOT}/upstream/hil-serl/serl_launcher" \
        "${SERL_FR3_ROOT}/upstream/serl/serl_launcher"; do
        if [ -d "$candidate" ]; then
            SERL_LAUNCHER="$candidate"
            break
        fi
    done

    if [ -n "$SERL_LAUNCHER" ]; then
        log_info "Installing serl_launcher from ${SERL_LAUNCHER}..."
        pip install -e "$SERL_LAUNCHER" 2>/dev/null && \
            log_pass "serl_launcher installed" || \
            log_warn "serl_launcher install had issues"
    else
        log_warn "serl_launcher source not found"
        log_info "Expected at: ${SERL_FR3_ROOT}/upstream/hil-serl/serl_launcher"
        log_info "Clone it: git clone https://github.com/rail-berkeley/hil-serl.git ${SERL_FR3_ROOT}/upstream/hil-serl"
    fi

    # Install agentlace
    if python -c "import agentlace" 2>/dev/null; then
        AGENTLACE_VER=$(python -c "import agentlace; print(agentlace.__version__)" 2>/dev/null || echo "unknown")
        log_pass "agentlace already installed (v${AGENTLACE_VER})"
    else
        log_info "Installing agentlace..."
        pip install agentlace 2>/dev/null && \
            log_pass "agentlace installed" || \
            log_warn "agentlace install failed — try: pip install agentlace"
    fi

    # Install other SERL dependencies
    for pkg in gymnasium jax jaxlib flax optax; do
        if python -c "import ${pkg}" 2>/dev/null; then
            VER=$(python -c "import ${pkg}; print(getattr(${pkg}, '__version__', 'ok'))" 2>/dev/null || echo "ok")
            log_pass "${pkg} (${VER})"
        else
            log_warn "${pkg} not installed"
        fi
    done
fi

echo ""

# ─── 3. Verify PYTHONPATH ───────────────────────────────────────────────────
echo "── PYTHONPATH ──"

# Construct the required PYTHONPATH
REQUIRED_PATHS="${SERL_FR3_ROOT}:${SERL_FR3_ROOT}/experiments"
export PYTHONPATH="${REQUIRED_PATHS}:${PYTHONPATH:-}"

# Verify key imports
IMPORTS_OK=true
for module in "experiments.config" "experiments.plug_insertion.config"; do
    if python -c "import ${module}" 2>/dev/null; then
        log_pass "import ${module}"
    else
        log_fail "import ${module} failed"
        IMPORTS_OK=false
    fi
done

if [ "$IMPORTS_OK" = true ]; then
    log_pass "PYTHONPATH is correctly configured"
else
    log_warn "Some imports failed — add to your shell profile:"
    log_info "  export PYTHONPATH=\"${REQUIRED_PATHS}:\${PYTHONPATH}\""
fi

echo ""

# ─── 4. Verify JAX / GPU ───────────────────────────────────────────────────
echo "── JAX / GPU ──"

JAX_INFO=$(python -c "
import jax
devices = jax.devices()
gpu = [d for d in devices if 'gpu' in str(d.platform).lower()]
cpu = [d for d in devices if 'cpu' in str(d.platform).lower()]
print(f'version={jax.__version__}')
print(f'devices={len(devices)}')
print(f'gpu={len(gpu)}')
print(f'cpu={len(cpu)}')
if gpu:
    print(f'gpu_name={gpu[0]}')
" 2>/dev/null || echo "import_failed")

if echo "$JAX_INFO" | grep -q "import_failed"; then
    log_fail "JAX import failed"
else
    JAX_VER=$(echo "$JAX_INFO" | grep "version=" | cut -d= -f2)
    JAX_GPU=$(echo "$JAX_INFO" | grep "gpu=" | cut -d= -f2)
    JAX_GPU_NAME=$(echo "$JAX_INFO" | grep "gpu_name=" | cut -d= -f2 || echo "")

    log_pass "JAX ${JAX_VER}"

    if [ "$JAX_GPU" -gt 0 ] 2>/dev/null; then
        log_pass "JAX GPU available: ${JAX_GPU_NAME}"
    else
        log_warn "No JAX GPU detected (CPU-only mode)"
        log_info "Learner requires GPU — run on zktitan for training"
        log_info "Actor can run on CPU (fr3-desktop-ts)"
    fi
fi

# Check nvidia-smi separately
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    log_pass "nvidia-smi: ${GPU_NAME} (${GPU_MEM}, CUDA ${CUDA_VER})"
else
    log_info "nvidia-smi not available"
fi

echo ""

# ─── 5. Network connectivity ────────────────────────────────────────────────
echo "── Network connectivity ──"

# Determine which host we're on
HOSTNAME_STR=$(hostname)
IS_ZKTITAN=false
IS_FR3_DESKTOP=false

if echo "$HOSTNAME_STR" | grep -qi "zktitan\|gpu"; then
    IS_ZKTITAN=true
elif echo "$HOSTNAME_STR" | grep -qi "fr3\|desktop"; then
    IS_FR3_DESKTOP=true
fi

# Ping zktitan
if ping -c 1 -W 2 "$ZKTITAN_IP" &>/dev/null; then
    log_pass "zktitan reachable at ${ZKTITAN_IP}"
else
    if [ "$IS_ZKTITAN" = true ]; then
        log_info "Running on zktitan — self-ping skipped"
    else
        log_fail "Cannot reach zktitan at ${ZKTITAN_IP}"
        log_info "Learner runs on zktitan — network connectivity required"
    fi
fi

# Ping fr3-desktop-ts
if ping -c 1 -W 2 "$FR3_DESKTOP_IP" &>/dev/null; then
    log_pass "fr3-desktop-ts reachable at ${FR3_DESKTOP_IP}"
else
    if [ "$IS_FR3_DESKTOP" = true ]; then
        log_info "Running on fr3-desktop-ts — self-ping skipped"
    else
        log_warn "Cannot reach fr3-desktop-ts at ${FR3_DESKTOP_IP}"
        log_info "Actor runs on fr3-desktop-ts — network connectivity required"
    fi
fi

# Check agentlace port availability
LEARNER_PORT="${LEARNER_PORT:-50051}"
if ss -ltnp 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\\])${LEARNER_PORT}$"; then
    log_warn "Port ${LEARNER_PORT} already in use — may conflict with learner"
else
    log_pass "Port ${LEARNER_PORT} available for agentlace"
fi

echo ""

# ─── 6. Summary ──────────────────────────────────────────────────────────────
echo "=============================================="
echo " Summary"
echo "=============================================="
echo ""

echo -e "  conda env:       ${GREEN}${CONDA_ENV}${NC}"
echo -e "  JAX:             $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
echo -e "  Errors:          ${ERRORS}"
echo -e "  Warnings:        ${WARNINGS}"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}Some checks failed. Review errors above.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Fix errors above"
    echo "  2. Re-run: bash scripts/setup/04_training_env.sh"
    echo "  3. Start training:"
    echo "     Actor:    bash scripts/run_actor.sh"
    echo "     Learner:  bash scripts/run_learner.sh"
    exit 1
else
    echo -e "${GREEN}Training environment ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Collect classifier data (if needed):"
    echo "     python scripts/collect_classifier_images.py"
    echo "  2. Train reward classifier:"
    echo "     bash scripts/train_reward_classifier.sh"
    echo "  3. Start training:"
    echo "     On zktitan:        bash scripts/run_learner.sh"
    echo "     On fr3-desktop-ts: bash scripts/run_actor.sh"
    echo "  4. Deploy trained policy:"
    echo "     bash scripts/deploy_policy.sh --checkpoint_path <path>"
    exit 0
fi
