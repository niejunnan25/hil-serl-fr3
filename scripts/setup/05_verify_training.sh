#!/usr/bin/env bash
# =============================================================================
# 05_verify_training.sh — Verify Training Environment is Ready
#
# Checks:
#   1. JAX with GPU support
#   2. SERL library imports
#   3. Checkpoint loading (Orbax)
#   4. Classifier checkpoint exists
#   5. Experiment module imports
#   6. zktitan / robot server connectivity
#   7. Disk space for checkpoints
#   8. CUDA toolkit version
#
# Target: fr3-desktop-ts or local dev machine with GPU
# =============================================================================

set -euo pipefail

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
CHECKS=0

log_pass() { echo -e "  ${PASS} $1"; CHECKS=$((CHECKS + 1)); }
log_fail() { echo -e "  ${FAIL} $1"; ERRORS=$((ERRORS + 1)); CHECKS=$((CHECKS + 1)); }
log_warn() { echo -e "  ${WARN} $1"; WARNINGS=$((WARNINGS + 1)); CHECKS=$((CHECKS + 1)); }
log_info() { echo -e "  ${INFO} $1"; }

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"
ROBOT_SERVER="${ROBOT_SERVER:-http://127.0.0.2:5000/}"
CLASSIFIER_CKPT="${SERL_FR3_ROOT}/classifier_ckpt"
CHECKPOINT_DIR="${SERL_FR3_ROOT}/checkpoints"

# Try to find conda
CONDA_BASE=""
for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/home/robot/miniconda3"; do
    if [ -d "$candidate" ]; then
        CONDA_BASE="$candidate"
        break
    fi
done

PYTHON=""
if [ -n "$CONDA_BASE" ] && [ -d "${CONDA_BASE}/envs/${CONDA_ENV}" ]; then
    PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
fi

echo "=============================================="
echo " Training Environment Verification"
echo "=============================================="
echo ""
log_info "Project root: ${SERL_FR3_ROOT}"
log_info "Python:       ${PYTHON:-not found}"
log_info "Conda env:    ${CONDA_ENV}"
echo ""

# ─── 1. Python & Conda ──────────────────────────────────────────────────────
echo "── 1. Python Environment ──"

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    log_fail "Python not found. Install miniconda and create '${CONDA_ENV}' env"
else
    PY_VERSION=$("$PYTHON" --version 2>&1)
    log_pass "Python: ${PY_VERSION}"

    # Check if we're in the right conda env
    CURRENT_ENV=$("$PYTHON" -c "import os; print(os.environ.get('CONDA_DEFAULT_ENV', 'none'))" 2>/dev/null || echo "unknown")
    if [ "$CURRENT_ENV" = "$CONDA_ENV" ]; then
        log_pass "Active conda env: ${CURRENT_ENV}"
    elif [ "$CURRENT_ENV" = "none" ] || [ "$CURRENT_ENV" = "unknown" ]; then
        log_warn "Not in conda env '${CONDA_ENV}' (current: ${CURRENT_ENV}). Activate first."
    else
        log_warn "In conda env '${CURRENT_ENV}', expected '${CONDA_ENV}'"
    fi
fi

echo ""

# ─── 2. JAX / GPU ───────────────────────────────────────────────────────────
echo "── 2. JAX & GPU ──"

if [ -n "$PYTHON" ]; then
    # JAX version
    JAX_VER=$("$PYTHON" -c "import jax; print(jax.__version__)" 2>/dev/null || echo "")
    if [ -n "$JAX_VER" ]; then
        log_pass "JAX version: ${JAX_VER}"
    else
        log_fail "JAX not importable"
    fi

    # GPU devices
    GPU_INFO=$("$PYTHON" -c "
import jax
devices = jax.local_devices()
gpu = [d for d in devices if 'gpu' in d.platform.lower() or 'cuda' in d.platform.lower()]
if gpu:
    for g in gpu:
        print(f'{g.device_kind}: {g.platform}')
else:
    print(f'NO_GPU: {len(devices)} device(s): {[str(d) for d in devices]}')
" 2>/dev/null || echo "IMPORT_ERROR")

    if echo "$GPU_INFO" | grep -q "NO_GPU"; then
        log_fail "No GPU detected: ${GPU_INFO}"
    elif echo "$GPU_INFO" | grep -q "IMPORT_ERROR"; then
        log_fail "JAX GPU check failed"
    else
        log_pass "GPU: ${GPU_INFO}"
    fi

    # JAX sharding test
    SHARD_OK=$("$PYTHON" -c "
import jax
import jax.numpy as jnp
devices = jax.local_devices()
sharding = jax.sharding.PositionalSharding(devices)
x = jnp.ones((2, 2))
y = jax.device_put(x, sharding.replicate())
print(f'sharding_ok: {y.shape}')
" 2>/dev/null || echo "FAIL")

    if echo "$SHARD_OK" | grep -q "sharding_ok"; then
        log_pass "JAX sharding: working"
    else
        log_warn "JAX sharding test failed (multi-GPU may not work)"
    fi
else
    log_fail "Cannot check JAX — Python not available"
fi

echo ""

# ─── 3. CUDA Toolkit ────────────────────────────────────────────────────────
echo "── 3. CUDA Toolkit ──"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    log_pass "GPU: ${GPU_NAME} (${GPU_MEM}), CUDA ${CUDA_VER}"
else
    log_fail "nvidia-smi not found — no GPU driver"
fi

# Check cuDNN
if [ -n "$PYTHON" ]; then
    CUDNN_VER=$("$PYTHON" -c "
try:
    from jaxlib import version
    print(version)
except:
    print('unknown')
" 2>/dev/null || echo "unknown")
    log_info "jaxlib: ${CUDNN_VER}"
fi

echo ""

# ─── 4. SERL Imports ────────────────────────────────────────────────────────
echo "── 4. SERL Library ──"

if [ -n "$PYTHON" ]; then
    SERL_IMPORTS=(
        "serl_launcher.agents.continuous.sac"
        "serl_launcher.agents.continuous.bc"
        "serl_launcher.data.data_store"
        "serl_launcher.networks.reward_classifier"
        "serl_launcher.wrappers.serl_obs_wrappers"
        "serl_launcher.utils.launcher"
        "franka_env.envs.franka_env"
        "franka_env.envs.wrappers"
    )

    for mod in "${SERL_IMPORTS[@]}"; do
        OK=$("$PYTHON" -c "import ${mod}" 2>&1 && echo "OK" || echo "FAIL")
        if [ "$OK" = "OK" ]; then
            log_pass "import ${mod}"
        else
            log_fail "import ${mod}"
        fi
    done
else
    log_fail "Cannot check SERL imports — Python not available"
fi

echo ""

# ─── 5. Experiment Module ───────────────────────────────────────────────────
echo "── 5. Experiment Modules ──"

if [ -n "$PYTHON" ]; then
    EXP_IMPORTS=(
        "experiments.config"
        "experiments.plug_insertion.config"
        "experiments.plug_insertion.env"
        "experiments.plug_insertion.wrapper"
    )

    for mod in "${EXP_IMPORTS[@]}"; do
        OK=$("$PYTHON" -c "
import sys; sys.path.insert(0, '${SERL_FR3_ROOT}')
import ${mod}
" 2>&1 && echo "OK" || echo "FAIL")
        if [ "$OK" = "OK" ]; then
            log_pass "import ${mod}"
        else
            log_fail "import ${mod}"
        fi
    done
else
    log_fail "Cannot check experiment imports — Python not available"
fi

echo ""

# ─── 6. Checkpoint Loading ──────────────────────────────────────────────────
echo "── 6. Checkpoints ──"

# Classifier checkpoint
if [ -d "$CLASSIFIER_CKPT" ]; then
    CKPT_FILES=$(find "$CLASSIFIER_CKPT" -type f | wc -l)
    log_pass "Classifier checkpoint: ${CLASSIFIER_CKPT} (${CKPT_FILES} files)"
else
    log_warn "Classifier checkpoint not found: ${CLASSIFIER_CKPT}"
fi

# Policy checkpoints
if [ -d "$CHECKPOINT_DIR" ]; then
    CKPT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d | wc -l)
    log_pass "Checkpoint dir: ${CHECKPOINT_DIR} (${CKPT_COUNT} entries)"
else
    log_warn "Checkpoint dir not found: ${CHECKPOINT_DIR}"
fi

# Orbax checkpointer
if [ -n "$PYTHON" ]; then
    ORBAX_OK=$("$PYTHON" -c "from orbax.checkpoint import PyTreeCheckpointer; print('ok')" 2>/dev/null || echo "fail")
    if [ "$ORBAX_OK" = "ok" ]; then
        log_pass "Orbax PyTreeCheckpointer: importable"
    else
        log_fail "Orbax PyTreeCheckpointer: not importable"
    fi
fi

echo ""

# ─── 7. Robot Server Connectivity ───────────────────────────────────────────
echo "── 7. Robot Server Connectivity ──"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --connect-timeout 3 --max-time 5 \
    "${ROBOT_SERVER}getstate" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    log_pass "Robot server reachable: ${ROBOT_SERVER} (HTTP ${HTTP_CODE})"
elif [ "$HTTP_CODE" = "000" ]; then
    log_warn "Robot server unreachable: ${ROBOT_SERVER} (connection refused)"
    log_info "Start with: scripts/setup/03_start_franka_server.sh start"
else
    log_warn "Robot server responded: HTTP ${HTTP_CODE}"
fi

# Check zktitan if configured
ZKTITAN_HOST="${ZKTITAN_HOST:-}"
if [ -n "$ZKTITAN_HOST" ]; then
    if ping -c 1 -W 2 "$ZKTITAN_HOST" &>/dev/null; then
        log_pass "zktitan reachable: ${ZKTITAN_HOST}"
    else
        log_warn "zktitan unreachable: ${ZKTITAN_HOST}"
    fi
else
    log_info "ZKTITAN_HOST not set, skipping zktitan check"
fi

echo ""

# ─── 8. Disk Space ──────────────────────────────────────────────────────────
echo "── 8. Disk Space ──"

# Check available space on the project partition
AVAIL_KB=$(df -k "${SERL_FR3_ROOT}" 2>/dev/null | tail -1 | awk '{print $4}')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))

if [ "$AVAIL_GB" -ge 50 ]; then
    log_pass "Disk space: ${AVAIL_GB}GB available (>= 50GB recommended)"
elif [ "$AVAIL_GB" -ge 20 ]; then
    log_warn "Disk space: ${AVAIL_GB}GB available (50GB recommended for checkpoints)"
else
    log_fail "Disk space: ${AVAIL_GB}GB available (need at least 20GB)"
fi

echo ""

# ─── 9. ZED Cameras (optional) ─────────────────────────────────────────────
echo "── 9. ZED Cameras ──"

if command -v ZED_Explorer &>/dev/null || [ -f "/usr/local/zed/tools/ZED_Explorer" ]; then
    log_pass "ZED SDK tools found"
else
    log_info "ZED SDK tools not found (not required for training)"
fi

# Check USB devices for ZED
if command -v lsusb &>/dev/null; then
    ZED_USB=$(lsusb 2>/dev/null | grep -i "stereolabs\|zed\|2b03:f100" || true)
    if [ -n "$ZED_USB" ]; then
        log_pass "ZED camera detected via USB"
    else
        log_info "No ZED camera detected via USB"
    fi
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "=============================================="
echo " Summary"
echo "=============================================="
echo ""
echo -e "  Checks:   ${CHECKS}"
echo -e "  Passed:   ${GREEN}$((CHECKS - ERRORS - WARNINGS))${NC}"
echo -e "  Failed:   ${RED}${ERRORS}${NC}"
echo -e "  Warnings: ${YELLOW}${WARNINGS}${NC}"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}Training environment has ${ERRORS} critical issue(s). Fix before training.${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}${WARNINGS} warning(s). Review above. Training may work but with limitations.${NC}"
    exit 0
else
    echo -e "${GREEN}Training environment is ready. All checks passed.${NC}"
    exit 0
fi
