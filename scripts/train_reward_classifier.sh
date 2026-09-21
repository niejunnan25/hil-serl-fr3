#!/usr/bin/env bash
# =============================================================================
# train_reward_classifier.sh — Train binary reward classifier for plug_insertion
#
# Wrapper around this repo's scripts/train_reward_classifier.py. Trains a ResNet-based
# binary classifier on labeled positive/negative images to predict task
# completion (reward = 1 if inserted, 0 otherwise).
#
# Data layout:
#   data/classifier/
#   ├── positive/   # images of successful insertions
#   └── negative/   # images of failed attempts
#
# Output:
#   classifier_ckpt/   # trained classifier checkpoint (used by config.py)
#
# Usage:
#   bash scripts/train_reward_classifier.sh
#   bash scripts/train_reward_classifier.sh --data_dir data/classifier --epochs 100
#
# Can run on any machine with GPU (fr3-desktop-ts or zktitan)
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
CONDA_ENV="${CONDA_ENV:-hilserl-fr3}"

# Default paths (relative to SERL_FR3_ROOT)
DATA_DIR="${SERL_FR3_ROOT}/data/classifier"
OUTPUT_DIR="${SERL_FR3_ROOT}/classifier_ckpt"

# Training hyperparameters
BATCH_SIZE=64
EPOCHS=""
LEARNING_RATE="1e-3"

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --image_keys)
            warn "--image_keys is ignored by the local PyTorch trainer; it trains from positive/negative image folders."
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR        Classifier data directory (default: data/classifier/)"
            echo "  --output_dir DIR      Output checkpoint directory (default: classifier_ckpt/)"
            echo "  --image_keys KEYS     Accepted for legacy compatibility; ignored by local trainer"
            echo "  --batch_size N        Batch size (default: 64)"
            echo "  --epochs N            Number of training epochs (default: SERL default)"
            echo "  --lr RATE             Learning rate (default: 1e-3)"
            echo ""
            echo "Data layout:"
            echo "  <data_dir>/positive/  Positive examples (successful insertion)"
            echo "  <data_dir>/negative/  Negative examples (failed attempts)"
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
echo " Reward Classifier Training — plug_insertion"
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

# ─── 3. Validate data directory ─────────────────────────────────────────────
echo ""
echo "── Data validation ──"

if [ ! -d "$DATA_DIR" ]; then
    err "Data directory not found: ${DATA_DIR}"
    info "Expected layout:"
    info "  ${DATA_DIR}/positive/  (positive images)"
    info "  ${DATA_DIR}/negative/  (negative images)"
    info ""
    info "Use scripts/collect_classifier_images.py to collect data"
    exit 1
fi

N_POSITIVE=$(find "${DATA_DIR}/positive" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
N_NEGATIVE=$(find "${DATA_DIR}/negative" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
N_TOTAL=$((N_POSITIVE + N_NEGATIVE))

if [ "$N_TOTAL" -eq 0 ]; then
    err "No images found in ${DATA_DIR}/positive/ or ${DATA_DIR}/negative/"
    info "Use scripts/collect_classifier_images.py to collect data"
    exit 1
fi

log "Data found: ${N_POSITIVE} positive, ${N_NEGATIVE} negative (${N_TOTAL} total)"

if [ "$N_POSITIVE" -lt 10 ] || [ "$N_NEGATIVE" -lt 10 ]; then
    warn "Very few training samples — consider collecting more data"
fi

# ─── 4. Locate train_reward_classifier.py ───────────────────────────────────
echo ""

TRAIN_SCRIPT="${SERL_FR3_ROOT}/scripts/train_reward_classifier.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    err "train_reward_classifier.py not found: ${TRAIN_SCRIPT}"
    exit 1
fi
log "Found: ${TRAIN_SCRIPT}"

# ─── 5. Create output directory ─────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
info "Output checkpoint dir: ${OUTPUT_DIR}"

# ─── 6. Build training command ──────────────────────────────────────────────
CMD_ARGS=(
    --data-dir "${DATA_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --batch-size "${BATCH_SIZE}"
    --lr "${LEARNING_RATE}"
)

if [ -n "$EPOCHS" ]; then
    CMD_ARGS+=(--epochs "$EPOCHS")
fi

echo ""
info "Training command:"
info "  python ${TRAIN_SCRIPT} ${CMD_ARGS[*]}"
echo ""

# ─── 7. Run training ───────────────────────────────────────────────────────
cd "${SERL_FR3_ROOT}"

python "$TRAIN_SCRIPT" "${CMD_ARGS[@]}"

# ─── 8. Verify output ──────────────────────────────────────────────────────
echo ""
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    log "Classifier checkpoint saved to ${OUTPUT_DIR}"
    ls -la "$OUTPUT_DIR"
    echo ""
    info "The checkpoint is loaded by config.py via:"
    info "  load_classifier_func(checkpoint_path='classifier_ckpt/')"
    info ""
    info "Use it in training with: bash scripts/run_learner.sh"
else
    warn "Output directory is empty — training may have failed"
fi
