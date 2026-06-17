#!/usr/bin/env bash
# =============================================================================
# 02_adapt_serl_controllers.sh — Clone & patch serl_franka_controllers for FR3
#
# This script:
#   1. Clones serl_franka_controllers from GitHub
#   2. Patches arm_id from "panda" to "fr3"
#   3. Patches joint names from "panda_joint1-7" to "fr3_joint1-7"
#   4. Patches delta_tau_max_ from Panda (1000) to FR3 (2000) limits
#   5. Creates a catkin workspace (if ROS is available)
#   6. Does NOT run on the real robot
#
# Target host: fr3-desktop-ts (Ubuntu 22.04)
# Robot IP:    172.16.0.2
# =============================================================================

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
SERL_FR3_ROOT="${SERL_FR3_ROOT:-/home/robot/serl_projects/hil-serl-fr3}"
UPSTREAM_DIR="${SERL_FR3_ROOT}/upstream"
SERL_CTRL_REPO="https://github.com/rail-berkeley/serl_franka_controllers.git"
SERL_CTRL_PIN="1f140ef0d8e3fc443569c193d3ede1856e50d521"
SERL_CTRL_DIR="${UPSTREAM_DIR}/serl_franka_controllers"
CATKIN_WS="${SERL_FR3_ROOT}/catkin_ws_franka"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

echo "=============================================="
echo " FR3 Adaptation — serl_franka_controllers"
echo "=============================================="
echo ""

# ─── 1. Clone / Update ──────────────────────────────────────────────────────
echo "── Step 1: Clone serl_franka_controllers ──"

mkdir -p "$UPSTREAM_DIR"

if [ -d "$SERL_CTRL_DIR" ]; then
    info "Repository already exists at ${SERL_CTRL_DIR}"
    CURRENT_REV=$(cd "$SERL_CTRL_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
    info "Current revision: ${CURRENT_REV}"
    warn "Skipping clone (already exists). To re-clone, remove the directory first."
else
    info "Cloning serl_franka_controllers..."
    # Clear proxy vars to avoid localhost:7890 issues on fr3-desktop-ts
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
        git -c http.proxy= -c https.proxy= \
        clone "$SERL_CTRL_REPO" "$SERL_CTRL_DIR"

    # Pin to known-good revision
    cd "$SERL_CTRL_DIR"
    git checkout "$SERL_CTRL_PIN"
    log "Cloned and pinned to ${SERL_CTRL_PIN}"
fi

cd "$SERL_CTRL_DIR"
git rev-parse HEAD
echo ""

# ─── 2. Apply FR3 Patches ───────────────────────────────────────────────────
echo "── Step 2: Apply FR3 patches ──"

PATCH_COUNT=0

# ─── 2a. Patch arm_id ────────────────────────────────────────────────────────
info "Patching arm_id: 'panda' → 'fr3'..."

# Find files that reference arm_id with panda
ARM_ID_FILES=$(grep -rl "arm_id.*panda\|panda.*arm_id\|\"panda\"" \
    --include="*.launch" --include="*.yaml" --include="*.yml" \
    --include="*.xml" --include="*.cfg" --include="*.py" \
    "$SERL_CTRL_DIR" 2>/dev/null || true)

if [ -n "$ARM_ID_FILES" ]; then
    while IFS= read -r file; do
        # Backup
        cp "$file" "${file}.bak"
        # Replace arm_id: panda → arm_id: fr3
        sed -i 's/arm_id: panda/arm_id: fr3/g' "$file" 2>/dev/null || \
            sed -i '' 's/arm_id: panda/arm_id: fr3/g' "$file"  # macOS compat
        # Replace "panda" in ROS params (quoted)
        sed -i "s/\"panda\"/\"fr3\"/g" "$file" 2>/dev/null || \
            sed -i '' "s/\"panda\"/\"fr3\"/g" "$file"
        log "  Patched arm_id in: $(basename "$file")"
        PATCH_COUNT=$((PATCH_COUNT + 1))
    done <<< "$ARM_ID_FILES"
else
    info "  No arm_id: panda references found (may be in C++ headers)"
fi

# ─── 2b. Patch joint names ───────────────────────────────────────────────────
info "Patching joint names: 'panda_joint*' → 'fr3_joint*'..."

# Find all source/config files with panda_joint references
JOINT_FILES=$(grep -rl "panda_joint" \
    --include="*.launch" --include="*.yaml" --include="*.yml" \
    --include="*.xml" --include="*.cfg" --include="*.py" \
    --include="*.h" --include="*.hpp" --include="*.cpp" \
    --include="*.urdf" --include="*.xacro" \
    "$SERL_CTRL_DIR" 2>/dev/null || true)

if [ -n "$JOINT_FILES" ]; then
    while IFS= read -r file; do
        # Backup (if not already backed up above)
        [ ! -f "${file}.bak" ] && cp "$file" "${file}.bak"

        for i in 1 2 3 4 5 6 7; do
            sed -i "s/panda_joint${i}/fr3_joint${i}/g" "$file" 2>/dev/null || \
                sed -i '' "s/panda_joint${i}/fr3_joint${i}/g" "$file"
        done
        log "  Patched joint names in: $(basename "$file")"
        PATCH_COUNT=$((PATCH_COUNT + 1))
    done <<< "$JOINT_FILES"
else
    info "  No panda_joint references found in source files"
fi

# ─── 2c. Patch panda_link references ────────────────────────────────────────
info "Checking for panda_link references..."

LINK_FILES=$(grep -rl "panda_link" \
    --include="*.launch" --include="*.yaml" --include="*.yml" \
    --include="*.xml" --include="*.h" --include="*.hpp" \
    --include="*.cpp" --include="*.urdf" --include="*.xacro" \
    "$SERL_CTRL_DIR" 2>/dev/null || true)

if [ -n "$LINK_FILES" ]; then
    while IFS= read -r file; do
        [ ! -f "${file}.bak" ] && cp "$file" "${file}.bak"

        # Link names: panda_link0-8 → fr3_link0-8
        for i in 0 1 2 3 4 5 6 7 8; do
            sed -i "s/panda_link${i}/fr3_link${i}/g" "$file" 2>/dev/null || \
                sed -i '' "s/panda_link${i}/fr3_link${i}/g" "$file"
        done
        # panda_hand → fr3_hand
        sed -i "s/panda_hand/fr3_hand/g" "$file" 2>/dev/null || \
            sed -i '' "s/panda_hand/fr3_hand/g" "$file"
        log "  Patched link names in: $(basename "$file")"
        PATCH_COUNT=$((PATCH_COUNT + 1))
    done <<< "$LINK_FILES"
else
    info "  No panda_link references found"
fi

echo ""
info "Total patches applied: ${PATCH_COUNT}"

# ─── 2d. Patch delta_tau_max_ for FR3 ──────────────────────────────────────
# FR3 has different torque rate limits than Panda.
# Panda default: delta_tau_max_ = 1000.0 Nm/s (conservative)
# FR3 recommended: delta_tau_max_ = 2000.0 Nm/s (per Franka FR3 docs)
# Without this patch, the FR3 controller may reject fast impedance commands
# with a libfranka "motion_generator_violation" error.
info "Patching delta_tau_max_: Panda 1000 → FR3 2000 (Nm/s)..."

DELTA_TAU_FILES=$(grep -rl "delta_tau_max_\s*=\s*1000" \
    --include="*.cpp" --include="*.h" --include="*.hpp" \
    "$SERL_CTRL_DIR" 2>/dev/null || true)

if [ -z "$DELTA_TAU_FILES" ]; then
    # Also try searching for the literal value without the variable name
    DELTA_TAU_FILES=$(grep -rl "1000\.0" \
        --include="*.cpp" --include="*.h" --include="*.hpp" \
        "$SERL_CTRL_DIR" 2>/dev/null | \
        xargs grep -l "delta_tau" 2>/dev/null || true)
fi

if [ -n "$DELTA_TAU_FILES" ]; then
    while IFS= read -r file; do
        [ ! -f "${file}.bak" ] && cp "$file" "${file}.bak"

        # Patch delta_tau_max_ assignments from Panda (1000) to FR3 (2000)
        # Matches patterns like: delta_tau_max_{1000.0} or delta_tau_max_ = 1000.0
        sed -i 's/delta_tau_max_{\s*1000/delta_tau_max_{2000/g' "$file" 2>/dev/null || \
            sed -i '' 's/delta_tau_max_{\s*1000/delta_tau_max_{2000/g' "$file"
        sed -i 's/delta_tau_max_\s*=\s*1000/delta_tau_max_ = 2000/g' "$file" 2>/dev/null || \
            sed -i '' 's/delta_tau_max_\s*=\s*1000/delta_tau_max_ = 2000/g' "$file"

        log "  Patched delta_tau_max_ in: $(basename "$file")"
        PATCH_COUNT=$((PATCH_COUNT + 1))
    done <<< "$DELTA_TAU_FILES"
else
    warn "  No delta_tau_max_ references found in C++ sources"
    warn "  (may need manual patching if controller rejects fast commands)"
fi

echo ""

# ─── 3. Verify Patches ─────────────────────────────────────────────────────
echo ""
echo "── Step 3: Verify patches ──"

REMAINING_PANDA=$(grep -r "panda_joint\|panda_link\|arm_id.*panda\|delta_tau_max_.*1000" \
    --include="*.launch" --include="*.yaml" --include="*.h" \
    --include="*.hpp" --include="*.cpp" --include="*.cfg" \
    "$SERL_CTRL_DIR" 2>/dev/null | grep -v ".bak:" | head -20 || true)

if [ -n "$REMAINING_PANDA" ]; then
    warn "Remaining 'panda' references (may be in comments or non-critical paths):"
    echo "$REMAINING_PANDA" | head -10
else
    log "No remaining panda_joint/panda_link/arm_id references found"
fi

# Check that fr3 references exist
FR3_REFS=$(grep -r "fr3_joint\|fr3_link\|arm_id.*fr3" \
    --include="*.launch" --include="*.yaml" --include="*.h" \
    --include="*.hpp" --include="*.cpp" --include="*.cfg" \
    "$SERL_CTRL_DIR" 2>/dev/null | grep -v ".bak:" | wc -l)

if [ "$FR3_REFS" -gt 0 ]; then
    log "Found ${FR3_REFS} FR3 references in patched files"
else
    warn "No FR3 references found — patches may not have applied correctly"
fi

# ─── 4. Create catkin workspace structure ───────────────────────────────────
echo ""
echo "── Step 4: Prepare catkin workspace ──"

info "Creating catkin workspace at ${CATKIN_WS}..."
mkdir -p "${CATKIN_WS}/src"

# Symlink the patched controller into the catkin workspace
if [ ! -L "${CATKIN_WS}/src/serl_franka_controllers" ]; then
    ln -s "$SERL_CTRL_DIR" "${CATKIN_WS}/src/serl_franka_controllers"
    log "Symlinked serl_franka_controllers into catkin workspace"
else
    info "Symlink already exists"
fi

# ─── 5. Attempt catkin build (only if ROS available) ────────────────────────
echo ""
echo "── Step 5: catkin build ──"

ROS_SETUP=""
for candidate in /opt/ros/noetic/setup.bash /opt/ros/melodic/setup.bash; do
    if [ -f "$candidate" ]; then
        ROS_SETUP="$candidate"
        break
    fi
done

if [ -n "$ROS_SETUP" ]; then
    info "Sourcing ROS from ${ROS_SETUP}..."
    # shellcheck disable=SC1090
    source "$ROS_SETUP"

    info "Running catkin_make in ${CATKIN_WS}..."
    cd "$CATKIN_WS"
    catkin_make -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
    log "catkin_make completed"

    # Source the workspace
    source "${CATKIN_WS}/devel/setup.bash"
    log "catkin workspace sourced"
else
    warn "ROS not found — skipping catkin build"
    echo ""
    echo "  To build later, either:"
    echo "    A) Install ROS Noetic and run:"
    echo "       source /opt/ros/noetic/setup.bash"
    echo "       cd ${CATKIN_WS}"
    echo "       catkin_make"
    echo ""
    echo "    B) Use Docker with ROS Noetic:"
    echo "       docker run -it --net=host -v ${CATKIN_WS}:/catkin_ws \\"
    echo "         ros:noetic bash"
    echo ""
    echo "    C) Skip ROS entirely — use the direct libfranka bridge path"
    echo "       (recommended for the hilserl-fr3 project)"
fi

# ─── 6. Summary ─────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " FR3 Adaptation Summary"
echo "=============================================="
echo ""
echo "  Source:      ${SERL_CTRL_DIR}"
echo "  Catkin WS:   ${CATKIN_WS}"
echo "  Patches:     arm_id, joint names, link names, delta_tau_max_"
echo "  Status:      Patched (NOT built, NOT deployed to robot)"
echo ""
echo "  Next steps:"
echo "    1. Run 03_start_franka_server.sh to test the robot server"
echo "    2. Or use the direct libfranka bridge path (non-ROS)"
echo ""
echo "  ⚠  Do NOT run any live robot commands without safety approval."
echo "=============================================="
