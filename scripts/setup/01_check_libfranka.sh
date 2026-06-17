#!/usr/bin/env bash
# =============================================================================
# 01_check_libfranka.sh — Check libfranka, ROS Noetic, and franka_ros readiness
# for FR3 adaptation of serl_franka_controllers
#
# Target host: fr3-desktop-ts (Ubuntu 22.04, realtime kernel)
# Robot IP:    172.16.0.2
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

log_pass() { echo -e "  ${PASS} $1"; }
log_fail() { echo -e "  ${FAIL} $1"; ERRORS=$((ERRORS + 1)); }
log_warn() { echo -e "  ${WARN} $1"; WARNINGS=$((WARNINGS + 1)); }
log_info() { echo -e "  ${INFO} $1"; }

echo "=============================================="
echo " FR3 Adaptation — Environment Compatibility Check"
echo "=============================================="
echo ""

# ─── 1. libfranka ────────────────────────────────────────────────────────────
echo "── libfranka ──"

LIBFRANKA_MIN="0.13.0"
LIBFRANKA_VERSION=""
LIBFRANKA_STATUS="unknown"

# Check dpkg
if dpkg -l 2>/dev/null | grep -q "libfranka"; then
    LIBFRANKA_VERSION=$(dpkg -l 2>/dev/null | grep "libfranka" | awk '{print $3}' | head -1)
    log_info "dpkg reports libfranka version: ${LIBFRANKA_VERSION}"
else
    log_info "libfranka not found via dpkg"
fi

# Check pkg-config
if command -v pkg-config &>/dev/null && pkg-config --exists franka 2>/dev/null; then
    PKG_VERSION=$(pkg-config --modversion franka 2>/dev/null || echo "unknown")
    log_info "pkg-config reports franka version: ${PKG_VERSION}"
    if [ -z "$LIBFRANKA_VERSION" ]; then
        LIBFRANKA_VERSION="$PKG_VERSION"
    fi
fi

# Check for libfranka shared library
if ldconfig -p 2>/dev/null | grep -q "libfranka"; then
    FRANKA_LIB_PATH=$(ldconfig -p 2>/dev/null | grep "libfranka" | head -1 | awk '{print $NF}')
    log_pass "libfranka shared library found: ${FRANKA_LIB_PATH}"
else
    log_warn "libfranka shared library not in ldconfig cache"
fi

# Check franka version header
FRANKA_HEADER=""
for dir in /usr/include /usr/local/include /opt/libfranka/include; do
    if [ -f "${dir}/franka/robot.h" ]; then
        FRANKA_HEADER="${dir}/franka/robot.h"
        break
    fi
done

if [ -n "$FRANKA_HEADER" ]; then
    log_pass "franka header found: ${FRANKA_HEADER}"
else
    log_warn "franka/robot.h not found in standard include paths"
fi

# Version comparison
if [ -n "$LIBFRANKA_VERSION" ]; then
    # Extract major.minor for comparison (strip leading 0s and any suffix)
    ver_major=$(echo "$LIBFRANKA_VERSION" | cut -d. -f1 | sed 's/^0*//')
    ver_minor=$(echo "$LIBFRANKA_VERSION" | cut -d. -f2 | sed 's/^0*//')
    min_major=$(echo "$LIBFRANKA_MIN" | cut -d. -f1)
    min_minor=$(echo "$LIBFRANKA_MIN" | cut -d. -f2)

    if [ "$ver_major" -gt "$min_major" ] 2>/dev/null; then
        LIBFRANKA_STATUS="ok"
    elif [ "$ver_major" -eq "$min_major" ] 2>/dev/null && [ "$ver_minor" -ge "$min_minor" ] 2>/dev/null; then
        LIBFRANKA_STATUS="ok"
    else
        LIBFRANKA_STATUS="too_old"
    fi

    if [ "$LIBFRANKA_STATUS" = "ok" ]; then
        log_pass "libfranka ${LIBFRANKA_VERSION} >= ${LIBFRANKA_MIN} (FR3 compatible)"
    else
        log_fail "libfranka ${LIBFRANKA_VERSION} < ${LIBFRANKA_MIN} (FR3 requires >= ${LIBFRANKA_MIN})"
    fi
else
    log_fail "Could not determine libfranka version"
fi

echo ""

# ─── 2. Franka Examples / Tools ──────────────────────────────────────────────
echo "── Franka tools ──"

for tool in echo_robot_state print_joint_poses communication_test; do
    if [ -f "/usr/bin/${tool}" ]; then
        log_pass "/usr/bin/${tool} exists"
    elif [ -f "/usr/local/bin/${tool}" ]; then
        log_warn "/usr/local/bin/${tool} exists (may be stale build)"
    else
        log_info "${tool} not found in /usr/bin or /usr/local/bin"
    fi
done

echo ""

# ─── 3. Robot Network Connectivity ───────────────────────────────────────────
echo "── Robot network (172.16.0.2) ──"

if ping -c 1 -W 2 172.16.0.2 &>/dev/null; then
    log_pass "FR3 robot reachable at 172.16.0.2"
else
    log_warn "Cannot reach 172.16.0.2 (robot may be off or network not configured)"
fi

# Check local interface
if ip addr show 2>/dev/null | grep -q "172.16.0"; then
    LOCAL_IP=$(ip addr show 2>/dev/null | grep "172.16.0" | awk '{print $2}' | head -1)
    log_pass "Local interface on 172.16.0.x subnet: ${LOCAL_IP}"
else
    log_warn "No local interface found on 172.16.0.x subnet"
fi

echo ""

# ─── 4. ROS Noetic ──────────────────────────────────────────────────────────
echo "── ROS Noetic ──"

ROS_SETUP=""
ROS_STATUS="not_found"

for candidate in /opt/ros/noetic/setup.bash /opt/ros/melodic/setup.bash; do
    if [ -f "$candidate" ]; then
        ROS_SETUP="$candidate"
        break
    fi
done

if [ -n "$ROS_SETUP" ]; then
    ROS_DISTRO=$(basename "$(dirname "$ROS_SETUP")")
    log_pass "ROS found: ${ROS_DISTRO} at ${ROS_SETUP}"
    ROS_STATUS="found"

    if [ "$ROS_DISTRO" = "noetic" ]; then
        log_pass "ROS Noetic detected (compatible with serl_franka_controllers)"
    else
        log_warn "ROS ${ROS_DISTRO} found, but serl_franka_controllers targets Noetic"
    fi
else
    log_warn "No ROS installation found in /opt/ros/"
    log_info "serl_franka_controllers requires ROS Noetic for catkin build"
    log_info "Options:"
    log_info "  A) Install ROS Noetic (Ubuntu 20.04 required, or use Docker)"
    log_info "  B) Use an isolated Ubuntu 20.04 container with host networking"
    log_info "  C) Build a direct libfranka bridge (non-ROS path, recommended)"
    ROS_STATUS="missing"
fi

# Check roscore / roslaunch
if command -v roscore &>/dev/null; then
    log_pass "roscore found in PATH"
else
    log_info "roscore not in PATH"
fi

if command -v catkin_make &>/dev/null; then
    log_pass "catkin_make found in PATH"
else
    log_info "catkin_make not in PATH"
fi

echo ""

# ─── 5. franka_ros ──────────────────────────────────────────────────────────
echo "── franka_ros ──"

FRANKA_ROS_STATUS="not_found"

# Check common locations
for ws in /home/robot/ros_ws /home/robot/catkin_ws "$HOME/ros_ws" "$HOME/catkin_ws"; do
    if [ -d "${ws}/src/franka_ros" ]; then
        log_pass "franka_ros found at ${ws}/src/franka_ros"
        FRANKA_ROS_STATUS="found"
    fi
done

# Check if installed as ROS package
if command -v rospack &>/dev/null; then
    if rospack find franka_ros &>/dev/null; then
        FRANKA_ROS_PATH=$(rospack find franka_ros)
        log_pass "franka_ros package: ${FRANKA_ROS_PATH}"
        FRANKA_ROS_STATUS="found"
    fi
fi

if [ "$FRANKA_ROS_STATUS" = "not_found" ]; then
    log_warn "franka_ros not found"
    log_info "franka_ros is required for serl_franka_controllers ROS path"
    log_info "It provides: franka_control, franka_hw, franka_msgs, etc."
fi

echo ""

# ─── 6. serl_franka_controllers ─────────────────────────────────────────────
echo "── serl_franka_controllers ──"

SERL_CTRL_DIR=""
for candidate in \
    /home/robot/serl_projects/hil-serl-fr3/upstream/serl_franka_controllers \
    "$HOME/serl_projects/hil-serl-fr3/upstream/serl_franka_controllers"; do
    if [ -d "$candidate" ]; then
        SERL_CTRL_DIR="$candidate"
        break
    fi
done

if [ -n "$SERL_CTRL_DIR" ]; then
    log_pass "serl_franka_controllers found at ${SERL_CTRL_DIR}"

    # Check if it's been adapted for FR3
    if grep -rq "fr3_joint" "$SERL_CTRL_DIR" 2>/dev/null; then
        log_pass "FR3 joint names already present in source"
    else
        log_info "FR3 adaptation not yet applied (still uses panda_* names)"
    fi

    if grep -rq "arm_id.*fr3" "$SERL_CTRL_DIR" 2>/dev/null; then
        log_pass "FR3 arm_id already present"
    else
        log_info "arm_id still set to 'panda' — needs FR3 adaptation"
    fi
else
    log_warn "serl_franka_controllers not found in expected locations"
fi

echo ""

# ─── 7. CUDA / GPU (for downstream training) ────────────────────────────────
echo "── GPU / CUDA ──"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    log_pass "GPU: ${GPU_NAME}, CUDA support: ${CUDA_VER}"
else
    log_info "nvidia-smi not found (GPU not needed for controller build)"
fi

echo ""

# ─── 8. conda environment ───────────────────────────────────────────────────
echo "── conda hilserl-fr3 ──"

CONDA_ENV_DIR="/home/robot/miniconda3/envs/hilserl-fr3"
if [ -d "$CONDA_ENV_DIR" ]; then
    PYTHON_VER=$("${CONDA_ENV_DIR}/bin/python" --version 2>/dev/null || echo "unknown")
    log_pass "hilserl-fr3 conda env exists: ${PYTHON_VER}"

    # Check JAX
    if "${CONDA_ENV_DIR}/bin/python" -c "import jax; print('JAX', jax.__version__)" 2>/dev/null; then
        log_pass "JAX importable in hilserl-fr3"
    else
        log_warn "JAX not importable in hilserl-fr3"
    fi
else
    log_warn "hilserl-fr3 conda env not found at ${CONDA_ENV_DIR}"
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
echo "=============================================="
echo " Summary"
echo "=============================================="
echo ""

if [ "$LIBFRANKA_STATUS" = "ok" ]; then
    echo -e "  libfranka:       ${GREEN}FR3 compatible${NC} (${LIBFRANKA_VERSION:-detected})"
else
    echo -e "  libfranka:       ${RED}NOT FR3 compatible${NC}"
fi

echo -e "  ROS Noetic:      $([ "$ROS_STATUS" = "found" ] && echo -e "${GREEN}found${NC}" || echo -e "${YELLOW}not installed${NC}")"
echo -e "  franka_ros:      $([ "$FRANKA_ROS_STATUS" = "found" ] && echo -e "${GREEN}found${NC}" || echo -e "${YELLOW}not found${NC}")"
echo -e "  Robot network:   $(ping -c 1 -W 2 172.16.0.2 &>/dev/null && echo -e "${GREEN}reachable${NC}" || echo -e "${YELLOW}unreachable${NC}")"
echo -e "  Errors:   ${ERRORS}"
echo -e "  Warnings: ${WARNINGS}"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}Some checks failed. Review errors above before proceeding.${NC}"
    exit 1
elif [ "$WARNINGS" -gt 0 ]; then
    echo -e "${YELLOW}Some warnings found. Review above. Non-ROS direct bridge path is recommended.${NC}"
    exit 0
else
    echo -e "${GREEN}All checks passed. Ready for FR3 adaptation.${NC}"
    exit 0
fi
