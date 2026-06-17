#!/bin/bash
# ============================================================
# deploy.sh -- 将 plug_insertion env skeleton 部署到 fr3-desktop-ts
# ---
# 使用 scp 复制整个目录，设置权限，并验证 Python import。
#
# 用法:
#   bash deploy.sh [--dry-run]
# ============================================================

set -euo pipefail

# -- 配置 --
REMOTE_HOST="fr3-desktop-ts"
REMOTE_USER="robot"
REMOTE_BASE="/home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion"
LOCAL_SOURCE_DIR="/Users/tacyvan/.planning/hil-serl-plug/evidence/env-skeleton"
EXPECTED_FILES=(
    "__init__.py"
    "config.py"
    "env.py"
    "wrapper.py"
    "run_actor.sh"
    "run_learner.sh"
)

# -- 颜色 --
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
info() { echo -e "[INFO] $*"; }

# -- Dry-run 模式 --
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    warn "Dry-run 模式，不会实际执行远程操作"
fi

# ============================================================
# 1. 检查本地源目录
# ============================================================
info "检查本地源目录: ${LOCAL_SOURCE_DIR}"
if [[ ! -d "${LOCAL_SOURCE_DIR}" ]]; then
    fail "源目录不存在: ${LOCAL_SOURCE_DIR}"
    exit 1
fi

MISSING=()
for f in "${EXPECTED_FILES[@]}"; do
    if [[ ! -f "${LOCAL_SOURCE_DIR}/${f}" ]]; then
        MISSING+=("${f}")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    fail "源目录缺少文件: ${MISSING[*]}"
    exit 1
fi
ok "本地文件检查通过 (${#EXPECTED_FILES[@]} 个文件)"

# ============================================================
# 2. 检查远程主机连通性
# ============================================================
info "检查远程主机连通性: ${REMOTE_USER}@${REMOTE_HOST}"
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
    fail "无法连接到 ${REMOTE_USER}@${REMOTE_HOST}"
    echo "  请确认:"
    echo "    1. SSH 配置 (~/.ssh/config) 中已配置 fr3-desktop-ts"
    echo "    2. SSH key 已添加到远程主机"
    echo "    3. 远程主机在线"
    exit 1
fi
ok "远程主机连接正常"

# ============================================================
# 3. 创建远程目录结构
# ============================================================
info "创建远程目标目录: ${REMOTE_BASE}"
if [[ "${DRY_RUN}" == true ]]; then
    info "[dry-run] ssh ${REMOTE_USER}@${REMOTE_HOST} mkdir -p ${REMOTE_BASE}"
else
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_BASE}'"
    ok "远程目录已创建"
fi

# ============================================================
# 4. SCP 传输整个目录
# ============================================================
info "传输文件到远程服务器..."
if [[ "${DRY_RUN}" == true ]]; then
    info "[dry-run] scp -r ${LOCAL_SOURCE_DIR}/ ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
else
    scp -r "${LOCAL_SOURCE_DIR}/"* "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/"
    ok "文件传输完成"
fi

# ============================================================
# 5. 设置 shell 脚本可执行权限
# ============================================================
info "设置 run_actor.sh 和 run_learner.sh 可执行权限..."
if [[ "${DRY_RUN}" == true ]]; then
    info "[dry-run] ssh ${REMOTE_USER}@${REMOTE_HOST} chmod +x ${REMOTE_BASE}/run_actor.sh ${REMOTE_BASE}/run_learner.sh"
else
    ssh "${REMOTE_USER}@${REMOTE_HOST}" \
        "chmod +x '${REMOTE_BASE}/run_actor.sh' '${REMOTE_BASE}/run_learner.sh'"
    ok "可执行权限已设置"
fi

# ============================================================
# 6. 验证远程文件完整性
# ============================================================
info "验证远程文件完整性..."
if [[ "${DRY_RUN}" == true ]]; then
    info "[dry-run] 跳过远程验证"
else
    REMOTE_CHECK=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<'CHECK_EOF'
MISSING=()
for f in __init__.py config.py env.py wrapper.py run_actor.sh run_learner.sh; do
    if [[ ! -f "/home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion/${f}" ]]; then
        MISSING+=("${f}")
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "MISSING:${MISSING[*]}"
else
    echo "ALL_PRESENT"
fi

# 检查可执行权限
for f in run_actor.sh run_learner.sh; do
    if [[ -x "/home/robot/serl_projects/hil-serl-fr3/experiments/plug_insertion/${f}" ]]; then
        echo "EXEC_OK:${f}"
    else
        echo "EXEC_FAIL:${f}"
    fi
done
CHECK_EOF
    )

    if echo "${REMOTE_CHECK}" | grep -q "MISSING:"; then
        fail "远程文件缺失: $(echo "${REMOTE_CHECK}" | grep 'MISSING:' | cut -d: -f2)"
        exit 1
    fi
    ok "远程文件完整性验证通过"

    if echo "${REMOTE_CHECK}" | grep -q "EXEC_FAIL"; then
        warn "部分脚本未获得可执行权限"
    else
        ok "Shell 脚本可执行权限验证通过"
    fi
fi

# ============================================================
# 7. 验证 Python import
# ============================================================
info "验证 Python import..."
if [[ "${DRY_RUN}" == true ]]; then
    info "[dry-run] 跳过 Python import 验证"
else
    IMPORT_RESULT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<'IMPORT_EOF'
cd /home/robot/serl_projects/hil-serl-fr3
python -c "
from experiments.plug_insertion.config import TrainConfig, EnvConfig
print('IMPORT_OK')
print(f'  TrainConfig.image_keys = {TrainConfig.image_keys}')
print(f'  EnvConfig.SERVER_URL = {EnvConfig.SERVER_URL}')
print(f'  EnvConfig.MAX_EPISODE_LENGTH = {EnvConfig.MAX_EPISODE_LENGTH}')
" 2>&1
IMPORT_EOF
    )

    if echo "${IMPORT_RESULT}" | grep -q "IMPORT_OK"; then
        ok "Python import 验证通过"
        echo "${IMPORT_RESULT}" | grep "^  " | while read -r line; do
            echo "    ${line}"
        done
    else
        fail "Python import 验证失败"
        echo "${IMPORT_RESULT}"
        exit 1
    fi
fi

# ============================================================
# 部署总结
# ============================================================
echo ""
echo "============================================"
echo -e "${GREEN}部署完成${NC}"
echo "============================================"
echo "  源目录:   ${LOCAL_SOURCE_DIR}"
echo "  远程目标: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}"
echo "  文件数量: ${#EXPECTED_FILES[@]}"
echo ""
echo "下一步:"
echo "  1. 在 fr3-desktop-ts 上运行: bash ${REMOTE_BASE}/run_actor.sh"
echo "  2. 在 zktitan 上运行: bash run_learner.sh"
echo "  3. 运行 verify_deployment.sh 做完整验证"
echo "============================================"
