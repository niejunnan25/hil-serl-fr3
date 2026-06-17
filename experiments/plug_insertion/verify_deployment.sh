#!/bin/bash
# ============================================================
# verify_deployment.sh -- 验证 plug_insertion 部署状态
# ---
# SSH 到 fr3-desktop-ts，检查文件存在性、权限和 Python import。
#
# 用法:
#   bash verify_deployment.sh
# ============================================================

set -euo pipefail

# -- 配置 --
REMOTE_HOST="fr3-desktop-ts"
REMOTE_USER="robot"
REMOTE_PROJECT="/home/robot/serl_projects/hil-serl-fr3"
REMOTE_EXPERIMENT="${REMOTE_PROJECT}/experiments/plug_insertion"

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

PASS=0
FAIL=0
WARN=0

check_pass() { ((PASS++)); echo -e "${GREEN}[PASS]${NC} $*"; }
check_fail() { ((FAIL++)); echo -e "${RED}[FAIL]${NC} $*"; }
check_warn() { ((WARN++)); echo -e "${YELLOW}[WARN]${NC} $*"; }
info()       { echo -e "[INFO] $*"; }

echo "============================================"
echo "plug_insertion 部署验证"
echo "============================================"
echo "  目标: ${REMOTE_USER}@${REMOTE_HOST}"
echo "  路径: ${REMOTE_EXPERIMENT}"
echo "============================================"
echo ""

# ============================================================
# 1. SSH 连通性
# ============================================================
info "检查 SSH 连通性..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
    check_pass "SSH 连接到 ${REMOTE_HOST}"
else
    check_fail "无法连接到 ${REMOTE_USER}@${REMOTE_HOST}"
    echo "  验证终止。"
    exit 1
fi

# ============================================================
# 2. 项目目录存在性
# ============================================================
info "检查项目目录..."
if ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d '${REMOTE_PROJECT}'" 2>/dev/null; then
    check_pass "hil-serl-fr3 项目目录存在"
else
    check_fail "项目目录不存在: ${REMOTE_PROJECT}"
fi

if ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d '${REMOTE_EXPERIMENT}'" 2>/dev/null; then
    check_pass "plug_insertion 实验目录存在"
else
    check_fail "实验目录不存在: ${REMOTE_EXPERIMENT}"
fi

# ============================================================
# 3. 文件存在性检查
# ============================================================
info "检查文件存在性..."
FILE_CHECK=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<CHECK_EOF
for f in ${EXPECTED_FILES[*]}; do
    if [[ -f "${REMOTE_EXPERIMENT}/\$f" ]]; then
        echo "EXISTS:\$f"
    else
        echo "MISSING:\$f"
    fi
done
CHECK_EOF
)

for f in "${EXPECTED_FILES[@]}"; do
    if echo "${FILE_CHECK}" | grep -q "EXISTS:${f}"; then
        check_pass "文件存在: ${f}"
    else
        check_fail "文件缺失: ${f}"
    fi
done

# ============================================================
# 4. Shell 脚本可执行权限
# ============================================================
info "检查 shell 脚本权限..."
EXEC_CHECK=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<CHECK_EOF
for f in run_actor.sh run_learner.sh; do
    if [[ -x "${REMOTE_EXPERIMENT}/\$f" ]]; then
        echo "EXEC_OK:\$f"
    else
        echo "EXEC_FAIL:\$f"
    fi
done
CHECK_EOF
)

for f in run_actor.sh run_learner.sh; do
    if echo "${EXEC_CHECK}" | grep -q "EXEC_OK:${f}"; then
        check_pass "可执行权限: ${f}"
    else
        check_fail "缺少可执行权限: ${f}"
    fi
done

# ============================================================
# 5. Python import 验证
# ============================================================
info "验证 Python import..."
IMPORT_RESULT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<'IMPORT_EOF'
cd /home/robot/serl_projects/hil-serl-fr3
python -c "
from experiments.plug_insertion.config import TrainConfig
print('IMPORT_OK')
print(f'  image_keys:       {TrainConfig.image_keys}')
print(f'  classifier_keys:  {TrainConfig.classifier_keys}')
print(f'  proprio_keys:     {TrainConfig.proprio_keys}')
print(f'  encoder_type:     {TrainConfig.encoder_type}')
print(f'  setup_mode:       {TrainConfig.setup_mode}')
" 2>&1
IMPORT_EOF
)

if echo "${IMPORT_RESULT}" | grep -q "IMPORT_OK"; then
    check_pass "Python import: TrainConfig"
    # 打印配置摘要
    echo "${IMPORT_RESULT}" | grep "^  " | while read -r line; do
        echo "    ${line}"
    done
else
    check_fail "Python import 失败"
    echo "  详细错误:"
    echo "${IMPORT_RESULT}" | sed 's/^/    /'
fi

# ============================================================
# 6. 可选：检查 Python 依赖
# ============================================================
info "检查关键 Python 依赖..."
DEP_CHECK=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<'DEP_EOF'
cd /home/robot/serl_projects/hil-serl-fr3
python -c "
import importlib
deps = ['jax', 'jax.numpy', 'numpy', 'gymnasium', 'flax']
for d in deps:
    try:
        importlib.import_module(d)
        print(f'DEP_OK:{d}')
    except ImportError:
        print(f'DEP_FAIL:{d}')
" 2>&1
DEP_EOF
)

if echo "${DEP_CHECK}" | grep -q "DEP_FAIL"; then
    check_warn "部分 Python 依赖缺失:"
    echo "${DEP_CHECK}" | grep "DEP_FAIL" | sed 's/DEP_FAIL:/    缺失: /'
else
    check_pass "关键 Python 依赖检查通过 (jax, numpy, gymnasium, flax)"
fi

# ============================================================
# 汇总
# ============================================================
echo ""
echo "============================================"
echo "验证结果汇总"
echo "============================================"
echo -e "  ${GREEN}通过: ${PASS}${NC}"
echo -e "  ${RED}失败: ${FAIL}${NC}"
echo -e "  ${YELLOW}警告: ${WARN}${NC}"
echo "============================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo -e "${RED}验证未通过，请检查上述失败项。${NC}"
    exit 1
else
    echo -e "${GREEN}所有检查通过，部署验证成功。${NC}"
    exit 0
fi
