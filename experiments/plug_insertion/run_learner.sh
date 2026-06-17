#!/bin/bash
# ============================================================
# plug_insertion Learner 启动脚本
# ---
# 在 zktitan (fzt@162.105.195.74, 多 GPU 训练机) 上运行。
# Learner 负责：接收数据、训练 SAC 网络、发布更新后的参数。
#
# 用法:
#   bash run_learner.sh [--demo_path=demos/plug_insertion/*.pkl] [...]
# ============================================================

# -- JAX/XLA 内存配置 --
# Learner 需要更多 GPU 显存用于训练
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

# -- 训练实验根目录 --
# 注意：在 zktitan 上需要先将 hil-serl 仓库 clone 到本地
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${SCRIPT_DIR}/../.."

# -- 启动 Learner --
cd "${EXPERIMENT_ROOT}" && \
python train_rlpd.py "$@" \
    --exp_name=plug_insertion \
    --checkpoint_path="${SCRIPT_DIR}/checkpoints" \
    --demo_path="demos/plug_insertion/*.pkl" \
    --learner
    # --debug  # 取消注释以禁用 wandb 日志
