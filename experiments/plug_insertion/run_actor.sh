#!/bin/bash
# ============================================================
# plug_insertion Actor 启动脚本
# ---
# 在 fr3-desktop-ts (机器人工作站) 上运行。
# Actor 负责：采集数据、执行策略、发送 transition 给 learner。
#
# 用法:
#   bash run_actor.sh [--ip=<learner_ip>] [--demo_path=...] [...]
# ============================================================

# -- JAX/XLA 内存配置 --
# Actor 只做推理，不需要大量 GPU 显存
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

# -- 训练实验根目录 --
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="${SCRIPT_DIR}/../.."

# -- 启动 Actor --
cd "${EXPERIMENT_ROOT}" && \
python train_rlpd.py "$@" \
    --exp_name=plug_insertion \
    --checkpoint_path="${SCRIPT_DIR}/checkpoints" \
    --actor \
    --ip=162.105.195.74
    # --debug  # 取消注释以禁用 wandb 日志
