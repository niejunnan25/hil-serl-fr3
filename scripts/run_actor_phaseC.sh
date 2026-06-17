#!/usr/bin/env bash
# Phase C 在线训练 actor 启动器 — 匹配 zktitan 运行中的 _run_learner.py (agentlace 5588/5589)。
# 2026-06-16 新增(含代理绕过修复)。用本脚本,不要用 run_actor.sh(陈旧:experiments/10.192.4.249:50051)。
# 真机 motion:启动后 env.reset() 会把末端移到 RESET_POSE —— 操作者必须在场 + E-stop 在手。
set -uo pipefail
ROOT=/home/robot/serl_projects/hil-serl-fr3
LEARNER_IP="${LEARNER_IP:-162.105.195.74}"
CKPT="${CKPT:-$ROOT/artifacts/checkpoints/plug_insertion_phaseC_20260616}"
cd "$ROOT" || exit 1
source env/activation.sh
export SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1
export SDL_VIDEODRIVER=dummy        # 与 hubtest 一致;cv2(GTK/Qt) 不受影响
# 绕开 desktop 代理:franka_env 用裸 requests 访问 http://172.16.0.1:5000,
# no_proxy 的 CIDR(172.16.0.0/12) 不被 requests 认 → 会被 127.0.0.1:7890 劫持。用精确 IP + 清 proxy。
export no_proxy="172.16.0.1,localhost,127.0.0.1,::1"
export NO_PROXY="172.16.0.1,localhost,127.0.0.1,::1"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export MANUAL_RESET=1                                          # episode 结束后等操作者按 Enter 再 reset(插头需手动协助拔出)
export MANUAL_SUCCESS=1                                        # 在线真机由操作者输入 1/0 裁决成功, classifier 仅作提示
export RESET_STRICT="${RESET_STRICT:-1}"                       # reset 未收敛直接停,禁止带坏位姿进入下一 episode
export FRANKA_CLEARERR_ON_POSE="${FRANKA_CLEARERR_ON_POSE:-0}" # 不在每步 pose command 前清错; 只在 reset/recover 显式清错
export FRANKA_ACTION_MAX_STEP="${FRANKA_ACTION_MAX_STEP:-0.0065}"
export FRANKA_ACTION_MAX_ROT_STEP="${FRANKA_ACTION_MAX_ROT_STEP:-0.045}"
export FRANKA_ACTION_DBG_PERIOD="${FRANKA_ACTION_DBG_PERIOD:-1.0}"
export ACTOR_SAFETY_DQ_MAX="${ACTOR_SAFETY_DQ_MAX:-0.35}"
export ACTOR_SAFETY_FORCE_MAX="${ACTOR_SAFETY_FORCE_MAX:-45.0}"
export FRANKA_SAFETY_DQ_MAX="${FRANKA_SAFETY_DQ_MAX:-$ACTOR_SAFETY_DQ_MAX}"
export FRANKA_SAFETY_FORCE_MAX="${FRANKA_SAFETY_FORCE_MAX:-$ACTOR_SAFETY_FORCE_MAX}"
export ACTOR_RELZ_ABS_MAX="${ACTOR_RELZ_ABS_MAX:-0.35}"
export ACTOR_SAFETY_LOG_PERIOD="${ACTOR_SAFETY_LOG_PERIOD:-1.0}"
echo "[actor] learner_ip=$LEARNER_IP  ckpt=$CKPT  (proxy cleared for 172.16.0.1)"
echo "[actor] safety RESET_STRICT=$RESET_STRICT CLEARERR_ON_POSE=$FRANKA_CLEARERR_ON_POSE ACTION_MAX_STEP=$FRANKA_ACTION_MAX_STEP DQ_MAX=$ACTOR_SAFETY_DQ_MAX FORCE_MAX=$ACTOR_SAFETY_FORCE_MAX RELZ_ABS_MAX=$ACTOR_RELZ_ABS_MAX"
echo "[actor] manual_success: 运行中在本终端输入 1=成功提前结束当前 episode, 0=失败提前结束; 不必等自动结束。"
echo "[actor] 手柄/插头就位? E-stop 在手? 起来后第一个 env.reset 即真机 motion。"
python -u _run_actor.py --exp_name=plug_insertion --actor --ip="$LEARNER_IP" \
  --checkpoint_path="$CKPT" --seed=0 "$@" 2>&1 | tee /tmp/actor.log
