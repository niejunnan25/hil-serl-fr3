# Phase C 在线训练就绪 — 只读核验原始输出 (2026-06-16 ~14:00 CST)

> 方式：从 Mac 经 `ssh fr3-desktop-ts` 只读探测，zktitan 经 desktop 跳板 `ssh 162.105.195.74`。
> 零 motion。机器日志，不翻译。生成于 read-only 核验，未写任何远端、未动机器人。
> 触发上下文：操作者本会话已手动 go_home 复位 + reboot 恢复绿灯。

## R1 desktop 残留 actor/teleop/python 进程
```
pgrep -af "run_actor|run_learner|run_env|teleop|gello|hybrid|serl|train"  ->  (空)
```
desktop uptime: `up 1 day, 2:59`（未 reboot；reboot 的是机器人 Desk）。判定：GREEN。

## R3 franka_server / 控制栈 (laptop 172.16.0.1:5000)
```
curl -s -o ... -w "%{http_code}" -m6 -X POST http://172.16.0.1:5000/getstate  ->  http_code=000
ping -c1 -W2 172.16.0.1  ->  UP
ssh robot@172.16.0.1  ->  Permission denied (publickey,password)   # desktop 无法免密进 laptop
```
判定：RED。franka_server 未在服务（reboot 后控制栈未起）。重起 = 操作者在 laptop 的启动步。

## R2 learner (zktitan 162.105.195.74, user njn 跳板)
```
zk-host=zktitan user=njn
pgrep run_learner:
  3417680 (bash)  user=fzt  ELAPSED=01:51:35
  3417682 (python /nvme/fzt/envs/hilserl-fr3/bin/python _run_learner.py)
  cmd: --exp_name=plug_insertion --learner --debug --seed=0
       --checkpoint_path=/nvme/fzt/hilserl-deploy/ckpt
       --demo_path=<23 x /nvme/fzt/hilserl-deploy/serl19_insert/gello_demo_20260615_*_success.pkl>
       > /nvme/fzt/learner.log 2>&1
ps -p 3417680 -> ALIVE (user fzt, ELAPSED 01:51:35)
```
判定：GREEN（运行中，RLPD + 23 prior demo）。决策项：复用 vs 重起 = 操作者定。

## GPU snapshot (zktitan)
```
idx,name,mem_used,mem_total,util
0, RTX 5880 Ada,      2 MiB,  49140 MiB, 0%
1, RTX 5880 Ada,   1067 MiB,  49140 MiB, 0%
2, RTX PRO 6000,   90271 MiB, 97887 MiB, 0%   <- VLLM::EngineCore pid 2278578 (90262 MiB)  [别动]
3, RTX 5880 Ada,      2 MiB,  49140 MiB, 0%
4, RTX 5880 Ada,      2 MiB,  49140 MiB, 0%
5, RTX PRO 6000,    9027 MiB, 97887 MiB, 41%  <- learner pid 3417682 (9018 MiB)
compute-apps: 3443814/3447043 (njn 'sais', GPU0, 526MiB each)
```
判定：GREEN。GPU5=learner；GPU2=vLLM 未动。

## R5 actor<->learner 通信
```
desktop -> /dev/tcp/10.192.4.249/50051   ->  closed/unreachable
desktop -> /dev/tcp/162.105.195.74/50051 ->  closed/unreachable
learner 监听: run_learner.sh LEARNER_IP=0.0.0.0:50051
actor 连接:   run_actor.sh   LEARNER_IP=10.192.4.249:50051 (tailscale); ACTOR_IP=10.192.4.136
```
判定：AMBER。learner 在监听但 desktop 当前到 zktitan:50051 不通（tailscale/防火墙）。actor 启动前必验。

## Xbox 手柄 (desktop, RB bootstrap 必需)
```
/dev/input/js0  crw-rw-r--+ root input  (6/15 11:01)
lsusb: Bus 001 Device 008: ID 045e:0b12 Microsoft Corp. Xbox Wireless Controller (model 1914)
```
判定：GREEN（STATE.md "无 /dev/input/js*" 已过时）。

## R4 LIVE 训练配置 (experiments/plug_insertion/config.py, 校准 2026-06-16)
```
ACTION_SCALE = [0.015,0.015,0.015, 0.1,0.1,0.1, 1.0]
TARGET_POSE  = [0.6743,-0.0074,0.1355(seated z), 0.07951,-0.09165,0.01549]  (euler_2_quat domain)
RESET_POSE   = [0.6259,-0.0161,0.1925(=seated+0.057), 0.03517,0.03167,0.01783]
RANDOM_XY_RANGE=0.01  RANDOM_RZ_RANGE=0.1
ABS_POSE_LIMIT_HIGH=[0.724,0.062,0.267,pi,pi,pi]  LOW=[0.559,-0.059,0.039,-pi,-pi,-pi]
COMPLIANCE clips: translational_clip_z=0.0035, clip_x=0.006, clip_y=0.0059, neg~0.005; rot~0.02/0.015
image_keys=[side_policy,wrist_1]  classifier_keys=[wrist_1]  proprio=[tcp_pose,tcp_vel,tcp_force,tcp_torque,gripper_pose]
discount=0.98  setup_mode="single-arm-learned-gripper"
reward = int( (sigmoid(classifier(obs))>0.7) and (obs.state[2] < -0.05) )   # CALIBRATED 2026-06-16
wrapper stack: FrankaEnv -> HoldGripper -> [real] XboxIntervention -> RelativeFrame -> Quat2Euler
               -> SERLObs -> Chunking(1) -> [clf] MultiCameraBinaryRewardClassifier -> GripperPenalty(-0.02)
```
注：`plug_zed_insertion/config.py` 仅 4 行桩，非 live。GelloIntervention 已 import 但 plug_insertion 仅用 XboxIntervention。
判定：GREEN（ACTION_SCALE 正确；reward 已校准）。

## reward classifier
```
classifier_ckpt/reward_classifier.pt      44789387 bytes  (6/15 21:16)
classifier_ckpt/checkpoint_100/           (dir)
classifier_ckpt/training_history.json     43330 bytes
```
判定：GREEN（TRAIN-01 已完成）。classifier_keys=[wrist_1]（与 DECISION 文档 side_classifier 分歧，待核）。

## demos
```
/home/robot/hilserl-fr3/demos/hybrid/        20+ x gello_demo_20260615_*_success.pkl (+ _raw_success.npz, fails, _orphans)
zktitan: /nvme/fzt/hilserl-deploy/serl19_insert/  23 x *_success.pkl (learner 在用)
desktop training-side data/demos             空 (learner 用显式 --demo_path，非自动扫描)
```
判定：GREEN（DATA-03 大头已采）。

## launch surface (desktop /home/robot/serl_projects/hil-serl-fr3)
```
env:     source env/activation.sh   (SERL_FR3_ENV=/home/robot/miniconda3/envs/hilserl-fr3, ROBOT_PORT=5017,
                                      PYTHONPATH += upstream/hil-serl/serl_robot_infra, CUDA_ROOT for JAX 0.4.35)
actor:   scripts/run_actor.sh        (CONDA_ENV=hilserl-fr3, FRANKA_PORT=5000, LEARNER_IP=10.192.4.249:50051,
                                      EXPERIMENT=plug_insertion)  == python -m experiments.plug_insertion.run_actor
                                      需 SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1
learner: scripts/run_learner.sh      (LEARNER_IP=0.0.0.0:50051, ACTOR_IP=10.192.4.136, --resume_from N)
no-motion validator: tools/no_motion_validate.py (binds 127.0.0.1:5017, rejects /pose) -> "NO_MOTION_VALIDATION_OK"
verify_infra.py: has check_zktitan_ssh()
```

## desktop 旧 runbooks (5/31, Codex; 历史, 部分过时)
00_environment.md / 01_no_motion_validation.md / 02_fr3_control_bridge_options.md / 03_live_robot_gate.md
