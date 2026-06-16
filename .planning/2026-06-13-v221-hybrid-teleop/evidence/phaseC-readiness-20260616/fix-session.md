# Phase C 修复会话 — 只读核验 + 一处 live config 修复 (2026-06-16, 授权后)

> 授权范围：所有真机操作 + 后续修复。本会话执行：零 motion 的数据/通信/分类器核验与修复。
> 真机 motion(控制栈重起 / RB 引导训练)留待操作者现场握手。机器日志，不翻译。

## 通信真相（修正先前 R5）
- 运行中 learner = `_run_learner.py`（zktitan，user fzt，pid 3417680/3417682，GPU5，1h51m），
  agentlace 监听 **0.0.0.0:5588(port_number) + 5589(broadcast_port)**（make_trainer_config），**非 50051**。
- desktop→zktitan 公网可达实测：`162.105.195.74:5588 OPEN`，`:5589 OPEN`，`:5555 closed`，`:5556 closed`，`:50051 closed`，`:22 OPEN`。
- desktop tailscale tailnet **不含 zktitan**（`tailscale ping 10.192.4.249` → no matching peer；`ip route` 经 LAN 网关）。
- 结论：**无需 tailscale / 隧道**；actor 直连 `--ip 162.105.195.74`。先前"comms RED@50051"是测错端口。
- 匹配 actor = `_run_actor.py --exp_name=plug_insertion --actor --ip=162.105.195.74 --checkpoint_path=...`
  （absl flags：exp_name/seed/learner/actor/ip/demo_path/checkpoint_path/eval_*/save_video/debug；`ip` 默认 localhost）。
- `scripts/run_actor.sh` / `run_learner.sh` 走 `experiments.*.run_*`（ActorServer，LEARNER_IP=10.192.4.249:50051）=
  **另一套实现、与运行中 learner 不配对、且 10.192.4.249 不可达**。勿用。

## demo provenance（CHECKLIST #1/#2/#5）— 核为 GREEN
- desktop 源副本 `/home/robot/hilserl-fr3/demos/hybrid/*success.pkl`：30 条；首条 list len 1264。
- transition keys = `[observations, next_observations, actions, rewards, masks, dones]`（SERL 格式）。
- obs keys = `[state, side_policy, wrist_1, side_classifier]`；`state (25,) f32`；图像 `(3,128,128) uint8`。
  → **25D tcp-空间 obs + 3 路图，非 8D joint state**。
- actions `(N,7)`，逐维 min≈[-0.896,-1,-1,-0.5,-0.5,-0.128,-1]，max≈[1,1,1,0.5,0.5,0.165,1] → **归一化 [-1,1]**。
- gripper col6 ∈ {-1, +1} → **SERL ±1 约定（非 raw GELLO 1=closed）**。
- 结论：无 8D-joint / 夹爪反号 / ~50cm-FK 污染。demo buffer 可信。

## reward classifier 相机（CHECKLIST #6）— 实测 + 修复
分类器 = `classifier_ckpt/reward_classifier.pt`(PyTorch) + `checkpoint_100/`(JAX/Orbax)。
离线 CPU 推理测试（torch 来自 env evo-rl-5080；样本各 24，success vs failure pkl）：
```
side_classifier  succ=0.914  fail=0.104  sep=+0.810
wrist_1          succ=0.053  fail=0.045  sep=+0.007   <- 失明
side_policy      succ=0.914  fail=0.104  sep=+0.810
```
→ ckpt 训练于侧相机；**wrist_1 完全不可分**。
collect_classifier_images.py 默认 `camera_name="side_classifier"`，与上述一致。

**修复**：`experiments/plug_insertion/config.py:266`
`classifier_keys = ["wrist_1"]` → `classifier_keys = ["side_classifier"]`
（备份 `config.py.bak_classifierkeys_20260616`；`py_compile` OK；side_classifier 是 REALSENSE_CAMERAS 已配相机=外置 ZED 36276705 分类裁剪）。
未修则 Phase C reward = 噪声、训练不收敛。

## actor 运行时 classifier 加载
- actor env `hilserl-fr3`：jax 0.6.2 ✓，**无 torch**。
- `_load_classifier_adaptive` 优先 JAX 路径（`load_classifier_func` + Orbax `checkpoint_100`）；`.pt`(torch) 仅回退。
- 故 torch 缺失非阻塞；首次 actor 起动确认 classifier 加载即可。

## 仍 RED（留操作者现场握手）
- franka_server :5000 DOWN（http 000）。重起程序（HANDOFF 验证、sshpass 可用）见 RUNBOOK §2。motion-相邻，需在场 + E-stop。
