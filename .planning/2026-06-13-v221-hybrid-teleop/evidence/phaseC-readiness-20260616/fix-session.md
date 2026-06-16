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

## 真机操作执行结果（操作者授权 + 在场 + E-stop 确认后）
- **franka_server 全栈重起** ✅ 2026-06-16 14:50：经 desktop→laptop(sshpass) 执行 HANDOFF 程序，pid 135016，
  Flask serving；desktop→/getstate(`--noproxy`) http 200(裸 curl 000 是 desktop 代理劫持，actor `trust_env=False` 不受影响)；
  机器人 q=[0,0,0,-1.57,0,1.57,0] 安全位、j6 远离下限、gripper 全开、force 小、无 fault。
- **learner fresh 重起** ✅ 2026-06-16 15:05：操作者选"从头重起"(因 90k 步零奖励 demo-only 空转)。
  - venv = `/nvme/fzt/envs/hilserl-fr3`(pyvenv.cfg)，python abs-path 自洽含 jax 0.6.2 + CudaDevice@GPU5。
  - 旧 ckpt(涨到 ~98k) → `ckpt_demoonly_bak_20260616`；新 step 0、demo buffer 9632、classifier(side_classifier) 加载 OK、5588/5589 listen。pid 3488693，CUDA_VISIBLE_DEVICES=5。
  - 踩坑(已纠)：① /proc/exe 解析成系统 /usr/bin/python3.10(无 jax) → 必须用 venv 的 bin/python；
    ② 预建空 ckpt 触发 `Press Enter to resume` 交互、detached EOFError → 改为不预建(路径不存在=fresh)。
- **zktitan config classifier_keys 修复** ✅：`/nvme/fzt/hilserl-deploy/hil-serl-fr3/experiments/plug_insertion/config.py:265`
  `[wrist_1]→[side_classifier]`(备份 + py_compile OK)。learner 也加载 classifier(给 demo relabel)，故 desktop + zktitan 两侧都需修。
- **actor 启动脚本** ✅ 新写：desktop `scripts/run_actor_phaseC.sh`（无现成正确脚本——run_actor.sh 陈旧:
  experiments/10.192.4.249:50051，连不上当前 _run_learner.py）。内置 source activation +
  SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 + 代理绕过 + `python _run_actor.py --actor --ip=162.105.195.74 --seed=0`。
- **代理劫持修复(关键)** ✅：franka_env 用裸 requests 访问 http://172.16.0.1:5000(config SERVER_URL:128)，
  desktop 有 http_proxy=127.0.0.1:7890 且 no_proxy 是 CIDR 172.16.0.0/12——**requests 不认 CIDR → 会被劫持**。
  launcher 内 `export no_proxy=172.16.0.1,...` + `unset http_proxy...`。实测:修复后 requests POST /getstate → **200**。

## 起飞前检查 (preflight 2026-06-16, 零 motion, 全 PASS)
- franka_server: desktop→/getstate(noproxy) 200，机器人 q=[0,0,0,-1.57,0,1.57,0] 安全位、gripper 全开。
- ZED 相机: ZED-M(2b03:f682) + ZED 2i(2b03:f880) 在线，/dev/video0..4。
- actor HTTP: requests(代理绕过后) → franka_server 200；SERVER_URL=http://172.16.0.1:5000(非 localhost)。
- learner: pid 3488693 活、5588/5589 listen、日志无 error(等 actor)。
- comms: desktop→zktitan 5588/5589 OPEN；actor ckpt 不存在=fresh(不弹 resume 提示)。

- **episode 间手动复位门控** ✅(应操作者要求，因机械臂自主拔插头不可靠)：`_run_actor.py` 在 between-episode
  `env.reset()`(line 216,client.update() 之后)前插 `if os.environ.get("MANUAL_RESET"): input("[episode 结束]...按 Enter 复位")`
  (备份 `_run_actor.py.bak_manualreset_20260616`，py_compile OK)；launcher 加 `export MANUAL_RESET=1`。
  行为:episode 结束(成功/超时)→ 暂停 → 操作者手动协助拔插头/复位 → 按 Enter → env.reset 笛卡尔上提到 RESET_POSE。
  约束:actor 须前台跑(input 读终端);首次启动的 reset 不门控;去掉 MANUAL_RESET 即恢复自动 reset。
- **仍待(操作者，真机 motion)**：跑 `scripts/run_actor_phaseC.sh` 起 actor + RB 引导插入 + 每条 episode 结束按 Enter 复位。我不代为启动(需手在控制器 + E-stop)。
