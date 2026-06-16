# RUNBOOK — Phase C 在线训练 (hilserl-fr3 v2.2.1 hybrid teleop)

目的:把已运行的 learner(zktitan GPU5,RLPD + 23 条 demo buffer)与真机 actor 接通,用 RB deadman 引导 bootstrap,跑通 plug_insertion 在线训练直到 EVAL-01 Stage-1。

> 前置:你必须在场 + E-stop 触手可及。本 runbook 的所有 motion 步骤默认会动真机,任何一步出现 pose-limit / 力异常,先拍 E-stop 再排查。

---

## 0. 就绪门快照 (2026-06-16 只读核验)

下表为本会话只读探测结果(零 motion)。启动前逐项确认;RED 必须先修。

| item | 状态 | 实测 |
|---|---|---|
| 机器人复位 (go_home + 绿灯) | 🟢 | 本会话操作员已 reset 到 go_home,绿灯已恢复 |
| desktop 残留 actor/python 进程 | 🟢 | 无,环境干净 |
| Xbox 手柄 | 🟢 | /dev/input/js0 在位 + lsusb 045e:0b12 Microsoft Xbox Wireless Controller (model 1914),RB 可用(STATE.md "no /dev/input/js*" 阻塞为 STALE) |
| learner (zktitan GPU5) | 🟢 | RUNNING,pid 3417680(bash)/3417682(python),user fzt,ELAPSED 1h51m,exp_name=plug_insertion,23 条 *_success.pkl demo buffer,RLPD(非 BC) |
| reward classifier checkpoint | 🟢 | classifier_ckpt/checkpoint_100/(JAX/Orbax,首选)+ reward_classifier.pt(PyTorch 回退);actor env hilserl-fr3 有 jax 0.6.2、无 torch → 走 JAX 路径加载 |
| classifier 训练相机 | 🟢 已修 | 实测分类器仅侧相机可分(side_classifier sep +0.810 / wrist_1 sep +0.007=失明);已把 config.py:266 `classifier_keys` 由 `[wrist_1]` 改为 `[side_classifier]`(备份 config.py.bak_classifierkeys_20260616,py_compile OK) |
| demo buffer (30 *_success.pkl) | 🟢 | 已核 desktop 源副本:SERL transition 格式、obs=25D tcp state + side_policy/wrist_1/side_classifier 图、action∈[-1,1]、gripper∈{-1,+1}(SERL 约定)。无 8D-joint/夹爪反号/~50cm-FK 污染 |
| live training config | 🟢 | experiments/plug_insertion/config.py(2026-06-16 标定 + 本次 classifier_keys 修复);plug_zed_insertion/config.py 是 4 行 stub,非 live |
| franka_server :5000 (laptop) | 🔴 | DOWN,curl -X POST http://172.16.0.1:5000/getstate → http_code=000;laptop ping UP。控制栈必须由操作员在 laptop 重起(见 §2) |
| actor↔learner comms | 🟢 | 实测 desktop→zktitan 162.105.195.74:**5588+5589 OPEN**(agentlace port_number+broadcast_port,非 50051)。运行中 learner=`_run_learner.py`;匹配 actor=`_run_actor.py --ip 162.105.195.74`(见 §3/§4) |
| Desk System Image | 🟠 | UNKNOWN,需 Desk UI 人工核验(操作员) |

---

## 1. 复位 (已由你完成)

本会话你已完成 go_home + 绿灯恢复,**本节无需再操作**。

如后续需要再次复位:**只允许 Cartesian reset_to_home**。

- 禁用 `/jointreset`(真机死路:PositionJointInterface 找不到 fr3_joint1,impedance franka_control 占着 FCI,controller switch 失败)。
- 禁用 `restart_imp.sh`(会让 franka_server 的 self.imp 指向死进程)。

---

## 2. 重起控制栈 / franka_server [你, laptop 172.16.0.1] — 真机步骤

franka_server :5000 现在是 DOWN(http_code=000)。在 **laptop(RT 控制主机)** 上重起 **整个 franka_server**——它会自动拉起 roscore + impedance + franka_control。不要单独碰 restart_imp.sh,不要盲目重试 rosservice(会把 franka_control 卡死)。

PYTHONPATH 必须包含 serl_robot_infra,否则 franka_server 起不来。重起后 sleep ~34s 等 roscore+impedance+franka_control 全部就绪。

[laptop, 真机] 重起整个 franka_server(本会话已验证可用,逐字源 HANDOFF-2026-06-16):

```bash
# 经 desktop 跳板登录 laptop（口令 123456）
sshpass -p 123456 ssh robot@172.16.0.1

# --- 以下在 laptop 上执行,python=/usr/bin/python3.8 ---
# 1) kill 整棵树(franka_server 是 roscore+impedance+franka_control 的树根)
pkill -9 -f franka_server.py; pkill -9 -f impedance.launch; pkill -9 -f franka_control_node
pkill -9 -f controller_manager/spawner; pkill -9 -f joint_state_publisher; pkill -9 -f rosout
sleep 2; pkill -9 -f rosmaster; pkill -9 -f "roscore -p"; sleep 5

# 2) 重起(关键:PYTHONPATH 必须含 serl_robot_infra,否则 robot_servers import 失败)
cd /home/robot/serl_projects/hilserl-fr3-control/src/hil-serl/serl_robot_infra
source /opt/ros/noetic/setup.bash
source /home/robot/serl_projects/hilserl-fr3-control/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://localhost:11311
export PYTHONPATH=$PWD:$PYTHONPATH
setsid nohup python3 robot_servers/franka_server.py --robot_ip=172.16.0.2 \
  --gripper_type=Franka --flask_url=0.0.0.0 --ros_port=11311 > /tmp/fsrv.log 2>&1 < /dev/null &
disown
sleep 34   # 等 roscore+impedance+franka_control 起来
```

重起期间机器人靠 FCI 内部 hold 保持位置不动(安全);启动不 home 夹爪 → 插头不掉。

[desktop, 只读 liveness 探针] 从 desktop 验 franka_server 活着(期望非空 state,而非 http 000):

```bash
curl -s -m6 -X POST http://172.16.0.1:5000/getstate
```

- 期望:返回非空 state JSON。
- 失败判据:http_code=000 / 空响应 → 控制栈未起,**回到本节重起,不要继续往下**。
- 切记:不要 spam rosservice;不要盲目重试。

---

## 3. actor↔learner 通信 [已核验 GREEN]

运行中 learner = `_run_learner.py`(zktitan,user fzt,pid 3417680/3417682,GPU5,已跑 1h51m),agentlace 监听 **0.0.0.0:5588(port_number)+ 5589(broadcast_port)**(make_trainer_config 端口,**非** 50051)。

[desktop, 只读诊断] 已实测 desktop→zktitan 公网端口可达:

```bash
for p in 5588 5589; do timeout 5 bash -c "echo > /dev/tcp/162.105.195.74/$p" && echo "$p OPEN" || echo "$p closed"; done
```

- 实测:**5588 OPEN / 5589 OPEN** → 直接走公网 IP `162.105.195.74`,**无需 tailscale、无需 SSH 隧道**。
- 坑:`scripts/run_actor.sh` 配的是陈旧死路(`LEARNER_IP=10.192.4.249:50051`,desktop 不可达,且是另一套 ActorServer 实现,与运行中的 `_run_learner.py` 不配对)。**不要用 run_actor.sh**;匹配 actor 见 §4(`_run_actor.py --ip 162.105.195.74`)。

reuse vs restart learner:
- **优先 reuse** 现 learner(已有 23 条 demo buffer,checkpoint_path=`/nvme/fzt/hilserl-deploy/ckpt`)。直接接 actor 即可。
- 仅当 learner 已死 / checkpoint 异常 / 换配置时才重起。zktitan **GPU2 vLLM(pid 2278578,90GB)绝对不要碰**;只 kill 自己进程。

---

## 4. 起 actor + 进入在线训练 [你, motion] — 真机步骤

确认 §2 franka_server 活着、§3 通信(5588/5589)就绪后,在 **desktop(actor 主机)** 起 actor。**必须用 `_run_actor.py`(匹配运行中的 `_run_learner.py`),不要用 `run_actor.sh`(陈旧死路 10.192.4.249:50051)。** experiment=plug_insertion,RLPD,classifier 默认开,无 BC。

[desktop, 真机] 起 actor:

```bash
cd /home/robot/serl_projects/hil-serl-fr3
source env/activation.sh                       # conda hilserl-fr3 + PYTHONPATH + CUDA_ROOT(JAX)
export SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1   # 后台手柄事件(RB 死手必需)
python _run_actor.py --exp_name=plug_insertion --actor \
  --ip=162.105.195.74 \
  --checkpoint_path=<本机 actor ckpt 目录(与上一轮/采集约定一致)> \
  --seed=0
```

- `--ip=162.105.195.74` = learner 公网 IP(agentlace 5588/5589,已实测可达)。**不要**用 run_actor.sh / 10.192.4.249:50051。
- 依赖:`SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` + §2 franka_server up + 手柄在位(已 GREEN)。
- classifier 默认开;reward classifier 走 JAX/Orbax(checkpoint_100)路径(actor env 有 jax 无 torch);**classifier_keys 已修为 side_classifier**(见 §0)。
- checkpoint_path = actor 本机 buffer/ckpt 目录;learner 自己的 ckpt 在 zktitan `/nvme/fzt/hilserl-deploy/ckpt`。
- env 包装栈(顺序):FrankaEnv(fake_env) → HoldGripperWrapper → [if not fake_env] XboxIntervention → RelativeFrame → Quat2EulerWrapper → SERLObsWrapper(proprio_keys) → ChunkingWrapper(obs_horizon=1, act_exec_horizon=None) → [if classifier] MultiCameraBinaryRewardClassifierWrapper(reward_func) → GripperPenaltyWrapper(penalty=-0.02)。真机 actor fake_env=False(含 XboxIntervention);learner 端 fake_env=True(不连真机、无 Xbox)。GelloIntervention 已 import 但 Phase C **不启用**(Xbox-only)。

---

## 5. RB 引导 bootstrap [你, motion, human-in-loop] — 真机步骤

RB = deadman 接管。用它在开局把策略从随机种成"会插"。

RB deadman 语义:

- **按住 RB + 推摇杆 = 你接管**(策略被覆盖,只用你的输入)。
- **松开 RB = 交还策略**(0.4s debounce 去抖)。
- **按住 RB 不动 = 保持不动**。
- HoldGripperWrapper 锁夹爪,接管期间插头不会掉。
- **每一步接管都会被记录为训练数据**。

操作流程(显式 start/stop 信令,你在终端直接掌控时序):

1. **开局 seed**:多次按住 RB 把插头插进去,给随机策略喂正样本。每次接管前在心里(或口头)报"接管开始",松开 RB 报"交还"。
2. **逐步放手**:随训练进行,越来越多地松开 RB,让策略自己尝试,只在它要出界 / 卡死时再按住 RB 救场。
3. 全程盯住:任何异常先松手 / 拍 E-stop。

Reward 触发判据(2026-06-16 标定):仅当 `classifier > 0.7` **且** `rel-z < -0.05`(相对 RESET_POSE 真实下插 ≥5cm)才给 reward=1。`relz = obs["state"][2]`(RelativeFrame z vs RESET_POSE),`cls = sigmoid(classifier_fn(obs))[0]`。光满足分类器、没真插下去 = 不给奖励。

---

## 6. 监看 / 成功判据

成功判据:**EVAL-01 Stage-1 = ≥3 次自主插入成功**(松开 RB、策略自主完成的插入)。

监看入口:

- learner 日志(zktitan):`/nvme/fzt/learner.log`
- checkpoints(zktitan):`/nvme/fzt/hilserl-deploy/ckpt`

[desktop, 真机] eval 某个 checkpoint:

```bash
scripts/run_actor.sh --eval_checkpoint_step N --eval_n_trajs M
```

- `--eval_checkpoint_step N`:载入第 N 步 checkpoint。
- `--eval_n_trajs M`:跑 M 条评估轨迹。
- eval 仍是真机 motion,E-stop 在手。

---

## 7. ABORT / 安全

- **E-stop**:任何 pose-limit 触发、力异常、行为不可预测 → 先拍 E-stop。
- motion 默认 blocked,需操作员在场 + E-stop 在手 + 显式 approval env var 才放行。
- `max_step` 在任何路径上都**不可绕过**。
- 历史教训:`communication_test` 曾让 FR3 意外移动 —— 不要随手跑会发指令的脚本。
- **干净停机**:在 actor 终端 `Ctrl-C` 停 actor(motion 立即停)。learner 不动,继续保留 checkpoint 在 `/nvme/fzt/hilserl-deploy/ckpt`,下次可直接接回或 `--resume_from`。
- 复位禁令(重申):禁用 `/jointreset`、禁用 `restart_imp.sh`;复位只走 Cartesian reset_to_home。

---

## 范围声明

本 runbook 的 motion 步骤(§2 重起控制栈、§4 起 actor、§5 RB bootstrap、§6 eval)均为 **operator-only**,必须由操作员在场亲自执行。§0 就绪门快照与 §3 的可达性诊断为 **只读核验**(read-only,零 motion)。
