# STATE — v2.2.1 Hybrid Teleop（real 机器人线）

> **拆分说明（2026-06-13）**：v2.2.1（real）与 v2.1.1（sim）原共用顶层 `.planning/STATE.md`，
> 两会话并发编辑互相覆盖。现拆分：**v2.2.1 的权威状态在本文件**，v2.1.1（sim 会话）用它自己的
> milestone 目录 / 顶层 STATE。顶层 `.planning/STATE.md` 只保留跨 milestone 索引 + 指针。
> 本会话以后只更新本文件，不再写顶层 STATE 的 v2.2.1 内容。

## 是什么
v2.2.1 = FR3 真机插插头 HIL-SERL 的 real 机器人线（sim 线 = v2.1.1，独立会话）。
GELLO 抓取段示教 + Xbox 精插段示教/介入，纯端到端单 SAC policy。两线 fork，日后并主线。
设计决策见 `DECISIONS.md`；契约见 `B-RESEARCH.md`；review 见 `REVIEW-PhaseA.md`。

## 执行策略
- grill/discuss/plan/review：Opus 最高 effort（Fable 5 已停用，见全局 model-stage-policy）。
- execute：Opus 4.8 xhigh。安全关键改动做独立对抗式 review。
- 真机 motion：显式 approval 环境变量 + 用户现场 + E-stop；默认 blocked。max_step 任何路径不可绕过。
- 变更随做随 commit；显式 pathspec（git 根 /Users/tacyvan/Documents/Code），禁 `git add .`，不擅自 push。

## 进度

```
Phase A 遥操作工具链(软件)   ✅ CLOSED + A6 gap-closure ✅
Phase B 真机验收             🟡 B1a✅ B2a✅ | B1b(GELLO e2e)✅真机 | B2b(Xbox e2e)✅真机 | B3 切换/B4 实标 待
Phase C Stage-1 纯插入训成   ⏳ (依赖 B + 夹爪 actuation + 录制器 + ZED obs)
Phase D 完整过程端到端训成   ⏳ = 收官判据
```

### 真机验证 ✅ 2026-06-13（B1b/B2b）
两条 teleop 栈在真机 FR3 上验证通过（详见下"关键发现/修复"）：
- **GELLO e2e** ✅：拨主臂→FR3 跟随，净~12mm，方向正确，force<2N
- **Xbox e2e** ✅：左摇杆→X/Y(右→+X、前→−Y，顺手)、右摇杆→yaw、D-pad→pitch/roll、RB 死手；映射逐项实测
- **核心方案 = relative-pose（绕开坏 FK）**：repo 内 DH FK 与 server O_T_EE 差~50cm，故不用 FK(joint_target)；改为每 tick 读真机 currpos(/getstate) + 笛卡尔增量 → POST /pose（Xbox 摇杆→增量；GELLO 关节增量→机器人 Jacobian→增量）。每步 3mm 限幅。代码：scripts/relative_teleop.py(+test_relative_teleop.py 6 tests)。

### 关键发现/修复（真机集成踩的坑，已全部解决）
1. **控制栈在 fr3-laptop**(172.16.0.1, RT 内核)，**robot FCI=172.16.0.2**；desktop(172.16.0.4)=actor，经有线访问 laptop:5000。用户口述的 .1/.2 与实际相反，以 laptop ip addr 为准。
2. **断电上电默认 boot 非 RT 内核**→franka_control RealtimeException abort。已把 **GRUB 默认永久设为 5.9.1-rt20**(by entry-id)。
3. franka_server 启动：需 `FR3_REAL_FRANKA_SERVER_APPROVAL` + start_impedance 传 `allow_motion:=true`(补丁，清理时还原、engage 时重打) + `PYTHONPATH=...serl_robot_infra`(否则 robot_servers import 失败) + `--gripper_type=Franka --robot_ip=172.16.0.2 --flask_url=0.0.0.0`；缺 scipy 已 pip --user 装。
4. **desktop 代理劫持内网**：desktop 有 http_proxy(127.0.0.1:7890)，no_proxy 用 CIDR(172.16.0.0/12)不被 curl/requests 认 → :5000 被劫持。修：actor 用 `requests Session.trust_env=False`(relative_teleop 已内置)/ `--noproxy`。
5. **Xbox(teleop_hub)两个 bug**：pygame 只 joystick.init() 没 pygame.init()→event pump 空操作→读不到输入(修:SDL dummy+pygame.init)；轴布局错位(修:右摇杆=轴3/4、扳机=轴2/5 remap、D-pad=hat0、View/Menu=6/7)。
6. SSH：laptop 重置过 host key + 用户加了外置 WiFi(192.168.0.135，后断开)；现经 desktop 有线跳板 + 已装我的 pubkey 访问。

### 示教采集前置已就绪 ✅ 2026-06-15（GELLO 全流程录制器）
- **夹爪 actuation** ✅：relative_teleop gripper_edge + post_gripper（真机 5 次开合验证，提交 ad4d541）。
- **录制器** ✅：`scripts/gello_demo_recorder.py`（relative GELLO 驱动 + 线程化双 ZED 抓帧 + 每 tick
  state(25D)/action(7D)/两路图/原始全量/ts → SERL `.pkl` + 原始 `.npz`）。纯函数 TDD（obs_state/
  normalize_action/image_to_obs/build_transitions/validate）；真机 dry-check 实测 10.2Hz/overruns=0、
  validate PASS、npz 与 pkl 对齐。
- **ZED obs** ✅：vendored `scripts/fr3_zed_capture.py`（外置 2i 36276705→side_policy/side_classifier、
  腕 ZED-M 13132609→wrist_1）；**关键修复：fr3_zed_capture 的 channels="RGB" 实际返回 BGR（只丢 alpha
  无 B/R 互换），录制器 image_to_obs 加 BGR→RGB（真机帧实证通道互换正确）**。
- **ultracode 对抗 review（4 lens）→ 6C/9I/8m**：真问题全修并实证——旋转 drotvec 客户端钳制
  (apply_cartesian_delta 加 max_rot_step=0.1 + 返回 applied drotvec，3 元返回)、action 用 applied
  rotation、None 帧复用、每拍异常隔离 + 连续错误熔断、停机零增量重锚、信号处理器前置、控制超时 0.5s、
  相机 close 先 join、npz 截齐。全量 272 passed。

### GELLO 跟随重做 = Route E（2026-06-15，取代速度积分）
- **速度积分（relative_teleop/gello_demo_recorder gello_twist）真机失败**：摇动大、机器人仅 2.28cm、
  3mm 限幅从未触发 → 用静止 Jacobian 映射有损。**已弃**。
- **Route E（采用）= 关节空间锚定 + 正确 FK，跑现有 cartesian_impedance /pose**（同训练控制器/动作空间，
  无 Polymetis、无切栈）。`scripts/hybrid_teleop.py`：
  - `q_gello_est = q0_robot + (raw-raw0)·joint_signs·leader_scale` → `CorrectFK` → 期望 EE 位姿 → /pose。
  - **CorrectFK = pinocchio(panda_link8) ∘ T_offset[z 0.1034m, Rz -45° = Franka hand F_T_EE]**；实测 vs
    server O_T_EE = 0.13mm。pinocchio 已装 hilserl-fr3（清代理走清华镜像）。repo fk_converter 错 52cm，弃。
  - **混合仲裁**：☰(Xbox Menu) 边沿切 GELLO↔XBOX；**Xbox 激活时 GELLO 完全失效**；切回 GELLO 重锚（无跳变）。
  - 复用 ZED/contract 录制/限幅(3mm/0.1rad)/夹爪边沿。9 测试；全量 286 passed。
- **真机验证 ✅ 2026-06-15**：手臂跟随 10.12cm/0.75mm 每拍（vs 速度积分 2.28cm/0.06mm）；夹爪 close+open
  触发成功；10Hz/overruns=0/validate PASS。Xbox 段尚未真机走通（待用户用 teach.sh 采全流程）。
- **用户自驱采集 CLI**：`scripts/teach.sh`（→ teach_session.py）。在 desktop 终端直接跑：连通检查 →
  /jointreset 复位 HOME[0,0,0,-1.9,0,2,0]（自主运动，先 Enter 确认）→ 设备初始化 → "▶ 示教现在开始" →
  操作(GELLO 抓取/☰ 切 Xbox 精插)→ Ctrl-C 停 → 问 成功/失败/丢弃 + 备注 → 标注入 demos/hybrid/index.jsonl。
  起止信号在终端内，操作者直接掌控时机。

### 真机环境恢复（断电重启后，2026-06-15）
- **desktop GPU 驱动**：断电后 boot 错内核（5.15-realtime，NVIDIA 580 只为 6.8.0-generic 构建）→ nvidia
  模块加载不了 → ZED CUDA 失效。修：`/etc/default/grub.d/99-realtime.cfg` GRUB_FLAVOUR_ORDER 改 generic
  + GRUB_DEFAULT 按 entry-id 指 6.8.0-111-generic + 重启 → RTX 5080 + 驱动 580.159.03 恢复。
- **ZED USB 枚举**：重启后外置 2i 软件 USB reset 恢复；腕 ZED-M 视频接口缺失需**物理重插**(USB3 口)后枚举。

### Phase A ✅（详见 evidence/phase-a-A6-gate.md）
A0 基线回迁入库 / A1 TeleopDeviceHub+XboxIntervention / A2 GelloIntervention 改造+TeleopArbiter /
A3 record_hybrid_demos / A4 motion driver+e2e 脚手架 / A5 gate。
A6 修复 review 的 2 critical+7 important，全量 225 passed。

### Phase B（plans/PLAN-B1..B4.md）
- **B1a /pose 绝对位姿跟随（软件）✅**：gello_pose_follow.py + driver full 模式真实契约实现
  （/getstate → FK-bias 门 rc11 → /clearerr + /pose {"arr"}），approval 门保留。232 passed，
  Opus 对抗复核 APPROVE。详见 evidence/phase-b-B1a.md。
- **B2a Xbox 标定脚手架（软件）✅**：xbox_calibrate.py（纯估计器 deadzone/range/RT-LT 阈值
  + hub 采样器 + 交互 CLI + load_calibration；无设备 raise DeviceUnavailable）。**闭环**：
  XboxIntervention 加 rt_threshold/lt_threshold 参数 + `from_calibration(env,hub,cal)`，
  RT/LT 阈值不再硬编码 0.05。245 passed。手柄接入后一条命令产出标定档 → from_calibration 直接加载。
- **B1b / B2b / B3 / B4 真机**：见下"真机恢复条件"。

## Pending Blockers（真机门控）
1. Xbox 手柄 USB 接入 fr3-desktop-ts（当前无 /dev/input/js*）— 物理动作
2. 用户现场 + E-stop 就位 + 显式 approval 环境变量（FR3_GELLO/XBOX/HYBRID_E2E_APPROVAL）
3. Desk System Image actual 未记录（需 Desk UI 读）
4. 安全事件约定：communication_test 曾误动 FR3，motion wrapper 默认 blocked
5. zktitan learner ✅ 已验证（6 GPU，经 fr3-desktop-ts 跳板）— C/D 训练端就绪

## 真机恢复条件（B1b 起，待一次 motion session）
B1b (GELLO e2e ramp: dry-run→no-op→micro→full + live 契约核对) → B2b (Xbox 实标) →
B3 (切换 e2e) → B4 (安全参数实标 = Phase B 关闭)。
live-verify 清单（B-RESEARCH）：/healthz 可能 404(用 /getstate 探活)、payload key 'arr'、
FK 偏置(可能触发 rc11→标定 FK/flange)、四元数约定、是否需先 /startimp、fork config 是否覆盖
ACTION_SCALE/safety box/gripper。

## Decisions Log（v2.2.1）
- 2026-06-09: Xbox→GELLO（采集精度 <1mm vs ~5mm）
- 2026-06-13: v2.2.1 立项，接管 v2.2 Phase 3–5；核心决策见 DECISIONS.md（端到端单 SAC、
  GELLO 抓取→RB 切→Xbox 精插混合示教、单终点 classifier reward、π0.5 留 v2.3 RLDG）
- 2026-06-13: A6 修 Phase A review（C1 设备异常/C2 motion fail-closed/I1-I8）
- 2026-06-13: Phase B 契约 grounding（B-RESEARCH）：franka_server 唯一运动命令 = 绝对 /pose；
  C2 解决；B1a 按 record_gello_demos_serl 真实契约实现（关节目标→FK→/pose），加 FK-bias 门
- 2026-06-13: **拆分 v2.2.1 / v2.1.1 STATE**（本文件为 v2.2.1 权威）

## Next Step（本会话）
纯软件已基本见底：Phase A + A6 + B1a 完成，本轮 B2a。之后 v2.2.1 剩余全部硬件门控
（B1b 起需真机 + 手柄 + 你现场 + E-stop；C/D 需数据采集）。
