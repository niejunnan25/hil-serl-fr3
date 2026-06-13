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
Phase B 真机验收             🟡 已规划(PLAN-B1..B4) | B1a(/pose 跟随软件) ✅ | B2a(Xbox 标定软件) ✅ | B1b/B2b/B3/B4 待真机
Phase C Stage-1 纯插入训成   ⏳ (依赖 B)
Phase D 完整过程端到端训成   ⏳ = 收官判据
```

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
