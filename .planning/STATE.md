# STATE — hilserl-fr3

> 重建说明：原 v2.2 STATE.md 于 2026-06-12 13:42–14:57 的 working-tree reset 中丢失（从未 commit）。
> 本文件 2026-06-13 从 session 28c86798 transcript 重建，并叠加 v2.2.1 milestone 定义。
> Phase 1/2 的原始 evidence 仍在 fr3-desktop-ts:/home/robot/.planning/2026-06-11-fr3-serl-control-runtime/evidence/
> 与 fr3-desktop-ts:/home/robot/hilserl-fr3/{scripts,tests}/（245 passed local / 113 passed desktop gate）。

## Current Milestone: v2.1.1 — Sim Runtime Fix

立项日期：2026-06-13（v2.1 sim-build-fork 审计后，见 `2026-06-13-v211-sim-runtime-fix/AUDIT.md`）
状态：executing（Phase 1/2/3/4 ✅；Phase 5 暂缓。见 `2026-06-13-v211-sim-runtime-fix/PROGRESS.md`）

```
v2.1.1 progress: Phase 1 ✅ | Phase 2 ✅ (全场景 FR3+桌+插头+插座渲染) | Phase 3 ✅ (schema 3键,99 passed) | Phase 4 ✅ (文档校正) | Phase 5 ⏳暂缓(P6并发) (~4/5)
v2.2.1 progress (real 线, 本会话推进): Phase A ✅ CLOSED + A6 ✅；Phase B 已规划(PLAN-B1..B4 + B-RESEARCH 契约 grounding) + B1a(/pose 绝对位姿跟随软件) ✅ 实现验证(232 passed, opus 对抗 APPROVE)；B1b/B2b/B3/B4 待真机 motion session（用户现场/E-stop/Xbox），物理前置未就绪
v2.2 progress:   Phase 1 ✅ CLOSED | Phase 2 🟡 5/6 (P2-T3 待 motion 批准) | Phase 6 ⏳ (sim, 留 v2.2)
```

### v2.1.1 问题清单（P1-P7，详见 2026-06-13-v211-sim-runtime-fix/AUDIT.md）
1. P1 IsaacLab 插头场景黑屏/卡启动（desktop，阻塞）
2. P2 上次死会话 orphan Isaac 进程仍占 ~4.6GB GPU（PID 2876313，~20h）
3. P3 repo `sim/` ↔ desktop `plug_insertion_scene.py` 分叉
4. P4 图像 schema `pixels` vs 3 键不一致（verify 拒绝自家产物，真 bug 未修）
5. P5 sim-build-fork 文档丢失 + "111 passed" vs 实测 93/91 漂移
6. P6 分支混杂（v2.1 sim + v2.2.1 Phase A 同分支）+ worktree/残留分支
7. P7 v2.1 deferred runtime 项（A8 DR/scene 实例化/A12 ABI）

## v2.2 已完成部分（保留，不重做）

- **Phase 1 基础设施** ✅ CLOSED 2026-06-12 03:30 CST
  - INFRA-01~07 全部 [x]：libfranka 0.15.3-1 / ROS Noetic + franka_ros 0.10.1 /
    serl_franka_controllers FR3 适配 / franka_server guarded HTTP / ZED 2i external
    (36276705) + ZED-M wrist (13132609) frame gates / ZEDCapture adapter
  - PHASE1_READINESS_RC=0
- **Phase 2 GELLO 控制栈** 🟡 5/6
  - P2-T1 GelloIntervention wrapper ✅（delta-jog 模式，34 tests）
  - P2-T2 GELLO→FK→Cartesian delta ✅（19 tests）
  - P2-T4 夹爪映射 ✅ / P2-T5 安全标定 ✅（max_step=2.98mm, max_total_delta=29.93mm @10Hz）
  - P2-T6 record_gello_demos_serl HTTP API 迁移 ✅（33 tests）
  - **P2-T3 端到端 motion** 🟡 scaffolded（16_gello_e2e_motion_test.sh, 4 modes, approval-gated），
    motion driver 未实现，纳入 v2.2.1 Phase A/B

## Pending Blockers

1. **Motion 批准**（联合验收制）：`FR3_GELLO_E2E_APPROVAL=I_APPROVE_P2T3_FULL_E2E_MOTION` 模式沿用；
   一次现场 session 依次验收 GELLO e2e → Xbox e2e → 切换 e2e（E-stop 就位、用户在场）
2. **Xbox 手柄未接**：fr3-desktop-ts 无 /dev/input/js*；用户确认有手柄，USB 有线接入（物理动作）
3. **Desk System Image actual 未记录**（沿 v2.2，需从 Desk UI 读取）
4. **安全事件约定沿用**：communication_test 曾误动 FR3，所有 motion wrapper 默认 blocked
5. zktitan learner ✅ 已验证（6 GPU: 4×RTX 5880 Ada 48GB + 2×RTX PRO 6000 96GB，经 fr3-desktop-ts 跳板）

## v2.2.1 Phase A ✅ CLOSED 2026-06-13

- A0 baseline 回迁与入库：rsync desktop→local, 17/18_*.sh 恢复 (git blob ee4b589),
  本地+desktop 17_phase2_readiness_gate 双 RC=0, 113 passed
- A1 TeleopDeviceHub + XboxIntervention：hub 单例+mock backend, RB-deadman wrapper
  with deadzone/scale-toggle/gripper, 26 tests
- A2 GelloIntervention 介入改造 + TeleopArbiter：LB 使能 + per-engagement budget
  reset + env.reset + 预算超限降级; arbiter 优先级真值表 (RB>xbox, only-LB>gello,
  双按>xbox, 无→policy); 18 tests
- A3 record_hybrid_demos：状态机 GELLO_FOLLOW --RB rising edge--> XBOX_DELTA,
  切换零跳变, pkl schema 字段等价; 11 tests
- A4 P2-T3 motion driver + 17_xbox / 18_hybrid e2e shell scaffolds：driver
  dry-run/micro/full, max_step/max_total_delta 内置, approval 不可绕过; 13 tests
- A5 同步+gate+收尾：rsync local→desktop, 19_phase_a_readiness_gate 双 RC=0
  (217 passed local+desktop), REQUIREMENTS XBOX-01/HYBRID-01/HYBRID-02 勾选

边界声明：Phase A 期间零真机 motion。所有 e2e shell 仅在 dry-run/preflight
模式被 CI/测试覆盖；motion 模式需要显式 approval env var（FR3_GELLO_E2E_APPROVAL /
FR3_XBOX_E2E_APPROVAL / FR3_HYBRID_E2E_APPROVAL）且用户现场 + E-stop 就位。

Evidence: .planning/2026-06-13-v221-hybrid-teleop/evidence/phase-a-{baseline,gate}-{local,desktop}.log
desktop-side raw 证据: fr3-desktop-ts:/home/robot/.planning/2026-06-13-v221-hybrid-teleop/evidence/phase-a-gate-desktop-raw/

## Decisions Log

- 2026-06-09: Xbox→GELLO（数据采集精度 <1mm vs ~5mm）
- 2026-06-12: v2.2 Phase 1 关闭；Phase 2 软件部分 5/6；zktitan 连通验证
- 2026-06-13: **禁用 fan-out subagents / ultracode dynamic workflows**（Claude Desktop >40GB 内存事故）；
  执行方式改为 Superpowers 规范 + 顺序执行 + GSD
- 2026-06-13: **v2.2.1 立项**，接管 v2.2 Phase 3–5。核心决策：
  - 纯端到端单 SAC policy（Stage-1 数据不复用，拼接作退路）
  - 混合示教：GELLO 抓取段 → RB 手动单向切换 → Xbox 精插段，切换事件入 metadata
  - 介入：前半程 GELLO（需预算改造+每 episode 重置+LB 使能+互斥仲裁）、后半程 Xbox
  - Reward：单终点 classifier + 升级预案单向门
  - 数据量：Stage-1 Xbox demo 20 / 完整混合 demo ≥25 / classifier 200+正 600+负
  - π0.5 不进 v2.2.1；v2.3 候选 = RLDG 蒸馏 SAC specialist 数据进 pi05（zktitan OpenPI 现成）
- 2026-06-13: planning 文件丢失事故记录在案；此后 .planning 变更随做随 commit
- 2026-06-13: **模型分工决议** — grill/discuss/plan/review 用 Fable 5（最高 effort），
  execute-phase 全部用 Opus 4.8 xhigh；安全关键 plan（review_gate: fable-5）执行后须过
  Fable 5 review；切换在 phase 边界换会话完成
- 2026-06-13: Phase A 规划完成 — plans/PLAN-A0..A5.md 六份（含 inline plan-checker 自检
  通过），A-RESEARCH.md 存档；发现并纳入 A0：Phase 1/2 代码仅存 desktop、从未入库
- 2026-06-13（晚）: **执行/模型策略反转** — Fable 5 使用权限被切断；原 Fable 5 的 grill/discuss/plan/review
  改由 Opus Max/ultracode 承担；**解除 2026-06-12 的"禁用 fan-out subagents/workflows"限制，按需调用**。
  全局记忆 prefer-workflows / model-stage-policy 已同步更新。
- 2026-06-13（晚）: **v2.1.1 立项** — v2.1 (sim-build-fork) 审计发现 sim-code-ready 仅对 repo 单测成立，
  IsaacLab runtime 未达成（desktop 黑屏 + repo↔desktop 分叉）+ 真 schema bug（pixels vs 3 键）+ 工程债。
  v2.1.1 设为 current、暂停 v2.2.1、全量 P1-P7（见 2026-06-13-v211-sim-runtime-fix/）。

## Next Step — v2.1.1 收尾（Phase 5 暂缓）

**已完成（Phase 1-4）**：
- Phase 1 止血：orphan 清掉、GPU 释放、anti-orphan 启动方式。
- Phase 2：黑屏根因（场景从未实例化）已修；**全场景 viewer `plug_fullscene_viewer.py` 渲染出
  FR3+桌+插头+插座**（复用 droid FR3/桌资产，RGB mean 130），见 capture_fullscene_01.png。P3=desktop viewer canonical。
- Phase 3 (SD-01)：producers 产 3 键 schema（sim-to-real），99 passed。
- Phase 4 (DOC-01)：sim-build-fork 文档死引用 + 计数校正。

**剩余**：
- Phase 5 (P6 worktree/分支清理) **暂缓**：并发 ultracode 会话在用同一 working tree，删 worktree/分支有风险，待其结束。
- 可选打磨（非阻塞）：插头/插座 USD 仍为粗 STL→USD 白块——若要逼真需重做/补 USD 细节 + 接腕相机第二视角。
- SR-04 deferred runtime 项（A8 DR / scene 实例化 / A12 ABI）：GUI 已可用，可顺带闭合或给结论。

**v2.1.1 不需要**：真机 motion、Xbox 手柄、E-stop（纯 sim/软件 + desktop GUI）。

---

### （暂停）v2.2.1 Phase B 恢复条件 — 待 v2.1.1 完成
B1 (GELLO e2e P2-T3) → B2 (Xbox e2e) → B3 (切换 e2e) → B4 (实标)。前置：
- [ ] Xbox 手柄 USB 接入 fr3-desktop-ts
- [ ] 用户现场 + E-stop 就位
- [ ] 显式 approval 环境变量
- [ ] 一次联合 motion session
