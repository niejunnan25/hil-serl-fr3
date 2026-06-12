# STATE — hilserl-fr3

> 重建说明：原 v2.2 STATE.md 于 2026-06-12 13:42–14:57 的 working-tree reset 中丢失（从未 commit）。
> 本文件 2026-06-13 从 session 28c86798 transcript 重建，并叠加 v2.2.1 milestone 定义。
> Phase 1/2 的原始 evidence 仍在 fr3-desktop-ts:/home/robot/.planning/2026-06-11-fr3-serl-control-runtime/evidence/
> 与 fr3-desktop-ts:/home/robot/hilserl-fr3/{scripts,tests}/（245 passed local / 113 passed desktop gate）。

## Current Milestone: v2.2.1 — Hybrid Teleop + 完整过程训练

定义日期：2026-06-13（grill-me 九分支决策，见 `2026-06-13-v221-hybrid-teleop/DECISIONS.md`）

```
v2.2 progress:   Phase 1 ✅ CLOSED | Phase 2 🟡 5/6 (P2-T3 待 motion 批准) | Phase 6 ⏳ (sim, 留 v2.2)
v2.2.1 progress: Phase A ⏳ | Phase B ⏳ | Phase C ⏳ | Phase D ⏳ (0/4)
```

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

## Next Step

v2.2.1 Phase A（纯软件，无需 motion 批准）：XboxIntervention wrapper → GelloIntervention
介入改造 → 混合示教录制器 → P2-T3 motion driver 补全 → desktop 同步。
