# REQUIREMENTS — v2.1.1 Sim Runtime Fix

> 范围：修复 v2.1 (sim-build-fork) 留下的 P1-P7（见 `AUDIT.md`）。phase6-ready 仍 DEFERRED，不在本里程碑。

## v2.1.1 Requirements（按类别 + REQ-ID）

### SIM-RUNTIME（IsaacLab 真能跑）
- [ ] **SR-01**: fr3-desktop-ts 上插头场景在 IsaacLab 中渲染出可见的 FR3 + 桌 + 插头 + 插座，用户确认非黑屏截图；不再卡启动/自动关闭。 [P1]
- [ ] **SR-02**: 确定并落地 repo `sim/scenes/plug_scene.py` ↔ desktop `plug_insertion_scene.py` 的单一 canonical 源 + 单向收敛策略。 [P3]
- [ ] **SR-03**: 黑屏根因（Y-up/Z-up、`panda_hand` prim 路径、DistantLight 无 direction、USD 路径）逐项排查并修复，留证据。 [P1]
- [ ] **SR-04**: v2.1 deferred 的 runtime 项（A8 DR runtime、scene 实例化、A12 ABI 在 mainline conda 实跑）在 GUI 可用后闭合或给明确结论。 [P7]

### SIM-DATA（数据 schema 一致）
- [ ] **SD-01**: `failure_scenario_generator` / `gello_replay` 的 pkl 产物通过 `verify_sim_data` schema 校验（`pixels` ↔ 3 键一致），TDD 测试绿。 [P4]

### INFRA-HYGIENE（工程债）
- [ ] **IH-01**: 清理上次会话 orphan Isaac 进程（释放 ~4.6GB GPU）；GUI launcher 加进程组/PID 管理避免再孤儿。 [P2]
- [ ] **IH-02**: 厘清 `ai/20260228-optimize-cge` 上 v2.1 sim 与 v2.2.1 Phase A 的边界（fork 合并债处理策略）。 [P6]
- [ ] **IH-03**: 清理 `.claude/worktrees/wf_*` 残留与 `sim/v2.1-a*` 残留分支。 [P6]

### DOCS（证据一致）
- [ ] **DOC-01**: 重建/校正 sim-build-fork 文档（SYNC.md 死引用 DESIGN/PLAN/REVIEW、"111 passed"→实测 93/91），与现实一致。 [P5]

## Traceability（→ ROADMAP 阶段）

| 阶段 | REQ-IDs |
|------|---------|
| Phase 1 Desktop 止血 | IH-01, SR-01(诊断), SR-03(诊断) |
| Phase 2 黑屏修复 + 对齐 | SR-01, SR-02, SR-03 |
| Phase 3 schema 一致性 | SD-01 |
| Phase 4 文档重建 | DOC-01 |
| Phase 5 分支清理 + runtime 收尾 | IH-02, IH-03, SR-04 |

## Out of Scope（显式排除）

- phase6-ready 评估（real pkl + balanced fixture + precision/recall ≥ 0.85 + 用户签字）—— 留 real-machine 后续。
- v2.2.1 Phase A-D 推进 —— 已暂停。
- background domain randomization runtime 调优 —— 仍归 v2.2 Phase 6 / mainline。
