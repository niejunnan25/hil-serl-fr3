# ROADMAP — hilserl-fr3

> 重建说明：原 v2.2 ROADMAP.md 在 2026-06-12 working-tree reset 中丢失。Phase 1/2/6 部分
> 2026-06-13 从 session 28c86798 transcript（subagent wf_64eb4c25 存档）重建；Phase 3–5
> （GELLO-only 旧设计）被 v2.2.1 Phase A–D 取代，不再保留。

## Milestone 总览

- **v1.0** — archived（dry-run PROVEN）
- **v2.1** — sim 主线 fork（sim-code-ready PASS，见 `sim-build-fork/`）；审计发现 IsaacLab runtime 未达成 + 真 bug，见下 v2.1.1
- **v2.1.1（当前）** — Sim Runtime Fix：修 v2.1 留下的 P1-P7（IsaacLab 黑屏 / repo↔desktop 分叉 / 图像 schema / 工程债），见 `2026-06-13-v211-sim-runtime-fix/`
- **v2.2** — FR3 真机原生 SERL 控制 runtime：Phase 1 ✅ / Phase 2 🟡 5/6 / Phase 6 ⏳
- **v2.2.1（暂停，待 v2.1.1 完成后恢复）** — Hybrid Teleop + 完整过程训练：接管原 v2.2 Phase 3–5 范围
- **v2.3（候选）** — RLDG：SAC specialist 数据蒸馏进 pi05（zktitan OpenPI/LoRA）；
  备选研究线：ConRFT/π_RL 式直接 VLA 在线 RL

---

## v2.1.1 Sim Runtime Fix（当前）

**Goal**: 让 IsaacLab 插头场景在 fr3-desktop-ts 真能渲染并跑起来，修掉 v2.1 留下的真 bug 与工程债。
**立项依据**: `2026-06-13-v211-sim-runtime-fix/AUDIT.md`（P1-P7） + `DECISIONS.md` + `REQUIREMENTS.md`。
**范围外**: phase6-ready 评估（real pkl + precision/recall ≥ 0.85 + 用户签字）仍 DEFERRED。
**执行**: Opus Max/ultracode（Fable 5 已停用）；fan-out/workflows 按需启用。

| Phase | 内容 | REQ-IDs | Exit 判据 |
|-------|------|---------|-----------|
| 1 | Desktop 止血：清 orphan Isaac 进程 + launcher 进程组管理；干净重启采首帧诊断 | IH-01, SR-01/03(诊断) | GPU 无孤儿；GUI 能干净启停；拿到首帧截图 + 启动日志 |
| 2 | 黑屏根因修复 + repo↔desktop 对齐：Y-up/Z-up、panda_hand prim、DistantLight 方向、USD 路径逐项修；定 canonical 源单向收敛 | SR-01, SR-02, SR-03 | 用户确认的非黑屏截图（见 FR3+桌+插头+插座）；对齐策略落地 |
| 3 | 图像 schema 一致性：`pixels` ↔ 3 键统一，verify 通过真实生产链产物 + TDD | SD-01 | failure/gello pkl 过 verify_sim_data；测试绿 |
| 4 | 文档/证据重建：SYNC.md 死引用、"111 passed" 校正 | DOC-01 | sim-build-fork 文档与现实一致，无死引用 |
| 5 | 分支拆分 + worktree/分支清理 + deferred runtime 收尾 | IH-02, IH-03, SR-04 | 分支边界清晰；worktree/残留分支清理；A8 DR/scene 实例化/A12 ABI 有结论 |

**完成 = v2.1.1 关闭，恢复 v2.2.1 Phase A。**

---

## v2.2 Phase 1: 基础设施搭建 ✅ CLOSED (2026-06-12)

INFRA-01~07 全部完成。Evidence: fr3-desktop-ts:/home/robot/.planning/2026-06-11-fr3-serl-control-runtime/evidence/。

## v2.2 Phase 2: GELLO 控制栈集成 🟡 5/6

P2-T1/T2/T4/T5/T6 ✅（113 tests desktop gate RC=0）。P2-T3（端到端 motion）转入 v2.2.1 Phase A4/B1。

---

## v2.2.1 Phase A: 遥操作工具链（纯软件，无 motion）

**Goal**: Xbox 路径、GELLO 介入改造、混合示教录制器全部代码就绪 + 测试通过。
**Exit**: desktop pytest 套件全绿；所有新 wrapper mock 模式可运行；无任何真机 motion。

已规划（plans/PLAN-A0..A5.md，2026-06-13，wave 顺序执行）：

| Plan | Wave | 内容 | 安全关键 |
|------|------|------|----------|
| A0 | 1 | 基线回迁与入库（desktop→local rsync + 首次 git commit；Phase 1/2 代码现仅存 desktop） | — |
| A1 | 2 | TeleopDeviceHub + XboxIntervention（RB deadman、deadzone、档位、TDD mock 后端） | — |
| A2 | 3 | GelloIntervention 介入改造（LB 使能、按次接入预算、env.reset 重置）+ TeleopArbiter 互斥仲裁 | ✔ Fable 5 review gate |
| A3 | 4 | record_hybrid_demos：GELLO 段→RB 单向切换→Xbox 段，零跳变 + metadata + pkl schema 等价 | — |
| A4 | 4 | P2-T3 motion driver + 17/18 e2e 脚手架（全部 approval-gated，Phase A 内仅 dry-run/mock） | ✔ Fable 5 review gate |
| A5 | 5 | 双向同步 + 19_phase_a_readiness_gate.sh（本地+desktop 双 RC=0）+ planning 翻转入库 | — |

## v2.2.1 Phase B: 联合真机验收（一次 motion 批准 session）

**Goal**: 三条遥操作链路真机验证 + 安全参数实标。
**Exit**: GELLO e2e / Xbox e2e / GELLO→Xbox 切换 e2e 全部通过；scale/预算实标完成。
**前置**: 用户现场、E-stop 就位、Xbox USB 接入、明确 approval 环境变量。

| ID | Task |
|----|------|
| B1 | GELLO e2e（原 P2-T3）：跟随精度、10Hz 稳定性、夹爪映射 |
| B2 | Xbox e2e：映射手感、fine-scale 档位、deadzone 实标 |
| B3 | 切换 e2e：RB 切换无跳变、metadata 正确、介入仲裁验证 |
| B4 | 安全实标：leader_scale / Xbox scale / max_step / max_total_delta 按真实示教幅度重标（当前值按微动测试标定，偏保守） |

## v2.2.1 Phase C: Stage-1 纯插入训成（Xbox）

**Goal**: 夹爪固定闭合、复位点在插座上方的纯插入 policy 训成。
**Exit**: ≥3 次自主插入成功；classifier P/R ≥ 0.90。

| ID | Task | 责任 |
|----|------|------|
| C1 | IMAGE_CROP 标定（真实 ZED 视野） | session |
| C2 | classifier 数据：≥200 正 / ≥600 负 | human (Xbox) |
| C3 | classifier 训练 + P/R 验证 | session (zktitan) |
| C4 | Stage-1 demo：Xbox 示教 20 条 | human |
| C5 | SAC 训练（actor=desktop, learner=zktitan）；介入 Xbox | session+human |
| C6 | ≥3 次自主插入验收 + 失败模式记录 | session |

## v2.2.1 Phase D: 完整过程训成（端到端）

**Goal**: 抓取→搬运→插入单一端到端 policy 训成。
**Exit**: ≥3 次完整自主 抓取→插入 成功 = v2.2.1 完成。

| ID | Task | 责任 |
|----|------|------|
| D1 | 混合示教 ≥25 条（GELLO 抓取段 → RB → Xbox 精插段） | human |
| D2 | classifier 复核（终点定义同 Stage-1，必要时补图） | session |
| D3 | 端到端 SAC 训练；介入：前半程 GELLO / 后半程 Xbox | session+human |
| D4 | ≥3 次完整自主成功 + 失败模式记录 | session |

**升级预案（单向门，按序触发）**：抓取段学不会 → 加大前半程介入 → 补录 demo →
两阶段 reward（需另标定抓取判据）→ 最后退路：两段 policy 拼接（Stage-1 产物直接复用）。

## v2.2 Phase 6: Sim 数据增强 ⏳（留在 v2.2，可与 v2.2.1 Phase C/D 并行）

SIM-01~07：IsaacLab 失败场景负样本 ≥500，混合训练 classifier P/R ≥ 0.85。
背景 DR（纹理/天空/光照扩展）沿 v2.1 决议仍归此 phase。
