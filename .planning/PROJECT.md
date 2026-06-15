# PROJECT — hilserl-fr3

> 2026-06-13 重建（原文件如存在已随 2026-06-12 reset 丢失，unknown）。

## 是什么

FR3 真机插插头（CN 两脚插头→插座）HIL-SERL 项目：SERL/RLPD + reward classifier +
人工示教与介入，actor 在 fr3-desktop-ts，learner 在 zktitan（经 desktop SSH 跳板）。
长期目标衔接 FR3-8010：sim（IsaacLab）与 pi05（OpenPI, zktitan:8010）能力对齐。

## 当前 milestone

**v2.1.1 Sim Runtime Fix**（2026-06-13 立项）—— 修 v2.1 (sim-build-fork) 审计发现的 P1-P7：
IsaacLab 插头场景黑屏/卡启动、repo `sim/` ↔ desktop 场景脚本分叉、图像 schema 不一致、
orphan 进程、分支混杂、文档丢失。详见 `2026-06-13-v211-sim-runtime-fix/`（AUDIT/DECISIONS/REQUIREMENTS）+ ROADMAP.md。

**v2.2.1 Hybrid Teleop + 完整过程训练**（接管 v2.2 Phase 3–5）—— **暂停**，待 v2.1.1 完成后恢复。
GELLO 抓取段示教 + Xbox 精插段示教/介入，纯端到端单 SAC policy。
详见 2026-06-13-v221-hybrid-teleop/DECISIONS.md。
v2.3 候选：RLDG 蒸馏 SAC specialist 数据进 pi05。

## 执行策略（2026-06-13 决议；2026-06-15 对齐 Fable-5 切断 + fan-out 解禁）

| 阶段 | 模型 / 模式 |
|------|------|
| grill-me / discuss-phase / plan-phase / review | Opus Max（最高 effort）；按需 ultracode + fan-out 多视角 |
| execute-phase（全部） | Opus 4.8 xhigh |

- **Fable 5 权限已切断**：原指派 Fable 5 的高杠杆思考（grill/discuss/plan/review）改由 Opus Max/ultracode 承担。
- 安全关键 plan（A2/A4 等标 `review_gate: opus-adversarial`）执行后必须过**独立对抗式 review pass**
  （Opus Max/ultracode，可 fan-out 多视角校验）才算完成——**不再要求 Fable 5**（已无可回退）。
- **fan-out subagents 与 ultracode workflows 已解禁、按需调用**（2026-06-13 覆盖 2026-06-12 禁用令）；
  GSD 官方流程 + Superpowers 纪律（TDD、verification-before-completion）仍为执行骨架。
  审慎提醒：2026-06-12 曾因 fan-out + ultracode + 遗留 worktree 致 Claude Desktop >40GB，
  扩大并行前先清 `.claude/worktrees/wf_*`。
- 真机运动执行（teleop/示教/运动）仍按下方「安全边界」+ human-in-loop 握手起停（顺序、人机同步，不做无人值守 fan-out）；
  批量 fan-out / ultracode 用于 sim / 研究 / 评审。
- 现单一 Opus 模型：阶段切换体现在 effort / ultracode 档位与是否 fan-out，"换会话 /model" 基本作废。
- 权威依据：全局记忆 model-stage-policy / prefer-workflows、v2.1.1 DECISIONS.md（lines 11-13）、STATE.md（2026-06-13 晚 反转）。

## 安全边界（不可妥协）

- 任何真机 motion 需显式 approval 环境变量 + 用户现场 + E-stop；默认全部 blocked
- max_step=3mm 逐步上限任何代码路径不可绕过
- .planning 与代码变更随做随 commit（2026-06-12 丢失事故教训），显式 pathspec，
  git 根是 /Users/tacyvan/Documents/Code，禁 `git add .`，未经批准不 push

## 关键路径

- 本地 repo：~/Documents/Code/hilserl-fr3（planning 权威）
- 运行时：fr3-desktop-ts:/home/robot/hilserl-fr3（A0 后与本地保持同步）
- 上游参考：fr3-desktop-ts:/home/robot/serl_projects/hil-serl-fr3/upstream/hil-serl
- learner：zktitan 162.105.195.74（经 fr3-desktop-ts 跳板；6 GPU）
- Phase 1/2 evidence：fr3-desktop-ts:/home/robot/.planning/2026-06-11-fr3-serl-control-runtime/
