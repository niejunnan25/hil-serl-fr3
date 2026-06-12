# PROJECT — hilserl-fr3

> 2026-06-13 重建（原文件如存在已随 2026-06-12 reset 丢失，unknown）。

## 是什么

FR3 真机插插头（CN 两脚插头→插座）HIL-SERL 项目：SERL/RLPD + reward classifier +
人工示教与介入，actor 在 fr3-desktop-ts，learner 在 zktitan（经 desktop SSH 跳板）。
长期目标衔接 FR3-8010：sim（IsaacLab）与 pi05（OpenPI, zktitan:8010）能力对齐。

## 当前 milestone

**v2.2.1 Hybrid Teleop + 完整过程训练**（接管 v2.2 Phase 3–5）——
GELLO 抓取段示教 + Xbox 精插段示教/介入，纯端到端单 SAC policy。
详见 ROADMAP.md 与 2026-06-13-v221-hybrid-teleop/DECISIONS.md。
v2.3 候选：RLDG 蒸馏 SAC specialist 数据进 pi05。

## 执行策略（用户 2026-06-13 决议）

| 阶段 | 模型 |
|------|------|
| grill-me / discuss-phase / plan-phase / review | Fable 5（最高 effort） |
| execute-phase（全部） | Opus 4.8 xhigh |

- 安全关键 plan（A2/A4 等标 `review_gate: fable-5`）执行后必须过 Fable 5 review 才算完成
- **禁用 fan-out subagents 与 ultracode dynamic workflows**（2026-06-12 内存事故决议）；
  顺序执行 + Superpowers 纪律（TDD、verification-before-completion）+ GSD 官方流程
- 模型切换发生在 GSD phase 边界（换会话 /model）

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
