---
plan: A5
phase: v2.2.1-A
title: 同步、Phase A gate 与收尾
wave: 5
depends_on: [A2, A3, A4]
files_modified: [scripts/setup/19_phase_a_readiness_gate.sh, .planning/STATE.md, .planning/REQUIREMENTS.md]
autonomous: true
requirements: [XBOX-01, HYBRID-01, HYBRID-02]
---

<objective>
代码双向同步、Phase A readiness gate 固化、planning 状态翻转、全部入库。
</objective>

<context>
范式：fr3-phase1-readiness-gate / phase2 readiness gate（RC=0 综合门）。
git 纪律：显式 pathspec，从 /Users/tacyvan/Documents/Code 提交，禁 `git add .`。
</context>

<tasks>
<task id="A5-1">
rsync local→desktop（scripts/tests/franka_env，排除缓存），desktop 上跑全套
pytest gate 并保存输出到 .planning/2026-06-13-v221-hybrid-teleop/evidence/
phase-a-gate-desktop.log。
</task>
<task id="A5-2">
scripts/setup/19_phase_a_readiness_gate.sh：综合门 = 既有 4 套 GELLO 套件 +
新增 5 套（hub/xbox/arbiter/hybrid-recorder/motion-scaffold）全绿 → RC=0。
本地与 desktop 各跑一次，RC 与计数写 evidence。
</task>
<task id="A5-3">
planning 翻转与提交：REQUIREMENTS.md 勾选 XBOX-01/HYBRID-01/HYBRID-02；
STATE.md Phase A → complete、Next Step → Phase B（列出 Phase B 前提：手柄 USB 接入、
用户现场、E-stop、approval 变量）；commit "feat(hilserl-fr3): v2.2.1 Phase A
teleop toolchain (+gate evidence)"。
</task>
</tasks>

<verification>
- 19_phase_a_readiness_gate.sh 本地 + desktop 双 RC=0，evidence 留档
- A2/A4 的 Fable 5 review 记录已在 evidence/（gate 的前置检查项）
- git log 含基线 commit（A0）与 Phase A commit，diff 仅含 in-scope 文件
</verification>

<must_haves>
- desktop 与本地代码逐字节一致（gate 内 diff -rq 检查）
- Phase A 结束时真机零 motion 的声明写入 STATE（boundary 段）
</must_haves>
