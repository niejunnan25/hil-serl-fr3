---
plan: A0
phase: v2.2.1-A
title: 基线回迁与入库
wave: 1
depends_on: []
files_modified: [scripts/, tests/, franka_env/]
autonomous: true
requirements: []
---

<objective>
把 fr3-desktop-ts 上唯一存活的 Phase 1/2 代码基线回迁到本地 repo 并首次 git 入库，
恢复本地可开发/可测试状态。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/A-RESEARCH.md（代码存活状态表）
本地 GELLO 文件已随 2026-06-12 working-tree reset 丢失且从未 commit；desktop 副本为唯一真源。
</context>

<tasks>
<task id="A0-1">
rsync 回迁（desktop→local，排除 __pycache__/artifacts）：
`rsync -av --exclude __pycache__ --exclude '*.pyc' fr3-desktop-ts:/home/robot/hilserl-fr3/scripts/ ~/Documents/Code/hilserl-fr3/scripts/`
同样回迁 tests/ 与 franka_env/。回迁前先 `git status --short hilserl-fr3/` 记录现状，
不得覆盖本地 sim/ 与 .planning/。
</task>
<task id="A0-2">
本地建测试环境并跑通：用现有 conda/venv 或 `pip install -r` 等价物；
`python -m pytest hilserl-fr3/tests -q` 预期与上次会话基线同量级（GELLO 4 套件 ≈113 passed；
若本地缺 ROS/SDK 依赖导致个别 collection error，记录并以 desktop gate 为准）。
</task>
<task id="A0-3">
git 入库：从 /Users/tacyvan/Documents/Code 用显式 pathspec
`git add hilserl-fr3/scripts hilserl-fr3/tests hilserl-fr3/franka_env`，
commit "chore(hilserl-fr3): restore Phase1/2 code baseline from fr3-desktop-ts (never committed)"。
禁止 `git add .`。
</task>
</tasks>

<verification>
- 本地 pytest GELLO 套件通过数与 desktop gate 一致（113）或差异有书面解释
- git show --stat 显示基线文件已入库
</verification>

<must_haves>
- 本地与 desktop 的 scripts/tests/franka_env 内容一致（diff -rq 抽查无差异）
- 基线 commit 存在，后续任何 plan 不在未入库基线上开发
</must_haves>
