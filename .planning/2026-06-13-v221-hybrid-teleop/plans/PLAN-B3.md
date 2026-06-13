---
plan: B3
phase: v2.2.1-B
title: GELLO→Xbox 切换 e2e（真机）
wave: 3
depends_on: [B1, B2]
files_modified: []
autonomous: false(real-machine)
requirements: [HYBRID-03]
---

<objective>
真机验证一条 episode 内 GELLO 抓取段 → RB 单向切换 → Xbox 精插段：无命令跳变、
仲裁正确、metadata 正确。状态机/仲裁/录制器逻辑已在 A2/A3/A6 单测覆盖（含切换零跳变断言）；
B3 是真机端到端确认。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/DECISIONS.md（决策 5/6）
A2 TeleopArbiter（RB>xbox / only-LB>gello / 双按>xbox / downgrade→none）、
A3 record_hybrid_demos（GELLO_FOLLOW --RB rising--> XBOX_DELTA，切换 tick 冻结当前关节）。
</context>

<tasks>
<task id="B3-1">真机录一条混合 episode：GELLO 跟随抓取 → 按 RB 切 Xbox → 精插。确认切换 tick 机器人无跳变（量化：相邻命令 ≤max_step）。</task>
<task id="B3-2">仲裁真机确认：训练态 RB 按住=Xbox 介入、仅 LB=GELLO 介入、双按=Xbox；downgrade(预算超限)不误标介入。</task>
<task id="B3-3">A2-F3 真机确认：Xbox 段后松 RB 回 GELLO，首个 GELLO tick 不误拒（re-seed 生效）。</task>
<task id="B3-4">录制产物 schema + metadata（active_device/switch_step、Xbox 段夹爪取自 RT/LT）真机数据核对。</task>
</tasks>

<must_haves>
- approval + E-stop + 用户在场
- 切换零跳变真机复现；介入设备标注与实际一致
</must_haves>
