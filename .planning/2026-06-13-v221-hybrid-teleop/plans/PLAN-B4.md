---
plan: B4
phase: v2.2.1-B
title: 安全参数真机实标
wave: 4
depends_on: [B1, B2, B3]
files_modified: [scripts/ (配置常量), .planning/.../evidence/]
autonomous: false(real-machine)
requirements: [XBOX-04]
---

<objective>
当前安全参数（max_step=2.98mm/max_total_delta=29.93mm @10Hz、leader_scale=0.50、Xbox cap=3mm）
是按微动测试标的、偏保守。B4 按真实示教幅度在真机上重标，平衡安全与可操作性。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/B-RESEARCH.md（ACTION_SCALE 参考：usb_pickup 0.015/0.1）
B1b/B2b/B3 现场记录的 leader_scale/fine-scale 实感。
</context>

<tasks>
<task id="B4-1">leader_scale 重标：grasp 段大范围（0.5 可调）、临近插座建议 0.25/0.10 档（DECISIONS 已记），真机确认。</task>
<task id="B4-2">max_step / max_total_delta 重标：在不牺牲安全前提下放宽到真实示教需要的幅度，留够 impedance 软化余量。</task>
<task id="B4-3">Xbox per-step 平移 cap（A6 默认 3mm）与 pos_scale 对齐真实 env ACTION_SCALE；coarse/fine 档实标。</task>
<task id="B4-4">把实标值写回配置 + evidence 记录（含 before/after 对比 + 安全论证）。</task>
</tasks>

<must_haves>
- 任何放宽都有真机证据支撑 + 安全论证；E-stop 在场
- 实标值落盘可复现，B4 完成 = Phase B 关闭，进 Phase C
</must_haves>
