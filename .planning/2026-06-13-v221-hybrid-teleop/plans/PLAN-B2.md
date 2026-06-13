---
plan: B2
phase: v2.2.1-B
title: Xbox e2e — B2a 标定脚手架(软件) / B2b 真机手感实标
wave: 2
depends_on: [B1]
files_modified: [scripts/xbox_calibrate.py, tests/test_xbox_calibrate.py]
autonomous: B2a=部分(需手柄插入才能跑出真值) / B2b=false
requirements: [XBOX-03, XBOX-04]
---

<objective>
Xbox 手柄真机遥操作链路验证 + fine-scale/deadzone 实标。映射/clip/cap 逻辑已在 A1/A6
单测覆盖；B2 新增的是**真实设备标定**（需手柄物理接入）。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/B-RESEARCH.md
A6 已做：XboxIntervention RB-deadman + deadzone + 档位 + 3mm 平移 cap（mock 测过）。
前置硬件：Xbox 手柄 USB 接 fr3-desktop-ts（当前无 /dev/input/js*）。
</context>

## B2a — 标定脚手架（软件，手柄插上即可跑）
<tasks>
<task id="B2a-1">
scripts/xbox_calibrate.py：手柄接入后自动量取 ① 各轴静息噪声（→ deadzone 建议值）、
② 满量程范围、③ RT/LT 触发阈值；输出 JSON 标定档。无手柄时优雅报"设备缺失"。
</task>
<task id="B2a-2">
tests/test_xbox_calibrate.py（mock backend）：注入静息/满杆序列 → 断言 deadzone/范围估计正确、
无设备降级不崩。纯软件可测。
</task>
</tasks>

## B2b — 真机实标（needs 手柄 + robot，Phase B session）
<tasks>
<task id="B2b-1">手柄 USB 接 fr3-desktop-ts，确认 /dev/input/js* + pygame 识别。</task>
<task id="B2b-2">跑 xbox_calibrate.py 出真实 deadzone/scale，写回 XboxIntervention 配置。</task>
<task id="B2b-3">真机 Xbox e2e：RB 介入驱动 FR3，确认 3mm/步 cap 真实生效、映射方向、fine/coarse 档手感。</task>
<task id="B2b-4">插入精细段手感判定（靠真机），记录 fine-scale 推荐值（交 B4 收敛）。</task>
</tasks>

<must_haves>
- 真机段沿用 B1 的 approval + E-stop + no-op→micro→full ramp
- 标定档落盘可复现，不硬编码到代码
</must_haves>
