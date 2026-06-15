---
plan: A4
phase: v2.2.1-A
title: Motion driver + e2e 脚手架（安全关键，全部 approval-gated）
wave: 4
depends_on: [A1]
files_modified: [scripts/p2_t3_e2e_motion_driver.py, scripts/setup/17_xbox_e2e_motion_test.sh, scripts/setup/18_hybrid_switch_e2e_test.sh, tests/test_p2t3_motion_driver.py, tests/test_e2e_scaffolding_xbox_hybrid.py]
autonomous: true
requirements: [GELLO-03(prep), XBOX-03(prep), HYBRID-03(prep)]
review_gate: opus-adversarial
---

<objective>
补全 Phase B 联合验收所需的全部驱动与脚手架。本 plan 产物在 Phase A 内
只允许 dry-run/mock 运行；真机 motion 留给 Phase B。安全关键 — Opus 对抗 review gate。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/A-RESEARCH.md（事实 5）
既有范式：scripts/setup/16_gello_e2e_motion_test.sh（4 modes, approval-gated）。
安全事件约定：所有 motion 默认 blocked，需显式 approval 环境变量。
</context>

<tasks>
<task id="A4-1">
scripts/p2_t3_e2e_motion_driver.py：GELLO open → GelloCartesianDeltaAgent →
POST /pose @10Hz（guarded franka_server）。模式：`--dry-run`（默认，仅打印
将发送的 pose，不 POST）/ `--micro`（3 步 ×1mm）/ `--full`。
非 dry-run 需 `FR3_GELLO_E2E_APPROVAL=I_APPROVE_P2T3_FULL_E2E_MOTION`；
缺失即拒绝启动（RC≠0 + 明确提示）。接入 16_*.sh 的 mode 4。
tests：approval gate 拒绝逻辑、dry-run 输出契约、micro 模式步数/幅度上限。
</task>
<task id="A4-2">
17_xbox_e2e_motion_test.sh / 18_hybrid_switch_e2e_test.sh：镜像 16_*.sh 四模式结构
（device-echo / dry-run / micro / full），各自独立 approval 变量
`FR3_XBOX_E2E_APPROVAL` / `FR3_HYBRID_E2E_APPROVAL`，full 模式校验 E-stop 确认提示。
tests/test_e2e_scaffolding_xbox_hybrid.py 仿 test_gello_e2e_scaffolding.py（21 tests 范式）。
</task>
</tasks>

<verification>
- 全部测试绿；`bash 17_... device-echo` 与 `18_... device-echo` 在无手柄环境下
  优雅报"设备缺失"而非崩溃
- 三个 approval 变量缺失时，任何脚本路径都无法到达 POST /pose（测试断言）
- **Opus 对抗 review gate**：重点审 approval gate 不可绕过性与 dry-run 默认性
</verification>

<must_haves>
- Phase A 全程零真机 motion：CI/测试证据中不存在任何实际 POST /pose 到真实 server
- approval 变量名与值精确匹配（前缀/子串不通过）
</must_haves>
