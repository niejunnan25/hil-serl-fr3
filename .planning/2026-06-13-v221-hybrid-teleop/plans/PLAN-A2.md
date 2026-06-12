---
plan: A2
phase: v2.2.1-A
title: GelloIntervention 介入改造 + 双设备仲裁（安全关键）
wave: 3
depends_on: [A1]
files_modified: [scripts/gello_intervention.py, scripts/gello_cartesian_delta_agent.py, scripts/teleop_arbiter.py, tests/test_gello_intervention_contract.py, tests/test_teleop_arbiter.py]
autonomous: true
requirements: [HYBRID-01]
review_gate: fable-5
---

<objective>
让 GELLO 介入可在长 horizon 训练中安全使用：LB 使能、按次接入预算、每 episode 重置、
与 Xbox 的互斥仲裁。安全关键 — 完成后需 Fable 5 review gate 方可进入 A3。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/A-RESEARCH.md（事实 1：现有预算缺陷）
@.planning/2026-06-13-v221-hybrid-teleop/DECISIONS.md（决策 6 及配套工程清单）
</context>

<tasks>
<task id="A2-1">
GelloIntervention 改造（保持既有接口与默认行为向后兼容，新行为由参数开启）：
- `arming_hub: Optional[TeleopDeviceHub]` — 提供时，介入需 **LB 按住**（与 1mm 移动
  阈值取 AND）；不提供时保持旧语义（既有 34 tests 不破坏）
- **engagement 语义**：LB 从松到按 = 新接入 → `_agent.reset(当前 gello joints)`，
  本次接入的累积预算 30mm 从零起算；LB 松开 = 接入结束
- wrapper 重载 `reset()`：env.reset 时调用 `_agent.reset()` 并清 `last_intervene`
- 预算超限行为从「裸 RuntimeError」改为：本次接入失效 + 告警日志 + 返回 policy 动作
  （训练循环不可因预算崩溃）
扩展 tests/test_gello_intervention_contract.py：engagement 重置、预算按次、
env.reset 重置、超限降级、旧语义回归。
</task>
<task id="A2-2">
scripts/teleop_arbiter.py — TeleopArbiter：组合两个 intervention wrapper 的仲裁层
（或等价的组合 wrapper），单一优先级真值表：
  RB 按住 → Xbox 介入（无视 GELLO）；
  仅 LB 按住 → GELLO 介入；
  皆无 → policy。
同 tick 双输入只允许一个生效；`info["intervene_device"] ∈ {xbox, gello, none}`。
先写 tests/test_teleop_arbiter.py：完整真值表 ×（gripper on/off）×（设备不可用降级）。
</task>
</tasks>

<verification>
- 全套 pytest 绿（含既有 GELLO 套件零回归）
- 仲裁真值表测试逐行对应 DECISIONS.md 决策 6
- **Fable 5 review gate**：plan 执行完后由 Fable 5 会话审查 diff（重点：预算逻辑、
  reset 路径、异常降级），review 记录写入 evidence/ 后方可标记完成
</verification>

<must_haves>
- 任何代码路径下 max_step=3mm 逐步上限不可被绕过
- LB/RB 同按时只有 Xbox 动作到达 env.step
- 预算耗尽/设备异常永不中断训练循环（降级为 policy 动作 + 日志）
</must_haves>
