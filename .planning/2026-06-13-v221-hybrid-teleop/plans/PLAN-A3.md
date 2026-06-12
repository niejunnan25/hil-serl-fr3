---
plan: A3
phase: v2.2.1-A
title: 混合示教录制器 record_hybrid_demos
wave: 4
depends_on: [A2]
files_modified: [scripts/record_hybrid_demos.py, tests/test_record_hybrid_demos.py]
autonomous: true
requirements: [HYBRID-02, DATA-05]
---

<objective>
单 episode 内 GELLO 抓取段 → RB 单向切换 → Xbox 精插段的连续示教录制，
输出与 record_gello_demos_serl 相同的 SERL pkl 格式 + 混合 metadata。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/A-RESEARCH.md（事实 2：相对跟随机制）
@.planning/2026-06-13-v221-hybrid-teleop/DECISIONS.md（决策 5）
基底：scripts/record_gello_demos_serl.py（P2-T6，33 tests）。
</context>

<tasks>
<task id="A3-1">
状态机：`GELLO_FOLLOW --RB按下(沿)--> XBOX_DELTA`（单向，不可逆）。
- GELLO 段：复用 record_gello_demos_serl 的 joint 相对跟随
  （q0/raw_gello0 锚定、leader_scale、check_max_step/total）
- 切换瞬间：target 冻结为当前实际关节位（从 robot.get_joint_positions() 读），
  Xbox 段以当前 EE 位姿为零点开始累积 delta —— **切换 tick 不产生跳变**
- Xbox 段：经 TeleopDeviceHub 读手柄，复用 XboxIntervention 的映射常量
  （单一来源 import，不复制）
</task>
<task id="A3-2">
数据与 metadata：逐步记录沿用既有字段；新增 per-step `active_device`、
episode 级 `switch_step`、`leader_scale`、`xbox_scale_mode`；成功标记按既有
空格键约定；pkl 输出走既有序列化路径（格式不分叉）。
</task>
<task id="A3-3">
tests/test_record_hybrid_demos.py（全 mock：假 robot HTTP、mock hub、合成 GELLO 流）：
- 切换连续性：切换前后相邻 target 差 ≤ max_step
- 单向性：RB 再按不回切；切换前 Xbox 输入被忽略
- metadata 正确性：switch_step 与 active_device 数组一致
- pkl 与 record_gello_demos_serl 输出 schema 等价（字段级 diff）
</task>
</tasks>

<verification>
- 新套件 + 既有 33 个录制测试全绿
- mock 全流程跑出一条含切换的 episode，schema 校验通过
</verification>

<must_haves>
- 切换零跳变（量化断言，不是目测）
- 任一段安全检查（max_step/total）失败 → episode 标记 aborted 且不写成功样本
</must_haves>
