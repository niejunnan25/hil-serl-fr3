---
plan: A1
phase: v2.2.1-A
title: TeleopDeviceHub + XboxIntervention wrapper
wave: 2
depends_on: [A0]
files_modified: [scripts/teleop_hub.py, scripts/xbox_intervention.py, tests/test_teleop_hub.py, tests/test_xbox_intervention_contract.py]
autonomous: true
requirements: [XBOX-01]
---

<objective>
实现 Xbox 介入路径：单点设备状态 hub + 与 GelloIntervention/SpacemouseIntervention
同接口的 XboxIntervention wrapper。TDD，mock joystick 后端，零真机。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/A-RESEARCH.md（事实 1/3）
@.planning/2026-06-13-v221-hybrid-teleop/DECISIONS.md（决策 5/6）
参考实现：scripts/gello_intervention.py（接口范式）、上游 spacemouse 目录。
</context>

<tasks>
<task id="A1-1">
scripts/teleop_hub.py — TeleopDeviceHub：进程内单例持有唯一 pygame joystick 实例；
暴露 `poll() -> XboxState`（axes 6, buttons dict: RB/LB/Y/A/B, dpad, triggers），
线程安全；`backend="pygame"|"mock"`，mock 后端可由测试注入状态序列。
pygame 不可用/无设备时 hub 构造不抛错、`available=False`（desktop 无头环境兼容）。
先写 tests/test_teleop_hub.py（mock 注入、单例、无设备降级），再实现。
</task>
<task id="A1-2">
scripts/xbox_intervention.py — XboxIntervention(gym.ActionWrapper)：
- 接口与 GelloIntervention 完全一致：`action(a)->(a,replaced)`、`step()` 注
  `info["intervene_action"]`、6D 动作空间时 gripper_enabled=False
- 介入条件：**按住 RB**（deadman，显式使能；不用移动阈值/hold 窗）
- 映射（常量集中一处，B4 实标）：左摇杆→dx,dy；右摇杆 Y→dz；右摇杆 X→dyaw；
  D-pad→dpitch/droll；RT/LT→gripper 闭/开；Y 键→coarse/fine 档切换
  （fine=0.3 默认档，coarse=1.0）；deadzone=0.15
- RB 按住但摇杆中位 → 输出零动作（悬停，仍算介入）
先写 tests/test_xbox_intervention_contract.py：接口对等（与
test_gello_intervention_contract.py 同结构）、deadman 真值表、deadzone、
档位切换、gripper 映射、6D 退化。
</task>
</tasks>

<verification>
- 新增两套 pytest 全绿；既有套件无回归
- 接口对等断言：XboxIntervention 与 GelloIntervention 的 action()/step() 签名与
  info 键一致（contract 测试中显式断言）
</verification>

<must_haves>
- RB 未按住时 wrapper 绝不改写 policy 动作（含摇杆被碰歪的情况）
- 全部输出经 clip 到 [-1,1]^7；mock 后端下确定性可测
</must_haves>
