---
plan: B1
phase: v2.2.1-B
title: GELLO e2e（/pose 跟随）— B1a 软件可先做 / B1b 真机验证
wave: 1
depends_on: [A5]
files_modified: [scripts/p2_t3_e2e_motion_driver.py, scripts/gello_pose_follow.py, tests/test_gello_pose_follow.py, tests/test_p2t3_motion_driver.py]
autonomous: B1a=true / B1b=false(real-machine)
requirements: [GELLO-03]
review_gate: opus-adversarial
---

<objective>
让 motion driver 按 **已验证的真实契约**（record_gello_demos_serl：关节目标→FK→绝对 /pose
`{"arr"}`）驱动 GELLO 跟随，替换 A6 的 fail-closed。B1a 纯软件（mock server，无真机）；
B1b 真机现场验证。
</objective>

<context>
@.planning/2026-06-13-v221-hybrid-teleop/B-RESEARCH.md（权威契约 + live-verify 清单）
@.planning/2026-06-13-v221-hybrid-teleop/REVIEW-PhaseA.md（C2：原 bug = 归一化 delta 当绝对位姿 POST）
proven 路径：record_gello_demos_serl.py（serl:301-331）。FK：fk_converter.forward_kinematics。
</context>

## B1a — 软件（autonomous，本阶段先做）

<tasks>
<task id="B1a-1">
scripts/gello_pose_follow.py：纯函数 + 类，复用 record_gello_demos_serl 的关节跟随数学
（不依赖 GelloCartesianDeltaAgent 的归一化 delta）：
- `joint_target(q0, raw_gello0, raw, joint_signs, leader_scale)` → clip(FR3 limits)
- 命令限幅：`command += clip(target-command, -max_step, max_step)`；`max_total_delta` 累积检查
- `abs_pose(command) = forward_kinematics(command)` → [x,y,z,qx,qy,qz,qw]
全部可纯 numpy 单测，无网络。
</task>
<task id="B1a-2">
重写 driver `_send_full`（解除 fail-closed，但仍受 approval gate）：
1. `POST /getstate` 读 `q0=state['q']`、`currpos=state['pose']`（key 'arr' 不用于读）。
2. **启动 bias 门**：`‖forward_kinematics(q0)[:3] - currpos[:3]‖ > FK_BIAS_LIMIT(默认 0.005m)`
   → 拒绝（rc 11 + 日志 "FK/EE-frame mismatch, calibrate before live motion"），绝不在 FK 偏置大时命令跳变。
3. 每 tick：`POST /clearerr`（镜像 env）→ 关节目标→限幅→`POST /pose {"arr": abs_pose}` @ HZ。
4. payload key 用 `'arr'`（权威）；4xx/5xx → rc 9。
approval gate 不变（full 仍需 `FR3_GELLO_E2E_APPROVAL`）。
</task>
<task id="B1a-3">
tests/test_gello_pose_follow.py + 扩展 test_p2t3_motion_driver.py（mock HTTP server，无真机）：
- /pose payload 恒为 `{"arr":[7 floats]}`，姿态单位四元数（‖q‖≈1）
- 首个命令 = FK(q0)（当 bias 在容差内 ≈ currpos，无跳变）
- bias 门：mock /getstate 返回与 FK(q0) 偏离 >5mm → 拒绝 rc 11、零 /pose
- 安全限幅：单步 ≤max_step、累积 ≤max_total_delta、关节在限位内
- 无 approval → rc 5（保留）；/getstate 不可达 → 优雅退出非零
</task>
</tasks>

<verification>
- 新 + 既有 driver 套件全绿；mock 下零真机、payload 契约断言通过
- review_gate：Opus 独立对抗复核（重点：bias 门不可绕过、payload key、四元数有效性、限幅）
</verification>

<must_haves>
- 任何 mock 路径下绝不 POST 非单位四元数 / 非绝对位姿 / 错 key
- FK 偏置超限时拒绝启动，绝不命令跳变
- full 模式无 approval 不可达 /pose
</must_haves>

## B1b — 真机现场验证（needs robot + 用户 + E-stop，Phase B session）

<tasks>
<task id="B1b-1">live 探活：`POST /getstate` 成功（不靠 /healthz，可能 404）；记录 q0/currpos/force。</task>
<task id="B1b-2">契约核对：先 `POST /pose {"arr": currpos}`（命令当前位姿=应无运动）确认 key/四元数/frame 正确、机器人不跳。</task>
<task id="B1b-3">FK 偏置实测：读 currpos vs FK(q0)，记录偏置；超 5mm 则现场标定 FK/flange 再继续。</task>
<task id="B1b-4">micro：3 步 ×1mm GELLO 跟随，确认方向/符号、impedance 是否需先 /startimp。</task>
<task id="B1b-5">full：GELLO 实时跟随 10Hz 稳定、跟踪误差、夹爪映射；记录 leader_scale 实感（交 B4 重标）。</task>
</tasks>

<must_haves>
- 每次启动 controller/motion 前用户在场 + E-stop 触手可及 + 显式 approval
- 先 no-op（当前位姿）→ micro → full 单向 ramp，任一步异常立即停
</must_haves>
