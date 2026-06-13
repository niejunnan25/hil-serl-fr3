# Phase A6 gap-closure — verification evidence (2026-06-13)

修复 REVIEW-PhaseA.md 的 2 critical + 7 important + 卫生项。执行：Opus 4.8 xhigh，
顺序 TDD subagent workflow（C1/ARB/XBOX/REC）+ DRV 与对抗式复核在主会话内完成
（verify 子代理因 API ECONNRESET 中断，按 model-stage-policy 由 Opus 主会话兜底复核）。

## 全量测试

```
cd hilserl-fr3 && python3 -m pytest tests/ -q   ->   225 passed
```
（A6 前 203 passed；+22 个新回归测试覆盖每条修复的 must_have。）

per-suite（A6 触及的）：
- test_gello_intervention_a2_contract  16 passed  (C1: +6 device-error 测试)
- test_teleop_arbiter                  15 passed  (ARB: I2/I3/A2-F3 新增+改写)
- test_xbox_intervention_contract      23 passed  (XBOX: I1 cap +6, 3 改写)
- test_record_hybrid_demos             14 passed  (REC: I4 gripper + I7 CLI)
- test_p2t3_motion_driver              13 passed  (DRV: C2 fail-closed 改写)
- 既有未回归：gello_contract 15 / delta_agent 19 / safety_calibration 25 /
  e2e_scaffolding 21 / record_gello_demos_serl 33 / teleop_hub 9

## finding → fix → 状态

| ID | 严重度 | 修复 | 文件 | 状态 |
|----|--------|------|------|------|
| C1 | critical | action() 宽 except（保留 RuntimeError 在前）+ 短读 ValueError + `_device_error_logged`；设备故障降级 `(policy,False)` 不抛 | gello_intervention.py | ✅ 实测 OSError/IOError/短读不崩 |
| C2 | critical | `--mode full` fail-closed：返回 rc10、不 POST、删除 `_post_pose`；归一化 delta→绝对 /pose 重建留 Phase B | p2_t3_e2e_motion_driver.py | ✅ rc10，无 /pose |
| I1 | important | XboxIntervention 内置 per-step 平移 cap：`‖action[:3]·pos_scale‖≤max_step`，方向保留 | xbox_intervention.py | ✅ 满杆 coarse 也 ≤3mm |
| I2 | important | arbiter.reset() 只 reset base env 一次（委托 gello.reset），不再二次 env.reset | teleop_arbiter.py | ✅ CountEnv==1 |
| I3 | important | 传播 wrapper 的 replaced；downgrade→DEVICE_NONE，不写 intervene_action | teleop_arbiter.py | ✅ 不再污染 buffer |
| A2-F3 | important | 引擎更新每 tick 恰好一次；XBOX→GELLO 恢复时 re-seed agent 到当前 leader 位姿 | teleop_arbiter.py | ✅ 恢复不误拒 |
| I4 | important | Xbox 段夹爪从 RT/LT 取（RT>0.05→+1, LT>0.05→-1），GELLO 段仍取第 8 通道 | record_hybrid_demos.py | ✅ |
| I5 | important→minor | docstring 改为 joint-space 占位（非 XboxIntervention._state_to_action）；删未用 import | record_hybrid_demos.py | ✅ |
| I6 | important | 16_*.sh motion() 显式 `--mode full` + rc10→exit32 + banner 注明 fail-closed | 16_gello_e2e_motion_test.sh | ✅ |
| I7 | important | CLI dry-run q0=FR3_DEFAULT_JOINTS（不再 tick-0 abort）+ run_dry None 时打印 [ABORT] | record_hybrid_demos.py | ✅ |
| I8 | important | evidence 目录改名去掉 `.log` 后缀（可跟踪）；本 .md 为持久证据；raw per-suite *.log 仍按 .gitignore 忽略（可再生） | evidence/ | ✅ |
| 卫生 | minor | 删死代码 `_clip_to_limits`；XBOX/REC 清未用 import | — | ✅ |
| REFUTED | — | XCQ-03（多次 poll）原 review 已对抗校验降级 minor，不修 | — | — |

## 已知遗留（非本轮范围，记录待 B4/后续）

- FR3 关节限位常量（FR3_LOWER/UPPER_LIMITS）仍在 4+ 文件各自重定义；抽公共模块是
  跨文件重构，留待后续统一（cosmetic，非功能问题）。
- A2-F3 的 re-seed 仅覆盖 XBOX→GELLO 恢复；NONE→GELLO（中性态直接按 LB）的 stale-anchor
  是 pre-existing 轻微项，留 B4 真机标定时连同 leader_scale 一并处理。
- Xbox 段动作空间（关节）与 policy 期 XboxIntervention（笛卡尔）不一致仍是 B4 占位，
  待测量 Jacobian 后统一（I5 已在 docstring 明示）。

## 边界

A6 全程零真机 motion：DRV full 模式 fail-closed，所有测试 mock/dry-run。
未改动 v2.1.1 sim 工作的任何文件。
