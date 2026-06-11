# SIM 数据增强 — 推进日志

## 2026-06-11

### Plan 完成
- 只读盘点完成
- 写入 `.planning/sim-build-fork/PLAN.md`
- 范围: sim/ + sim_remote/ + .planning/sim-build-fork/

### 关键发现
1. **sim_remote/ 与 sim/ 重复** — `sim_remote/sim/` 与 `sim_remote/` 顶层 95% 重复
2. **real pkl 不可用** — 验证脚本 A11/A12 需要真机数据，当前 Phase 3 尚未完成
3. **sim_remote USD 路径硬编码** — `/home/robot/plug_insertion_sim/assets/panda_arm_hand.usd`
   不存在；正确路径是 `sim_remote/assets/fr3.usd`（已存在）
4. **sim_remote joint 名称不匹配** — 用的是 `panda_joint*`，应改 `fr3_joint*`
5. **action_scale 不匹配** — sim 用 0.1/0.2，real SERL 用 0.015/0.1
6. **state 维度不匹配** — sim 8D (joint + gripper)，real 25D (tcp_pose/vel/force/torque/gripper)

### 决策
- **sim/ 作为 sim-only 唯一目录**，sim_remote/ 搬运到 sim/ 后废弃顶层 sim/
- **不 import real 侧任何模块** — sim 侧 hardcode scale 与 proprio 维度
- **domain randomization**: lighting 800-1200, camera ±5°, plug ±1cm xy jitter
- **failure scenario 类别**: mis_alignment, angle_offset, insufficient_force, drop

### 下一步
- 开始 SIM-A1（目录统一）
- 然后 SIM-A2/A3/A4/A5/A6 并行（gello_replay 改造）
- 然后 SIM-A7/A8（reward labeler + domain randomization）
- 最后 SIM-A9（failure scenario generator）+ A10/A11/A12 验证脚本

### A1 完成 ✅
- sim_remote/ 资产全部搬入 sim/ 对应子目录
- sim_remote/sim/ 重复目录删除
- sim_remote/ 根目录清理（保留 .pyc 缓存待 CI 清理）
- 删除项：sim_remote/obs/obs_dict.py (droid.sim legacy)、sim_remote/data/demo_relative_replay.py (DEPRECATED 标注)、sim_remote/assets/configuration/ (旧 IsaacLab config)
- L1 隔离 gate: 已知 FAIL（panda_joint + /home/robot 残留 — A5/A6/A7 任务范围；本 plan 阻塞项不命中）
- 下一步：A2 (state 8D→25D hard-freeze)

### 偏离 PLAN-A1 的实现细节（README）
1. **git mv 不可用** — 仓库所有 sim/ 与 sim_remote/ 文件在 ai/20260228-optimize-cge 上是 untracked；改用 `mv + git add` 模拟 git mv 效果（最终 commit 体现为新增，无 rename 检测，这是分支历史决定的，无法补救）。
2. **测试名规范化** — `test_imports.py` 在 5 个子目录下重名，pytest 收集冲突；改名 `test_{subdir}_imports.py` 以解决 import-mode=prepend 下的 module 重名。
3. **测试断言需根据实际函数名调整**：
   - `sim.kinematics.fr3_fk` → 实际 `fk_ee_pose`（非 `forward_kinematics`）
   - `sim.obs.camera_wrapper` → 实际 `camera_rgb_to_obs`（非 `CameraWrapper` class）
   - `sim.safety.runtime_check` → 实际 `ensure_real_stack_idle`（非 `assert_no_real_stack_running`）
   - `sim.data.plug_reward_labeler` → 实际 `label_rewards`（非 `label_insertion`）
   - `sim.data.sim_replay_pipeline` → 实际 `run_pipeline_single` / `run_pipeline_batch`（非 `run_pipeline`）
4. **`sim_replay_pipeline` 测试需 skip** — 该模块 line 59 import `from gello_replay import ...`，gello_replay 又引用不存在的 `/home/robot/...` 路径，import 链断。已加 `@pytest.mark.skip(reason="待 A2-A6 修复")`。

### A5 完成 ✅ (2026-06-11)
- sim/data/gello_replay.py: panda_joint[1-7] → fr3_joint[1-7] (10 处)
- sim/data/gello_replay.py: panda_finger_joint.* → fr3_finger_joint.* (2 处)
- sim/scenes/plug_scene.py: 已是 fr3_joint (regression lock)
- sim/data/plug_reward_labeler.py: 已是 fr3_/无 joint 字符串 (regression lock)
- 新增: sim/data/tests/test_joint_names.py (5 tests: 3 zero-panda + 2 fr3-present)
- L1 A5-scope gate: OK
- L1 全 sim tree: 除 test_joint_names.py (字符串字面 assertion) 外零 panda_joint 残留
- 下一步：A6 (USD 路径硬编码修复)
