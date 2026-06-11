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
- 下一步：A6 (USD 路径硬编码修复) / A4 (action_scale 对齐)

### A4 完成 ✅
- sim/data/gello_replay.py DEFAULT_POS_SCALE: 0.1→0.015
- sim/data/gello_replay.py DEFAULT_RPY_SCALE: 0.2→0.1
- sim/data/gello_replay.py 新增 DEFAULT_GRIPPER_SCALE: 1.0
- 3 个常量改为从 sim.data.contract.ACTION_SCALE 派生 (single source of truth)
- replay_in_sim / replay_pure_fk 内部 action_scale 从 3D [pos, rpy, 0.0] 改为 7D list(ACTION_SCALE)
- L1 isolation gate (A4 范围内, 新文件 + diff): clean

### A4 偏离 PLAN 的实现细节（README）
1. **创建 sim/data/contract.py** — plan 假设 A2 Task 1 已经建好，但 A2 实际未跑。contract.py 内的 ACTION_SCALE = (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0) 按 spec 钉值，未 import experiments/scripts/droid（保持 L1 隔离）。
2. **fk_converter / normalize_action imports 包装 try/except** — plan 未要求，但 gello_replay 在 vanilla dev / CI 上 import 会因 /home/robot/... 路径不存在而失败，导致所有常量测试因 ModuleNotFoundError 而非断言失败而 RED。包装后 surfaces None；replay_*() 入口在真正调用时再 raise 清晰错误。
3. **A4 test 套 5 项全部 PASS**（plan 期望 22 passed 的 22 = contract 9 + state_25d 4 + image_aliases 4 + action_scale 5；后三项属 A2/A3 plan，A4 隔离执行时不在本分支上）。

### A3 完成 ✅
- sim/scenes/plug_scene.py PlugSceneCfg 加 side_policy_cam + wrist_1_cam (TiledCameraCfg; local 无 isaaclab 时用 _StubTiledCamera)
- sim/data/contract.py 加 IMAGE_KEY_ALIAS_MAP (side_classifier→side_policy) + VALID_PKL_IMAGE_KEYS (3 键)
- L1 isolation gate: OK (per-file gate on contract.py / plug_scene.py 中我添加的部分；pre-existing /home/robot 引用属 A5/A6 范围)
- IsaacLab runtime test 推迟到 mainline agent / desktop env (本地无 isaaclab 安装)
- 下一步：A8 (domain randomization) 现在可启动 (依赖 A3 cfg 字段)

### A2 完成 ✅
- sim/data/contract.py STATE_DIMS=25 hard-freeze (verifies 7+6+3+3+6=25, gripper 6D tiled)
- sim/data/gello_replay.py capture_observation 改 25D (per STATE_KEYS_ORDERED, sim.kinematics.fr3_fk FK)
- sim/data/gello_replay.py replay_pure_fk + replay_in_sim 同步产 25D (用 capture_observation 统一入口)
- sim/data/gello_replay.py validate_output 改用 STATE_DIMS (不再 hardcode 8)
- sim/data/tests/test_contract.py 新增 9 个断言 (含 STATE_DIMS=25 / STATE_KEYS_ORDERED 等)
- sim/data/tests/test_state_25d.py 新增 4 个断言 (含 arith 20 vs wrapper 25 矛盾的 warning)
- L1 A2-scope isolation gate: clean (无 droid/EnvConfig/franka_env/scripts import; 预存 /home/robot 是 A6 USD 范围)
- L1 全 sim tree: 24 passed, 2 skipped, 1 warning
- VERIFY.md hard-freeze 标签: A2-STATE-HARD-FREEZE: PASS
- 下一步：A9 (failure_scenario_generator) 现在可启动；A10 (verify_sim_data.py) 现在可启动
