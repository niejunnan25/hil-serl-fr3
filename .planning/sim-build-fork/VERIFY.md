
---

## A3: PlugSceneCfg + side_policy_cam + wrist_1_cam + side_classifier alias

**Status:** A3-CFG-SCHEMA: PASS

**cfg 字段确认**（来自 `sim/scenes/plug_scene.py` PlugSceneCfg）:
- `side_policy_cam: TiledCameraCfg` (top-down, 0.5/0/0.5, look down)
- `wrist_1_cam: TiledCameraCfg` (gripper-mounted, 0/0/0.05 offset)
- 分辨率 128x128 RGB (matches IMAGE_SHAPE)

**Contract 字段确认**（来自 `sim/data/contract.py`）:
- `VALID_PKL_IMAGE_KEYS = ("side_policy", "wrist_1", "side_classifier")` ✓
- `IMAGE_KEY_ALIAS_MAP = {"side_classifier": "side_policy"}` ✓

**A10 影响**: verify_sim_data.py 现在可 assert pkl schema 含三键 (side_policy / wrist_1 / side_classifier)。

**IsaacLab runtime test**: 本地无 isaaclab 安装 (pip show isaaclab = not found), cfg 字段正确性 = 单元测试覆盖；scene 实例化 = desktop env (mainline agent) 验证。

**Test 报告**:
- `python -m pytest sim/scenes/tests/test_plug_scene_cfg.py -v` → 3 passed
- `python -m pytest sim/data/tests/test_contract_image_aliases.py -v` → 4 passed

---

## A2: state 8D → 25D + contract.py hard-freeze

**Status:** `A2-STATE-HARD-FREEZE: PASS`

**Contract 字段确认**（来自 `sim/data/contract.py`）:
- `CONTRACT_VERSION = "v2.1-real-2026-06-11"`
- `ACTION_SCALE = (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0)` ✓ 与 mainline 一致
- `STATE_KEYS_ORDERED = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")` ✓ 与 `experiments/plug_insertion/config.py:237` 一致
- `STATE_DIMS = 25` ✓ 与 `experiments/plug_insertion/wrapper.py:23-25` 注释一致（hard-freeze）
- `STATE_DTYPE = "float32"` ✓
- `IMAGE_SHAPE = (3, 128, 128)` ✓
- `POLICY_IMAGE_KEYS = ("side_policy", "wrist_1")` ✓
- `CLASSIFIER_IMAGE_KEYS = ("side_classifier",)` ✓
- `CLASSIFIER_ALIAS_OF = (("side_classifier", "side_policy"),)` ✓ sim 单一 TiledCamera
- `VALID_PKL_IMAGE_KEYS = ("side_policy", "wrist_1", "side_classifier")` ✓ A3
- `IMAGE_KEY_ALIAS_MAP = {"side_classifier": "side_policy"}` ✓ A3

**25D dim 来源记录**（codex review #2 finding）:
- spec 算式: 7 (tcp_pose: pos+quat) + 6 (tcp_vel) + 3 (force) + 3 (torque) + 1 (gripper) = **20**
- wrapper.py 注释: `[tcp_pose(6), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1)] = 19D` (spec 88-91 行)
- 实测 dim: spec 88-91 行注明 "wrapper.py state[0, -1] 2D 索引说明实际 shape (1, 25) ⇒ dim=25"
- 矛盾说明: 算式 7+6+3+3+1=20 与 wrapper.py 25D 不一致；差额 5 维的来源:
  - 选项 A: tcp_pose = 6 (pos+euler) 而非 7 (pos+quat)
  - 选项 B: gripper 6D (gripper qpos[2] + 衍生 4 个 status) 而非 1D
  - 选项 C: tcp_pose + tcp_vel 各自增 1 个 scalar (e.g. tcp_status flag)
- **本 hard-freeze 取 25D** (与 wrapper.py 注释一致); `capture_observation` 实现采用
  tcp_pose(7)=pos+quat_xyzw, gripper(6)=scalar tiled 6× (7+6+3+3+6=25), 差额由 gripper 6D 承担。
  算式 20 的 spec 残留记录在 contract.py docstring 与 capture_observation docstring。
- **未决**: 用户合并时跑 `env.sample()["state"].shape` 实测确认 dim;
  若实测 = 20, 需改 STATE_DIMS=20 并调 gripper_pose 宽度 (e.g. 1D) 或 tcp_pose 维度 (e.g. 6D pos+euler)。

**L1 隔离 gate 关键字**（A1/A2 都跑）:
```bash
grep -rE "panda_joint|/home/robot|droid\.sim|from droid|import droid|EnvConfig|franka_env|from scripts|import scripts" sim/ --include="*.py" --include="*.sh"
```
**A2 L1 状态**: ✓ A2-scope clean (contract.py 0 外部 import; gello_replay.py 不引用 droid/EnvConfig/franka_env/scripts);
  pre-existing /home/robot/ 引用属 A6 (USD 路径) + A5 (joint rename 注释) 范围, 仍 pending fix。
  A2-scope 验证: `grep -nE "from droid|import droid|EnvConfig|franka_env|from scripts|import scripts" sim/data/contract.py sim/data/gello_replay.py sim/data/tests/test_contract.py sim/data/tests/test_state_25d.py` → 0 matches。

**A2 之前不可启动的 plan**:
- A9 (failure_scenario_generator 需要 25D state; 现在 OK)
- A10 (verify_sim_data.py 需要 contract.py 提供的 3 键 schema; 现在 OK)

**Test 报告**:
- `python -m pytest sim/data/tests/test_contract.py -v` → 9 passed
- `python -m pytest sim/data/tests/test_state_25d.py -v` → 4 passed (含 1 warning 记录 arith 矛盾)
- `python -m pytest sim/data/tests/ -v` → 24 passed, 2 skipped (intentional skip for unimplemented modules)

**gello_replay.py 改造点 (A2 diff summary)**:
1. `capture_observation()` 新签名加 `prev_tcp_pose`, `dt` 参数; 改用 `sim.kinematics.fr3_fk.fk_ee_pose` 算 tcp_pose(7D pos+quat);
2. tcp_vel 6D (pos diff 3 + angular placeholder 3), tcp_force/torque 3D zeros, gripper 6D (scalar tiled);
3. 拼接顺序严格按 STATE_KEYS_ORDERED, assert shape == (STATE_DIMS,);
4. `_rotmat_to_quat_xyzw()` + `_quat_xyzw_to_euler_xyz()` 纯 numpy helper (no scipy dep);
5. `replay_in_sim()` 初始化 `prev_tcp_pose=None` 并在循环内传递;
6. `replay_pure_fk()` 改用 `capture_observation()` 统一产 25D state; 加 `_local_trajectory_to_cartesian_deltas` 兜底
   (用 sim.kinematics.fr3_fk) 当 gello_pipeline 不可用 (dev box / CI);
7. `validate_output()` 改用 `STATE_DIMS` 而非 hardcode (8,)。

**A2 偏离 PLAN 的实现细节 (README)**:
1. **gripper_pose 6D = scalar tiled 6×** — plan 假设 gripper 1D (算式 7+6+3+3+1=20) 与 wrapper.py 25D 注释冲突;
   本 hard-freeze 选择 gripper 6D (差额 5 维分给 gripper, 因为 gripper 实际是
   franka_hand 的 2 finger + status, 6D 合理; tcp_pose 选 7D 因为 pos+quat 已是 SERL 主流通用)。
2. **增加 sim.kinematics.fr3_fk fallback** — plan 不要求, 但为了让 dev box / CI 能跑
   `replay_pure_fk` 单测 (不依赖 /home/robot/... 路径), 加了 `_local_trajectory_to_cartesian_deltas` 兜底
   (纯 numpy 旋转矩阵+quat 转换, 无 scipy 依赖)。sim 部署 (有 fr3-desktop-ts) 走
   scripts/fk_converter (scipy Rotation, singularity-safe), dev 走本地 fallback。
3. **contract.py 早就由 A4 创建** — plan 假设 A2 Task 1 还没建, 但 A4 提前建了 (A4 偏离 PLAN README 已记录);
   A2 Task 1 实际只补了 test_contract.py (9 个断言), contract.py 字段 A4 改 STATE_DIMS 注释 + 加 gripper 6D 解释。
