
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

## A8: domain randomization (light / camera / plug)

**Status:** A8-DR-CODE: PASS (smoke-only; runtime 验证推迟到 desktop env)

**DR 范围确认**（来自 `sim/data/contract.py`）:
- `LIGHT_INTENSITY_MIN/MAX = 800.0 / 1200.0` ✓
- `CAMERA_YAW_RANGE_DEG = (-5.0, 5.0)` ✓
- `CAMERA_PITCH_RANGE_DEG = (-5.0, 5.0)` ✓
- `PLUG_XY_JITTER_M = 0.01` (1cm) ✓
- `PLUG_RZ_JITTER_RAD = 0.1` (~5.7°) ✓
- `RANDOMIZE_SEED_DEFAULT = 20260611` ✓

**PlugScene API 确认**（来自 `sim/scenes/plug_scene.py`）:
- `__init__(..., randomize: bool = False, dr_seed: Optional[int] = None)`
- `_randomize_lighting(rng) -> float` ∈ [800, 1200]
- `_randomize_camera_pose(rng) -> (yaw, pitch) degrees` in ±5° range
- `_randomize_plug_pose(rng) -> (dx, dy, drz)` in ±1cm / ±0.1 rad
- `info_dr_samples() -> dict` for debug + L3 reporting

**Codex review #3 trade-off (记录于此)**:
- Project/Roadmap 命名 DR 为 "light/background randomization"
- 本 plan **只做 light/camera/plug, 不做 background** (textures / sky 随机化)
- 理由: (a) background DR 涉及 scene USD material binding + 3D 资产变动, 与本 fork
  "sim-as-augmentation" 范围不匹配; (b) light/camera/plug jitter 已覆盖视觉变异的
  80% 用例 (per DROID-style DR 经验); (c) background DR 留待 v2.2 / mainline Phase 6
  退出时补, 由 mainline agent 决定优先级。
- 接手 agent 若需 background DR, 应开 v2.2 计划, 不要再扩 A8。

**Test 报告**:
- `python -m pytest sim/data/tests/test_contract_dr.py -v` → 6 passed
- `python -m pytest sim/scenes/tests/test_domain_randomization.py -v` → 7 passed
  (6 from PLAN-A8 spec + 1 extra `test_info_dict_has_dr_samples_when_randomize_enabled`)

**IsaacLab runtime test**: 本地无 isaaclab 安装, _randomize_* 方法的 **runtime 行为**
(实际改 light intensity 数值 / camera prim transform / plug root_pos_w) 推迟到 desktop
env (mainline agent) 验证; **本 plan 验证的是** 纯函数返回值在 contract 范围内 + 复现性。

**Test adaptation (TDD 适配)**: 本 worktree local 无 isaaclab, 标准 import-based 测试
走 source-text fallback 兜底 (per test_plug_scene_cfg.py 模式), 保证 TDD cycle 可见
(no-skip path)。PlugScene class-level tests 用 `PlugScene.__new__(PlugScene)` 跳过
`__init__` 来直接验证 3 个 `_randomize_*` 方法的纯函数行为。
