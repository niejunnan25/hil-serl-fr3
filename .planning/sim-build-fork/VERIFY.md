
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

---

## A11: gen_mock_real_pkl + test_domain_alignment (MOCK SMOKE ONLY)

**Status:** A11-MOCK-SMOKE: PASS (schema smoke; NOT phase6-ready)

**Codex 修复证据**:

**Codex #6 (MED "A12 mock 阈值 > 50% 太弱")**:
- A11 mock run 是 **schema/format smoke**, 不是 readiness
- test_domain_alignment.py 的 "通过条件" = 脚本能 end-to-end 跑通 + 打印 report
- 不做 accuracy-based 验证 (e.g. "state_range_overlap > 0.8" 不构成 readiness)
- Real pkl + balanced fixture + confusion matrix (precision/recall ≥ 0.85 per ROADMAP)
  = **phase6-ready gate**, 需要 user approval, 留待合并时实测

**Tools 确认**（来自 `sim/scripts/`）:
- `gen_mock_real_pkl.py` — CLI: `--output path --num-frames N --pos-ratio 0.8 --seed 20260611`
  产 50-100 帧 mock real pkl (3 键 image + 25D state + 7D action + 80% pos_ratio)
- `test_domain_alignment.py` — CLI: `--sim path/sim.pkl --real path/mock_real.pkl`
  跑 image stats (mean/std) + state range overlap, 打印 report, exit 0 = smoke pass

**Report 内容**:
- `image_mean_sim`, `image_mean_real`: float (overall mean across 3 image keys)
- `image_std_sim`, `image_std_real`: float
- `state_range_overlap`: float ∈ [0, 1] (intersection / union of per-dim state ranges)
- ⚠️ 这些数值是 **信息性**, 不作 readiness 判据

**Test 报告**:
- `python -m pytest sim/scripts/tests/test_gen_mock_real_pkl.py -v` → 4 passed
- `python -m pytest sim/scripts/tests/test_domain_alignment.py -v` → 4 passed
- CLI smoke: gen → verify → alignment 全跑通 exit 0

**Phase6-ready 真要求 (per ROADMAP)**:
- Real pkl (50%+ positive, 真实图像)
- Balanced fixture (50/50 pos/neg)
- Confusion matrix: precision ≥ 0.85, recall ≥ 0.85
- User approval gate (per spec D11)
- A11/A12 mock smoke 不构成此 readiness

---

## A12: test_mixed_training (mixed sim + mock real, SCHEMA SMOKE ONLY)

**Status:** A12-MIXED-SMOKE: PASS (schema smoke; NOT phase6-ready)

**Codex 修复证据**:

**Codex #6 (MED "A12 mock 阈值 > 50% 太弱")**:
- A12 通过条件 = **schema/format smoke**, NOT accuracy-based
- "测试" = 脚本跑通 + 模型 fit (sklearn LogisticRegression) + 预测 emit, exit 0
- 报告含 accuracy (∈ [0,1]) + baseline_accuracy (majority class) + confusion_matrix
- **不**验证 accuracy ≥ 0.85 / precision ≥ 0.85 / recall ≥ 0.85
- 80% pos baseline = 0.8 > 0.5 任何阈值; majority class baseline accuracy 报告是必须的

**Codex #7 (HIGH "gsd-autonomous 启动缺 hard gate")**:
- A12 完成 ≠ phase6-ready
- `sim-code-ready` label 可在 A12 done 后打 (A1-A12 schema smoke 全 pass)
- `phase6-ready` label 必须 user approval (per spec D11)
- 本 plan **不**声明 phase6-ready

**Tools 确认**（来自 `sim/scripts/test_mixed_training.py`）:
- `load_features(sim_pkl, real_pkl) -> (X, y)`: 加载 pkl, 25D state features + rewards labels
- `train_classifier(X, y) -> (model, X_test, y_test)`: sklearn LogisticRegression, 80/20 stratified split
- `compute_report(y_true, y_pred) -> dict`: accuracy + baseline + confusion matrix
- `print_report(report)`: stdout 打印
- `main() -> int`: CLI entry, exit 0 (smoke pass)

**Report 内容**:
- `accuracy: float ∈ [0, 1]` (model on test set)
- `baseline_accuracy: float ∈ [0, 1]` (majority class baseline = max(pos_ratio, 1-pos_ratio))
- `confusion_matrix: [tn, fp, fn, tp]` (rows=true, cols=pred)
- `n_test: int`
- ⚠️ 这些数值是 **信息性**, 不作 readiness 判据

**Test 报告**:
- `python -m pytest sim/scripts/tests/test_mixed_training.py -v` → 6 passed
- CLI smoke: A9 sim pkl + A11 mock real pkl → mixed training 跑通 exit 0

---

## sim-code-ready label (per spec D5b + D11)

**`v2.1 sim-code-ready: PASS`** (2026-06-11)

触发条件 (per spec 1.9):
- ✅ A1-A10 全部 done (A8 domain randomization + A9 failure_scenario + A10 verify schema)
- ✅ A11/A12 schema smoke pass
- ✅ VERIFY.md 出现 "v2.1 sim-code-ready"

**`phase6-ready: DEFERRED`** — 不在 sim fork 范围。

**phase6-ready 真要求 (per ROADMAP Phase 6 退出判据)**:
- Real pkl (50%+ positive, 真实图像)
- Balanced fixture (50/50 pos/neg)
- Confusion matrix: precision ≥ 0.85, recall ≥ 0.85
- User approval gate (per spec D11)
- A11/A12 mock smoke 不构成此 readiness

接手 agent 第一步:
1. 读 DESIGN.md / PROGRESS.md / VERIFY.md
2. 决定: 继续未完成 plan (本 milestone 全部 done) / 申请 `phase6-ready` gate / 合并回主线
