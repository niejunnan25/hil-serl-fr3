
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
