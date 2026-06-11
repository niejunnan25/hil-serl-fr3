# VERIFY.md — sim-build-fork v2.1

Per-plan validation labels, append-only.

---

## A4: action_scale 对齐 (0.015, 0.1, 1.0) — 从 contract 强制

**Status:** A4-ACTION-SCALE: PASS

**常量确认**（来自 `sim/data/gello_replay.py`）:
- `DEFAULT_POS_SCALE = ACTION_SCALE[0] = 0.015` ✓
- `DEFAULT_RPY_SCALE = ACTION_SCALE[3] = 0.1` ✓
- `DEFAULT_GRIPPER_SCALE = ACTION_SCALE[6] = 1.0` ✓ (新加)
- `replay_in_sim` / `replay_pure_fk` 内 `action_scale = list(ACTION_SCALE)` (7D, 非 3D)

**Source of truth 链**:
  gello_replay → sim.data.contract.ACTION_SCALE = (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0)
  contract.ACTION_SCALE → 手动 hardcode (与 mainline EnvConfig.ACTION_SCALE 一致; 不 import)

**Test 报告**:
- `python -m pytest sim/data/tests/test_action_scale.py -v` → 5 passed
- `python -m pytest sim/data/tests/` → 6 passed, 2 skipped (pre-existing skips)
