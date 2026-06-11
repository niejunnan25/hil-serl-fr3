"""A9: failure scenario 范围常量必须定义在 sim/data/contract.py.

4 类失败:
  1. mis_alignment: plug 偏移 ±3cm
  2. angle_offset:  z 旋转 ±10°
  3. insufficient_force: 最后 N 帧提前 close_gripper
  4. drop:           中段 release_gripper
"""
import pytest


def test_misalignment_xy_m():
    from sim.data.contract import FAILURE_MISALIGNMENT_XY_M
    assert FAILURE_MISALIGNMENT_XY_M == 0.03  # 3 cm


def test_angle_offset_deg():
    from sim.data.contract import FAILURE_ANGLE_OFFSET_DEG
    assert FAILURE_ANGLE_OFFSET_DEG == 10.0


def test_insufficient_force_frames():
    from sim.data.contract import FAILURE_INSUFFICIENT_FORCE_FRAMES
    assert FAILURE_INSUFFICIENT_FORCE_FRAMES == 10
    assert isinstance(FAILURE_INSUFFICIENT_FORCE_FRAMES, int)


def test_drop_frame_ratio():
    from sim.data.contract import FAILURE_DROP_FRAME_RATIO
    assert FAILURE_DROP_FRAME_RATIO == 0.5
    assert 0.0 < FAILURE_DROP_FRAME_RATIO < 1.0


def test_failure_class_names_complete():
    """A9 必须定义 4 类 failure class 名称 (作为 enum 字符串列表)."""
    from sim.data.contract import FAILURE_CLASSES
    assert FAILURE_CLASSES == (
        "mis_alignment", "angle_offset", "insufficient_force", "drop",
    )


def test_failure_outputs_reward_zero():
    """failure 输出的 reward 必须全 0.0 (与 real positive 混合训练用)."""
    from sim.data.contract import FAILURE_REWARD
    assert FAILURE_REWARD == 0.0
