"""A7: plug_reward_labeler.py 阈值与主线 8mm/2mm/5° 一致.

数值参考: ~/.planning/hil-serl-plug/evidence/sim-scene/insertion_detector.py
  - insertion_depth_threshold = 0.008 (8mm)
  - xy_tolerance = 0.002 (2mm)
  - angle_tolerance_deg = 5.0 (5°)

本 test 锁死 3 个常量 + 5 类 synthetic-pose:
  1. 完美插入 (15mm 深, 0mm XY, 0° 倾角) → success
  2. 浅插入 (5mm 深) → fail (depth)
  3. XY 偏移 (3mm) → fail (xy)
  4. 角度倾斜 (10°) → fail (angle)
  5. 边界 (8mm 深, 2mm XY, 5° 倾角) → success (>= / <= 边界)
"""
import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 常量断言（直接 import 锁死数值）
# ---------------------------------------------------------------------------
def test_insertion_depth_threshold_is_8mm():
    from sim.data.plug_reward_labeler import INSERTION_DEPTH_THRESHOLD
    assert INSERTION_DEPTH_THRESHOLD == pytest.approx(0.008, abs=1e-9), (
        f"INSERTION_DEPTH_THRESHOLD 必须 = 8mm (0.008m), "
        f"实际 {INSERTION_DEPTH_THRESHOLD}"
    )

def test_xy_tolerance_is_2mm():
    from sim.data.plug_reward_labeler import XY_TOLERANCE
    assert XY_TOLERANCE == pytest.approx(0.002, abs=1e-9), (
        f"XY_TOLERANCE 必须 = 2mm (0.002m), 实际 {XY_TOLERANCE}"
    )

def test_angle_tolerance_deg_is_5():
    from sim.data.plug_reward_labeler import ANGLE_TOLERANCE_DEG
    assert ANGLE_TOLERANCE_DEG == pytest.approx(5.0, abs=1e-9), (
        f"ANGLE_TOLERANCE_DEG 必须 = 5.0°, 实际 {ANGLE_TOLERANCE_DEG}"
    )

def test_angle_tolerance_rad_derived_from_deg():
    from sim.data.plug_reward_labeler import (
        ANGLE_TOLERANCE_DEG, ANGLE_TOLERANCE_RAD,
    )
    assert ANGLE_TOLERANCE_RAD == pytest.approx(
        math.radians(ANGLE_TOLERANCE_DEG), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Synthetic-pose 5 类 case（用 check_insertion 验证判定正确性）
# ---------------------------------------------------------------------------
@pytest.fixture
def socket_identity():
    """Socket 在原点, identity quat [w,x,y,z] = [1,0,0,0]."""
    return (
        np.array([0.0, 0.0, 0.0]),           # socket_pos
        np.array([1.0, 0.0, 0.0, 0.0]),      # socket_quat wxyz
    )


@pytest.fixture
def identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])  # wxyz


def _y_rotation_quat(angle_rad: float) -> np.ndarray:
    """绕 Y 轴旋转 angle_rad 的四元数 [w,x,y,z]."""
    half = angle_rad / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0])


def test_case1_perfect_insertion_succeeds(socket_identity, identity_quat):
    """完美插入: 深度 15mm, XY 0mm, 角度 0° → success=True."""
    from sim.data.plug_reward_labeler import check_insertion
    socket_pos, socket_quat = socket_identity
    # 沿 socket -Z 方向插入 15mm
    plug_pos = np.array([0.0, 0.0, -0.015])
    result = check_insertion(plug_pos, identity_quat, socket_pos, socket_quat)
    # 用 bool(...) 避免 numpy.bool_ 与 Python bool 的 is 比较陷阱
    assert bool(result.success) is True, (
        f"完美插入应判定 success, depth={result.depth*1000:.2f}mm, "
        f"xy_err={result.xy_error*1000:.2f}mm, angle_err={result.angle_error_deg:.2f}°"
    )


def test_case2_shallow_insertion_fails(socket_identity, identity_quat):
    """浅插入: 深度 5mm (< 8mm 阈值) → success=False, depth_ok=False."""
    from sim.data.plug_reward_labeler import check_insertion
    socket_pos, socket_quat = socket_identity
    plug_pos = np.array([0.0, 0.0, -0.005])  # 5mm
    result = check_insertion(plug_pos, identity_quat, socket_pos, socket_quat)
    assert bool(result.success) is False
    assert bool(result.depth_ok) is False
    assert result.depth == pytest.approx(0.005, abs=1e-6)


def test_case3_xy_offset_fails(socket_identity, identity_quat):
    """XY 偏移 3mm (> 2mm 阈值) → success=False, xy_ok=False."""
    from sim.data.plug_reward_labeler import check_insertion
    socket_pos, socket_quat = socket_identity
    plug_pos = np.array([0.003, 0.0, -0.015])  # X 偏 3mm, 深度 15mm
    result = check_insertion(plug_pos, identity_quat, socket_pos, socket_quat)
    assert bool(result.success) is False
    assert bool(result.xy_ok) is False
    assert result.xy_error == pytest.approx(0.003, abs=1e-6)


def test_case4_angle_tilt_fails(socket_identity, identity_quat):
    """角度倾斜 10° (> 5° 阈值) → success=False, angle_ok=False."""
    from sim.data.plug_reward_labeler import check_insertion
    socket_pos, socket_quat = socket_identity
    plug_pos = np.array([0.0, 0.0, -0.015])  # 15mm 深
    # 绕 Y 轴 10° 倾斜
    plug_quat = _y_rotation_quat(math.radians(10.0))
    result = check_insertion(plug_pos, plug_quat, socket_pos, socket_quat)
    assert bool(result.success) is False
    assert bool(result.angle_ok) is False
    assert result.angle_error_deg == pytest.approx(10.0, abs=0.1)


def test_case5_boundary_insertion_succeeds(socket_identity, identity_quat):
    """边界条件: 深度 8mm, XY 2mm, 角度 5° → success=True (>= / <= 边界)."""
    from sim.data.plug_reward_labeler import check_insertion
    socket_pos, socket_quat = socket_identity
    # 8mm 深 + 2mm XY 偏移 + 5° 倾角（边界值）
    plug_pos = np.array([0.002, 0.0, -0.008])
    plug_quat = _y_rotation_quat(math.radians(5.0))
    result = check_insertion(plug_pos, plug_quat, socket_pos, socket_quat)
    assert bool(result.success) is True, (
        f"边界条件应判定 success (>= / <= 阈值), depth={result.depth*1000:.2f}mm "
        f"(need >= 8), xy_err={result.xy_error*1000:.2f}mm (need <= 2), "
        f"angle_err={result.angle_error_deg:.2f}° (need <= 5)"
    )


# ---------------------------------------------------------------------------
# reward 计算 sanity (sparse / dense 模式)
# ---------------------------------------------------------------------------
def test_sparse_reward_is_binary(socket_identity, identity_quat):
    """sparse reward 在 success/fail 下分别为 1.0 / 0.0."""
    from sim.data.plug_reward_labeler import (
        check_insertion, compute_sparse_reward,
    )
    socket_pos, socket_quat = socket_identity
    # 成功 case
    ok_result = check_insertion(
        np.array([0.0, 0.0, -0.015]), identity_quat, socket_pos, socket_quat,
    )
    assert compute_sparse_reward(ok_result) == 1.0
    # 失败 case
    fail_result = check_insertion(
        np.array([0.0, 0.0, -0.005]), identity_quat, socket_pos, socket_quat,
    )
    assert compute_sparse_reward(fail_result) == 0.0


def test_dense_reward_in_unit_interval(socket_identity, identity_quat):
    """dense reward 必须 ∈ [0, 1]."""
    from sim.data.plug_reward_labeler import (
        check_insertion, compute_dense_reward,
    )
    socket_pos, socket_quat = socket_identity
    for pos in [
        np.array([0.0, 0.0, -0.015]),   # 完美
        np.array([0.0, 0.0, -0.005]),   # 浅
        np.array([0.003, 0.0, -0.015]), # XY 偏
        np.array([0.0, 0.0, 0.0]),      # 完全在外面
    ]:
        result = check_insertion(pos, identity_quat, socket_pos, socket_quat)
        r = compute_dense_reward(result)
        assert 0.0 <= r <= 1.0, f"dense reward {r} 超出 [0,1] for pos={pos}"
