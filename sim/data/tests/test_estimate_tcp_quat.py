"""C2: estimate_tcp_from_state 必须返回真实 TCP 朝向 (xyzw->wxyz), 非 identity 占位.

回归点:
  - gello_replay.capture_observation 把 tcp_pose 写为 pos(3)+quat_xyzw(4) 作为
    25D state 的前 7 维 (scalar-last)。
  - check_insertion 约定 tcp_quat 为 [w,x,y,z] (scalar-first), 用于角度对齐判定。
  - 旧实现返回 identity [1,0,0,0] 占位, 导致任意倾斜的 plug 角度误差恒为 0,
    可能把倾斜失败帧误判为 success。
"""
import math
import numpy as np

from sim.data.plug_reward_labeler import (
    estimate_tcp_from_state,
    check_insertion,
    DEFAULT_SOCKET_POS,
    DEFAULT_SOCKET_QUAT,
)


def _state_with_quat_xyzw(tcp_pos, q_xyzw):
    """构造 25D state, 前 3 维为 tcp_pos, 第 3:7 维为 quat_xyzw."""
    state = np.zeros(25, dtype=np.float64)
    state[0:3] = tcp_pos
    state[3:7] = q_xyzw
    return state


def test_estimate_returns_non_identity_quat():
    """非 identity 的 state 朝向不应被压成 identity 占位."""
    half = math.radians(10.0) / 2.0
    # 绕 Y 轴 10°: wxyz=[cos,0,sin,0]  =>  xyzw=[0,sin,0,cos]
    q_xyzw = np.array([0.0, math.sin(half), 0.0, math.cos(half)])
    state = _state_with_quat_xyzw([0.0, 0.0, -0.015], q_xyzw)

    _tcp_pos, tcp_quat = estimate_tcp_from_state(
        state, DEFAULT_SOCKET_POS, DEFAULT_SOCKET_QUAT
    )

    assert not np.allclose(tcp_quat, [1.0, 0.0, 0.0, 0.0]), (
        f"tcp_quat 不应为 identity 占位, 实际 {tcp_quat}"
    )


def test_estimate_converts_xyzw_to_wxyz():
    """state[3:7]=xyzw 必须被转换为 check_insertion 约定的 [w,x,y,z]."""
    # 任意可区分分量, 便于校验 convention 顺序
    q_xyzw = np.array([0.1, 0.2, 0.3, 0.9])  # x,y,z,w
    norm = np.linalg.norm(q_xyzw)
    q_xyzw = q_xyzw / norm
    state = _state_with_quat_xyzw([0.01, 0.02, 0.03], q_xyzw)

    _tcp_pos, tcp_quat = estimate_tcp_from_state(
        state, DEFAULT_SOCKET_POS, DEFAULT_SOCKET_QUAT
    )

    expected_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    assert np.allclose(tcp_quat, expected_wxyz), (
        f"tcp_quat 应为 wxyz={expected_wxyz}, 实际 {tcp_quat}"
    )


def test_tilted_plug_angle_error_recovered():
    """10° 倾斜的 plug 经 estimate->check_insertion 应得到 ~10° 角度误差.

    旧 identity 占位会给出 0°, 把倾斜帧误判为角度对齐 (angle_ok=True)。
    """
    half = math.radians(10.0) / 2.0
    q_xyzw = np.array([0.0, math.sin(half), 0.0, math.cos(half)])
    state = _state_with_quat_xyzw([0.0, 0.0, -0.015], q_xyzw)

    socket_pos = np.array([0.0, 0.0, 0.0])
    socket_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity

    tcp_pos, tcp_quat = estimate_tcp_from_state(state, socket_pos, socket_quat)
    result = check_insertion(tcp_pos, tcp_quat, socket_pos, socket_quat)

    assert abs(result.angle_error_deg - 10.0) < 0.1, (
        f"倾斜 10° 应恢复 ~10° 角度误差, 实际 {result.angle_error_deg:.3f}°"
    )
    assert bool(result.angle_ok) is False, (
        "10° 倾斜 (> 5° 阈值) 不应通过角度对齐判定"
    )
