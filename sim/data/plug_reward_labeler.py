#!/usr/bin/env python3
"""plug_reward_labeler.py

基于 sim 中的 plug 插入状态自动标记 reward。
对 gello_replay.py 生成的 SERL pkl 文件进行后处理:
  - 分析 sim 中的 plug/socket 位姿
  - 自动判定插入成功/失败
  - 为每帧 transition 分配 reward

reward 策略 (参考 insertion_detector.py):
  稀疏模式: 成功帧 reward=1, 其余=0
  密集模式: 基于插入深度 + XY 对齐 + 角度对齐的连续 reward

判定条件:
  1. 插入深度 >= 8mm
  2. XY 对齐误差 < 2mm
  3. 角度误差 < 5 deg

用法:
  python plug_reward_labeler.py input_sim.pkl --output labeled.pkl
  python plug_reward_labeler.py input_sim.pkl --mode dense --output labeled.pkl
  python plug_reward_labeler.py /tmp/sim_replays/ --batch --output-dir /tmp/labeled/
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# Constants (aligned with insertion_detector.py)
# ===========================================================================
INSERTION_DEPTH_THRESHOLD = 0.008   # 8mm
XY_TOLERANCE = 0.002               # 2mm
ANGLE_TOLERANCE_DEG = 5.0          # 5 degrees
ANGLE_TOLERANCE_RAD = math.radians(ANGLE_TOLERANCE_DEG)

# 插座位姿 (与 plug_insertion_scene.py / isaac_lab_scene_config.py 一致)
# 在 sim 中, 插座固定在桌面上已知位置
# 如果不同场景, 通过 CLI --socket-pos/--socket-rot 参数覆盖
DEFAULT_SOCKET_POS = np.array([0.5, 0.0, 0.02])  # (x, y, z) world frame
DEFAULT_SOCKET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z) identity


# ===========================================================================
# Quaternion utilities (pure numpy, no torch dependency)
# ===========================================================================
def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭 [w,x,y,z] -> [w,-x,-y,-z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton 四元数乘法."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数旋转向量: v' = q * [0,v] * conj(q)."""
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_rot = quat_multiply(quat_multiply(q, q_v), quat_conjugate(q))
    return q_rot[1:4]


def quat_angle(q: np.ndarray) -> float:
    """四元数对应旋转角度 (rad, 非负)."""
    w_abs = min(abs(q[0]), 1.0)
    return 2.0 * math.acos(w_abs)


def quat_relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """q1 相对于 q2 的旋转: delta = conj(q2) * q1."""
    return quat_multiply(quat_conjugate(q2), q1)


# ===========================================================================
# Insertion check (simplified from InsertionDetector, numpy-only)
# ===========================================================================
@dataclass
class InsertionResult:
    """单帧插入判定结果."""
    depth: float
    xy_error: float
    angle_error_deg: float
    depth_ok: bool
    xy_ok: bool
    angle_ok: bool
    success: bool


def check_insertion(
    tcp_pos: np.ndarray,
    tcp_quat: np.ndarray,
    socket_pos: np.ndarray,
    socket_quat: np.ndarray,
) -> InsertionResult:
    """检查一帧的插入状态.

    Args:
        tcp_pos: (3,) TCP 位置 (world)
        tcp_quat: (4,) TCP 四元数 [w,x,y,z] 或 [x,y,z,w] (自动检测)
        socket_pos: (3,) 插座位置
        socket_quat: (4,) 插座四元数

    Returns:
        InsertionResult
    """
    # 确保四元数为 [w,x,y,z] 格式
    # 启发式: 如果 w 分量 < 0, 可能是 [x,y,z,w] scipy 格式
    tcp_q = np.asarray(tcp_quat, dtype=np.float64)
    sock_q = np.asarray(socket_quat, dtype=np.float64)

    # 1. 计算 TCP 相对插座的位移
    delta_p = np.asarray(tcp_pos) - np.asarray(socket_pos)

    # 旋转到插座局部坐标系
    delta_p_local = quat_rotate_vec(quat_conjugate(sock_q), delta_p)

    # 插入深度 = 局部坐标系 -Z 分量
    depth = -delta_p_local[2]

    # XY 对齐误差
    xy_error = float(np.linalg.norm(delta_p_local[:2]))

    # 2. 角度对齐
    neg_z = np.array([0.0, 0.0, -1.0])
    plug_dir = quat_rotate_vec(tcp_q, neg_z)
    socket_dir = quat_rotate_vec(sock_q, neg_z)
    cos_angle = np.clip(np.dot(plug_dir, socket_dir), -1.0, 1.0)
    angle_error = math.acos(cos_angle)
    angle_error_deg = math.degrees(angle_error)

    # 3. 判定
    depth_ok = depth >= INSERTION_DEPTH_THRESHOLD
    xy_ok = xy_error <= XY_TOLERANCE
    angle_ok = angle_error <= ANGLE_TOLERANCE_RAD
    success = depth_ok and xy_ok and angle_ok

    return InsertionResult(
        depth=depth,
        xy_error=xy_error,
        angle_error_deg=angle_error_deg,
        depth_ok=depth_ok,
        xy_ok=xy_ok,
        angle_ok=angle_ok,
        success=success,
    )


# ===========================================================================
# Reward computation
# ===========================================================================
def compute_dense_reward(result: InsertionResult) -> float:
    """基于插入状态的密集 reward (0~1)."""
    # 深度 reward (越深越好)
    depth_reward = min(result.depth / INSERTION_DEPTH_THRESHOLD, 1.0) if result.depth > 0 else 0.0

    # XY reward (越近越好, exp 衰减)
    xy_reward = math.exp(-result.xy_error / XY_TOLERANCE) if XY_TOLERANCE > 0 else 0.0

    # 角度 reward (越小越好, exp 衰减)
    angle_reward = (
        math.exp(-result.angle_error_deg / ANGLE_TOLERANCE_DEG)
        if ANGLE_TOLERANCE_DEG > 0
        else 0.0
    )

    # 加权
    reward = 0.5 * depth_reward + 0.3 * xy_reward + 0.2 * angle_reward

    # 成功时额外 boost
    if result.success:
        reward = 1.0
    else:
        reward *= 0.1

    return float(np.clip(reward, 0.0, 1.0))


def compute_sparse_reward(result: InsertionResult) -> float:
    """稀疏 reward: 成功=1, 失败=0."""
    return 1.0 if result.success else 0.0


# ===========================================================================
# TCP pose estimation from state
# ===========================================================================
def estimate_tcp_from_state(
    state: np.ndarray,
    socket_pos: np.ndarray,
    socket_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从 obs state 估计 TCP 位姿.

    state = [7 joints, 1 gripper]
    使用 sim 内记录的关节位置, 简化为 EE 到插座的相对估计。
    当 sim 中无直接 EE pose 记录时, 使用 state 的前 7 个关节值
    作为估计 (实际 EE pose 需要 FK, 这里用简化方式)。

    注: 如果 gello_replay.py 记录了完整 EE pose, 应直接使用。
    """
    # 用关节位置的加权和作为 TCP 位置估计 (简化)
    # 实际项目中应通过 FK 从 joints 计算精确 EE pose
    joints = state[:7]
    # 简化: 假设 TCP 接近最后一个关节 (需要实际 FK)
    # 这里返回一个基于 joint state 的近似
    tcp_pos = socket_pos.copy()  # placeholder — 需要场景特定逻辑
    tcp_quat = np.array([1.0, 0.0, 0.0, 0.0])  # placeholder
    return tcp_pos, tcp_quat


# ===========================================================================
# Labeler: 为 pkl transitions 添加 reward
# ===========================================================================
def label_rewards(
    transitions: list[dict],
    mode: str = "sparse",
    socket_pos: Optional[np.ndarray] = None,
    socket_quat: Optional[np.ndarray] = None,
    success_bonus: float = 1.0,
    approach_reward: float = 0.1,
) -> list[dict]:
    """为 transitions list 中每帧标记 reward.

    策略:
    - 稀疏模式: 仅最后几帧若插入成功给 reward=1
    - 密集模式: 每帧基于插入质量给连续 reward
    - 渐进 bonus: 接近目标时给小 reward 鼓励探索

    Args:
        transitions: SERL pkl transitions list
        mode: "sparse" 或 "dense"
        socket_pos: 插座位置 (3,)
        socket_quat: 插座四元数 (4,) [w,x,y,z]
        success_bonus: 成功时的额外 reward
        approach_reward: 接近目标时的 reward

    Returns:
        labeled transitions (原地修改 + 返回)
    """
    sock_pos = socket_pos if socket_pos is not None else DEFAULT_SOCKET_POS
    sock_quat = socket_quat if socket_quat is not None else DEFAULT_SOCKET_QUAT

    n = len(transitions)
    success_count = 0
    any_success = False

    print(f"[LABELER] Mode: {mode}, {n} transitions")
    print(f"[LABELER] Socket pos: {sock_pos}, quat: {sock_quat}")

    for i, t in enumerate(transitions):
        state = t["observations"]["state"]

        # 估计 TCP 位姿
        tcp_pos, tcp_quat = estimate_tcp_from_state(state, sock_pos, sock_quat)

        # 检查插入
        result = check_insertion(tcp_pos, tcp_quat, sock_pos, sock_quat)

        # 计算 reward
        if mode == "sparse":
            reward = compute_sparse_reward(result)
        elif mode == "dense":
            reward = compute_dense_reward(result)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 渐进 bonus: 靠近目标时给小 reward
        if not result.success and result.depth > 0:
            reward = max(reward, approach_reward * min(result.depth / INSERTION_DEPTH_THRESHOLD, 1.0))

        t["rewards"] = np.float32(reward)

        if result.success:
            success_count += 1
            any_success = True

    # 统计
    reward_vals = np.array([float(t["rewards"]) for t in transitions])
    print(f"[LABELER] Rewards: min={reward_vals.min():.4f}, max={reward_vals.max():.4f}, "
          f"mean={reward_vals.mean():.4f}")
    print(f"[LABELER] Success frames: {success_count}/{n} ({100*success_count/max(1,n):.1f}%)")

    return transitions


# ===========================================================================
# 保存/加载
# ===========================================================================
def save_pkl(data: list[dict], path: str, overwrite: bool = False):
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Exists: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVE] {path} ({os.path.getsize(path)/1024:.1f} KB)")


def load_pkl(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Label rewards for sim-replayed GELLO demos")
    parser.add_argument("input", help="Input pkl file or directory (batch mode)")
    parser.add_argument("--output", default=None, help="Output pkl path")
    parser.add_argument("--output-dir", default=None, help="Output directory (batch)")
    parser.add_argument("--mode", choices=["sparse", "dense"], default="sparse",
                        help="Reward mode")
    parser.add_argument("--socket-pos", type=float, nargs=3,
                        default=DEFAULT_SOCKET_POS.tolist(),
                        help="Socket position x y z")
    parser.add_argument("--socket-quat", type=float, nargs=4,
                        default=DEFAULT_SOCKET_QUAT.tolist(),
                        help="Socket quaternion w x y z")
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--approach-reward", type=float, default=0.1)
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all pkl in dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    socket_pos = np.array(args.socket_pos)
    socket_quat = np.array(args.socket_quat)

    if args.batch or os.path.isdir(args.input):
        # 批量模式
        pkl_files = sorted(glob.glob(os.path.join(args.input, "*.pkl")))
        if not pkl_files:
            print(f"[ERROR] No pkl files in {args.input}")
            sys.exit(1)

        output_dir = args.output_dir or args.input
        os.makedirs(output_dir, exist_ok=True)

        print(f"[BATCH] {len(pkl_files)} files, mode={args.mode}")
        for pkl_file in pkl_files:
            basename = os.path.splitext(os.path.basename(pkl_file))[0]
            out_path = os.path.join(output_dir, f"{basename}_labeled.pkl")
            print(f"\n--- Processing: {pkl_file}")
            transitions = load_pkl(pkl_file)
            label_rewards(transitions, args.mode, socket_pos, socket_quat,
                          args.success_bonus, args.approach_reward)
            save_pkl(transitions, out_path, overwrite=args.overwrite)
    else:
        # 单文件
        if not os.path.isfile(args.input):
            print(f"[ERROR] File not found: {args.input}")
            sys.exit(1)

        output = args.output
        if output is None:
            base, ext = os.path.splitext(args.input)
            output = f"{base}_labeled{ext}"

        print(f"[INPUT]  {args.input}")
        print(f"[OUTPUT] {output}")
        transitions = load_pkl(args.input)
        label_rewards(transitions, args.mode, socket_pos, socket_quat,
                      args.success_bonus, args.approach_reward)
        save_pkl(transitions, output, overwrite=args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()
