#!/usr/bin/env python3
"""load_pkl_to_serl.py

将 convert_to_pkl.py 输出的 demo PKL 转换为 SERL 训练管线兼容格式。

问题:
  convert_to_pkl.py 输出的 PKL 使用:
    - state: [joint_poses(7), gripper(1)] = 8D
    - pixels: zeros(3, 128, 128)

  SERL 训练管线 (config.py) 期望:
    - proprio: [tcp_pose(7), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1)] = 25D
    - images: side_policy(3, 128, 128), wrist_1(3, 128, 128)

  本脚本将 demo PKL 转换为 SERL 可加载的格式，使用 FK 从关节角度计算
  Cartesian 位姿，并适配观测空间。

用法:
  python scripts/load_pkl_to_serl.py demo.pkl -o demo_serl.pkl
  python scripts/load_pkl_to_serl.py data/demos/ -o data/demos_serl/ --image-dir data/images/

注意:
  - 转换后的 PKL 可直接用于 SERL MemoryEfficientReplayBufferDataStore
  - 如果没有图像数据，pixels 将保持为零 (需要后续补充)
  - tcp_vel, tcp_force, tcp_torque 用零填充 (demo 数据中不包含)
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# 确保 scripts 目录在 import path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fk_converter import forward_kinematics


# ---------------------------------------------------------------------------
# SERL 观测空间维度 (来自 config.py TrainConfig)
# ---------------------------------------------------------------------------
SERL_PROPRIO_KEYS = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
SERL_PROPRIO_DIMS = {
    "tcp_pose": 7,      # xyz + quaternion
    "tcp_vel": 6,       # linear(3) + angular(3)
    "tcp_force": 3,     # fx, fy, fz
    "tcp_torque": 3,    # tx, ty, tz
    "gripper_pose": 1,  # gripper position
}
SERL_STATE_DIM = sum(SERL_PROPRIO_DIMS.values())  # 25D

SERL_IMAGE_KEYS = ["side_policy", "wrist_1"]
SERL_IMAGE_SHAPE = (3, 128, 128)


def joints_to_tcp_pose(joint_poses: np.ndarray) -> np.ndarray:
    """将关节角度转换为 TCP 位姿 (xyz + quaternion)。

    使用 FK 计算绝对 TCP pose。不要通过单帧 delta 累积；单帧输入的
    trajectory_to_cartesian_deltas() 首帧 delta 必然为 0，会把 pose 压成原点。

    Args:
        joint_poses: (N, 7) 关节角度

    Returns:
        (N, 7) TCP 位姿 [x, y, z, qw, qx, qy, qz]
    """
    q_arr = np.asarray(joint_poses, dtype=np.float64)
    if q_arr.ndim != 2 or q_arr.shape[1] != 7:
        raise ValueError(f"joint_poses must have shape (N, 7), got {q_arr.shape}")
    return np.stack([forward_kinematics(q) for q in q_arr], axis=0).astype(np.float32)


def convert_transition_to_serl(
    transition: dict,
    image_dir: str | None = None,
) -> dict | None:
    """将单条 demo transition 转换为 SERL 兼容格式。

    Args:
        transition: 原始 demo transition dict
        image_dir: 可选图像目录

    Returns:
        SERL 兼容的 transition dict，或 None (如果转换失败)
    """
    obs = transition.get("observations", {})
    next_obs = transition.get("next_observations", {})

    # 提取关节数据
    state = obs.get("state")
    if state is None:
        return None

    # state 可能是 8D [joints(7), gripper(1)] 或 7D [joints(7)]
    if state.shape[0] >= 8:
        joint_poses = state[:7]
        gripper = state[7]
    elif state.shape[0] == 7:
        joint_poses = state
        gripper = 0.0
    else:
        return None

    # 构建 SERL proprio (25D)
    tcp_pose = joints_to_tcp_pose(joint_poses.reshape(1, -1))[0]  # (7,)
    tcp_vel = np.zeros(6, dtype=np.float32)      # demo 中无速度数据
    tcp_force = np.zeros(3, dtype=np.float32)     # demo 中无力数据
    tcp_torque = np.zeros(3, dtype=np.float32)    # demo 中无力矩数据
    gripper_pose = np.array([gripper], dtype=np.float32)

    serl_state = np.concatenate([
        tcp_pose, tcp_vel, tcp_force, tcp_torque, gripper_pose
    ]).astype(np.float32)

    # 构建 SERL images
    serl_images = {}
    for img_key in SERL_IMAGE_KEYS:
        serl_images[img_key] = np.zeros(SERL_IMAGE_SHAPE, dtype=np.uint8)

    # next_obs 同样处理
    next_state_raw = next_obs.get("state", state)
    if next_state_raw.shape[0] >= 8:
        next_joint = next_state_raw[:7]
        next_gripper = next_state_raw[7]
    else:
        next_joint = next_state_raw
        next_gripper = gripper

    next_tcp_pose = joints_to_tcp_pose(next_joint.reshape(1, -1))[0]
    next_serl_state = np.concatenate([
        next_tcp_pose, tcp_vel, tcp_force, tcp_torque,
        np.array([next_gripper], dtype=np.float32)
    ]).astype(np.float32)

    next_serl_images = {}
    for img_key in SERL_IMAGE_KEYS:
        next_serl_images[img_key] = np.zeros(SERL_IMAGE_SHAPE, dtype=np.uint8)

    # 构建 SERL transition
    serl_transition = {
        "observations": {
            "state": serl_state,
            **serl_images,
        },
        "next_observations": {
            "state": next_serl_state,
            **next_serl_images,
        },
        "actions": transition["actions"].astype(np.float32),
        "rewards": np.float32(transition.get("rewards", 0.0)),
        "masks": np.float32(transition.get("masks", 1.0)),
        "dones": bool(transition.get("dones", False)),
    }

    return serl_transition


def convert_pkl_to_serl(
    pkl_path: str,
    output_path: str | None = None,
    image_dir: str | None = None,
    overwrite: bool = False,
) -> str:
    """将 demo PKL 转换为 SERL 兼容格式。

    Args:
        pkl_path: 输入 PKL 文件路径
        output_path: 输出 PKL 路径 (默认: input_serl.pkl)
        image_dir: 可选图像目录
        overwrite: 是否覆盖已有文件

    Returns:
        输出文件路径
    """
    if output_path is None:
        base = os.path.splitext(pkl_path)[0]
        output_path = f"{base}_serl.pkl"

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite")

    # Load input
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    if not isinstance(transitions, list):
        raise ValueError(f"Expected list of transitions, got {type(transitions)}")

    print(f"  Input: {pkl_path} ({len(transitions)} transitions)")

    # Convert
    serl_transitions = []
    skipped = 0

    for t in transitions:
        serl_t = convert_transition_to_serl(t, image_dir=image_dir)
        if serl_t is not None:
            serl_transitions.append(serl_t)
        else:
            skipped += 1

    print(f"  Converted: {len(serl_transitions)} (skipped {skipped})")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(serl_transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(output_path)
    print(f"  Output: {output_path} ({file_size / 1024:.1f} KB)")

    return output_path


def verify_serl_pkl(pkl_path: str) -> bool:
    """验证转换后的 PKL 是否符合 SERL 观测空间。"""
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    if not transitions:
        print("  [FAIL] Empty transitions")
        return False

    t0 = transitions[0]
    ok = True

    print(f"\n  Verifying SERL compatibility: {pkl_path}")
    print(f"  {'=' * 50}")

    # Check state dim
    state_dim = t0["observations"]["state"].shape[0]
    dim_ok = state_dim == SERL_STATE_DIM
    print(f"  [{'OK' if dim_ok else 'FAIL'}] state dim: {state_dim} (expect {SERL_STATE_DIM})")
    if not dim_ok:
        ok = False

    # Check image keys
    for img_key in SERL_IMAGE_KEYS:
        if img_key in t0["observations"]:
            shape = t0["observations"][img_key].shape
            print(f"  [OK] obs['{img_key}']: shape={shape}")
        else:
            print(f"  [FAIL] Missing obs['{img_key}']")
            ok = False

    # Check action dim
    act_dim = t0["actions"].shape[0]
    print(f"  [{'OK' if act_dim == 7 else 'FAIL'}] action dim: {act_dim} (expect 7)")

    print(f"\n  {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert demo PKL to SERL-compatible observation space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=str, help="Input PKL file or directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (batch mode)")
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory for ZED frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")

    args = parser.parse_args()
    input_path = args.input

    if os.path.isdir(input_path):
        # Batch mode
        pkl_files = sorted(glob.glob(os.path.join(input_path, "*.pkl")))
        # Filter out already-converted _serl files
        pkl_files = [f for f in pkl_files if "_serl.pkl" not in f]
        if not pkl_files:
            print(f"[ERROR] No PKL files found in: {input_path}")
            sys.exit(1)

        output_dir = args.output_dir or input_path
        os.makedirs(output_dir, exist_ok=True)

        print(f"Batch mode: {len(pkl_files)} PKL files")
        print(f"Output dir: {output_dir}")
        print()

        success = 0
        for pkl_file in pkl_files:
            basename = os.path.splitext(os.path.basename(pkl_file))[0]
            out_path = os.path.join(output_dir, f"{basename}_serl.pkl")

            print(f"Converting: {pkl_file}")
            try:
                convert_pkl_to_serl(
                    pkl_file, out_path,
                    image_dir=args.image_dir,
                    overwrite=args.overwrite,
                )
                if not args.no_verify:
                    verify_serl_pkl(out_path)
                success += 1
            except Exception as e:
                print(f"  [ERROR] {e}")
            print()

        print(f"DONE: {success}/{len(pkl_files)} files converted")
    else:
        # Single file
        if not os.path.isfile(input_path):
            print(f"[ERROR] File not found: {input_path}")
            sys.exit(1)

        output_path = args.output
        if output_path is None:
            base = os.path.splitext(input_path)[0]
            output_path = f"{base}_serl.pkl"

        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print()

        convert_pkl_to_serl(
            input_path, output_path,
            image_dir=args.image_dir,
            overwrite=args.overwrite,
        )

        if not args.no_verify:
            verify_serl_pkl(output_path)

        print("\nDone.")


if __name__ == "__main__":
    main()
