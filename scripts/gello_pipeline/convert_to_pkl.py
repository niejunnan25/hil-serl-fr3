#!/usr/bin/env python3
"""convert_to_pkl.py

将 record_gello_demos.py 生成的 npz 文件转换为 SERL pickle (.pkl) 格式。

Phase 3 发现: SERL 使用 pickle 格式存储 demo 数据，不是 zarr。

pkl 格式: 一个 list，每个元素是一个 transition dict:
{
    "observations": {
        "state": np.ndarray(8,),           # 7 joints + 1 gripper
        "pixels": np.ndarray(3,128,128),   # 图像 (Phase 暂为空)
    },
    "next_observations": {
        "state": np.ndarray(8,),
        "pixels": np.ndarray(3,128,128),
    },
    "actions": np.ndarray(7,),   # normalized [-1,1]
    "rewards": np.float32,       # demo 数据为 0
    "masks": np.float32,         # 1 - done
    "dones": bool,
}

转换流程:
  1. 加载 npz: joint_poses, gripper_states, timestamps
  2. FK 转换: joints_to_cartesian_delta(q[t-1], q[t]) -> 6D delta
  3. 归一化 action: normalize_action(cartesian_delta, gripper) -> 7D [-1, 1]
  4. 构建 transitions list (含 next_observations)
  5. 过滤零动作 (norm(actions) > 0.0)
  6. 保存为 pkl

依赖:
  - fk_converter.py   (FK 转换)
  - normalize_action.py (动作归一化)

用法:
  python convert_to_pkl.py input.npz output.pkl
  python convert_to_pkl.py input.npz output.pkl --pos-scale 0.1 --rpy-scale 0.2
  python convert_to_pkl.py /tmp/gello_demos/ --output-dir /tmp/pkl_demos
  python convert_to_pkl.py /tmp/gello_demos/ --output-dir /tmp/pkl_demos --no-filter
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# 确保 scripts 目录在 import path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fk_converter import trajectory_to_cartesian_deltas
from normalize_action import normalize_action


# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
DEFAULT_POS_SCALE = 0.1    # xyz 归一化分母 (meters)
DEFAULT_RPY_SCALE = 0.2    # roll/pitch/yaw 归一化分母 (radians)
IMAGE_H = 128
IMAGE_W = 128
IMAGE_C = 3


# ---------------------------------------------------------------------------
# 空图像占位
# ---------------------------------------------------------------------------
def _empty_pixels() -> np.ndarray:
    """返回空白图像占位 (3, 128, 128) uint8，CHW 格式。"""
    return np.zeros((IMAGE_C, IMAGE_H, IMAGE_W), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 核心转换逻辑
# ---------------------------------------------------------------------------
def convert_npz_to_pkl_data(
    npz_path: str,
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    filter_zero_actions: bool = True,
) -> list[dict]:
    """将单个 npz 文件转换为 SERL pkl transitions list。

    Args:
        npz_path: 输入 npz 文件路径
        pos_scale: xyz 归一化分母
        rpy_scale: rpy 归一化分母
        filter_zero_actions: 是否过滤零动作

    Returns:
        list[dict]: transitions list，每个元素是一个 transition dict

    Raises:
        ValueError: 输入数据格式不正确
    """
    # --- 1. 加载 npz ---
    data = np.load(npz_path)

    required_keys = ["joint_poses", "gripper_states", "timestamps"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"npz 缺少必需字段 '{key}'. 可用字段: {list(data.keys())}")

    joint_poses = data["joint_poses"]       # (N, 7)
    gripper_states = data["gripper_states"] # (N,)
    timestamps = data["timestamps"]         # (N,)

    N = len(joint_poses)
    if N < 2:
        raise ValueError(f"npz 数据太短 ({N} 帧), 至少需要 2 帧来计算 delta")

    print(f"  Loaded npz: {N} frames")
    print(f"  joint_poses shape: {joint_poses.shape}")
    print(f"  gripper_states shape: {gripper_states.shape}")

    # --- 2. FK: 计算 Cartesian delta ---
    print("  Computing Cartesian deltas via FK...")
    cartesian_deltas = trajectory_to_cartesian_deltas(joint_poses)  # (N, 6)
    print(f"  cartesian_deltas shape: {cartesian_deltas.shape}")

    # --- 3. 归一化 action ---
    action_scale = [pos_scale, rpy_scale, 0.0]  # 第三项 unused in normalize_action
    actions = np.zeros((N, 7), dtype=np.float32)

    for i in range(N):
        actions[i] = normalize_action(
            cartesian_delta=cartesian_deltas[i],
            action_scale=action_scale,
            gripper=float(gripper_states[i]),
        )

    actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
    print(f"  actions shape: {actions.shape}")
    print(f"  action range: [{actions.min():.4f}, {actions.max():.4f}]")

    # --- 4. 构建 observations ---
    # state = [joint_poses(7), gripper(1)] = 8D
    states = np.zeros((N, 8), dtype=np.float32)
    states[:, :7] = joint_poses.astype(np.float32)
    states[:, 7] = gripper_states.astype(np.float32)

    pixels = _empty_pixels()  # Phase: 暂不含图像，共享同一占位

    # --- 5. 构建 transitions list ---
    transitions = []
    num_filtered = 0

    for i in range(N):
        # 过滤零动作: norm > 0.0 才保留
        action = actions[i]
        if filter_zero_actions and np.linalg.norm(action) <= 0.0:
            num_filtered += 1
            continue

        done = (i == N - 1)
        mask = np.float32(1.0 - float(done))

        transition = {
            "observations": {
                "state": states[i].copy(),
                "pixels": pixels.copy(),
            },
            "next_observations": {
                # next_state: 最后一帧指向自身
                "state": states[min(i + 1, N - 1)].copy(),
                "pixels": pixels.copy(),
            },
            "actions": action.copy(),
            "rewards": np.float32(0.0),
            "masks": mask,
            "dones": done,
        }
        transitions.append(transition)

    print(f"  transitions count: {len(transitions)} (filtered {num_filtered} zero-action frames)")

    return transitions


def save_pkl(transitions: list[dict], output_path: str, overwrite: bool = False) -> str:
    """将 transitions list 保存为 pickle 文件。

    Args:
        transitions: transition dicts list
        output_path: 输出 pkl 文件路径
        overwrite: 是否覆盖已存在的文件

    Returns:
        str: 保存路径
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"输出文件已存在: {output_path}. 使用 --overwrite 覆盖"
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(output_path)
    print(f"  Saved: {output_path} ({file_size / 1024:.1f} KB)")

    return output_path


# ---------------------------------------------------------------------------
# 验证工具
# ---------------------------------------------------------------------------
def verify_pkl(pkl_path: str) -> bool:
    """验证输出 pkl 文件的结构和数据范围。

    Args:
        pkl_path: pkl 文件路径

    Returns:
        bool: 验证是否通过
    """
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    ok = True

    print(f"\n  Verifying pkl: {pkl_path}")
    print(f"  {'=' * 50}")

    # 基本结构
    N = len(transitions)
    print(f"  [INFO] transitions count: {N}")

    if N == 0:
        print(f"  [FAIL] Empty transitions list")
        return False

    # 检查第一个 transition 的结构
    t0 = transitions[0]
    required_top_keys = ["observations", "next_observations", "actions", "rewards", "masks", "dones"]
    for key in required_top_keys:
        if key not in t0:
            print(f"  [FAIL] Missing key: '{key}'")
            ok = False
        else:
            print(f"  [OK]   Key present: '{key}'")

    # observations 子结构
    for obs_key in ["observations", "next_observations"]:
        obs = t0[obs_key]
        for sub_key in ["state", "pixels"]:
            if sub_key not in obs:
                print(f"  [FAIL] Missing {obs_key}/{sub_key}")
                ok = False
            else:
                print(f"  [OK]   {obs_key}/{sub_key}: shape={obs[sub_key].shape}, dtype={obs[sub_key].dtype}")

    # actions shape 和 range
    all_actions = np.array([t["actions"] for t in transitions])
    print(f"  [INFO] actions shape: {all_actions.shape}")
    a_min, a_max = float(all_actions.min()), float(all_actions.max())
    in_range = a_min >= -1.0 - 1e-5 and a_max <= 1.0 + 1e-5
    range_str = "OK" if in_range else "FAIL"
    print(f"  [{range_str}] actions range: [{a_min:.4f}, {a_max:.4f}] (expect [-1, 1])")
    if not in_range:
        ok = False

    # state 维度
    state_dim = t0["observations"]["state"].shape[0]
    dim_str = "OK" if state_dim == 8 else "FAIL"
    print(f"  [{dim_str}] state dim: {state_dim} (expect 8)")
    if state_dim != 8:
        ok = False

    # action 维度
    action_dim = t0["actions"].shape[0]
    adim_str = "OK" if action_dim == 7 else "FAIL"
    print(f"  [{adim_str}] action dim: {action_dim} (expect 7)")
    if action_dim != 7:
        ok = False

    # masks / dones 一致性
    all_masks = np.array([t["masks"] for t in transitions])
    all_dones = np.array([t["dones"] for t in transitions])
    last_done = bool(all_dones[-1])
    last_mask = float(all_masks[-1])
    ld_str = "OK" if last_done else "FAIL"
    lm_str = "OK" if abs(last_mask) < 1e-6 else "FAIL"
    print(f"  [{ld_str}] dones[-1] = {last_done} (expect True)")
    print(f"  [{lm_str}] masks[-1] = {last_mask:.4f} (expect 0.0)")
    if not last_done:
        ok = False
    if abs(last_mask) >= 1e-6:
        ok = False

    # 检查所有 masks = 1 - dones
    masks_ok = np.allclose(all_masks, 1.0 - all_dones.astype(np.float32))
    m_str = "OK" if masks_ok else "FAIL"
    print(f"  [{m_str}] masks == 1 - dones for all transitions")
    if not masks_ok:
        ok = False

    # 零动作检查
    zero_actions = int(np.sum(np.linalg.norm(all_actions, axis=1) <= 0.0))
    z_str = "OK" if zero_actions == 0 else "WARN"
    print(f"  [{z_str}] zero-action count: {zero_actions}")

    print(f"\n  {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GELLO demo npz to SERL pickle (.pkl) format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=str,
        help="Input npz file or directory containing npz files",
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        default=None,
        help="Output pkl path (default: input_name.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (places .pkl files here)",
    )
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=DEFAULT_POS_SCALE,
        help=f"Position normalization scale (default: {DEFAULT_POS_SCALE})",
    )
    parser.add_argument(
        "--rpy-scale",
        type=float,
        default=DEFAULT_RPY_SCALE,
        help=f"RPY normalization scale (default: {DEFAULT_RPY_SCALE})",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Keep zero-action frames (default: filter them out)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pkl output",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after conversion",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input
    is_batch = os.path.isdir(input_path)
    filter_zero = not args.no_filter

    if is_batch:
        # 批量模式: 处理目录中所有 npz 文件
        npz_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
        if not npz_files:
            print(f"[ERROR] 目录中没有 npz 文件: {input_path}")
            sys.exit(1)

        output_dir = args.output_dir or input_path
        os.makedirs(output_dir, exist_ok=True)

        print(f"Batch mode: {len(npz_files)} npz files")
        print(f"Output dir: {output_dir}")
        print(f"pos_scale={args.pos_scale}, rpy_scale={args.rpy_scale}")
        print(f"filter_zero_actions={filter_zero}")
        print()

        success_count = 0
        for npz_file in npz_files:
            basename = os.path.splitext(os.path.basename(npz_file))[0]
            pkl_path = os.path.join(output_dir, f"{basename}.pkl")

            print(f"Converting: {npz_file}")
            try:
                transitions = convert_npz_to_pkl_data(
                    npz_file,
                    pos_scale=args.pos_scale,
                    rpy_scale=args.rpy_scale,
                    filter_zero_actions=filter_zero,
                )
                save_pkl(transitions, pkl_path, overwrite=args.overwrite)

                if not args.no_verify:
                    verify_pkl(pkl_path)

                success_count += 1
            except Exception as e:
                print(f"  [ERROR] {e}")

            print()

        print(f"DONE: {success_count}/{len(npz_files)} files converted")

    else:
        # 单文件模式
        if not os.path.isfile(input_path):
            print(f"[ERROR] 文件不存在: {input_path}")
            sys.exit(1)

        if args.output:
            pkl_path = args.output
        else:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                pkl_path = os.path.join(args.output_dir, f"{basename}.pkl")
            else:
                pkl_path = os.path.join(
                    os.path.dirname(input_path), f"{basename}.pkl"
                )

        print(f"Input:  {input_path}")
        print(f"Output: {pkl_path}")
        print(f"pos_scale={args.pos_scale}, rpy_scale={args.rpy_scale}")
        print(f"filter_zero_actions={filter_zero}")
        print()

        transitions = convert_npz_to_pkl_data(
            input_path,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            filter_zero_actions=filter_zero,
        )
        save_pkl(transitions, pkl_path, overwrite=args.overwrite)

        if not args.no_verify:
            verify_pkl(pkl_path)

        print("\nDone.")


if __name__ == "__main__":
    main()
