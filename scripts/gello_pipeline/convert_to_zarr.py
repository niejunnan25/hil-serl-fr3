#!/usr/bin/env python3
"""convert_to_zarr.py

将 record_gello_demos.py 生成的 npz 文件转换为 HIL-SERL zarr 格式。

转换流程:
  1. 加载 npz: joint_poses, gripper_states, timestamps
  2. FK 转换: joints_to_cartesian_delta(q[t-1], q[t]) -> 6D delta
  3. 归一化 action: normalize_action(cartesian_delta, gripper) -> 7D [-1, 1]
  4. 构建 observations: state = [joint_poses(7), gripper(1)] (legacy only)
  5. 保存为 zarr 格式

目标 zarr 结构:
  {
      "observations": {
          "images": {"wrist_1": (N, H, W, 3) uint8},  # Phase 2 暂为空
          "state":    (N, 8) float32,                   # legacy 7 joints + 1 gripper
      },
      "actions":  (N, 7) float32,   # normalized [-1, 1]
      "rewards":  (N,)   float32,   # zeros (demo data has no rewards)
      "dones":    (N,)   bool,      # last frame = True
  }

依赖:
  - fk_converter.py   (Phase 2 FK Retry)
  - normalize_action.py (Phase 2)
  - zarr (pip install zarr)

用法:
  python convert_to_zarr.py input.npz output.zarr
  python convert_to_zarr.py input.npz output.zarr --allow-legacy-8d
  python convert_to_zarr.py /tmp/gello_demos/ --output-dir /tmp/zarr_demos
  python convert_to_zarr.py /tmp/gello_demos/ --output-dir /tmp/zarr_demos --overwrite
"""

import argparse
import glob
import os
import sys

import numpy as np

# 确保 scripts 目录在 import path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fk_converter import trajectory_to_cartesian_deltas
from normalize_action import normalize_action


# ---------------------------------------------------------------------------
# 默认 action scale 参数
# ---------------------------------------------------------------------------
DEFAULT_POS_SCALE = 0.015  # xyz 归一化分母 (meters), EnvConfig.ACTION_SCALE[0]
DEFAULT_RPY_SCALE = 0.1    # rotvec 归一化分母 (radians), EnvConfig.ACTION_SCALE[3]
LEGACY_8D_WARNING = (
    "convert_to_zarr currently emits legacy 8D joint-state observations "
    "([joint_poses(7), gripper(1)]), not the live SERL19 flat state. "
    "Pass allow_legacy_8d=True or --allow-legacy-8d only for explicit offline legacy use."
)


# ---------------------------------------------------------------------------
# 核心转换逻辑
# ---------------------------------------------------------------------------
def convert_npz_to_zarr_data(
    npz_path: str,
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    allow_legacy_8d: bool = False,
) -> dict:
    """将单个 npz 文件转换为 zarr 可写入的 dict 结构。

    Args:
        npz_path: 输入 npz 文件路径
        pos_scale: xyz 归一化分母
        rpy_scale: rpy 归一化分母

    Returns:
        dict: 包含 observations, actions, rewards, dones 的字典

    Raises:
        ValueError: 输入数据格式不正确
    """
    if not allow_legacy_8d:
        raise RuntimeError(LEGACY_8D_WARNING)

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
    state = np.zeros((N, 8), dtype=np.float32)
    state[:, :7] = joint_poses.astype(np.float32)
    state[:, 7] = gripper_states.astype(np.float32)

    # images: Phase 2 暂不包含, 留空占位
    # wrist_1 = np.zeros((N, 1, 1, 3), dtype=np.uint8)

    # --- 5. rewards / dones ---
    # demo 数据无 reward, 设为 0
    rewards = np.zeros(N, dtype=np.float32)
    # 最后一帧标记为 episode 终止
    dones = np.zeros(N, dtype=bool)
    dones[-1] = True

    return {
        "observations": {
            "state": state,
        },
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        # 元数据
        "_meta": {
            "source_npz": os.path.basename(npz_path),
            "num_frames": N,
            "pos_scale": pos_scale,
            "rpy_scale": rpy_scale,
            "state_layout": "legacy_joint_gripper_8d",
            "allow_legacy_8d": True,
            "timestamps": timestamps,
        },
    }


def save_zarr(data: dict, output_path: str, overwrite: bool = False) -> str:
    """将转换后的数据保存为 zarr 格式。

    兼容 zarr v2 和 v3 API。

    Args:
        data: convert_npz_to_zarr_data 返回的字典
        output_path: 输出 zarr 目录路径
        overwrite: 是否覆盖已存在的目录

    Returns:
        str: 保存路径
    """
    import zarr
    from numcodecs import Zstd

    if os.path.exists(output_path):
        if overwrite:
            import shutil
            shutil.rmtree(output_path)
            print(f"  Removed existing zarr: {output_path}")
        else:
            raise FileExistsError(
                f"输出路径已存在: {output_path}. 使用 --overwrite 覆盖"
            )

    N = len(data["actions"])
    chunk_1d = min(1024, N)

    try:
        root = zarr.open_group(output_path, mode="w", zarr_format=2)
        use_v3_api = True
    except TypeError:
        root = zarr.open_group(output_path, mode="w")
        use_v3_api = False
    compressor = Zstd(level=3)

    def _ds(group, name, arr, chunks):
        if use_v3_api:
            return group.create_array(
                name, data=arr, chunks=chunks, compressors=compressor,
            )
        return group.create_dataset(
            name, data=arr, chunks=chunks, compressor=compressor,
        )

    # observations/state
    obs_group = root.create_group("observations")
    _ds(obs_group, "state", data["observations"]["state"].astype(np.float32),
        chunks=(chunk_1d, 8))

    # actions
    _ds(root, "actions", data["actions"].astype(np.float32),
        chunks=(chunk_1d, 7))

    # rewards
    _ds(root, "rewards", data["rewards"].astype(np.float32),
        chunks=(chunk_1d,))

    # dones
    _ds(root, "dones", data["dones"].astype(bool),
        chunks=(chunk_1d,))

    # 元数据 (作为 zarr attributes)
    root.attrs["source_npz"] = data["_meta"]["source_npz"]
    root.attrs["num_frames"] = data["_meta"]["num_frames"]
    root.attrs["pos_scale"] = data["_meta"]["pos_scale"]
    root.attrs["rpy_scale"] = data["_meta"]["rpy_scale"]
    root.attrs["state_layout"] = data["_meta"].get("state_layout", "unknown")
    root.attrs["allow_legacy_8d"] = bool(data["_meta"].get("allow_legacy_8d", False))

    return output_path


# ---------------------------------------------------------------------------
# 验证工具
# ---------------------------------------------------------------------------
def verify_zarr(zarr_path: str, allow_legacy_8d: bool = False) -> bool:
    """验证输出 zarr 文件的结构和数据范围。

    Args:
        zarr_path: zarr 目录路径

    Returns:
        bool: 验证是否通过
    """
    import zarr

    root = zarr.open(zarr_path, mode="r")
    ok = True

    print(f"\n  Verifying zarr: {zarr_path}")
    print(f"  {'=' * 50}")

    # 检查必需的 group/dataset
    required = [
        ("observations/state", True),
        ("actions", True),
        ("rewards", True),
        ("dones", True),
    ]

    for path, is_dataset in required:
        try:
            node = root[path]
            if is_dataset:
                print(f"  [OK]   {path}: shape={node.shape}, dtype={node.dtype}")
        except KeyError:
            print(f"  [FAIL] {path}: MISSING")
            ok = False

    # 检查 actions 范围
    actions = root["actions"][:]
    a_min, a_max = actions.min(), actions.max()
    in_range = a_min >= -1.0 - 1e-5 and a_max <= 1.0 + 1e-5
    range_str = "OK" if in_range else "FAIL"
    print(f"  [{range_str}] actions range: [{a_min:.4f}, {a_max:.4f}] (expect [-1, 1])")
    if not in_range:
        ok = False

    # 检查 shapes 一致性
    N = root["actions"].shape[0]
    state_N = root["observations/state"].shape[0]
    rewards_N = root["rewards"].shape[0]
    dones_N = root["dones"].shape[0]

    consistent = state_N == N and rewards_N == N and dones_N == N
    cons_str = "OK" if consistent else "FAIL"
    print(f"  [{cons_str}] N consistency: actions={N}, state={state_N}, rewards={rewards_N}, dones={dones_N}")
    if not consistent:
        ok = False

    # 检查 state 维度
    state_dim = root["observations/state"].shape[1]
    if state_dim == 19:
        print(f"  [OK] state dim: {state_dim} (live SERL19)")
    elif state_dim == 8 and allow_legacy_8d:
        print(f"  [OK] state dim: {state_dim} (explicit legacy 8D)")
    elif state_dim == 8:
        print(f"  [FAIL] state dim: {state_dim} is legacy 8D; pass allow_legacy_8d=True only for explicit legacy use")
        ok = False
    else:
        print(f"  [FAIL] state dim: {state_dim} (expect live 19D)")
        ok = False

    # 检查 actions 维度
    action_dim = root["actions"].shape[1]
    adim_str = "OK" if action_dim == 7 else "FAIL"
    print(f"  [{adim_str}] action dim: {action_dim} (expect 7)")
    if action_dim != 7:
        ok = False

    # 检查 dones 最后一帧
    dones = root["dones"][:]
    last_true = bool(dones[-1])
    lt_str = "OK" if last_true else "FAIL"
    print(f"  [{lt_str}] dones[-1] = {last_true} (expect True)")
    if not last_true:
        ok = False

    # 元数据
    print(f"\n  Metadata:")
    for key in ["source_npz", "num_frames", "pos_scale", "rpy_scale", "state_layout", "allow_legacy_8d"]:
        val = root.attrs.get(key, "MISSING")
        print(f"    {key}: {val}")

    print(f"\n  {'PASSED' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert GELLO demo npz to HIL-SERL zarr format.",
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
        help="Output zarr path (default: input_name.zarr)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch mode (places .zarr files here)",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing zarr output",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after conversion",
    )
    parser.add_argument(
        "--allow-legacy-8d",
        action="store_true",
        help="acknowledge that output uses legacy 8D joint-state observations, not live SERL19",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input
    is_batch = os.path.isdir(input_path)

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
        print()

        success_count = 0
        for npz_file in npz_files:
            basename = os.path.splitext(os.path.basename(npz_file))[0]
            zarr_path = os.path.join(output_dir, f"{basename}.zarr")

            print(f"Converting: {npz_file}")
            try:
                data = convert_npz_to_zarr_data(
                    npz_file,
                    pos_scale=args.pos_scale,
                    rpy_scale=args.rpy_scale,
                    allow_legacy_8d=args.allow_legacy_8d,
                )
                save_zarr(data, zarr_path, overwrite=args.overwrite)
                print(f"  Saved: {zarr_path}")

                if not args.no_verify:
                    verify_zarr(zarr_path, allow_legacy_8d=args.allow_legacy_8d)

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
            zarr_path = args.output
        else:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                zarr_path = os.path.join(args.output_dir, f"{basename}.zarr")
            else:
                zarr_path = os.path.join(
                    os.path.dirname(input_path), f"{basename}.zarr"
                )

        print(f"Input:  {input_path}")
        print(f"Output: {zarr_path}")
        print(f"pos_scale={args.pos_scale}, rpy_scale={args.rpy_scale}")
        print()

        data = convert_npz_to_zarr_data(
            input_path,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            allow_legacy_8d=args.allow_legacy_8d,
        )
        save_zarr(data, zarr_path, overwrite=args.overwrite)
        print(f"\n  Saved: {zarr_path}")

        if not args.no_verify:
            verify_zarr(zarr_path, allow_legacy_8d=args.allow_legacy_8d)

        print("\nDone.")


if __name__ == "__main__":
    main()
