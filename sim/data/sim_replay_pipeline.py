#!/usr/bin/env python3
"""sim_replay_pipeline.py

端到端 pipeline: GELLO demo npz → sim 回放 → 记录 obs → 转 pkl → SERL 训练。

集成:
  1. load_gello_npz()          — 加载 GELLO demo
  2. fk_converter              — FK 转换 (关节 → 笛卡尔 delta)
  3. replay_in_sim()           — IsaacLab sim 中回放 + 记录 observation
  4. label_rewards()           — 自动 reward 标记
  5. save as SERL pkl          — 输出标准格式

依赖:
  - gello_replay.py            (sim 回放 + obs 记录; resolves its own
                                 fk_converter / normalize_action deps
                                 from scripts/gello_pipeline/ via its
                                 own candidate list — no need to expose
                                 that path here)
  - plug_reward_labeler.py     (reward 自动标记)
  - fk_converter.py            (FK 转换, 来自 gello_pipeline/)
  - normalize_action.py        (动作归一化, 来自 gello_pipeline/)

用法 (on fr3-desktop-ts):

    # Conda environment activation is the caller's responsibility:
    #   conda activate isaaclab
    # Run this script inside that environment (or any environment where
    # `python` resolves to one with the sim-side deps installed). Do not
    # hardcode the path to miniconda here — the host layout varies across
    # workstations and CI sandboxes.

    # 单文件模式
    python sim_replay_pipeline.py --npz /tmp/gello_demos/demo.npz

    # 批量模式
    python sim_replay_pipeline.py --input-dir /tmp/gello_demos/ --output-dir /tmp/sim_replays/

    # 纯 FK 模式 (无 GPU)
    python sim_replay_pipeline.py --npz /tmp/gello_demos/demo.npz --pure-fk

    # 自定义 reward 参数
    python sim_replay_pipeline.py --npz demo.npz \\
        --reward-mode dense --socket-pos 0.5 0.0 0.02
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np

# ===========================================================================
# 项目 import path
# ===========================================================================
# NOTE: fk_converter / normalize_action are imported transitively from
# `gello_replay`, which keeps its own (relative, not hardcoded) list of
# candidate paths. We deliberately do NOT inject a hardcoded absolute
# gello_pipeline path here — the repo root layout is the caller's
# responsibility (set PYTHONPATH or use the repo-relative search list in
# gello_replay).
_SIM_DATA = os.path.dirname(os.path.abspath(__file__))
if _SIM_DATA not in sys.path:
    sys.path.insert(0, _SIM_DATA)

# 从 gello_replay 导入核心功能
from gello_replay import (
    load_gello_npz,
    replay_in_sim,
    replay_pure_fk,
    save_transitions,
    validate_output,
    HAS_ISAACLAB,
    HAS_SIM,
    DEFAULT_POS_SCALE,
    DEFAULT_RPY_SCALE,
)

# 从 plug_reward_labeler 导入 reward 标记
from plug_reward_labeler import (
    label_rewards,
    save_pkl,
    DEFAULT_SOCKET_POS,
    DEFAULT_SOCKET_QUAT,
)


# ===========================================================================
# Pipeline: 单文件
# ===========================================================================
def run_pipeline_single(
    npz_path: str,
    output_path: Optional[str] = None,
    reward_mode: str = "sparse",
    socket_pos: Optional[np.ndarray] = None,
    socket_quat: Optional[np.ndarray] = None,
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    sub_steps: int = 10,
    max_frames: Optional[int] = None,
    pure_fk: bool = False,
    skip_reward: bool = False,
    overwrite: bool = False,
    no_verify: bool = False,
) -> str:
    """单文件 pipeline.

    Args:
        npz_path: GELLO demo npz 路径
        output_path: 输出 pkl 路径 (None=自动生成)
        reward_mode: "sparse" 或 "dense"
        socket_pos/quat: 插座位姿
        pos_scale/rpy_scale: 归一化参数
        sub_steps: sim 子步数
        max_frames: 最大帧数
        pure_fk: 是否用纯 FK 模式
        skip_reward: 跳过 reward 标记
        overwrite: 覆盖已有文件
        no_verify: 跳过验证

    Returns:
        输出 pkl 路径
    """
    start = time.time()
    basename = os.path.splitext(os.path.basename(npz_path))[0]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(npz_path), f"{basename}_sim_labeled.pkl")

    print(f"\n{'='*60}")
    print(f"  SIM REPLAY PIPELINE")
    print(f"{'='*60}")
    print(f"  Input:     {npz_path}")
    print(f"  Output:    {output_path}")
    print(f"  Mode:      {'pure FK' if pure_fk or not HAS_ISAACLAB else 'IsaacLab sim'}")
    print(f"  Reward:    {reward_mode} {'(skipped)' if skip_reward else ''}")
    print(f"  FK scale:  pos={pos_scale}, rpy={rpy_scale}")

    # --- Step 1: 加载 demo ---
    print(f"\n{'='*60}")
    print(f"  Step 1: Load GELLO demo")
    print(f"{'='*60}")
    demo = load_gello_npz(npz_path)

    # --- Step 2: Sim 回放 + 记录 obs ---
    print(f"\n{'='*60}")
    print(f"  Step 2: Replay in sim & record observations")
    print(f"{'='*60}")
    if pure_fk or not HAS_ISAACLAB:
        if not pure_fk and not HAS_ISAACLAB:
            print("[WARN] IsaacLab unavailable, falling back to pure FK")
        transitions = replay_pure_fk(
            demo,
            pos_scale=pos_scale,
            rpy_scale=rpy_scale,
            max_frames=max_frames,
        )
    else:
        transitions = replay_in_sim(
            demo,
            pos_scale=pos_scale,
            rpy_scale=rpy_scale,
            sub_steps=sub_steps,
            max_frames=max_frames,
        )

    if not transitions:
        print("[ERROR] No transitions generated!")
        return ""

    # --- Step 3: Reward 标记 ---
    if not skip_reward:
        print(f"\n{'='*60}")
        print(f"  Step 3: Label rewards ({reward_mode})")
        print(f"{'='*60}")
        sock_pos = socket_pos if socket_pos is not None else DEFAULT_SOCKET_POS
        sock_quat = socket_quat if socket_quat is not None else DEFAULT_SOCKET_QUAT
        label_rewards(transitions, reward_mode, sock_pos, sock_quat)
    else:
        print(f"\n[SKIP] Step 3: Reward labeling skipped")

    # --- Step 4: 保存 pkl ---
    print(f"\n{'='*60}")
    print(f"  Step 4: Save SERL pkl")
    print(f"{'='*60}")
    save_transitions(transitions, output_path, overwrite=overwrite)

    # --- Step 5: 验证 ---
    if not no_verify:
        print(f"\n{'='*60}")
        print(f"  Step 5: Validate")
        print(f"{'='*60}")
        validate_output(output_path)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Transitions: {len(transitions)}")
    print(f"{'='*60}")

    return output_path


# ===========================================================================
# Pipeline: 批量
# ===========================================================================
def run_pipeline_batch(
    input_dir: str,
    output_dir: str,
    reward_mode: str = "sparse",
    socket_pos: Optional[np.ndarray] = None,
    socket_quat: Optional[np.ndarray] = None,
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    sub_steps: int = 10,
    max_frames: Optional[int] = None,
    pure_fk: bool = False,
    skip_reward: bool = False,
    overwrite: bool = False,
    no_verify: bool = False,
) -> list[str]:
    """批量处理目录中所有 npz 文件。"""
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not npz_files:
        print(f"[ERROR] No npz files in {input_dir}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[BATCH] {len(npz_files)} files")
    print(f"[BATCH] Input:  {input_dir}")
    print(f"[BATCH] Output: {output_dir}")

    outputs = []
    success = 0
    for i, npz_file in enumerate(npz_files):
        basename = os.path.splitext(os.path.basename(npz_file))[0]
        out_path = os.path.join(output_dir, f"{basename}_sim_labeled.pkl")
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(npz_files)}] {basename}")
        print(f"{'#'*60}")

        try:
            result = run_pipeline_single(
                npz_path=npz_file,
                output_path=out_path,
                reward_mode=reward_mode,
                socket_pos=socket_pos,
                socket_quat=socket_quat,
                pos_scale=pos_scale,
                rpy_scale=rpy_scale,
                sub_steps=sub_steps,
                max_frames=max_frames,
                pure_fk=pure_fk,
                skip_reward=skip_reward,
                overwrite=overwrite,
                no_verify=no_verify,
            )
            if result:
                outputs.append(result)
                success += 1
        except Exception as e:
            print(f"[ERROR] {npz_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Batch complete: {success}/{len(npz_files)} succeeded")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")

    return outputs


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="GELLO → sim replay → obs → pkl → SERL training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 输入 (互斥)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--npz", type=str, help="Single GELLO demo npz path")
    input_group.add_argument("--input-dir", type=str, help="Directory of npz files (batch)")

    # 输出
    parser.add_argument("--output", type=str, default=None, help="Output pkl path (single)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (batch)")

    # 模式
    parser.add_argument("--pure-fk", action="store_true", help="Use pure FK (no sim)")
    parser.add_argument("--reward-mode", choices=["sparse", "dense"], default="sparse",
                        help="Reward mode (default: sparse)")

    # 插座位姿
    parser.add_argument("--socket-pos", type=float, nargs=3,
                        default=DEFAULT_SOCKET_POS.tolist(),
                        help="Socket position x y z")
    parser.add_argument("--socket-quat", type=float, nargs=4,
                        default=DEFAULT_SOCKET_QUAT.tolist(),
                        help="Socket quaternion w x y z")

    # 参数
    parser.add_argument("--pos-scale", type=float, default=DEFAULT_POS_SCALE,
                        help="Position normalization scale")
    parser.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE,
                        help="RPY normalization scale")
    parser.add_argument("--sub-steps", type=int, default=10,
                        help="Sim sub-steps per frame")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to replay")

    # 控制
    parser.add_argument("--skip-reward", action="store_true", help="Skip reward labeling")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--no-verify", action="store_true", help="Skip validation")

    args = parser.parse_args()

    socket_pos = np.array(args.socket_pos)
    socket_quat = np.array(args.socket_quat)

    if args.npz:
        # 单文件
        run_pipeline_single(
            npz_path=args.npz,
            output_path=args.output,
            reward_mode=args.reward_mode,
            socket_pos=socket_pos,
            socket_quat=socket_quat,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            sub_steps=args.sub_steps,
            max_frames=args.max_frames,
            pure_fk=args.pure_fk,
            skip_reward=args.skip_reward,
            overwrite=args.overwrite,
            no_verify=args.no_verify,
        )
    else:
        # 批量
        output_dir = args.output_dir or args.input_dir
        run_pipeline_batch(
            input_dir=args.input_dir,
            output_dir=output_dir,
            reward_mode=args.reward_mode,
            socket_pos=socket_pos,
            socket_quat=socket_quat,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            sub_steps=args.sub_steps,
            max_frames=args.max_frames,
            pure_fk=args.pure_fk,
            skip_reward=args.skip_reward,
            overwrite=args.overwrite,
            no_verify=args.no_verify,
        )


if __name__ == "__main__":
    main()
