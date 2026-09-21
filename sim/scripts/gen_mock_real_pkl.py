"""gen_mock_real_pkl.py — 合成 mock real pkl (per PLAN-A11).

⚠️ CRITICAL: 这是 MOCK 数据, 仅用于 A10/A12 schema smoke 测试。
不是 real data; real pkl + balanced fixture + 真 domain alignment = phase6-ready gate (用户批准)。

输出: SERL pkl with 3 image keys (side_policy + wrist_1 + side_classifier)
+ live float32 state + 7D float32 action + 80% pos_ratio (正样本 reward=1.0).

Usage:
  python -m sim.scripts.gen_mock_real_pkl --output /tmp/mock.pkl --num-frames 50 --pos-ratio 0.8
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Optional

import numpy as np

from sim.data.contract import (
    ACTION_SCALE, IMAGE_DTYPE, IMAGE_SHAPE, STATE_DIMS, TRANSITION_KEYS,
    VALID_PKL_IMAGE_KEYS,
)


def _make_image_set(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Make 3 image keys (side_policy + wrist_1 + side_classifier) with same shape."""
    # 用同一 source 数组 (alias 机制: side_classifier 与 side_policy 相同数据)
    base = rng.integers(0, 256, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE)
    return {
        "side_policy": base.copy(),
        "wrist_1": rng.integers(0, 256, size=IMAGE_SHAPE, dtype=IMAGE_DTYPE),
        "side_classifier": base.copy(),  # alias of side_policy
    }


def generate_mock_real_pkl(
    output_path: str,
    num_frames: int = 50,
    pos_ratio: float = 0.8,
    seed: int = 20260611,
) -> list[dict]:
    """Generate a mock real pkl with given num_frames + pos_ratio.

    Args:
        output_path: pkl file path to write
        num_frames: number of transitions to generate
        pos_ratio: fraction of positive samples (reward=1.0)
        seed: RNG seed for reproducibility

    Returns:
        list of transition dicts (also written to output_path)
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(num_frames * pos_ratio))
    n_neg = num_frames - n_pos
    labels = [1.0] * n_pos + [0.0] * n_neg
    rng.shuffle(labels)

    transitions: list[dict] = []
    for i in range(num_frames):
        # live flat state (gaussian)
        state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        next_state = rng.normal(size=(STATE_DIMS,)).astype(np.float32)
        # 7D action
        action = rng.uniform(-1, 1, size=(7,)).astype(np.float32)
        # image set (3 keys)
        obs_imgs = _make_image_set(rng)
        next_imgs = _make_image_set(rng)
        reward = float(labels[i])
        # 最后一个 transition: done=True
        is_last = (i == num_frames - 1)
        transition = {
            "observations": {"state": state, **obs_imgs},
            "next_observations": {"state": next_state, **next_imgs},
            "actions": action,
            "rewards": np.float32(reward),
            "masks": np.float32(0.0 if is_last else 1.0),
            "dones": is_last,
        }
        transitions.append(transition)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(transitions, f)
    return transitions


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mock real pkl (smoke only)")
    parser.add_argument("--output", required=True, help="Output pkl path")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--pos-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()
    transitions = generate_mock_real_pkl(
        output_path=args.output,
        num_frames=args.num_frames,
        pos_ratio=args.pos_ratio,
        seed=args.seed,
    )
    n_pos = sum(1 for t in transitions if float(t["rewards"]) == 1.0)
    print(f"[gen_mock_real_pkl] wrote {len(transitions)} transitions to {args.output}")
    print(f"  pos: {n_pos}, neg: {len(transitions) - n_pos}, pos_ratio: {n_pos / len(transitions):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
