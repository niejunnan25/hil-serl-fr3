"""test_domain_alignment.py — sim vs mock real 域对齐 smoke (per PLAN-A11).

⚠️ CRITICAL: 本测试是 SCHEMA SMOKE, 不是 READINESS。
- 跑通不 crash = PASS
- 打印报告含 image mean/std + state range overlap
- 真实 domain alignment (KL divergence / 图像分布同分布) = phase6-ready gate (用户批准)

本 plan 不依赖 real pkl, 也不做 balanced fixture / confusion matrix。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from typing import Any

import numpy as np

from sim.data.contract import VALID_PKL_IMAGE_KEYS


def _load_pkl(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _collect_image_stats(pkl_path: str) -> dict[str, np.ndarray]:
    """收集 pkl 中所有 image 帧的 per-key mean/std 统计."""
    transitions = _load_pkl(pkl_path)
    stats: dict[str, list[np.ndarray]] = {k: [] for k in VALID_PKL_IMAGE_KEYS}
    for t in transitions:
        for k in VALID_PKL_IMAGE_KEYS:
            img = t["observations"][k]
            stats[k].append(float(img.mean()))
            stats[k].append(float(img.std()))
    return {k: np.array(v) for k, v in stats.items()}


def _collect_state_range(pkl_path: str) -> tuple[np.ndarray, np.ndarray]:
    """收集 pkl 中 state 的 (min, max) per-dim."""
    transitions = _load_pkl(pkl_path)
    states = np.stack([t["observations"]["state"] for t in transitions])  # (N, 25)
    return states.min(axis=0), states.max(axis=0)


def compute_alignment_report(sim_pkl_path: str, real_pkl_path: str) -> dict[str, Any]:
    """Compute image + state range comparison between sim and real pkls.

    Returns:
        dict with keys:
          - image_mean_sim, image_mean_real: float (overall mean across 3 image keys)
          - image_std_sim, image_std_real: float (overall std across 3 image keys)
          - state_range_overlap: float ∈ [0, 1] (intersection / union of per-dim ranges)
    """
    sim_image_stats = _collect_image_stats(sim_pkl_path)
    real_image_stats = _collect_image_stats(real_pkl_path)
    sim_state_min, sim_state_max = _collect_state_range(sim_pkl_path)
    real_state_min, real_state_max = _collect_state_range(real_pkl_path)
    # Image overall mean/std
    sim_image_mean = float(np.mean([sim_image_stats[k].mean() for k in VALID_PKL_IMAGE_KEYS]))
    real_image_mean = float(np.mean([real_image_stats[k].mean() for k in VALID_PKL_IMAGE_KEYS]))
    sim_image_std = float(np.mean([sim_image_stats[k].std() for k in VALID_PKL_IMAGE_KEYS]))
    real_image_std = float(np.mean([real_image_stats[k].std() for k in VALID_PKL_IMAGE_KEYS]))
    # State range overlap (per-dim, then average)
    intersection = np.maximum(0, np.minimum(sim_state_max, real_state_max) - np.maximum(sim_state_min, real_state_min))
    union = np.maximum(sim_state_max, real_state_max) - np.minimum(sim_state_min, real_state_min)
    # 避免除零
    safe_union = np.where(union > 0, union, 1.0)
    per_dim_overlap = intersection / safe_union
    state_range_overlap = float(per_dim_overlap.mean())
    return {
        "image_mean_sim": sim_image_mean,
        "image_mean_real": real_image_mean,
        "image_std_sim": sim_image_std,
        "image_std_real": real_image_std,
        "state_range_overlap": state_range_overlap,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print alignment report to stdout."""
    print("\n=== test_domain_alignment report ===")
    print(f"  image_mean_sim:       {report['image_mean_sim']:.4f}")
    print(f"  image_mean_real:      {report['image_mean_real']:.4f}")
    print(f"  image_std_sim:        {report['image_std_sim']:.4f}")
    print(f"  image_std_real:       {report['image_std_real']:.4f}")
    print(f"  state_range_overlap:  {report['state_range_overlap']:.4f}")
    print(f"  --- smoke pass (script ran end-to-end) ---")


def main() -> int:
    """CLI entry point. Returns 0 (smoke pass) if script runs end-to-end."""
    parser = argparse.ArgumentParser(description="Sim vs real domain alignment smoke test")
    parser.add_argument("--sim", required=True, help="Path to sim pkl")
    parser.add_argument("--real", required=True, help="Path to mock real pkl")
    args = parser.parse_args()
    try:
        report = compute_alignment_report(args.sim, args.real)
        print_report(report)
        return 0
    except Exception as e:
        print(f"\n[FAIL] test_domain_alignment raised: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
