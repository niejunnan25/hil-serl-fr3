#!/usr/bin/env python3
"""No-motion preflight for the SERL-aligned plug insertion training line."""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
BAD_DEMO_MARKERS = (
    "/classifier_data/",
    "classifier_data/",
    "plug_insertion_success.pkl",
    "plug_insertion_failure.pkl",
)
OLD_CKPT_MARKER = "plug_insertion_phaseC_local5080_state6_008_20260618"


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _load_pickle_len(path: str) -> tuple[int | None, str | None]:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return len(obj), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _expand_demo_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for item in paths:
        matches = sorted(glob.glob(item))
        if matches:
            out.extend(matches)
        else:
            out.append(item)
    return out


def _sample_reset_candidates(config, n: int, seed: int) -> dict:
    from experiments.plug_insertion.env import sample_reset_pose

    rng = random.Random(seed)

    class RngAdapter:
        @staticmethod
        def uniform(low, high, size=None):
            if size is None:
                return rng.uniform(float(low), float(high))
            return np.array([rng.uniform(float(low), float(high)) for _ in range(int(np.prod(size)))])

    samples = np.array([sample_reset_pose(config, RngAdapter) for _ in range(n)], dtype=float)
    return {
        "n": int(samples.shape[0]),
        "min": samples.min(axis=0).round(6).tolist(),
        "max": samples.max(axis=0).round(6).tolist(),
        "mean": samples.mean(axis=0).round(6).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", action="append", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--classifier-ckpt", default="classifier_ckpt")
    parser.add_argument("--expect-random-reset", action="store_true")
    parser.add_argument("--min-success-demos", type=int, default=10)
    parser.add_argument("--min-total-transitions", type=int, default=1000)
    parser.add_argument("--sample-reset-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    demo_paths = _expand_demo_paths(args.demo_path)

    if not demo_paths:
        _fail(errors, "no demo paths after glob expansion")

    if OLD_CKPT_MARKER in args.checkpoint_path:
        _fail(errors, f"checkpoint path is the paused old line: {args.checkpoint_path}")

    ckpt_path = Path(args.checkpoint_path)
    if ckpt_path.exists() and any(ckpt_path.glob("checkpoint_*")):
        _fail(errors, f"checkpoint path already contains checkpoint_* files: {ckpt_path}")

    classifier_ckpt = Path(args.classifier_ckpt)
    if not classifier_ckpt.is_absolute():
        classifier_ckpt = REPO_ROOT / classifier_ckpt
    if not classifier_ckpt.exists() or not any(classifier_ckpt.iterdir()):
        _fail(errors, f"classifier checkpoint is missing or empty: {classifier_ckpt}")

    demo_summaries = []
    success_demo_count = 0
    total_transitions = 0
    for path in demo_paths:
        normalized = path.replace("\\", "/")
        if any(marker in normalized for marker in BAD_DEMO_MARKERS):
            _fail(errors, f"demo path looks like classifier frame data, not full trajectory: {path}")
        if not Path(path).exists():
            _fail(errors, f"demo path does not exist: {path}")
            continue
        n, err = _load_pickle_len(path)
        if err:
            _fail(errors, f"failed to read demo pickle {path}: {err}")
            continue
        reward_sum = None
        done_last = None
        try:
            with open(path, "rb") as f:
                transitions = pickle.load(f)
            rewards = [float(t.get("rewards", 0.0)) for t in transitions]
            dones = [bool(t.get("dones", False)) for t in transitions]
            reward_sum = float(sum(rewards))
            done_last = bool(dones[-1]) if dones else None
        except Exception:
            pass
        total_transitions += int(n or 0)
        is_success = "success" in Path(path).name and (reward_sum is None or reward_sum > 0)
        success_demo_count += int(is_success)
        demo_summaries.append({
            "path": path,
            "len": n,
            "reward_sum": reward_sum,
            "done_last": done_last,
            "is_success_name": "success" in Path(path).name,
        })

    if success_demo_count < args.min_success_demos:
        _fail(errors, f"only {success_demo_count} success demos; need >= {args.min_success_demos}")
    if total_transitions < args.min_total_transitions:
        _fail(errors, f"only {total_transitions} demo transitions; need >= {args.min_total_transitions}")

    if errors:
        report = {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "demo_count": len(demo_summaries),
            "success_demo_count": success_demo_count,
            "total_transitions": total_transitions,
            "checkpoint_path": str(ckpt_path),
            "classifier_ckpt": str(classifier_ckpt),
            "demo_summaries": demo_summaries,
        }
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "demo_summaries"}, indent=2, sort_keys=True))
        return 2

    from experiments.plug_insertion.config import EnvConfig
    from experiments.plug_insertion.state_layout import FLAT_GRIPPER_INDEX, FLAT_TCP_POSE_Z_INDEX

    config = EnvConfig()
    if args.expect_random_reset and not bool(config.RANDOM_RESET):
        _fail(errors, "HILSERL_RANDOM_RESET is not enabled")
    if bool(config.RANDOM_RESET):
        if float(config.RANDOM_XY_RANGE) <= 0 or float(config.RANDOM_RZ_RANGE) <= 0:
            _fail(errors, "random reset enabled but xy/rz range is non-positive")
    else:
        warnings.append("random reset disabled; this is diagnostic, not SERL-aligned main training")

    reset_summary = _sample_reset_candidates(config, args.sample_reset_n, args.seed)
    lows = np.asarray(config.ABS_POSE_LIMIT_LOW, dtype=float)
    highs = np.asarray(config.ABS_POSE_LIMIT_HIGH, dtype=float)
    mins = np.asarray(reset_summary["min"], dtype=float)
    maxs = np.asarray(reset_summary["max"], dtype=float)
    if np.any(mins < lows) or np.any(maxs > highs):
        _fail(errors, f"sampled reset poses exceed ABS_POSE_LIMIT bounds: {reset_summary}")

    if FLAT_GRIPPER_INDEX != 0 or FLAT_TCP_POSE_Z_INDEX != 6:
        _fail(errors, f"unexpected flat state layout: gripper={FLAT_GRIPPER_INDEX}, tcp_pose_z={FLAT_TCP_POSE_Z_INDEX}")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "demo_count": len(demo_summaries),
        "success_demo_count": success_demo_count,
        "total_transitions": total_transitions,
        "checkpoint_path": str(ckpt_path),
        "classifier_ckpt": str(classifier_ckpt),
        "random_reset": bool(config.RANDOM_RESET),
        "random_xy_range": float(config.RANDOM_XY_RANGE),
        "random_rz_range": float(config.RANDOM_RZ_RANGE),
        "reset_summary": reset_summary,
        "state_layout": {
            "flat_gripper_index": FLAT_GRIPPER_INDEX,
            "flat_tcp_pose_z_index": FLAT_TCP_POSE_Z_INDEX,
        },
        "demo_summaries": demo_summaries,
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "demo_summaries"}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
