#!/usr/bin/env python3
"""Collect one insert-only SERL demo after the formal random reset.

Run on the FR3 desktop runtime. The script performs exactly one episode:
operator places the plug in the gripper, confirms, the env runs the same
hold-grip random reset as formal training, then GELLO recording starts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    try:
        sys.path.remove(_path)
    except ValueError:
        pass
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))


def _confirm(prompt: str) -> bool:
    try:
        input(prompt)
        return True
    except (EOFError, KeyboardInterrupt):
        print("\n[collect] cancelled before motion.", flush=True)
        return False


def _label_path(path: str, label: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}_{label}{ext}"


def _parse_label(value: str) -> str:
    value = (value or "").strip().lower()
    if value.startswith("s"):
        return "success"
    if value.startswith("d"):
        return "discard"
    return "fail"


def _count_label(index_path: Path, label: str) -> int:
    count = 0
    try:
        with index_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    count += int(json.loads(line).get("label") == label)
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return count


def _close_env(env) -> None:
    for name in ("close_cameras", "close"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # best-effort cleanup only
                print(f"[collect] cleanup {name} warning: {type(exc).__name__}: {exc}", flush=True)


def _load_gello_recorder_run():
    module_path = SCRIPT_DIR / "gello_demo_recorder.py"
    spec = importlib.util.spec_from_file_location("hilserl_gello_demo_recorder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError(f"{module_path} does not define callable run")
    return run


def _load_xbox_recorder_run():
    module_path = SCRIPT_DIR / "xbox_demo_recorder.py"
    spec = importlib.util.spec_from_file_location("hilserl_xbox_demo_recorder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError(f"{module_path} does not define callable run")
    return run


def _select_recorder_run(controller: str):
    if controller == "gello":
        return _load_gello_recorder_run()
    return _load_xbox_recorder_run()


def run_formal_random_reset(expect_random_reset: bool) -> None:
    import numpy as np

    from experiments.plug_insertion.config import EnvConfig
    from experiments.plug_insertion.env import PlugInsertionEnv

    cfg = EnvConfig()
    print(
        "[collect] reset config "
        f"random={bool(cfg.RANDOM_RESET)} xy={float(cfg.RANDOM_XY_RANGE):.4f} "
        f"rz={float(cfg.RANDOM_RZ_RANGE):.4f} reset_pose={np.round(cfg.RESET_POSE, 4).tolist()}",
        flush=True,
    )
    if expect_random_reset and not bool(cfg.RANDOM_RESET):
        raise RuntimeError("HILSERL_RANDOM_RESET is disabled; refusing SERL-aligned collection.")

    env = PlugInsertionEnv(fake_env=False, save_video=False, config=cfg)
    try:
        env.reset()
    finally:
        _close_env(env)


def write_label(out_dir: Path, result: dict, label: str, notes: str, args) -> None:
    index_path = out_dir / "index.jsonl"
    if label == "discard":
        for key in ("pkl", "npz"):
            try:
                os.remove(result[key])
            except OSError:
                pass
        print("[collect] discarded this take.", flush=True)
        return

    labeled = {}
    for key in ("pkl", "npz"):
        dst = _label_path(result[key], label)
        os.rename(result[key], dst)
        labeled[key] = dst

    record = {
        "label": label,
        "notes": notes,
        "pkl": os.path.basename(labeled["pkl"]),
        "npz": os.path.basename(labeled["npz"]),
        "n_transitions": int(result["n_transitions"]),
        "elapsed_s": float(result["elapsed_s"]),
        "validate_ok": bool(result["validate_ok"]),
        "validate_msg": result.get("validate_msg", ""),
        "abort_reason": result.get("abort_reason", ""),
        "server": args.server,
        "random_reset": True,
        "random_xy_range": float(os.environ.get("HILSERL_RANDOM_XY_RANGE", "nan")),
        "random_rz_range": float(os.environ.get("HILSERL_RANDOM_RZ_RANGE", "nan")),
        "hold_grip_action": float(args.hold_grip_action),
        "insert_only": True,
    }
    with index_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[collect] labeled={label} pkl={record['pkl']}", flush=True)
    print(f"[collect] index={index_path} success_count={_count_label(index_path, 'success')}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SERL-aligned insert-only demo collection")
    parser.add_argument("--server", default="http://172.16.0.1:5000/")
    parser.add_argument("--out-dir", default="/home/robot/serl_projects/hil-serl-fr3/demos/insert_recollect_20260624")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--max-step", type=float, default=0.003)
    parser.add_argument("--leader-scale", type=float, default=0.50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hold-grip-action", type=float, default=0.0)
    parser.add_argument("--controller", choices=("xbox", "gello"), default="xbox")
    parser.add_argument("--no-reset", action="store_true", help="skip env reset; diagnostic only")
    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"SERL insert-only demo collection: plug already held -> random reset -> {args.controller} record")
    print("=" * 72)
    print("[collect] This uses the local FR3 desktop runtime only; no zktitan path is involved.")
    if args.no_reset:
        print("[collect] Keep E-stop in hand. The next confirmation starts recorder/ZED initialization; no reset will run.")
        prompt = "确认插头已经放在夹爪中且已夹稳；工作区安全后按 Enter 开始 recording init: "
    else:
        print("[collect] Keep E-stop in hand. The next confirmation starts robot reset motion.")
        prompt = "确认插头已经放在夹爪中且已夹稳；工作区安全后按 Enter 开始 random reset: "
    if not _confirm(prompt):
        return 1

    if args.no_reset:
        print("[collect] WARNING: --no-reset skips SERL-aligned random reset.", flush=True)
    else:
        run_formal_random_reset(expect_random_reset=True)

    run_recorder = _select_recorder_run(args.controller)

    print("[collect] Reset complete. Recording starts after recorder/ZED initialization.", flush=True)
    if args.controller == "xbox":
        print("[collect] Use Xbox RB-deadman for insertion; gripper commands are disabled.", flush=True)
    else:
        print("[collect] Use GELLO for insertion; press Ctrl-C in this terminal to stop and save.", flush=True)
    result = run_recorder(
        server=args.server,
        hz=args.hz,
        duration=args.duration,
        out_dir=str(out_dir),
        max_step=args.max_step,
        leader_scale=args.leader_scale,
        gripper=False,
        fps=args.fps,
        dry_run=False,
        hold_grip_action=args.hold_grip_action,
        return_result=True,
    )
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if not result:
        print("[collect] no saved demo.", flush=True)
        return 0

    print(
        f"[collect] saved raw take transitions={result['n_transitions']} "
        f"validate={'PASS' if result['validate_ok'] else 'FAIL'}",
        flush=True,
    )
    try:
        label = _parse_label(input("结果 成功(s)/失败(f)/丢弃(d)? "))
    except (EOFError, KeyboardInterrupt):
        label = "fail"
        print("\n[collect] default label=fail", flush=True)
    try:
        notes = input("备注(可选，回车跳过): ").strip()
    except (EOFError, KeyboardInterrupt):
        notes = ""
    write_label(out_dir, result, label, notes, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
