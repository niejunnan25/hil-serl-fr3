#!/usr/bin/env python3
"""DROID 记录 → SERL pkl 格式转换器 (方案C)"""

import argparse
import glob
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fk_converter import trajectory_to_cartesian_deltas
from normalize_action import normalize_action

LEGACY_JOINT_STATE_WARNING = (
    "droid_to_serl_converter currently emits legacy 8D joint-state observations "
    "with zero image placeholders, not the plug_insertion 19D SERL observation. "
    "Pass allow_legacy_joint_state=True or --allow-legacy-joint-state only for "
    "explicit offline legacy use."
)


def load_droid_npz(npz_path):
    data = np.load(npz_path)
    return {
        "joint_poses": data["joint_poses"],
        "gripper_states": data["gripper_states"],
        "timestamps": data["timestamps"],
    }


def _normalize_transition_actions(cartesian_deltas, gripper_states, pos_scale, rpy_scale):
    cartesian_deltas = np.asarray(cartesian_deltas, dtype=np.float32)
    gripper_states = np.asarray(gripper_states, dtype=np.float32)
    if cartesian_deltas.ndim != 2 or cartesian_deltas.shape[1] != 6:
        raise ValueError(f"Expected cartesian_deltas shape (N, 6), got {cartesian_deltas.shape}")
    if len(cartesian_deltas) != len(gripper_states):
        raise ValueError(
            "cartesian_deltas and gripper_states must have the same transition count: "
            f"{len(cartesian_deltas)} vs {len(gripper_states)}"
        )

    action_scale = [float(pos_scale), float(rpy_scale), 0.0]
    actions = np.zeros((len(cartesian_deltas), 7), dtype=np.float32)
    for i, (delta, gripper) in enumerate(zip(cartesian_deltas, gripper_states)):
        actions[i] = normalize_action(delta, action_scale, float(gripper))
    return actions


def convert_droid_to_serl(
    npz_path,
    pos_scale=0.05,
    rpy_scale=0.1,
    filter_zero=True,
    allow_legacy_joint_state=False,
):
    if not allow_legacy_joint_state:
        raise RuntimeError(LEGACY_JOINT_STATE_WARNING)

    droid = load_droid_npz(npz_path)
    joint_poses = droid["joint_poses"]
    gripper_states = droid["gripper_states"]
    n = len(joint_poses)
    if n < 2:
        raise ValueError(f"Need >=2 frames, got {n}")
    if len(gripper_states) != n:
        raise ValueError(f"Expected {n} gripper states, got {len(gripper_states)}")

    # FK 转换
    cartesian_deltas = trajectory_to_cartesian_deltas(joint_poses)
    if len(cartesian_deltas) != n:
        raise ValueError(f"Expected {n} FK deltas, got {len(cartesian_deltas)}")
    # 归一化
    actions = _normalize_transition_actions(
        cartesian_deltas[1:],
        gripper_states[1:],
        pos_scale,
        rpy_scale,
    )

    transitions = []
    for i in range(len(actions)):
        action = actions[i]
        if filter_zero and np.linalg.norm(action) < 1e-6:
            continue
        state = np.concatenate([joint_poses[i], [gripper_states[i]]]).astype(np.float32)
        next_state = np.concatenate([joint_poses[i+1], [gripper_states[i+1]]]).astype(np.float32)
        done = (i == len(actions) - 1)
        transitions.append({
            "observations": {"state": state, "pixels": np.zeros((3, 128, 128), dtype=np.uint8)},
            "next_observations": {"state": next_state, "pixels": np.zeros((3, 128, 128), dtype=np.uint8)},
            "actions": action.astype(np.float32),
            "rewards": np.float32(0.0),
            "masks": np.float32(0.0 if done else 1.0),
            "dones": bool(done),
        })
    return transitions

def convert_directory(input_dir, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "*.npz"))
    total = 0
    errors = []
    for f in files:
        try:
            transitions = convert_droid_to_serl(f, **kwargs)
            out_name = os.path.splitext(os.path.basename(f))[0] + ".pkl"
            with open(os.path.join(output_dir, out_name), "wb") as fh:
                pickle.dump(transitions, fh, protocol=pickle.HIGHEST_PROTOCOL)
            total += len(transitions)
            print(f"  {f} -> {len(transitions)} transitions")
        except Exception as e:
            print(f"  {f} ERROR: {e}")
            errors.append(f"{f}: {e}")
    print(f"Total: {total} transitions from {len(files)} files")
    if errors:
        raise RuntimeError("Failed to convert one or more files:\n" + "\n".join(errors))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--pos-scale", type=float, default=0.05)
    p.add_argument("--rpy-scale", type=float, default=0.1)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument(
        "--allow-legacy-joint-state",
        action="store_true",
        help="acknowledge that output uses legacy 8D joint-state observations with zero image placeholders",
    )
    args = p.parse_args()

    if os.path.isdir(args.input):
        out = args.output_dir or args.output or os.path.join(args.input, "pkl_out")
        convert_directory(
            args.input,
            out,
            filter_zero=not args.no_filter,
            allow_legacy_joint_state=args.allow_legacy_joint_state,
        )
    else:
        out = args.output or args.input.replace(".npz", ".pkl")
        transitions = convert_droid_to_serl(
            args.input,
            filter_zero=not args.no_filter,
            allow_legacy_joint_state=args.allow_legacy_joint_state,
        )
        with open(out, "wb") as f:
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(transitions)} transitions to {out}")
