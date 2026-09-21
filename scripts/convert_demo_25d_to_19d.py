#!/usr/bin/env python3
"""Convert demo observations from raw 25D state to 19D env-compatible format.

Raw 25D = pose(7,xyz+quat) + vel(6) + force(3) + torque(3) + gripper_tiled(6)
Env 19D = gripper_pose(1) + tcp_force(3) + tcp_pose(6,xyz+euler) + tcp_torque(3) + tcp_vel(6)

Alphabetical key order after SERLObsWrapper flatten: gripper_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel
"""
import glob
import os
import pickle
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# State layout constants (from obs_state / gello_demo_recorder.py)
POSE_SLICE = slice(0, 7)    # xyz + quat(xyzw)
VEL_SLICE = slice(7, 13)    # 6D twist
FORCE_SLICE = slice(13, 16) # 3D force
TORQUE_SLICE = slice(16, 19) # 3D torque
GRIPPER_START = 19           # 6x tiled gripper scalar


def quat_to_euler_xyzw(q):
    """Convert quaternion (xyzw) to euler (xyz, extrinsic xyz)."""
    r = Rotation.from_quat(q)  # scipy uses xyzw
    return r.as_euler("xyz", degrees=False)


def convert_state_25d_to_19d(state_25):
    """Convert a single 25D state to 19D env format."""
    xyz = state_25[POSE_SLICE.start:POSE_SLICE.start+3]  # (3,)
    quat_xyzw = state_25[POSE_SLICE.start+3:POSE_SLICE.stop]  # (4,)
    euler = quat_to_euler_xyzw(quat_xyzw)  # (3,)
    tcp_pose = np.concatenate([xyz, euler])  # (6,)
    
    vel = state_25[VEL_SLICE]     # (6,)
    force = state_25[FORCE_SLICE]  # (3,)
    torque = state_25[TORQUE_SLICE] # (3,)
    gripper = state_25[GRIPPER_START:GRIPPER_START+1]  # (1,) -- take first of 6x tiled
    
    # SERLObsWrapper flatten order: alphabetical keys
    # gripper_pose(1) + tcp_force(3) + tcp_pose(6) + tcp_torque(3) + tcp_vel(6) = 19
    return np.concatenate([gripper, force, tcp_pose, torque, vel]).astype(np.float32)


def convert_transition(t):
    """Convert a single transition's observations to 19D format."""
    new_t = dict(t)  # shallow copy
    for side in ("observations", "next_observations"):
        obs = dict(t[side])
        state_25 = np.asarray(obs["state"], dtype=np.float32).ravel()
        if state_25.shape == (25,):
            obs["state"] = convert_state_25d_to_19d(state_25)
        # else: already converted or unexpected shape, skip
        new_t[side] = obs
    return new_t


def convert_pkl(in_path, out_path=None, dry_run=False):
    """Convert a demo pkl file from 25D to 19D."""
    with open(in_path, "rb") as f:
        transitions = pickle.load(f)
    
    n = len(transitions)
    t0 = transitions[0]
    old_shape = t0["observations"]["state"].shape
    if old_shape == (19,) or old_shape == (1, 19):
        print(f"  already 19D, skipping")
        return 0
    
    converted = [convert_transition(t) for t in transitions]
    new_shape = converted[0]["observations"]["state"].shape
    
    if out_path is None:
        out_path = in_path
    
    if not dry_run:
        with open(out_path, "wb") as f:
            pickle.dump(converted, f)
    
    print(f"  {n} transitions: {old_shape} -> {new_shape}  {'[DRY RUN]' if dry_run else '[SAVED]'}")
    return n


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--suffix", default="_19d")
    args = parser.parse_args()
    
    pkls = sorted(glob.glob(os.path.join(args.demo_dir, "*_success.pkl")))
    print(f"Found {len(pkls)} success demos in {args.demo_dir}")
    
    total = 0
    for p in pkls:
        print(f"  {os.path.basename(p)}")
        out = p.replace("_success.pkl", f"_success{args.suffix}.pkl") if args.suffix else p
        n = convert_pkl(p, out_path=out if args.suffix else None, dry_run=args.dry_run)
        total += n
    
    print(f"Total: {total} transitions converted")


if __name__ == "__main__":
    main()
