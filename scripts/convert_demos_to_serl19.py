#!/usr/bin/env python
"""convert_demos_to_serl19.py

Standalone numpy/scipy converter: turn a recorded GELLO demo
(list[transition]) into the SERL/RLPD format the live plug_insertion env
+ replay buffer expect.

Pipeline replicated (no gym, no robot, pure numpy/scipy):
    PlugInsertionEnv
      -> RelativeFrame(include_relative_pose=True)   # tcp_pose -> reset-relative; tcp_vel -> body frame
      -> Quat2EulerWrapper                            # tcp_pose quat -> euler 'xyz'
      -> SERLObsWrapper(proprio_keys=[tcp_pose,tcp_vel,tcp_force,tcp_torque,gripper_pose])
      -> ChunkingWrapper(obs_horizon=1)               # prepend leading axis
      -> GripperPenaltyWrapper                        # learner reads infos.grasp_penalty

DEMO source obs state (25,) f32 layout (empirically verified):
    [0:3]   pos
    [3:7]   quat SCALAR-FIRST (w, x, y, z)   <-- must reorder to (x,y,z,w) for scipy
    [7:13]  tcp_vel (6)
    [13:16] tcp_force (3)
    [16:19] tcp_torque (3)
    [19:25] gripper_pose tiled 6x
images (3,128,128) u8 CHW for keys side_policy / wrist_1 / side_classifier

TARGET (live observation_space, RunSetup-locked):
    state           (1, 19) f32 = gripper(1)+tcp_force(3)+tcp_pose_euler(6)+tcp_torque(3)+tcp_vel(6)
    side_policy     (1, 128, 128, 3) u8 (HWC, chunk dim=1)
    wrist_1         (1, 128, 128, 3) u8
    side_classifier (1, 128, 128, 3) u8
action (7,) f32 reprojected into the per-transition body frame (matches RelativeFrame.transform_action_inv)
"""
import argparse
import glob
import os
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

# image observation keys (order: policy cams then classifier)
IMG_KEYS = ["side_policy", "wrist_1", "side_classifier"]

# demo state slice layout
SL_POS = slice(0, 3)
SL_QUAT = slice(3, 7)      # scalar-first w,x,y,z
SL_VEL = slice(7, 13)
SL_FORCE = slice(13, 16)
SL_TORQUE = slice(16, 19)
SL_GRIP = slice(19, 25)    # tiled 6x


# --------------------------------------------------------------------------
# primitive transforms (each independently tested)
# --------------------------------------------------------------------------
def quat_wxyz_to_xyzw(q_wxyz):
    """Reorder scalar-first (w,x,y,z) -> scalar-last (x,y,z,w) for scipy."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_xyzw_to_euler(q_xyzw):
    """Quat (scalar-last) -> euler 'xyz' (matches Quat2EulerWrapper)."""
    return R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_euler("xyz")


def construct_homogeneous_matrix(pos, q_xyzw):
    """4x4 homogeneous transform; mirrors franka_env transformations.construct_homogeneous_matrix."""
    T = np.zeros((4, 4))
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = pos
    T[3, 3] = 1.0
    return T


def construct_transform_matrix(q_xyzw):
    """6x6 block-diag rotation (no translation/skew); mirrors construct_transform_matrix.
    Used to rotate tcp_vel (linear+angular) into the body frame."""
    rot = R.from_quat(q_xyzw).as_matrix()
    M = np.zeros((6, 6))
    M[:3, :3] = rot
    M[3:, 3:] = rot
    return M


def detile_gripper(state25):
    """gripper_pose is tiled 6x in the demo; de-tile = take index [19]."""
    return float(np.asarray(state25)[19])


def chw_to_chunked_hwc(img_chw):
    """(3,128,128) CHW u8 -> (1,128,128,3) HWC u8 (prepend chunk axis)."""
    a = np.asarray(img_chw)
    hwc = np.transpose(a, (1, 2, 0))            # C,H,W -> H,W,C
    return hwc[None, ...].astype(np.uint8)       # prepend chunk axis


def relative_tcp_pose(pos, q_xyzw, T_r_o_inv):
    """Reproduce RelativeFrame relative pose: T_b_r = T_r_o_inv @ T_b_o.
    Returns (rel_pos(3), rel_quat_xyzw(4))."""
    T_b_o = construct_homogeneous_matrix(pos, q_xyzw)
    T_b_r = T_r_o_inv @ T_b_o
    rel_pos = T_b_r[:3, 3]
    rel_quat = R.from_matrix(T_b_r[:3, :3]).as_quat()  # xyzw
    return rel_pos, rel_quat


def reproject_action(a_base, q_xyzw):
    """Action reprojection into the body frame.

    Mirrors RelativeFrame.transform_action_inv applied with the per-transition
    pose: a_new[:6] = inv(transform_matrix) @ a_base[:6] = blockdiag(R.T,R.T)@a_base[:6].
    Because the matrix is block-diagonal orthonormal, this equals
    R_curr.T @ a_base[:3] and R_curr.T @ a_base[3:6] (norm-preserving). Gripper unchanged.
    """
    a_base = np.asarray(a_base, dtype=np.float64)
    R_curr = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_matrix()
    a_new = np.array(a_base, dtype=np.float64)
    a_new[:3] = R_curr.T @ a_base[:3]
    a_new[3:6] = R_curr.T @ a_base[3:6]
    a_new[6] = a_base[6]
    return a_new


# --------------------------------------------------------------------------
# obs conversion
# --------------------------------------------------------------------------
def _reset_inv_from_obs(obs0):
    """Build T_r_o_inv (reset-relative tcp_pose frame) from the FIRST transition obs.

    Only tcp_pose is reset-relative. tcp_vel is NOT a reset-frame quantity: the live
    RelativeFrame sets transform_matrix from EACH step's own current pose
    (relative_env.py L51/L62) before transform_observation rotates tcp_vel by its
    inverse (L77-78). So tcp_vel is rotated by each transition's own same-step pose,
    handled per-obs in convert_obs — not here.
    """
    s0 = np.asarray(obs0["state"], dtype=np.float64)
    pos0 = s0[SL_POS]
    q0_xyzw = quat_wxyz_to_xyzw(s0[SL_QUAT])
    T_r_o = construct_homogeneous_matrix(pos0, q0_xyzw)
    T_r_o_inv = np.linalg.inv(T_r_o)
    return T_r_o_inv


def convert_obs(obs, T_r_o_inv):
    """Convert a single demo obs dict -> SERL chunked obs dict."""
    s = np.asarray(obs["state"], dtype=np.float64)

    pos = s[SL_POS]
    q_xyzw = quat_wxyz_to_xyzw(s[SL_QUAT])

    # (a) relative tcp_pose (RelativeFrame), (b) quat->euler (Quat2EulerWrapper)
    rel_pos, rel_quat = relative_tcp_pose(pos, q_xyzw, T_r_o_inv)
    tcp_pose_euler = np.concatenate([rel_pos, quat_xyzw_to_euler(rel_quat)])  # (6,)

    # tcp_vel rotated into the body frame of THIS obs's OWN same-step pose.
    # Live RelativeFrame: transform_matrix = construct_transform_matrix(current obs
    # tcp_pose) each step, then tcp_vel = inv(transform_matrix) @ tcp_vel. No reset
    # frame, no one-step lag. q_xyzw above IS this obs's own quat (incl. next_obs).
    tcp_vel = np.linalg.inv(construct_transform_matrix(q_xyzw)) @ s[SL_VEL]   # (6,)

    # force/torque untouched by RelativeFrame
    tcp_force = s[SL_FORCE]                                                   # (3,)
    tcp_torque = s[SL_TORQUE]                                                 # (3,)

    gripper = np.array([detile_gripper(s)])                                   # (1,)

    # (c) SERLObsWrapper flattens gymnasium Dict keys alphabetically:
    # gripper_pose, tcp_force, tcp_pose, tcp_torque, tcp_vel.
    state_vec = np.concatenate([
        gripper, tcp_force, tcp_pose_euler, tcp_torque, tcp_vel
    ]).astype(np.float32)                                                     # (19,)
    assert state_vec.shape == (19,), state_vec.shape

    out = {"state": state_vec[None, ...]}                                      # (1,19)
    # (d) images CHW->HWC, (e) chunk axis
    for k in IMG_KEYS:
        out[k] = chw_to_chunked_hwc(obs[k])                                    # (1,128,128,3)
    return out


# --------------------------------------------------------------------------
# top-level demo conversion
# --------------------------------------------------------------------------
def insertion_start_index(demo, margin=0.05):
    """Insert-only trim point: start of the FINAL contiguous run where the world
    TCP x stays within `margin` of the seated x — i.e. the EE has arrived above the
    socket and the final (near-vertical) insertion begins. Grasp + transport before
    this index is dropped for insert-only RL. seated x = median of the last 5 frames."""
    xs = np.array([np.asarray(tr["observations"]["state"], dtype=np.float64)[SL_POS][0]
                   for tr in demo])
    seated_x = float(np.median(xs[-5:]))
    thr = seated_x - margin
    i = len(xs) - 1
    while i > 0 and xs[i] >= thr:
        i -= 1
    return i + 1


def convert_demo(demo, is_success, start=0):
    """Convert a full demo (list[transition]) -> list[serl transition].

    start: trim to demo[start:] (insert-only); the RelativeFrame reset is the pose at
           demo[start] (the above-socket insertion-start), so the converted relative
           frame matches the live RESET_POSE (not the far grasp-start).
    reward/done:
      success demo: terminal transition reward=1.0 done=True; others reward=0.
      fail demo:    all reward=0.
    infos: every transition gets {'grasp_penalty': 0.0}.
    """
    assert len(demo) > 0, "empty demo"
    sub = demo[start:] if start else demo
    assert len(sub) > 0, f"empty demo after trim (start={start}, len={len(demo)})"

    # reset frame (tcp_pose only) from the FIRST KEPT transition's observation pose
    T_r_o_inv = _reset_inv_from_obs(sub[0]["observations"])

    n = len(sub)
    out = []
    for i, tr in enumerate(sub):
        is_terminal = (i == n - 1)

        obs = convert_obs(tr["observations"], T_r_o_inv)
        nobs = convert_obs(tr["next_observations"], T_r_o_inv)

        # action reprojected with THIS transition's own obs tcp_pose
        s = np.asarray(tr["observations"]["state"], dtype=np.float64)
        q_xyzw = quat_wxyz_to_xyzw(s[SL_QUAT])
        a_new = reproject_action(tr["actions"], q_xyzw)
        a_new[:6] = np.clip(a_new[:6], -1.0, 1.0)
        a_new = a_new.astype(np.float32)

        if is_success and is_terminal:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = bool(is_terminal)  # preserve episode boundary; fail terminal also done

        mask = 0.0 if done else 1.0

        out.append({
            "observations": obs,
            "next_observations": nobs,
            "actions": a_new,
            "rewards": float(reward),
            "masks": float(mask),
            "dones": bool(done),
            "infos": {"grasp_penalty": 0.0},
        })
    return out


def _is_success_path(path):
    base = os.path.basename(path).lower()
    if "success" in base:
        return True
    if "fail" in base:
        return False
    raise ValueError(f"cannot infer success/fail from filename: {path}")


def main():
    ap = argparse.ArgumentParser(description="Convert GELLO demos -> SERL19 format")
    ap.add_argument("--in_glob", required=True, help="glob of input *.pkl demos")
    ap.add_argument("--out_dir", required=True, help="output dir for converted pkls")
    ap.add_argument("--combined", default=None,
                    help="optional path to write all transitions combined into one pkl")
    ap.add_argument("--insert-only", dest="insert_only", action="store_true",
                    help="trim each demo to its insertion segment (reset = above-socket)")
    ap.add_argument("--insert-margin", type=float, default=0.05,
                    help="insertion-start = final run with world x within this of seated x")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings; matching input files are skipped")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    excludes = [e for e in args.exclude.split(",") if e]
    paths = sorted(glob.glob(args.in_glob))
    paths = [p for p in paths if not any(e in os.path.basename(p) for e in excludes)]
    if not paths:
        raise SystemExit(f"no demos matched {args.in_glob} (after --exclude)")

    all_tr = []
    for p in paths:
        is_succ = _is_success_path(p)
        with open(p, "rb") as f:
            demo = pickle.load(f)
        start = insertion_start_index(demo, args.insert_margin) if args.insert_only else 0
        conv = convert_demo(demo, is_success=is_succ, start=start)
        all_tr.extend(conv)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        with open(out_p, "wb") as f:
            pickle.dump(conv, f)
        print(f"[OK] {p} ({'success' if is_succ else 'fail'}, start={start}, {len(conv)} tr) -> {out_p}")

    if args.combined:
        with open(args.combined, "wb") as f:
            pickle.dump(all_tr, f)
        print(f"[OK] combined {len(all_tr)} transitions -> {args.combined}")


if __name__ == "__main__":
    main()
