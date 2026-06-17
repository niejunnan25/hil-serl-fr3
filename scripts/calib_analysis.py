#!/usr/bin/env python3
"""Calibration analysis for insert-only RL from the demo cluster (demos 6-30).

Uses the fast .npz proprio ('pose' world [x,y,z, qw,qx,qy,qz] wxyz; 'gripper_pos').
Computes, for the consistent cluster (drop first-5 + the y-outlier):
  - TARGET_POSE  = median seated world pose (last-5 frames)
  - RESET_POSE   = median insertion-start pose (start of final descent, ~8cm above seat)
  - ABS_POSE_LIMIT box = insertion-segment span + margin
  - seated rel-z relative to insertion-start  -> sets the rel-z reward gate
Prints per-demo so the cluster + trim points are auditable. READ-ONLY.
"""
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

HYBRID = "/home/robot/hilserl-fr3/demos/hybrid"
SEAT_K = 5          # seated = last K frames
DESCENT_ABOVE = 0.08   # insertion-start = last idx (from end) with z >= seated_z + this


def wxyz_to_euler(q_wxyz):
    q = np.asarray(q_wxyz, float)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return R.from_quat(q_xyzw).as_euler("xyz")


def insertion_start_idx(x, seated_x, margin=0.05):
    """Start of the FINAL contiguous run where x stays within `margin` of the socket x.

    = the moment the EE arrives directly above the socket and begins the final
    (near-vertical) insertion. Everything before is grasp+transport (dropped for
    insert-only RL)."""
    thr = seated_x - margin
    n = len(x)
    i = n - 1
    while i > 0 and x[i] >= thr:
        i -= 1
    return i + 1


def main():
    npzs = sorted(glob.glob(os.path.join(HYBRID, "*success*.npz")))
    rows = []
    for p in npzs:
        z = np.load(p, allow_pickle=True)
        pose = np.asarray(z["pose"], float)        # (N,7) wxyz
        grip = np.asarray(z["gripper_pos"], float)  # (N,)
        seat = np.median(pose[-SEAT_K:], axis=0)
        seated_xyz = seat[:3]
        is_i = insertion_start_idx(pose[:, 0], seated_xyz[0])
        rows.append(dict(
            name=os.path.basename(p).replace("gello_demo_", "").replace("_success.npz", ""),
            n=len(pose), seated=seated_xyz, seated_q=seat[3:7],
            ins_i=is_i, ins_len=len(pose) - is_i, ins_pose=pose[is_i],
            grip_min=float(grip.min()), grip_end=float(grip[-1]),
        ))

    print("idx  ts          n     seated_xyz                 ins_i ins_len  ins_start_xyz")
    for k, r in enumerate(rows):
        print("%2d  %-12s %4d  [%6.3f %6.3f %6.3f]  %4d  %4d   [%6.3f %6.3f %6.3f]" % (
            k, r["name"], r["n"], *r["seated"], r["ins_i"], r["ins_len"], *r["ins_pose"][:3]))

    # cluster: by seated-y. The dominant cluster + drop first-5 + y-outliers.
    ys = np.array([r["seated"][0:3][1] for r in rows])
    ymed = np.median(ys)
    # user decision: keep demos index 5..29 (6-30), drop the y-outlier(s) > 4cm from median
    kept = [k for k in range(len(rows)) if k >= 5 and abs(ys[k] - ymed) < 0.04]
    dropped = [rows[k]["name"] for k in range(len(rows)) if k not in kept]
    print("\nseated-y median=%.4f ; first5 y=%s" % (ymed, np.round(ys[:5], 4).tolist()))
    print("KEPT %d demos (idx %s) ; DROPPED %s" % (len(kept), kept, dropped))

    K = [rows[k] for k in kept]
    seated = np.array([r["seated"] for r in K])
    seated_q = np.array([r["seated_q"] for r in K])
    ins = np.array([r["ins_pose"][:3] for r in K])
    ins_q = np.array([r["ins_pose"][3:7] for r in K])

    def stat(a):
        return np.median(a, axis=0), (np.percentile(a, 75, axis=0) - np.percentile(a, 25, axis=0))

    t_med, t_iqr = stat(seated)
    r_med, r_iqr = stat(ins)
    print("\n=== CALIBRATION (kept cluster, n=%d) ===" % len(K))
    print("TARGET_POSE xyz  median=%s  IQR=%s" % (np.round(t_med, 4).tolist(), np.round(t_iqr, 4).tolist()))
    print("TARGET euler(rad) median=%s" % np.round(wxyz_to_euler(np.median(seated_q, axis=0)), 4).tolist())
    print("RESET_POSE  xyz  median=%s  IQR=%s" % (np.round(r_med, 4).tolist(), np.round(r_iqr, 4).tolist()))
    print("RESET euler(rad)  median=%s" % np.round(wxyz_to_euler(np.median(ins_q, axis=0)), 4).tolist())
    print("RESET above seat (z):  %.4f m" % (r_med[2] - t_med[2]))
    print("seated rel-z vs insertion-start (per demo median): %.4f  (-> rel-z reward gate)" %
          np.median(seated[:, 2] - ins[:, 2]))
    # safety box = insertion-segment span across kept demos
    seg_lo = np.array([1e9, 1e9, 1e9]); seg_hi = np.array([-1e9, -1e9, -1e9])
    for r in K:
        # recompute segment span from the npz file
        z = np.load(os.path.join(HYBRID, "gello_demo_" + r["name"] + "_success.npz"), allow_pickle=True)
        seg = np.asarray(z["pose"], float)[r["ins_i"]:, :3]
        seg_lo = np.minimum(seg_lo, seg.min(0)); seg_hi = np.maximum(seg_hi, seg.max(0))
    print("INSERT-SEG span  low=%s  high=%s" % (np.round(seg_lo, 4).tolist(), np.round(seg_hi, 4).tolist()))
    print("  suggested ABS_POSE_LIMIT (+/-0.03 m margin): low=%s high=%s" % (
        np.round(seg_lo - 0.03, 4).tolist(), np.round(seg_hi + 0.03, 4).tolist()))
    print("grip: kept demos grip_end median=%.3f (closed holds plug)" %
          np.median([r["grip_end"] for r in K]))


if __name__ == "__main__":
    main()
