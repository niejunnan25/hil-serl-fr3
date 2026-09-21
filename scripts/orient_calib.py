#!/usr/bin/env python3
"""Invert euler_2_quat for TARGET/RESET orientation from the 24-demo cluster.

The env commands pose via euler_2_quat(config_euler) (a NON-standard map, not the
inverse of quat_2_euler). So config TARGET/RESET euler must satisfy
euler_2_quat(euler) == demo_quat. We numerically invert per pose. READ-ONLY."""
import glob
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from franka_env.utils.rotations import euler_2_quat, quat_2_euler

HYBRID = "/home/robot/hilserl-fr3/demos/hybrid"
EXCL = ["194411", "194646", "194847", "195923", "200908", "204527"]
DEFAULT_QUAT_ATOL = 2e-3


def insertion_start(x, seated_x, margin=0.05):
    thr = seated_x - margin
    i = len(x) - 1
    while i > 0 and x[i] >= thr:
        i -= 1
    return i + 1


def chordal_mean_quat(qs):
    """sign-align to the first, normalized mean (xyzw)."""
    qs = np.asarray(qs, float).copy()
    for i in range(len(qs)):
        if np.dot(qs[i], qs[0]) < 0:
            qs[i] = -qs[i]
    m = qs.mean(0)
    return m / np.linalg.norm(m)


def invert(q):
    q = q / np.linalg.norm(q)

    def resid(e):
        qq = euler_2_quat(e)
        if np.dot(qq, q) < 0:
            qq = -qq
        return qq - q
    sol = least_squares(resid, x0=quat_2_euler(q))
    check = euler_2_quat(sol.x)
    if np.dot(check, q) < 0:
        check = -check
    residual_norm = float(np.linalg.norm(check - q))
    if not sol.success or residual_norm > DEFAULT_QUAT_ATOL:
        raise RuntimeError(
            "orientation inversion failed: "
            f"success={sol.success} residual_norm={residual_norm:.6g} "
            f"cost={float(sol.cost):.6g} optimality={float(sol.optimality):.6g} "
            f"message={sol.message}"
        )
    return sol.x, check


def main():
    npzs = sorted(glob.glob(os.path.join(HYBRID, "*success*.npz")))
    npzs = [p for p in npzs if not any(e in os.path.basename(p) for e in EXCL)]
    seated_q, ins_q = [], []
    for p in npzs:
        z = np.load(p, allow_pickle=True)
        pose = np.asarray(z["pose"], float)
        seated_wxyz = np.median(pose[-5:], axis=0)[3:7]
        k = insertion_start(pose[:, 0], float(np.median(pose[-5:, 0])))
        ins_wxyz = pose[k][3:7]
        # wxyz -> xyzw
        seated_q.append([seated_wxyz[1], seated_wxyz[2], seated_wxyz[3], seated_wxyz[0]])
        ins_q.append([ins_wxyz[1], ins_wxyz[2], ins_wxyz[3], ins_wxyz[0]])
    print("kept demos:", len(npzs))
    for name, qs in [("TARGET (seated)", seated_q), ("RESET (insertion-start)", ins_q)]:
        qm = chordal_mean_quat(qs)
        e, check = invert(qm)
        print(f"\n{name}")
        print("  target quat xyzw      :", np.round(qm, 4).tolist())
        print("  config euler (for euler_2_quat):", np.round(e, 5).tolist())
        print("  euler_2_quat(euler)   :", np.round(check, 4).tolist())
        print("  match:", bool(np.allclose(check, qm, atol=DEFAULT_QUAT_ATOL)))


if __name__ == "__main__":
    main()
