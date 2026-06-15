"""relative_teleop.py — v2.2.1 B1b/B2

FK-free relative-pose teleop for GELLO / Xbox -> FR3 via franka_server /pose.

The in-repo absolute FK (fk_converter) is ~50 cm off vs the server's O_T_EE
frame (proven live), so we do NOT command FK(joint_target). Instead, every
tick we read the robot's own currpos from /getstate and command
``currpos + small Cartesian delta`` to /pose — the same relative scheme the
upstream SERL FrankaEnv uses, which sidesteps the FK error entirely.

- Xbox : stick -> XboxIntervention normalized action -> de-normalized Cartesian
         delta (only while RB held, deadman).
- GELLO: per-tick joint delta -> robot Jacobian (from /getstate) -> Cartesian
         twist (continuous follow).

Pure math (apply_cartesian_delta, gello_twist) is unit-tested. The live loop is
dry-run validated against the real robot (no POST) before any motion.

Proxy-safe: uses a requests Session with trust_env=False so the desktop's
HTTP proxy env (which lacks CIDR no_proxy for 172.16.0.1) never intercepts the
internal control link.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Defaults mirror gello_pose_follow / upstream usb_pickup_insertion.
DEFAULT_MAX_STEP = 0.003          # m, per-tick Cartesian translation cap
DEFAULT_POS_SCALE = 0.1           # normalized -> m (Xbox)
DEFAULT_RPY_SCALE = 0.2           # normalized -> rad (Xbox)
DEFAULT_LEADER_SCALE = 0.50
DEFAULT_JOINT_SIGNS = np.array([1, -1, 1, 1, 1, -1, 1], dtype=float)


# ---------------------------------------------------------------------------
# Pure math (unit-tested)
# ---------------------------------------------------------------------------
def apply_cartesian_delta(currpos, dxyz, drotvec, max_step):
    """currpos (7: x,y,z,qx,qy,qz,qw scalar-last) + a Cartesian delta.

    Translation is clipped so ||dxyz|| <= max_step (direction preserved).
    Orientation is left-composed: q_next = rotvec(drotvec) * q_curr.
    Returns (nextpos7, applied_translation_norm).
    """
    currpos = np.asarray(currpos, dtype=float)
    dxyz = np.asarray(dxyz, dtype=float).copy()
    n = float(np.linalg.norm(dxyz))
    if n > max_step:
        dxyz *= max_step / n
        applied = max_step
    else:
        applied = n

    nextpos = currpos.copy()
    nextpos[:3] = currpos[:3] + dxyz

    drotvec = np.asarray(drotvec, dtype=float)
    if float(np.linalg.norm(drotvec)) > 0.0:
        from scipy.spatial.transform import Rotation as R
        q = (R.from_rotvec(drotvec) * R.from_quat(currpos[3:])).as_quat()  # scalar-last
        nextpos[3:] = q
    return nextpos, applied


def gello_twist(dq_gello, joint_signs, leader_scale, jacobian):
    """Map a GELLO joint delta to a Cartesian twist via the robot Jacobian.

    dq_fr3 = dq_gello * joint_signs * leader_scale ; twist = J(6x7) @ dq_fr3.
    Returns (dxyz3, drotvec3). FK-free — uses the robot's own Jacobian.
    """
    dq_fr3 = np.asarray(dq_gello, dtype=float)[:7] * np.asarray(joint_signs, dtype=float) * float(leader_scale)
    twist = np.asarray(jacobian, dtype=float).reshape(6, 7) @ dq_fr3
    return twist[:3], twist[3:]


# ---------------------------------------------------------------------------
# Live HTTP (proxy-safe)
# ---------------------------------------------------------------------------
def _session():
    import requests
    s = requests.Session()
    s.trust_env = False  # ignore HTTP(S)_PROXY/NO_PROXY env -> direct to internal IP
    return s


def get_state(session, url, timeout=5.0):
    r = session.post(url.rstrip("/") + "/getstate", json={}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_pose(session, url, pose7, timeout=5.0):
    r = session.post(url.rstrip("/") + "/pose", json={"arr": [float(x) for x in pose7]}, timeout=timeout)
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Live loop (dry-run validated before motion)
# ---------------------------------------------------------------------------
def run(device, server, hz, duration, dry_run, max_step, pos_scale, rpy_scale, leader_scale):
    session = _session()
    state = get_state(session, server)
    currpos = np.asarray(state["pose"], dtype=float)
    print(f"[init] device={device} dry_run={dry_run} currpos={np.round(currpos,4).tolist()}")

    xbox = gello_dev = prev_gello = None
    if device == "xbox":
        import gymnasium as gym
        from teleop_hub import TeleopDeviceHub
        from xbox_intervention import XboxIntervention

        class _Env(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)
                self.observation_space = gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros(7, np.float32), {}

            def step(self, action):
                return np.zeros(7, np.float32), 0.0, False, False, {}

        hub = TeleopDeviceHub(backend="pygame")
        if not hub.available:
            print("[ERR] no Xbox device on hub"); return 2
        # Proper init reuses the tested stick->action mapping + in-toolchain 3mm cap.
        xbox = XboxIntervention(_Env(), hub, max_step=max_step, pos_scale=pos_scale)
    elif device == "gello":
        from gello.dynamixel.driver import DynamixelDriver
        gello_dev = DynamixelDriver(list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
                                    max_retries=1, use_fake_fallback=False)
        prev_gello = np.asarray(gello_dev.get_joints(), dtype=float)
    else:
        print(f"[ERR] unknown device {device}"); return 2

    dt = 1.0 / hz
    end = time.monotonic() + duration
    tick = 0
    posted = 0
    try:
        while time.monotonic() < end:
            state = get_state(session, server)
            currpos = np.asarray(state["pose"], dtype=float)

            if device == "xbox":
                st = xbox.hub.poll()
                if not st.rb:                       # RB deadman: no command unless held
                    dxyz = np.zeros(3); drotvec = np.zeros(3)
                else:
                    action = xbox._state_to_action(st)   # normalized, 3mm-capped in xyz
                    dxyz = np.asarray(action[:3], float) * pos_scale
                    drotvec = np.asarray(action[3:6], float) * rpy_scale
            else:  # gello
                raw = np.asarray(gello_dev.get_joints(), dtype=float)
                dq = raw[:7] - prev_gello[:7]
                prev_gello = raw
                J = np.asarray(state["jacobian"], dtype=float)
                dxyz, drotvec = gello_twist(dq, DEFAULT_JOINT_SIGNS, leader_scale, J)

            nextpos, step = apply_cartesian_delta(currpos, dxyz, drotvec, max_step)
            if dry_run:
                if step > 1e-6 or np.linalg.norm(drotvec) > 1e-6:
                    print(f"[DRY t={tick}] step={step*1000:.2f}mm "
                          f"dxyz={np.round(dxyz,4).tolist()} -> next={np.round(nextpos[:3],4).tolist()}")
            else:
                post_pose(session, server, nextpos)
                posted += 1
            tick += 1
            time.sleep(dt)
    finally:
        if gello_dev is not None:
            try: gello_dev.close()
            except Exception: pass
    print(f"[done] ticks={tick} posted={posted} (dry_run={dry_run})")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Relative-pose teleop (GELLO/Xbox -> FR3 /pose)")
    p.add_argument("--device", choices=["xbox", "gello"], required=True)
    p.add_argument("--server", default="http://172.16.0.1:5000/")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true", help="compute + print targets, never POST")
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    p.add_argument("--pos-scale", type=float, default=DEFAULT_POS_SCALE)
    p.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE)
    p.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    a = p.parse_args(argv)
    return run(a.device, a.server, a.hz, a.duration, a.dry_run,
               a.max_step, a.pos_scale, a.rpy_scale, a.leader_scale)


if __name__ == "__main__":
    sys.exit(main())
