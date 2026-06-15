"""hybrid_teleop.py — v2.2.1 Route E GELLO + Xbox hybrid teleop recorder.

GELLO grasp/transport -> (toggle) -> Xbox precise insert, BOTH driving the same
serl cartesian_impedance controller via POST /pose (same action space as the
policy — zero controller mismatch, no Polymetis, no stack switch).

GELLO following (Route E) — joint-space anchor + CORRECT FK:
    q_gello_est = q0_robot + (raw_gello - raw_gello0) * joint_signs * leader_scale
    desired_pose = correct_fk(q_gello_est)     # pinocchio(panda_link8) o T_offset
    dxyz/drotvec = desired_pose (-) currpos    # -> apply_cartesian_delta (3mm/0.1rad clamp) -> /pose
The in-repo DH FK is 50cm wrong; pinocchio + the constant Franka-hand F_T_EE
(z 0.1034 m, Rz -45deg) reproduces the server's O_T_EE to ~0 error (verified
live 2026-06-15). Anchoring (raw_gello0, q0_robot at activation) mirrors the
GELLO's MOTION 1:1 with no jump on takeover.

Arbiter / takeover (user requirement): an edge-triggered toggle button flips
GELLO<->XBOX. When XBOX is active the GELLO is IGNORED (no interference);
switching back to GELLO RE-ANCHORS so there is no jump.

Records the SERL contract format (reuses gello_demo_recorder): per tick
state(25)/side_policy/wrist_1/side_classifier + 7D Cartesian action + gripper,
-> .pkl (+ raw .npz with the per-tick control mode).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from relative_teleop import (  # noqa: E402
    DEFAULT_JOINT_SIGNS,
    DEFAULT_MAX_STEP,
    GRIPPER_CLOSE_BELOW,
    GRIPPER_OPEN_ABOVE,
    apply_cartesian_delta,
    gripper_edge,
)
from gello_demo_recorder import (  # noqa: E402
    SIDE_SERIAL,
    WRIST_SERIAL,
    _ThreadedZED,
    build_transitions,
    image_to_obs,
    normalize_action,
    obs_state,
    save_demo,
    validate_transitions,
)

DEFAULT_LEADER_SCALE = 1.0          # 1:1 joint -> EE (Route E, position-based, no loss)
URDF_PATH = "/home/robot/fairo/polymetis/polymetis/data/franka_panda/panda_arm.urdf"
# Franka-hand flange(panda_link8)->EE (F_T_EE), calibrated live vs server O_T_EE.
T_OFFSET_TRANS = np.array([0.0, 0.0, 0.1034])
T_OFFSET_RZ_DEG = -45.0
# franka_server /jointreset default home (flag reset_joint_target).
RESET_JOINT_TARGET = np.array([0.0, 0.0, 0.0, -1.9, 0.0, 2.0, 0.0])


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------
def gello_joint_target(raw_gello, raw_gello0, q0_robot, joint_signs, leader_scale=DEFAULT_LEADER_SCALE):
    """GELLO Dynamixel raw -> robot joint target, anchored at (raw_gello0, q0_robot).

    q_target = q0_robot + (raw[:7] - raw0[:7]) * joint_signs * leader_scale.
    The gripper channel (index 7) is ignored. Returns (7,) float.
    """
    raw_gello = np.asarray(raw_gello, dtype=float).reshape(-1)
    raw_gello0 = np.asarray(raw_gello0, dtype=float).reshape(-1)
    q0_robot = np.asarray(q0_robot, dtype=float).reshape(-1)[:7]
    signs = np.asarray(joint_signs, dtype=float).reshape(-1)[:7]
    delta = (raw_gello[:7] - raw_gello0[:7]) * signs * float(leader_scale)
    return q0_robot + delta


def arbiter_step(mode, toggle_now, toggle_prev) -> Tuple[str, bool]:
    """Edge-triggered GELLO<->XBOX toggle. Rising edge flips; held/released hold.

    Returns (new_mode, switched). When the mode flips the caller must re-anchor
    the newly-active device.
    """
    if toggle_now and not toggle_prev:
        return ("xbox" if mode == "gello" else "gello"), True
    return mode, False


def home_reached(q, target=RESET_JOINT_TARGET, tol=0.15) -> bool:
    """True if every joint of q is within tol (rad) of the home target."""
    q = np.asarray(q, dtype=float).reshape(-1)[:7]
    target = np.asarray(target, dtype=float).reshape(-1)[:7]
    return bool(np.max(np.abs(q - target)) <= tol)


# ---------------------------------------------------------------------------
# Correct FK (pinocchio, live/desktop-validated — needs pinocchio + URDF)
# ---------------------------------------------------------------------------
class CorrectFK:
    """FK_correct(q) = pinocchio_FK(panda_link8, q) o T_offset == server O_T_EE.

    Returns a 7D pose [x,y,z, qx,qy,qz,qw] (scalar-last) matching franka_server.
    """

    def __init__(self, urdf_path=URDF_PATH, ee_link="panda_link8",
                 t_trans=T_OFFSET_TRANS, t_rz_deg=T_OFFSET_RZ_DEG):
        import pinocchio as pin
        from scipy.spatial.transform import Rotation as R

        self._pin = pin
        self._R = R
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        self._fid = self._model.getFrameId(ee_link)
        self._Toff = np.eye(4)
        self._Toff[:3, :3] = R.from_euler("z", t_rz_deg, degrees=True).as_matrix()
        self._Toff[:3, 3] = np.asarray(t_trans, dtype=float)

    def fk(self, q):
        pin = self._pin
        q = np.asarray(q, dtype=float).reshape(-1)[: self._model.nq]
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        T = self._data.oMf[self._fid]
        Tl = np.eye(4)
        Tl[:3, :3] = T.rotation
        Tl[:3, 3] = T.translation
        Tee = Tl @ self._Toff
        quat = self._R.from_matrix(Tee[:3, :3]).as_quat()  # xyzw
        return np.concatenate([Tee[:3, 3], quat])


def pose_delta(currpos, desired):
    """Cartesian delta from currpos to desired (both 7D xyzw). -> (dxyz3, drotvec3)."""
    from scipy.spatial.transform import Rotation as R

    currpos = np.asarray(currpos, dtype=float)
    desired = np.asarray(desired, dtype=float)
    dxyz = desired[:3] - currpos[:3]
    dR = R.from_quat(desired[3:7]) * R.from_quat(currpos[3:7]).inv()
    return dxyz, dR.as_rotvec()


def reset_to_home(session, url, home_pose, max_step=0.006, hz=10.0, timeout=45.0,
                  pos_tol=0.01, rot_tol=0.06):
    """CARTESIAN reset: interpolate the EE to home_pose (7D xyzw) via /pose on the
    already-running cartesian_impedance controller, clamped to max_step per tick.

    This deliberately AVOIDS franka_server /jointreset: that switches to the
    joint_position_controller, which cannot claim the PositionJointInterface while
    impedance's franka_control still holds FCI ("Could not find resource fr3_joint1
    in PositionJointInterface" — verified live), so the joint move silently no-ops.
    The Cartesian path uses the controller that already works. The robot moves
    slowly; the operator must keep the area clear + E-stop in hand. RAISES if the
    home pose is not reached within timeout. Returns the reached pose.
    """
    from relative_teleop import get_state, post_pose

    # Clear any reflex/error state first — a stuck impedance silently ignores
    # /pose (the robot freezes), which otherwise makes the reset time out.
    try:
        session.post(url.rstrip("/") + "/clearerr", json={}, timeout=10.0)
        time.sleep(1.0)
    except Exception:
        pass

    home = np.asarray(home_pose, dtype=float).reshape(-1)[:7]
    dt = 1.0 / hz
    end = time.monotonic() + timeout
    currpos = None
    while time.monotonic() < end:
        currpos = np.asarray(get_state(session, url, timeout=1.0)["pose"], dtype=float)
        dxyz, drotvec = pose_delta(currpos, home)
        if float(np.linalg.norm(dxyz)) < pos_tol and float(np.linalg.norm(drotvec)) < rot_tol:
            return currpos
        nextpos, _s, _r = apply_cartesian_delta(currpos, dxyz, drotvec, max_step)
        post_pose(session, url, nextpos, timeout=1.0)
        time.sleep(dt)
    perr = float(np.linalg.norm(pose_delta(currpos, home)[0])) if currpos is not None else -1.0
    raise RuntimeError(f"cartesian reset timeout; remaining pos err {perr:.3f} m")


def drive_gello(session, url, leader_scale=DEFAULT_LEADER_SCALE, max_step=DEFAULT_MAX_STEP,
                hz=10.0, duration=600.0, gripper=True):
    """Drive the robot via GELLO (Route E) with NO recording until SIGINT/duration,
    for positioning (e.g. capturing a home pose). Returns the final (pose7, q7)."""
    import signal

    from gello.dynamixel.driver import DynamixelDriver
    from relative_teleop import get_state, post_gripper, post_pose

    stop = {"f": False}

    def _s(sig, frm):  # noqa: ARG001
        stop["f"] = True
    signal.signal(signal.SIGINT, _s)
    signal.signal(signal.SIGTERM, _s)

    fk = CorrectFK()
    g = DynamixelDriver(list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
                        max_retries=1, use_fake_fallback=False)
    s0 = get_state(session, url, timeout=1.0)
    q0 = np.asarray(s0["q"], dtype=float)
    raw0 = np.asarray(g.get_joints(), dtype=float)
    gc = float(np.asarray(s0["gripper_pos"]).reshape(-1)[0]) < 0.04
    dt = 1.0 / hz
    end = time.monotonic() + duration
    last = np.asarray(s0["pose"], dtype=float)
    lastq = q0
    print("[SET-HOME] ▶ 用 GELLO 把机器人摆到起始位姿；摆好后按 Ctrl-C 保存为 home", flush=True)
    try:
        while not stop["f"] and time.monotonic() < end:
            t0 = time.monotonic()
            st = get_state(session, url, timeout=1.0)
            currpos = np.asarray(st["pose"], dtype=float)
            last = currpos
            lastq = np.asarray(st["q"], dtype=float)
            raw = np.asarray(g.get_joints(), dtype=float)
            qt = gello_joint_target(raw, raw0, q0, DEFAULT_JOINT_SIGNS, leader_scale)
            dxyz, drot = pose_delta(currpos, fk.fk(qt))
            nextpos, _a, _b = apply_cartesian_delta(currpos, dxyz, drot, max_step)
            post_pose(session, url, nextpos, timeout=1.0)
            if gripper:
                gcmd, gc = gripper_edge(float(raw[7]), gc, GRIPPER_CLOSE_BELOW, GRIPPER_OPEN_ABOVE)
                if gcmd:
                    post_gripper(session, url, gcmd, timeout=1.0)
            sl = dt - (time.monotonic() - t0)
            if sl > 0:
                time.sleep(sl)
    finally:
        try:
            g.close()
        except Exception:
            pass
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    return last, lastq


# ---------------------------------------------------------------------------
# Live hybrid recorder (validated on the robot; not unit-tested)
# ---------------------------------------------------------------------------
def run(server, hz, duration, out_dir, max_step, leader_scale, fps, dry_run,
        pos_scale=0.1, rpy_scale=0.2, align_prompt=False):
    import signal

    from relative_teleop import _session, get_state, post_gripper, post_pose

    stop = {"flag": False, "reason": "duration"}

    def _sig(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        stop["reason"] = f"signal {signum}"
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    session = _session()
    gello_dev = cam_side = cam_wrist = hub = xbox = fk = None
    obs_list: List[dict] = []
    action_list: List[np.ndarray] = []
    raw = {k: [] for k in ("pose", "q", "force", "torque", "gripper_pos",
                           "raw_gello", "action", "ts", "mode")}
    dt = 1.0 / hz
    t_start = time.monotonic()
    gripper_closed = False
    posted = gripper_events = tick = overruns = 0
    last_currpos = None
    last_side = last_wrist = None
    result = None
    CTRL_T = 0.5

    try:
        fk = CorrectFK()
        s0 = get_state(session, server, timeout=CTRL_T)
        # Reset the Franka gripper to a known OPEN state. The gripper server tracks
        # its own binary open/closed flag that DESYNCS from the physical gripper (an
        # open-after-grasp can silently no-op, leaving the flag wrong so every later
        # open/close becomes a no-op — the "gripper stopped working" bug). /reset_gripper
        # re-homes it open + resyncs, so the GELLO/Xbox gripper edges work reliably.
        try:
            session.post(server.rstrip("/") + "/reset_gripper", json={}, timeout=8.0)
            time.sleep(2.0)
        except Exception:
            pass
        gripper_closed = False  # reset_gripper leaves the gripper OPEN

        # Contact compliance for insertion: cap the per-axis position error
        # (translational_clip) so the stiff (2000 N/m) impedance can't build more than
        # ~10N against the socket and trip the Franka collision reflex. clip 0.01 gave
        # 2000*0.01=20N which tripped on insertion (red light, arm frozen); 0.005 -> 10N.
        # Stiffness stays 2000 so free-space GELLO following doesn't droop.
        try:
            cfg = {"translational_stiffness": 2000.0}
            for _ax in ("x", "y", "z"):
                cfg["translational_clip_" + _ax] = 0.005
                cfg["translational_clip_neg_" + _ax] = 0.005
            session.post(server.rstrip("/") + "/update_param", json=cfg, timeout=8.0)
        except Exception:
            pass

        from gello.dynamixel.driver import DynamixelDriver
        gello_dev = DynamixelDriver(list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
                                    max_retries=1, use_fake_fallback=False)

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
        xbox = XboxIntervention(_Env(), hub, max_step=max_step, pos_scale=pos_scale)

        print("[init] opening ZED cameras (threaded)...", flush=True)
        cam_side = _ThreadedZED(SIDE_SERIAL, fps=fps)
        cam_wrist = _ThreadedZED(WRIST_SERIAL, fps=fps)

        # Leader-follower ALIGNMENT (critical): the GELLO and robot must start at
        # the SAME joint config. The mapping q_gello_est = q0_robot + signs*(raw-raw0)
        # only gives valid robot configs if q0_robot ~= the GELLO's config at raw0;
        # otherwise GELLO motion drives the target out of joint range -> wild motion
        # (the bug seen after a HOME reset, robot at HOME but GELLO elsewhere). So the
        # operator aligns the GELLO to the robot's current (HOME) pose, THEN we anchor.
        if align_prompt:
            print("\n[ALIGN] 把 GELLO 摆到和机器人当前一样的姿态（机器人在 HOME——照它摆：夹爪朝下、"
                  "手臂形状对齐），对齐后按 Enter 开始示教。", flush=True)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        raw_gello0 = np.asarray(gello_dev.get_joints(), dtype=float)
        q0_robot = np.asarray(get_state(session, server, timeout=CTRL_T)["q"], dtype=float)
        mode = "gello"
        toggle_prev = False

        t_start = time.monotonic()
        end = t_start + duration
        consec_err = 0
        print(f"[REC] ▶ recording started (mode={mode})", flush=True)
        while not stop["flag"] and time.monotonic() < end:
            tick_t0 = time.monotonic()
            try:
                state = get_state(session, server, timeout=CTRL_T)
                currpos = np.asarray(state["pose"], dtype=float)
                raw_g = np.asarray(gello_dev.get_joints(), dtype=float)
                st = hub.poll()

                toggle_now = bool(getattr(st, "start", False))
                mode, switched = arbiter_step(mode, toggle_now, toggle_prev)
                toggle_prev = toggle_now
                if switched:
                    print(f"[MODE t={tick}] -> {mode}", flush=True)
                    if mode == "gello":  # re-anchor on takeover back to GELLO
                        raw_gello0 = raw_g
                        q0_robot = np.asarray(state["q"], dtype=float)

                gcmd = None
                if mode == "gello":
                    q_target = gello_joint_target(raw_g, raw_gello0, q0_robot,
                                                  DEFAULT_JOINT_SIGNS, leader_scale)
                    desired = fk.fk(q_target)
                    dxyz, drotvec = pose_delta(currpos, desired)
                    gcmd, gripper_closed = gripper_edge(
                        float(raw_g[7]), gripper_closed, GRIPPER_CLOSE_BELOW, GRIPPER_OPEN_ABOVE)
                else:  # xbox — GELLO ignored entirely
                    if not st.rb:
                        dxyz = np.zeros(3); drotvec = np.zeros(3)
                    else:
                        action = xbox._state_to_action(st)
                        dxyz = np.asarray(action[:3], float) * pos_scale
                        drotvec = np.asarray(action[3:6], float) * rpy_scale
                    if st.rt > 0.5 and not gripper_closed:
                        gcmd = "close"; gripper_closed = True
                    elif st.lt > 0.5 and gripper_closed:
                        gcmd = "open"; gripper_closed = False

                nextpos, _step, applied_drot = apply_cartesian_delta(
                    currpos, dxyz, drotvec, max_step)
                applied_dxyz = nextpos[:3] - currpos[:3]
                gripper_pm = 1.0 if gripper_closed else -1.0
                action = normalize_action(applied_dxyz, applied_drot, gripper_pm)

                sf = cam_side.latest(); wf = cam_wrist.latest()
                side = image_to_obs(sf) if sf is not None else last_side
                wrist = image_to_obs(wf) if wf is not None else last_wrist
                if side is None or wrist is None:
                    consec_err = 0; continue
                last_side, last_wrist = side, wrist
            except Exception as e:
                consec_err += 1
                if consec_err <= 3 or consec_err % 20 == 0:
                    print(f"[WARN t={tick}] skipped ({consec_err}): {type(e).__name__}: {str(e)[:80]}", flush=True)
                if consec_err >= 30:
                    stop["flag"] = True; stop["reason"] = "30 consecutive errors"
                continue
            consec_err = 0
            last_currpos = currpos

            obs = {
                "state": obs_state(currpos, state["vel"], state["force"],
                                   state["torque"], state["gripper_pos"]),
                "side_policy": side, "wrist_1": wrist, "side_classifier": side.copy(),
            }
            obs_list.append(obs); action_list.append(action)
            raw["pose"].append(currpos)
            raw["q"].append(np.asarray(state["q"], float))
            raw["force"].append(np.asarray(state["force"], float))
            raw["torque"].append(np.asarray(state["torque"], float))
            raw["gripper_pos"].append(float(np.asarray(state["gripper_pos"]).reshape(-1)[0]))
            raw["raw_gello"].append(raw_g)
            raw["action"].append(action)
            raw["ts"].append(time.monotonic() - t_start)
            raw["mode"].append(0 if mode == "gello" else 1)

            if not dry_run:
                post_pose(session, server, nextpos, timeout=CTRL_T); posted += 1
                if gcmd:
                    post_gripper(session, server, gcmd, timeout=CTRL_T); gripper_events += 1
                    print(f"[GRIPPER t={tick}] -> {gcmd} (mode={mode})", flush=True)

            tick += 1
            # Auto-recover from a collision reflex (red light, arm frozen, /pose
            # silently ignored). /clearerr publishes a Franka error-recovery goal:
            # a no-op when there's no error, recovery when there is. Every ~3s the
            # arm self-recovers without the operator restarting.
            if not dry_run and tick % 30 == 0:
                try:
                    session.post(server.rstrip("/") + "/clearerr", json={}, timeout=0.5)
                except Exception:
                    pass
            work = time.monotonic() - tick_t0
            if work > dt:
                overruns += 1
            if tick % 50 == 0:
                print(f"[REC t={tick}] {tick/(time.monotonic()-t_start):.1f}Hz mode={mode} "
                      f"posted={posted} grip={gripper_events} overruns={overruns}", flush=True)
            sleep = dt - work
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if not dry_run and last_currpos is not None:
            try:
                cp = np.asarray(get_state(session, server, timeout=CTRL_T)["pose"], float)
                post_pose(session, server, cp, timeout=CTRL_T)
                print(f"[REC] re-anchored to currpos {np.round(cp[:3], 4).tolist()}", flush=True)
            except Exception as e:
                print(f"[WARN] re-anchor failed: {type(e).__name__}", flush=True)
        for dev in (gello_dev,):
            try:
                dev and dev.close()
            except Exception:
                pass
        for cam in (cam_side, cam_wrist):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
        elapsed = time.monotonic() - t_start
        print(f"[REC] ■ recording stopped ({stop['reason']}): ticks={tick} "
              f"elapsed={elapsed:.1f}s avg={tick/max(elapsed,1e-6):.1f}Hz posted={posted} "
              f"grip={gripper_events} overruns={overruns}", flush=True)
        if len(obs_list) >= 2:
            transitions = build_transitions(obs_list, action_list[: len(obs_list) - 1])
            ok, msg = validate_transitions(transitions)
            n = len(transitions)
            raw_np = {k: np.asarray(v)[:n] for k, v in raw.items()}
            raw_np["meta_hz"] = float(hz)
            raw_np["meta_leader_scale"] = float(leader_scale)
            raw_np["meta_max_step"] = float(max_step)
            raw_np["meta_dry_run"] = bool(dry_run)
            raw_np["meta_abort_reason"] = stop["reason"]
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            pkl, npz = save_demo(out_dir, ts_str, transitions, raw_np)
            print(f"[SAVE] transitions={n} validate={'PASS' if ok else 'FAIL'}: {msg}", flush=True)
            print(f"[SAVE] pkl={pkl}\n[SAVE] npz={npz}", flush=True)
            result = {"pkl": pkl, "npz": npz, "n_transitions": int(n),
                      "elapsed_s": round(float(elapsed), 1), "validate_ok": bool(ok),
                      "abort_reason": stop["reason"]}
        else:
            print("[SAVE] too few ticks; nothing saved.", flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="GELLO(Route E)+Xbox hybrid teleop recorder")
    p.add_argument("--server", default="http://172.16.0.1:5000/")
    p.add_argument("--out-dir", default="/home/robot/hilserl-fr3/demos/hybrid")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    p.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    res = run(a.server, a.hz, a.duration, a.out_dir, a.max_step, a.leader_scale, a.fps, a.dry_run)
    return 0 if res is not None or a.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
