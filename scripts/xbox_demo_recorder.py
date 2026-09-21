"""Xbox-only insert demo recorder for SERL-aligned Stage-C recollection.

The plug is clamped before this recorder starts. This module never initializes
GELLO and never sends gripper open/close commands; Xbox controls only the TCP
pose while the recorded gripper action channel stays fixed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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
from hybrid_teleop import (  # noqa: E402
    XBOX_DZ,
    XBOX_HOLD_CLIP,
    XBOX_INSERT_CLIP,
    XBOX_INSERT_REACH,
    XBOX_ROT_STEP,
    XBOX_SPIRAL_R,
    XBOX_SPIRAL_RATE,
    XBOX_SPIRAL_W,
    execute_stop_cleanup,
)
from relative_teleop import apply_cartesian_delta  # noqa: E402


CTRL_TIMEOUT = 0.5
MAX_CONSEC_ERR = 30


@dataclass
class XboxOnlyInsertState:
    z_target: float
    insert_prev: bool = False
    insert_ticks: int = 0
    insert_center_x: float = 0.0
    insert_center_y: float = 0.0
    insert_x: float = 0.0
    insert_y: float = 0.0


@dataclass
class XboxOnlyCommand:
    nextpos: np.ndarray
    action: np.ndarray
    param_update: Optional[dict]
    gripper_command: Optional[str] = None


def compute_xbox_insert_command(
    currpos,
    xbox_state,
    state: XboxOnlyInsertState,
    max_step: float,
    dt: float,
    hold_grip_action: float = 0.0,
) -> XboxOnlyCommand:
    """Compute one Xbox-only insert command without touching the gripper."""
    currpos = np.asarray(currpos, dtype=float).reshape(-1)[:7]
    param_update = None
    insert_now = bool(getattr(xbox_state, "rb", False) and getattr(xbox_state, "a", False))

    if insert_now and not state.insert_prev:
        state.insert_center_x = float(currpos[0])
        state.insert_center_y = float(currpos[1])
        state.insert_ticks = 0
        param_update = {
            "translational_clip_z": XBOX_INSERT_CLIP,
            "translational_clip_neg_z": XBOX_INSERT_CLIP,
        }
    elif state.insert_prev and not insert_now:
        param_update = {
            "translational_clip_z": XBOX_HOLD_CLIP,
            "translational_clip_neg_z": XBOX_HOLD_CLIP,
        }
    state.insert_prev = insert_now

    if not getattr(xbox_state, "rb", False):
        dxyz = np.zeros(3)
        drotvec = np.zeros(3)
    else:
        lx = 0.0 if abs(float(getattr(xbox_state, "left_x", 0.0))) < XBOX_DZ else float(xbox_state.left_x)
        ly = 0.0 if abs(float(getattr(xbox_state, "left_y", 0.0))) < XBOX_DZ else float(xbox_state.left_y)
        ry = 0.0 if abs(float(getattr(xbox_state, "right_y", 0.0))) < XBOX_DZ else float(xbox_state.right_y)
        rxx = 0.0 if abs(float(getattr(xbox_state, "right_x", 0.0))) < XBOX_DZ else float(xbox_state.right_x)
        dxy = np.array([-ly, lx]) * float(max_step)
        nrm = float(np.linalg.norm(dxy))
        if nrm > float(max_step):
            dxy *= float(max_step) / nrm
        rz = -ry * float(max_step)
        if rz != 0.0:
            state.z_target = float(currpos[2] + rz)
        if insert_now:
            state.insert_ticks += 1
            th = state.insert_ticks * float(dt)
            rr = min(XBOX_SPIRAL_R, XBOX_SPIRAL_RATE * th)
            ang = XBOX_SPIRAL_W * th
            state.insert_x = state.insert_center_x + rr * float(np.cos(ang))
            state.insert_y = state.insert_center_y + rr * float(np.sin(ang))
            state.z_target = float(currpos[2] - XBOX_INSERT_REACH)
        dxyz = np.array([dxy[0], dxy[1], 0.0])
        drotvec = np.array([
            float(getattr(xbox_state, "dpad_x", 0.0)),
            -float(getattr(xbox_state, "dpad_y", 0.0)),
            rxx,
        ]) * XBOX_ROT_STEP

    nextpos, _step, applied_drot = apply_cartesian_delta(currpos, dxyz, drotvec, max_step)
    if insert_now:
        nextpos[0] = state.insert_x
        nextpos[1] = state.insert_y
    nextpos[2] = state.z_target
    applied_dxyz = nextpos[:3] - currpos[:3]
    # Rotation lock: plug insertion is xy+z only; policy never needs roll/pitch/yaw.
    action = normalize_action(applied_dxyz, applied_drot, float(hold_grip_action))
    action[3:6] = 0.0
    return XboxOnlyCommand(nextpos=nextpos, action=action, param_update=param_update)


def _xbox_state_array(st):
    return np.asarray([
        st.left_x,
        st.left_y,
        st.right_x,
        st.right_y,
        st.dpad_x,
        st.dpad_y,
        st.rt,
        st.lt,
        float(st.a),
        float(st.b),
        float(st.x),
        float(st.y),
        float(st.lb),
        float(st.rb),
        float(st.start),
        float(st.back),
    ], dtype=np.float32)


def run(
    server,
    hz,
    duration,
    out_dir,
    max_step,
    leader_scale=None,  # kept for collector signature compatibility
    gripper=False,  # noqa: ARG001 - gripper is intentionally ignored
    fps=30,
    dry_run=False,
    hold_grip_action=0.0,
    return_result=False,
):
    import signal

    from relative_teleop import _session, get_state, post_pose
    from teleop_hub import TeleopDeviceHub

    stop = {"flag": False, "reason": "duration"}

    def _sig(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        stop["reason"] = f"signal {signum}"
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    session = _session()
    hub = None
    cam_side = None
    cam_wrist = None
    obs_list = []
    action_list = []
    raw = {k: [] for k in ("pose", "q", "dq", "vel", "force", "torque",
                           "gripper_pos", "xbox_state", "action", "ts")}
    dt = 1.0 / float(hz)
    t_start = time.monotonic()
    posted = tick = overruns = 0
    last_currpos = None
    last_side = None
    last_wrist = None
    result = None

    try:
        s0 = get_state(session, server, timeout=CTRL_TIMEOUT)
        curr0 = np.asarray(s0["pose"], dtype=float)
        control_state = XboxOnlyInsertState(z_target=float(curr0[2]))
        print(
            f"[init] xbox-only server={server} dry_run={dry_run} hz={hz} dur={duration}s "
            f"currpos={np.round(curr0, 4).tolist()} hold_grip_action={hold_grip_action}",
            flush=True,
        )

        hub = TeleopDeviceHub(backend="pygame")
        if not hub.available:
            print("[ERR] Xbox hub unavailable (pygame missing or no controller detected).", flush=True)
            return None if return_result else 2

        try:
            cfg = {"translational_stiffness": 2000.0}
            for axis in ("x", "y", "z"):
                cfg["translational_clip_" + axis] = XBOX_HOLD_CLIP
                cfg["translational_clip_neg_" + axis] = XBOX_HOLD_CLIP
            session.post(server.rstrip("/") + "/update_param", json=cfg, timeout=8.0)
        except Exception:
            pass

        print("[init] opening ZED cameras (threaded)...", flush=True)
        cam_side = _ThreadedZED(SIDE_SERIAL, fps=fps)
        cam_wrist = _ThreadedZED(WRIST_SERIAL, fps=fps)
        print("[init] cameras live.", flush=True)

        end = time.monotonic() + float(duration)
        consec_err = 0
        print("[REC] ▶ recording started (mode=xbox-only)", flush=True)
        while not stop["flag"] and time.monotonic() < end:
            tick_t0 = time.monotonic()
            try:
                state = get_state(session, server, timeout=CTRL_TIMEOUT)
                currpos = np.asarray(state["pose"], dtype=float)
                xbox_state = hub.poll()
                cmd = compute_xbox_insert_command(
                    currpos,
                    xbox_state,
                    control_state,
                    max_step=float(max_step),
                    dt=dt,
                    hold_grip_action=float(hold_grip_action),
                )
                if cmd.param_update and not dry_run:
                    try:
                        session.post(server.rstrip("/") + "/update_param", json=cmd.param_update, timeout=CTRL_TIMEOUT)
                    except Exception:
                        pass

                sf = cam_side.latest()
                wf = cam_wrist.latest()
                side = image_to_obs(sf) if sf is not None else last_side
                wrist = image_to_obs(wf) if wf is not None else last_wrist
                if side is None or wrist is None:
                    consec_err = 0
                    continue
                last_side, last_wrist = side, wrist
            except Exception as exc:
                consec_err += 1
                if consec_err <= 3 or consec_err % 20 == 0:
                    print(f"[WARN t={tick}] skipped ({consec_err}): {type(exc).__name__}: {str(exc)[:80]}", flush=True)
                if consec_err >= MAX_CONSEC_ERR:
                    stop["flag"] = True
                    stop["reason"] = f"{MAX_CONSEC_ERR} consecutive errors"
                continue
            consec_err = 0
            last_currpos = currpos

            obs = {
                "state": obs_state(currpos, state["vel"], state["force"], state["torque"], state["gripper_pos"]),
                "side_policy": side,
                "wrist_1": wrist,
                "side_classifier": side.copy(),
            }
            obs_list.append(obs)
            action_list.append(cmd.action)
            raw["pose"].append(currpos)
            raw["q"].append(np.asarray(state["q"], float))
            raw["dq"].append(np.asarray(state["dq"], float))
            raw["vel"].append(np.asarray(state["vel"], float))
            raw["force"].append(np.asarray(state["force"], float))
            raw["torque"].append(np.asarray(state["torque"], float))
            raw["gripper_pos"].append(float(np.asarray(state["gripper_pos"]).reshape(-1)[0]))
            raw["xbox_state"].append(_xbox_state_array(xbox_state))
            raw["action"].append(cmd.action)
            raw["ts"].append(time.monotonic() - t_start)

            if not dry_run:
                post_pose(session, server, cmd.nextpos, timeout=CTRL_TIMEOUT)
                posted += 1

            tick += 1
            if not dry_run and tick % 30 == 0:
                try:
                    session.post(server.rstrip("/") + "/clearerr", json={}, timeout=CTRL_TIMEOUT)
                except Exception:
                    pass
            work = time.monotonic() - tick_t0
            if work > dt:
                overruns += 1
            if tick % 50 == 0:
                print(f"[REC t={tick}] {tick/(time.monotonic()-t_start):.1f}Hz posted={posted} overruns={overruns}", flush=True)
            sleep = dt - work
            if sleep > 0:
                time.sleep(sleep)
    finally:
        if not dry_run:
            try:
                session.post(server.rstrip("/") + "/clearerr", json={}, timeout=CTRL_TIMEOUT)
            except Exception:
                pass

            def _current_pose():
                return np.asarray(get_state(session, server, timeout=CTRL_TIMEOUT)["pose"], float)

            execute_stop_cleanup(
                session,
                server,
                get_current_pose=_current_pose,
                post_pose_fn=post_pose,
                fallback_currpos=last_currpos,
                timeout=CTRL_TIMEOUT,
            )
        if hub is not None:
            try:
                hub.close()
            except Exception:
                pass
        for cam in (cam_side, cam_wrist):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass

        elapsed = time.monotonic() - t_start
        print(
            f"[REC] ■ recording stopped ({stop['reason']}): ticks={tick} "
            f"elapsed={elapsed:.1f}s avg={tick/max(elapsed,1e-6):.1f}Hz posted={posted} overruns={overruns}",
            flush=True,
        )
        if len(obs_list) >= 2:
            transitions = build_transitions(obs_list, action_list[: len(obs_list) - 1])
            ok, msg = validate_transitions(transitions)
            n = len(transitions)
            raw_np = {k: np.asarray(v)[:n] for k, v in raw.items()}
            raw_np["meta_hz"] = float(hz)
            raw_np["meta_elapsed"] = float(elapsed)
            raw_np["meta_max_step"] = float(max_step)
            raw_np["meta_dry_run"] = bool(dry_run)
            raw_np["meta_hold_grip_action"] = float(hold_grip_action)
            raw_np["meta_controller"] = "xbox_only"
            raw_np["meta_overruns"] = int(overruns)
            raw_np["meta_abort_reason"] = stop["reason"]
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            pkl, npz = save_demo(out_dir, ts_str, transitions, raw_np)
            print(f"[SAVE] transitions={n} validate={'PASS' if ok else 'FAIL'}: {msg}", flush=True)
            print(f"[SAVE] pkl={pkl}", flush=True)
            print(f"[SAVE] npz={npz}", flush=True)
            result = {
                "pkl": pkl,
                "npz": npz,
                "n_transitions": int(n),
                "elapsed_s": float(elapsed),
                "validate_ok": bool(ok),
                "validate_msg": msg,
                "abort_reason": stop["reason"],
            }
        else:
            print("[SAVE] too few ticks; nothing saved.", flush=True)
            result = None
    return result if return_result else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Xbox-only insert demo recorder (-> SERL pkl)")
    parser.add_argument("--server", default="http://172.16.0.1:5000/")
    parser.add_argument("--out-dir", default="/home/robot/serl_projects/hil-serl-fr3/demos/insert_recollect_20260624")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--max-step", type=float, default=0.003)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hold-grip-action", type=float, default=0.0)
    args = parser.parse_args(argv)
    return run(
        server=args.server,
        hz=args.hz,
        duration=args.duration,
        out_dir=args.out_dir,
        max_step=args.max_step,
        fps=args.fps,
        dry_run=args.dry_run,
        hold_grip_action=args.hold_grip_action,
    )


if __name__ == "__main__":
    sys.exit(main())
