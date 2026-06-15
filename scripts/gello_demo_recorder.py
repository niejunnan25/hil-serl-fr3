"""gello_demo_recorder.py — v2.2.1 GELLO full-process demo recorder.

Records a GELLO-teleoperated FR3 trajectory (grasp -> transport -> insert) into
the SERL demo format the project contract (sim/data/contract.py) + replay buffer
expect, so a real-robot "first try" produces training-grade data:

  observations{
    state        : (25,)  float32  = tcp_pose(7)+tcp_vel(6)+tcp_force(3)+tcp_torque(3)+gripper(6)
    side_policy  : (3,128,128) uint8 RGB  (ZED 2i external, sn 36276705)
    wrist_1      : (3,128,128) uint8 RGB  (ZED-M wrist,     sn 13132609)
    side_classifier : alias/copy of side_policy
  }
  next_observations{...}, actions(7,) f32 in [-1,1], rewards, masks, dones.

Design
------
* Drive reuses relative_teleop (FK-free relative-pose: read robot currpos from
  /getstate, command currpos + small Cartesian delta to /pose; GELLO joint
  delta -> robot Jacobian -> twist; per-step 3mm translation clamp).
* The recorded ``action`` is the APPLIED (post-clamp) Cartesian delta normalized
  by ACTION_SCALE, so action <-> next_obs stay dynamically consistent.
* ZED cameras are read in BACKGROUND THREADS (each grabs continuously; the loop
  reads the latest frame non-blockingly). A synchronous ZED grab blocks ~1/fps
  (~63 ms @ 15 fps), so two sequential grabs would bust the 10 Hz control tick;
  threading decouples capture from control.
* Pure assembly/normalization (obs_state, normalize_action, image_to_obs,
  build_transitions) is unit-tested; the live loop is validated on the robot.

Both a SERL ``.pkl`` (training-ready transitions) and a raw sensor ``.npz``
(poses/q/dq/vel/force/torque/gripper/raw_gello/actions/timestamps, no images)
are written, so the episode is never lost even if the obs encoding is revised.

Stop: SIGINT/SIGTERM (or --duration) breaks the loop and saves in ``finally``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Reusable, hardware-free helpers (relative_teleop top-level imports numpy/stdlib
# only; requests/gello/pygame are imported lazily inside its run()).
from relative_teleop import (  # noqa: E402
    DEFAULT_JOINT_SIGNS,
    DEFAULT_LEADER_SCALE,
    DEFAULT_MAX_STEP,
    GRIPPER_CLOSE_BELOW,
    GRIPPER_OPEN_ABOVE,
    apply_cartesian_delta,
    gello_twist,
    gripper_edge,
)

# Per contract sim/data/contract.py:24 — normalized action -> Cartesian delta.
ACTION_SCALE = np.array([0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0], dtype=float)

# Real ZED serials (verified live): external side view + wrist view.
SIDE_SERIAL = 36276705   # ZED 2i  -> side_policy / side_classifier
WRIST_SERIAL = 13132609  # ZED-M   -> wrist_1

IMAGE_SIZE = 128


# ---------------------------------------------------------------------------
# Pure functions (unit-tested)
# ---------------------------------------------------------------------------
def obs_state(pose, vel, force, torque, gripper_pos):
    """Assemble the 25D contract state (float32), gripper scalar tiled 6x."""
    pose = np.asarray(pose, dtype=float).reshape(-1)[:7]
    vel = np.asarray(vel, dtype=float).reshape(-1)[:6]
    force = np.asarray(force, dtype=float).reshape(-1)[:3]
    torque = np.asarray(torque, dtype=float).reshape(-1)[:3]
    grip = float(np.asarray(gripper_pos, dtype=float).reshape(-1)[0])
    return np.concatenate([pose, vel, force, torque, np.full(6, grip)]).astype(np.float32)


def normalize_action(dxyz, drotvec, gripper_action, action_scale=ACTION_SCALE):
    """Cartesian delta + gripper -> normalized 7D action in [-1, 1] (float32)."""
    action_scale = np.asarray(action_scale, dtype=float)
    dxyz = np.asarray(dxyz, dtype=float).reshape(-1)[:3]
    drotvec = np.asarray(drotvec, dtype=float).reshape(-1)[:3]
    a = np.empty(7, dtype=float)
    a[:3] = dxyz / action_scale[:3]
    a[3:6] = drotvec / action_scale[3:6]
    a[6] = float(gripper_action)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def image_to_obs(frame_hwc, size=IMAGE_SIZE, bgr_to_rgb=True):
    """ZED frame -> (3,size,size) uint8 CHW (RGB).

    The ZED adapter (fr3_zed_capture, channels='RGB') returns **BGR**: VIEW.LEFT
    is BGRA and the 'RGB' path only drops alpha (frame[:, :, :3]) without a B<->R
    swap. The contract obs are RGB, so bgr_to_rgb swaps B<->R by default. Pass
    bgr_to_rgb=False for an already-RGB source.
    """
    arr = np.asarray(frame_hwc)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    arr = np.ascontiguousarray(arr).astype(np.uint8)
    if bgr_to_rgb and arr.ndim == 3 and arr.shape[2] == 3:
        arr = np.ascontiguousarray(arr[..., ::-1])
    try:
        import cv2

        r = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        r = np.asarray(Image.fromarray(arr).resize((size, size)))
    chw = np.transpose(r, (2, 0, 1))
    return np.ascontiguousarray(chw).astype(np.uint8)


def build_transitions(obs_list, action_list):
    """N+1 obs + N actions -> N SERL transition dicts; last done=True/mask=0."""
    trs: List[dict] = []
    n = len(action_list)
    assert len(obs_list) >= n + 1, "need len(obs) >= len(actions)+1"
    for i in range(n):
        done = i == n - 1
        trs.append(
            {
                "observations": obs_list[i],
                "next_observations": obs_list[i + 1],
                "actions": np.asarray(action_list[i], dtype=np.float32).reshape(7),
                "rewards": np.float32(0.0),
                "masks": np.float32(0.0 if done else 1.0),
                "dones": bool(done),
            }
        )
    return trs


def validate_transitions(trs):
    """Sanity-check a transition list against the contract. Returns (ok, msg)."""
    if not trs:
        return False, "empty transition list"
    keys = ("state", "side_policy", "wrist_1", "side_classifier")
    for idx in (0, len(trs) - 1):
        t = trs[idx]
        for side in ("observations", "next_observations"):
            o = t[side]
            if set(o.keys()) != set(keys):
                return False, f"t[{idx}].{side} keys {sorted(o.keys())} != {sorted(keys)}"
            if o["state"].shape != (25,) or o["state"].dtype != np.float32:
                return False, f"t[{idx}].{side}.state {o['state'].shape}/{o['state'].dtype}"
            for k in ("side_policy", "wrist_1", "side_classifier"):
                if o[k].shape != (3, 128, 128) or o[k].dtype != np.uint8:
                    return False, f"t[{idx}].{side}.{k} {o[k].shape}/{o[k].dtype}"
        a = t["actions"]
        if a.shape != (7,) or a.dtype != np.float32:
            return False, f"t[{idx}].actions {a.shape}/{a.dtype}"
        if float(np.max(np.abs(a))) > 1.0001:
            return False, f"t[{idx}].actions out of [-1,1]: max|a|={np.max(np.abs(a))}"
        if abs(float(t["masks"]) - (1.0 - float(t["dones"]))) > 1e-6:
            return False, f"t[{idx}] masks {t['masks']} != 1-dones {1.0 - float(t['dones'])}"
    return True, f"{len(trs)} transitions OK"


# ---------------------------------------------------------------------------
# Threaded ZED capture (live only)
# ---------------------------------------------------------------------------
class _ThreadedZED:
    """Background-thread ZED reader: grabs continuously, exposes latest frame."""

    def __init__(self, serial, fps=30, first_frame_timeout=8.0):
        import threading

        from fr3_zed_capture import ZEDCapture, ZEDCaptureConfig

        self.serial = serial
        self._fps = fps
        self._cap = ZEDCapture(
            ZEDCaptureConfig(serial_number=serial, resolution="HD720", fps=fps, channels="RGB")
        )
        self._latest = None
        self._lock = threading.Lock()
        self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        t0 = time.monotonic()
        while self.latest() is None and time.monotonic() - t0 < first_frame_timeout:
            time.sleep(0.02)
        if self.latest() is None:
            raise RuntimeError(f"ZED sn={serial}: no frame within {first_frame_timeout}s")

    def _loop(self):
        self.grabs = 0
        self.errors = 0
        while self._run:
            try:
                ok, img = self._cap.read()
            except Exception:
                ok, img = False, None
            if ok and img is not None:
                a = np.asarray(img)
                with self._lock:
                    self._latest = a
                self.grabs += 1
            else:
                self.errors += 1
                time.sleep(0.001)  # avoid 100% busy-spin on a stalled/erroring grab

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def close(self):
        self._run = False
        # Join the grab thread BEFORE freeing the native camera, else the thread
        # can call read() on a half-closed pyzed handle (use-after-free / segfault).
        try:
            self._t.join(timeout=max(2.0 / max(self._fps, 1), 0.2))
        except Exception:
            pass
        try:
            self._cap.close()
        except Exception:
            pass


def save_demo(out_dir, ts_str, transitions, raw):
    """Write SERL .pkl (transitions) + raw .npz (sensor log). Returns (pkl, npz)."""
    import pickle

    os.makedirs(out_dir, exist_ok=True)
    pkl = os.path.join(out_dir, f"gello_demo_{ts_str}.pkl")
    npz = os.path.join(out_dir, f"gello_demo_{ts_str}_raw.npz")
    with open(pkl, "wb") as f:
        pickle.dump(transitions, f)
    np.savez_compressed(npz, **raw)
    return pkl, npz


# ---------------------------------------------------------------------------
# Live recorder loop (validated on the robot; not unit-tested)
# ---------------------------------------------------------------------------
CTRL_TIMEOUT = 0.5        # s, control-path HTTP timeout (fail fast, don't freeze a tick)
MAX_CONSEC_ERR = 30       # bail after this many consecutive failed ticks (~3s @ 10Hz)


def run(server, hz, duration, out_dir, max_step, leader_scale, gripper, fps, dry_run):
    import signal

    from relative_teleop import _session, get_state, post_gripper, post_pose

    # Install stop handlers FIRST so a Ctrl-C / SIGTERM during the (blocking)
    # hardware bring-up still triggers the try/finally cleanup below.
    stop = {"flag": False, "reason": "duration"}

    def _sig(signum, frame):  # noqa: ARG001
        stop["flag"] = True
        stop["reason"] = f"signal {signum}"
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    session = _session()
    gello_dev = None
    cam_side = None
    cam_wrist = None
    obs_list: List[dict] = []
    action_list: List[np.ndarray] = []
    raw = {k: [] for k in ("pose", "q", "dq", "vel", "force", "torque",
                           "gripper_pos", "raw_gello", "action", "ts", "gcmd")}
    dt = 1.0 / hz
    t_start = time.monotonic()
    gripper_closed = False
    gripper_events = 0
    posted = 0
    tick = 0
    overruns = 0
    last_currpos = None
    last_side = None
    last_wrist = None

    try:
        s0 = get_state(session, server, timeout=CTRL_TIMEOUT)
        print(f"[init] server={server} dry_run={dry_run} hz={hz} dur={duration}s "
              f"currpos={np.round(np.asarray(s0['pose']), 4).tolist()}", flush=True)

        from gello.dynamixel.driver import DynamixelDriver

        gello_dev = DynamixelDriver(list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
                                    max_retries=1, use_fake_fallback=False)

        print("[init] opening ZED cameras (threaded)...", flush=True)
        cam_side = _ThreadedZED(SIDE_SERIAL, fps=fps)
        cam_wrist = _ThreadedZED(WRIST_SERIAL, fps=fps)
        print("[init] cameras live.", flush=True)

        # Re-sample the GELLO baseline AFTER the (slow) camera bring-up and right
        # before the loop, so a leader bump during init can't produce a huge
        # first-tick delta.
        prev_gello = np.asarray(gello_dev.get_joints(), dtype=float)

        t_start = time.monotonic()
        end = t_start + duration
        consec_err = 0
        print("[REC] ▶ recording started", flush=True)
        while not stop["flag"] and time.monotonic() < end:
            tick_t0 = time.monotonic()
            try:
                state = get_state(session, server, timeout=CTRL_TIMEOUT)
                currpos = np.asarray(state["pose"], dtype=float)
                raw_g = np.asarray(gello_dev.get_joints(), dtype=float)
                dq_g = raw_g[:7] - prev_gello[:7]
                prev_gello = raw_g  # advance baseline on every successful read
                J = np.asarray(state["jacobian"], dtype=float)
                dxyz, drotvec = gello_twist(dq_g, DEFAULT_JOINT_SIGNS, leader_scale, J)

                gcmd = None
                if gripper:
                    gcmd, gripper_closed = gripper_edge(
                        float(raw_g[7]), gripper_closed, GRIPPER_CLOSE_BELOW, GRIPPER_OPEN_ABOVE)

                nextpos, _step, applied_drotvec = apply_cartesian_delta(
                    currpos, dxyz, drotvec, max_step)
                applied_dxyz = nextpos[:3] - currpos[:3]
                gripper_pm = 1.0 if gripper_closed else -1.0
                action = normalize_action(applied_dxyz, applied_drotvec, gripper_pm)

                sf = cam_side.latest()
                wf = cam_wrist.latest()
                side = image_to_obs(sf) if sf is not None else last_side
                wrist = image_to_obs(wf) if wf is not None else last_wrist
                if side is None or wrist is None:
                    consec_err = 0  # not an error, just no frame yet
                    continue        # cannot form a complete obs without both images
                last_side, last_wrist = side, wrist
            except Exception as e:  # transient /getstate, GELLO, jacobian, etc.
                consec_err += 1
                if consec_err <= 3 or consec_err % 20 == 0:
                    print(f"[WARN t={tick}] tick skipped ({consec_err} consec): "
                          f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                if consec_err >= MAX_CONSEC_ERR:
                    stop["flag"] = True
                    stop["reason"] = f"{MAX_CONSEC_ERR} consecutive errors"
                continue
            consec_err = 0
            last_currpos = currpos

            obs = {
                "state": obs_state(currpos, state["vel"], state["force"],
                                   state["torque"], state["gripper_pos"]),
                "side_policy": side,
                "wrist_1": wrist,
                "side_classifier": side.copy(),
            }
            obs_list.append(obs)
            action_list.append(action)
            raw["pose"].append(currpos)
            raw["q"].append(np.asarray(state["q"], float))
            raw["dq"].append(np.asarray(state["dq"], float))
            raw["vel"].append(np.asarray(state["vel"], float))
            raw["force"].append(np.asarray(state["force"], float))
            raw["torque"].append(np.asarray(state["torque"], float))
            raw["gripper_pos"].append(float(np.asarray(state["gripper_pos"]).reshape(-1)[0]))
            raw["raw_gello"].append(raw_g)
            raw["action"].append(action)
            raw["ts"].append(time.monotonic() - t_start)
            raw["gcmd"].append(1 if gcmd == "close" else -1 if gcmd == "open" else 0)

            if not dry_run:
                post_pose(session, server, nextpos, timeout=CTRL_TIMEOUT)
                posted += 1
                if gcmd:
                    post_gripper(session, server, gcmd, timeout=CTRL_TIMEOUT)
                    gripper_events += 1
                    print(f"[GRIPPER t={tick}] -> {gcmd}", flush=True)

            tick += 1
            work = time.monotonic() - tick_t0
            if work > dt:
                overruns += 1
            if tick % 50 == 0:
                print(f"[REC t={tick}] {tick/(time.monotonic()-t_start):.1f}Hz "
                      f"posted={posted} grip_events={gripper_events} overruns={overruns}", flush=True)
            sleep = dt - work
            if sleep > 0:
                time.sleep(sleep)
    finally:
        # Re-anchor: command the robot's TRUE current pose so the impedance target
        # is where the arm actually is, not the last (slightly-ahead) commanded
        # target — a clean fail-safe stop. Best-effort; never blocks cleanup.
        if not dry_run and last_currpos is not None:
            try:
                cp = np.asarray(get_state(session, server, timeout=CTRL_TIMEOUT)["pose"], float)
                post_pose(session, server, cp, timeout=CTRL_TIMEOUT)
                print(f"[REC] re-anchored to currpos {np.round(cp[:3], 4).tolist()}", flush=True)
            except Exception as e:
                print(f"[WARN] re-anchor failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
        if gello_dev is not None:
            try:
                gello_dev.close()
            except Exception:
                pass
        for cam in (cam_side, cam_wrist):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
        elapsed = time.monotonic() - t_start
        print(f"[REC] ■ recording stopped ({stop['reason']}): "
              f"ticks={tick} elapsed={elapsed:.1f}s "
              f"avg={tick/max(elapsed,1e-6):.1f}Hz posted={posted} "
              f"grip_events={gripper_events} overruns={overruns}", flush=True)
        if len(obs_list) >= 2:
            transitions = build_transitions(obs_list, action_list[: len(obs_list) - 1])
            ok, msg = validate_transitions(transitions)
            n = len(transitions)
            # Trim per-tick raw arrays to exactly match the pkl transitions (the
            # final tick's action is dropped — no obs_{N+1} to be its next_obs).
            raw_np = {k: np.asarray(v)[:n] for k, v in raw.items()}
            raw_np["meta_hz"] = float(hz)
            raw_np["meta_elapsed"] = float(elapsed)
            raw_np["meta_leader_scale"] = float(leader_scale)
            raw_np["meta_max_step"] = float(max_step)
            raw_np["meta_dry_run"] = bool(dry_run)
            raw_np["meta_overruns"] = int(overruns)
            raw_np["meta_abort_reason"] = stop["reason"]
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            pkl, npz = save_demo(out_dir, ts_str, transitions, raw_np)
            print(f"[SAVE] transitions={n} validate={'PASS' if ok else 'FAIL'}: {msg}", flush=True)
            print(f"[SAVE] pkl={pkl}", flush=True)
            print(f"[SAVE] npz={npz}", flush=True)
        else:
            print("[SAVE] too few ticks; nothing saved.", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="GELLO full-process demo recorder (-> SERL pkl)")
    p.add_argument("--server", default="http://172.16.0.1:5000/")
    p.add_argument("--out-dir", default="/home/robot/hilserl-fr3/demos/gello_fullprocess")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    p.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    p.add_argument("--fps", type=int, default=30, help="ZED capture fps (threaded)")
    p.add_argument("--no-gripper", dest="gripper", action="store_false",
                   help="disable GELLO axis-7 -> FR3 gripper actuation")
    p.add_argument("--dry-run", action="store_true",
                   help="open cameras + read state + record, but never POST motion")
    a = p.parse_args(argv)
    return run(a.server, a.hz, a.duration, a.out_dir, a.max_step,
               a.leader_scale, a.gripper, a.fps, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
