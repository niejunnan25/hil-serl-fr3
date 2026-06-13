"""p2_t3_e2e_motion_driver.py — GELLO -> FK -> Cartesian delta -> POST /pose.

Used by 16_gello_e2e_motion_test.sh mode 4 (full E2E motion stream).
Phase A only exercises the driver in --dry-run and --micro modes;
--full is gated by the shell's approval env var and never invoked
without an operator in the loop with the E-stop.

dry-run / micro exercise the GelloCartesianDeltaAgent FK + safety
pipeline without touching the robot (no POST).

--full (B1a) drives the PROVEN record_gello_demos_serl contract:
read q0+currpos from /getstate, gate on the FK-vs-currpos bias, then
GELLO joint-follow -> forward_kinematics -> POST absolute /pose
{"arr":[x,y,z,qx,qy,qz,qw]} (franka_server's only motion command).
It is gated behind FR3_GELLO_E2E_APPROVAL; the FIRST live run must
still follow the on-site dry-run -> no-op(current pose) -> micro ->
full ramp with operator + E-stop (Phase B B1b live-verify). See
B-RESEARCH.md for the grounded contract.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# Ensure scripts is on the import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Try importing the agent and the requests lib. requests is only
# required for --full; dry-run and micro don't need it.
try:
    from gello_cartesian_delta_agent import GelloCartesianDeltaAgent
    _AGENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AGENT_AVAILABLE = False

try:
    import requests  # noqa: F401
    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False

# Approval phrase must match the shell's exactly. We re-verify in
# the driver so a Python caller cannot bypass the shell gate.
APPROVAL_PHRASE = "I_APPROVE_P2T3_FULL_E2E_MOTION"
APPROVAL_VAR = "FR3_GELLO_E2E_APPROVAL"

# Per-step safety caps (must not be relaxed outside the
# /home/robot/hilserl-fr3/scripts/setup approval flow).
MAX_STEP = 0.003          # m per step
MAX_TOTAL_DELTA = 0.03    # m cumulative
MAX_MICRO_DURATION_S = 0.5  # micro mode: 3 ticks * (1/10Hz) ~= 0.3s
MICRO_TICKS = 3


def _require_approval() -> None:
    if os.environ.get(APPROVAL_VAR) != APPROVAL_PHRASE:
        print(
            f"[ERR] approval required: {APPROVAL_VAR}={APPROVAL_PHRASE}",
            file=sys.stderr,
        )
        sys.exit(5)


def _open_gello(port: str, baudrate: int):
    """Try to open the real GELLO. If unavailable, fall back to a
    deterministic synthetic stream so dry-run / micro can still
    exercise the safety path on machines without /dev/ttyUSB0.
    """
    try:
        from gello.dynamixel.driver import DynamixelDriver
    except ImportError:
        return _SyntheticDriver()

    try:
        d = DynamixelDriver(
            list(range(8)),
            port=port,
            baudrate=baudrate,
            max_retries=1,
            use_fake_fallback=True,  # never hard-fail
        )
        return d
    except Exception as e:  # pragma: no cover - hardware path
        print(f"[WARN] gello open failed: {e}; using synthetic stream")
        return _SyntheticDriver()


class _SyntheticDriver:
    """Deterministic 8D output for dry-run / micro. Produces a
    slow walk so the agent's FK step_norm crosses MAX_STEP after
    a few ticks (worst-case safety rehearsal)."""

    def __init__(self) -> None:
        self._joints = np.zeros(8, dtype=np.float64)
        self._joints[7] = 0.5
        self._t = 0
        self.closed = False

    def get_joints(self) -> np.ndarray:
        # 0.005 rad per tick on joint 1 -> FK step ~0.5mm (sub-threshold
        # for 10 ticks, then the safety check fires).
        self._joints[0] += 0.005
        self._t += 1
        return self._joints.copy()

    def close(self) -> None:
        self.closed = True


def _send_dry(agent: GelloCartesianDeltaAgent, driver, hz: float, log_fp) -> int:
    """Mode: --dry-run. Print the would-be pose; never POST. Always
    safe to run — no approval needed by the driver itself, but the
    shell gate (mode 'dry-run-gello' in 16_*.sh) requires it anyway.
    """
    dt = 1.0 / hz
    # Drive the agent for 5 ticks so the cumulative delta is visible
    # but we deliberately keep it well under MAX_TOTAL_DELTA.
    for tick in range(5):
        joints = driver.get_joints()[:7]
        action, info = agent.step(joints, gripper=0.0)
        log_fp.write(
            f"[DRY] t={tick} joints0={joints[0]:.5f} "
            f"step={info['step_delta_norm']:.6f} "
            f"total={info['total_delta_norm']:.6f} "
            f"safe={info['safe']} action0={action[0]:.5f}\n"
        )
        time.sleep(dt)
    log_fp.flush()
    return 0


def _send_micro(agent: GelloCartesianDeltaAgent, driver, hz: float, log_fp) -> int:
    """Mode: --micro. 3 ticks x 1mm max deflection. The synthetic
    stream grows by 0.5mm per tick so 3 ticks stay under MAX_TOTAL_DELTA
    and under MAX_STEP. Real device would be commanded with explicit
    joint deltas that the FK chain translates to small cartesian moves.
    """
    dt = 1.0 / hz
    for tick in range(MICRO_TICKS):
        joints = driver.get_joints()[:7]
        action, info = agent.step(joints, gripper=0.0)
        log_fp.write(
            f"[MICRO] t={tick} step={info['step_delta_norm']:.6f} "
            f"total={info['total_delta_norm']:.6f} safe={info['safe']}\n"
        )
        if not info["safe"]:
            log_fp.write(f"[MICRO] safety violation: {info['violation']}\n")
            log_fp.flush()
            return 8  # safety violation
        time.sleep(dt)
    log_fp.flush()
    return 0


# FK-bias gate (B1a): if forward_kinematics(q0) disagrees with the server's
# reported current pose by more than this, the in-repo DH FK does not match the
# server's O_T_EE/flange frame, so the first FK'd absolute /pose would command a
# startup jump. Refuse rather than command it (verify/calibrate live in Phase B).
FK_BIAS_LIMIT = 0.005  # meters


def _post(url: str, route: str, body: dict, timeout: float = 5.0):
    import requests
    r = requests.post(url.rstrip("/") + route, json=body, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {}


def _send_full(
    agent: GelloCartesianDeltaAgent, driver, hz: float, duration: float,
    server_url: str, log_fp,
) -> int:
    """Mode: --full. GELLO joint-follow -> FK -> absolute POST /pose, the PROVEN
    record_gello_demos_serl contract (B-RESEARCH.md). Approval already verified
    by the caller. franka_server's only motion command is absolute /pose
    {"arr":[x,y,z,qx,qy,qz,qw]}; there is no joint command route.

    Safety ramp:
      1. read q0 + currpos from POST /getstate
      2. FK-bias gate: ||FK(q0)[:3] - currpos[:3]|| <= FK_BIAS_LIMIT else refuse (rc 11)
      3. per tick: POST /clearerr, then POST /pose {"arr": FK(joint_target)}
    """
    from gello_pose_follow import GelloPoseFollower
    from fk_converter import forward_kinematics

    # 1. Read current robot state to anchor the follow + gate the FK frame.
    try:
        state = _post(server_url, "/getstate", {})
        q0 = np.asarray(state["q"], dtype=np.float64).flatten()[:7]
        currpos = np.asarray(state["pose"], dtype=np.float64).flatten()[:7]
    except Exception as e:
        log_fp.write(f"[FULL] /getstate failed: {e}\n")
        log_fp.flush()
        return 6

    # 2. FK-bias gate — never command a startup jump from a mismatched FK frame.
    fk_q0 = forward_kinematics(q0)
    bias = float(np.linalg.norm(fk_q0[:3] - currpos[:3]))
    if bias > FK_BIAS_LIMIT:
        log_fp.write(
            f"[FULL] REFUSED: FK/EE-frame mismatch: ||FK(q0)-currpos||="
            f"{bias:.4f}m > {FK_BIAS_LIMIT}m. Calibrate FK/flange before live "
            f"motion (B-RESEARCH live-verify). No /pose issued.\n"
        )
        log_fp.flush()
        return 11

    raw_gello0 = np.asarray(driver.get_joints(), dtype=np.float64).flatten()[:7]
    # NOTE (unit hygiene): GelloPoseFollower's max_step / max_total_delta are
    # JOINT-RADIAN gates (faithful to record_gello_demos_serl). They are a
    # DIFFERENT unit system from the driver-level MAX_STEP / MAX_TOTAL_DELTA
    # below, which are Cartesian METERS caps consumed by GelloCartesianDeltaAgent
    # in dry-run/micro. Do NOT pass the meters constants here — use the
    # follower's own radian defaults so the two systems are never conflated.
    follower = GelloPoseFollower(q0=q0, raw_gello0=raw_gello0)
    log_fp.write(
        f"[FULL] start follow: bias={bias:.5f}m q0={np.round(q0,3).tolist()}\n"
    )

    dt = 1.0 / hz
    end = time.monotonic() + duration
    tick = 0
    while time.monotonic() < end:
        raw = driver.get_joints()
        target, abs_pose, ok, info = follower.step(raw)
        log_fp.write(
            f"[FULL] t={tick} step={info['step_delta']:.6f} "
            f"total={info['total_delta']:.6f} safe={ok}\n"
        )
        if not ok:
            log_fp.write(f"[FULL] safety violation: {info['violation']}\n")
            log_fp.flush()
            return 8
        try:
            _post(server_url, "/clearerr", {})
            _post(server_url, "/pose", {"arr": [float(x) for x in abs_pose]})
        except Exception as e:
            log_fp.write(f"[FULL] server rejected /pose: {e}\n")
            log_fp.flush()
            return 9
        tick += 1
        time.sleep(dt)
    log_fp.flush()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="P2-T3 GELLO E2E motion driver")
    p.add_argument("--mode", choices=["dry-run", "micro", "full"], default="dry-run")
    p.add_argument("--server", default="http://127.0.0.1:5000/")
    p.add_argument("--robot-ip", default="172.16.0.2")
    p.add_argument("--gello-port", default="/dev/ttyUSB0")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--log", required=True)
    args = p.parse_args()

    if not _AGENT_AVAILABLE:
        print("[ERR] gello_cartesian_delta_agent not importable", file=sys.stderr)
        return 4

    if args.mode == "full":
        _require_approval()
        if not _REQUESTS_AVAILABLE:
            print("[ERR] requests not installed; cannot run --full", file=sys.stderr)
            return 4

    driver = _open_gello(args.gello_port, baudrate=57600)
    agent = GelloCartesianDeltaAgent(
        max_step=MAX_STEP,
        max_total_delta=MAX_TOTAL_DELTA,
    )
    # Initial seed: prime the agent with the first read so the first
    # step_delta_norm reflects the actual movement from t0.
    seed = driver.get_joints()[:7]
    agent.reset(seed)

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w") as fp:
        fp.write(
            f"[INFO] mode={args.mode} hz={args.hz} duration={args.duration} "
            f"server={args.server} gello_port={args.gello_port}\n"
        )
        if args.mode == "dry-run":
            rc = _send_dry(agent, driver, args.hz, fp)
        elif args.mode == "micro":
            rc = _send_micro(agent, driver, args.hz, fp)
        else:
            rc = _send_full(agent, driver, args.hz, args.duration, args.server, fp)
        fp.write(f"[INFO] driver exited rc={rc}\n")

    try:
        driver.close()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
