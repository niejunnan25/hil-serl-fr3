"""p2_t3_e2e_motion_driver.py — GELLO -> FK -> Cartesian delta -> POST /pose.

Used by 16_gello_e2e_motion_test.sh mode 4 (full E2E motion stream).
Phase A only exercises the driver in --dry-run and --micro modes;
--full is gated by the shell's approval env var and never invoked
without an operator in the loop with the E-stop.

The driver is intentionally minimal: it opens GELLO, reads joints at
HZ, and feeds them through GelloCartesianDeltaAgent. dry-run / micro
exercise the FK + safety pipeline without touching the robot.

--full FAILS CLOSED (returns rc 10, no /pose) pending Phase B: the
agent emits a normalized [-1,1] delta but /pose expects an absolute
pose, so the GELLO-leader -> FR3-follower pose reconstruction must be
implemented and verified against the live server contract in Phase B
(B1) before any motion is streamed. See REVIEW-PhaseA C2.
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


def _send_full(
    agent: GelloCartesianDeltaAgent, driver, hz: float, duration: float,
    server_url: str, log_fp,
) -> int:
    """Mode: --full. FAIL CLOSED — never POST (REVIEW-PhaseA C2).

    The agent emits a NORMALIZED 7D delta action [dx,dy,dz,droll,dpitch,
    dyaw,gripper] in [-1,1], but franka_server's /pose endpoint expects an
    ABSOLUTE pose [x,y,z,qx,qy,qz,qw]. POSTing the normalized delta as an
    absolute pose would drive the EE toward the origin with a non-unit
    quaternion — a large, uncontrolled real-robot motion.

    The correct GELLO-leader -> FR3-follower absolute-pose reconstruction
    (read the follower's current pose, apply the raw Cartesian delta,
    compose a valid unit-quaternion target) must be implemented and
    verified against the LIVE /pose contract in Phase B (B1), with the
    robot + operator + E-stop in the loop. Until then full mode refuses
    to issue any motion command.
    """
    log_fp.write(
        "[FULL] DISABLED: GELLO->follower absolute /pose conversion pending "
        "Phase B server-contract verification (REVIEW-PhaseA C2); no /pose issued\n"
    )
    log_fp.flush()
    return 10


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
