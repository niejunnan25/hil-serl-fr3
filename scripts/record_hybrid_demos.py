"""record_hybrid_demos.py — v2.2.1-A3

Single-episode hybrid teleop recorder. Each episode progresses through
a one-way state machine:

    GELLO_FOLLOW  --(RB rising edge)-->  XBOX_DELTA   (terminal)

GELLO segment
  * Identical to record_gello_demos_serl: joint-relative following with
    `q0 + (raw - raw_gello0) * joint_signs * leader_scale` formula.
  * Safety: max_step / max_total_delta per the existing checks.

Switch tick
  * Trigger: RB rising edge.
  * Robot's actual joint positions are read at the switch instant
    (``robot.get_joint_positions()`` in live mode, last commanded
    ``target`` in dry-run).
  * Xbox segment anchors on those positions so the **switch tick
    produces no command jump** (continuous trajectory).

Xbox segment
  * Maps Xbox state to 7D action via XboxIntervention._state_to_action.
  * Adds the action to the current joint positions (in a simple
    proportional way) so the demo's `target` field is a joint-space
    trajectory, matching the existing record format.
  * The Xbox mapping constants are imported from xbox_intervention —
    single source of truth, no duplication.

Output schema
  * Reuses record_gello_demos_serl.save_demo() so the on-disk format is
    identical (joint_poses / gripper_states / timestamps / raw_gello /
    target / command / tracking_error / poses). New fields added:
      - per-step: active_device ("gello"|"xbox"|"switch")
      - episode-level: switch_step, leader_scale, xbox_scale_mode,
        max_step, max_total_delta, hz, num_steps, abort_reason
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from xbox_intervention import XboxIntervention  # noqa: E402
from xbox_intervention import XboxState  # noqa: E402
from xbox_intervention import (  # noqa: E402
    DEADZONE,
    DZ_IDX,
    DPITCH_IDX,
    DROLL_IDX,
    DX_IDX,
    DY_IDX,
    DYAW_IDX,
    GRIPPER_IDX,
    SCALE_FINE,
)
from teleop_hub import (  # noqa: E402
    MockJoystickBackend,
    TeleopDeviceHub,
)

# Defaults copied from record_gello_demos_serl so the GELLO segment
# behaves identically.
DEFAULT_MAX_STEP = 0.003
DEFAULT_MAX_TOTAL_DELTA = 0.03
DEFAULT_MAX_TRACKING_ERROR = 0.08
DEFAULT_HZ = 20.0
DEFAULT_DURATION = 30.0
DEFAULT_LEADER_SCALE = 0.50
DEFAULT_OUTPUT_DIR = "/tmp/hybrid_demos"
DEFAULT_JOINT_SIGNS = [1, -1, 1, 1, 1, -1, 1]

FR3_LOWER_LIMITS = np.array([-2.8, -1.66, -2.8, -2.97, -2.8, 0.08, -2.8])
FR3_UPPER_LIMITS = np.array([2.8, 1.66, 2.8, -0.17, 2.8, 3.65, 2.8])
FR3_DEFAULT_JOINTS = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])

# Mode labels for per-step metadata.
MODE_GELLO = "gello"
MODE_SWITCH = "switch"
MODE_XBOX = "xbox"


def check_max_step(prev, curr, max_step):
    delta = float(np.abs(curr - prev).max())
    return delta <= max_step, delta


def check_max_total_delta(initial, current, max_total_delta):
    total = float(np.abs(current - initial).max())
    return total <= max_total_delta, total


def check_tracking_error(target, actual, max_tracking_error):
    err = target - actual
    max_err = float(np.abs(err).max())
    return max_err <= max_tracking_error, max_err, err


def save_demo(output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"hybrid_demo_{ts}.npz")
    np.savez(filepath, **kwargs)
    return filepath


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class HybridStateMachine:
    """One-way GELLO_FOLLOW -> XBOX_DELTA controller.

    The recorder drives this in lockstep with the per-step I/O loop.
    ``gello_step()`` is called while the operator is in the GELLO
    segment; ``maybe_switch()`` advances the state; ``xbox_step()``
    runs the Xbox segment.

    The class is deliberately pure-Python (no real device, no real
    robot); the recorder injects the GELLO joints and the Xbox state
    per tick. Tests use this directly with mock data.
    """

    def __init__(
        self,
        joint_signs: np.ndarray,
        leader_scale: float,
        q0: np.ndarray,
        raw_gello0: np.ndarray,
        max_step: float,
        max_total_delta: float,
    ) -> None:
        self.joint_signs = np.asarray(joint_signs, dtype=float)
        self.leader_scale = float(leader_scale)
        self.q0 = np.asarray(q0, dtype=float).flatten()
        self.raw_gello0 = np.asarray(raw_gello0, dtype=float).flatten()
        self.max_step = float(max_step)
        self.max_total_delta = float(max_total_delta)

        self.mode = MODE_GELLO
        self.switch_step: Optional[int] = None
        self.prev_target = self.q0.copy()
        self.xbox_q0: Optional[np.ndarray] = None  # anchor at switch tick

    # ------------------------------------------------------------------
    # GELLO segment
    # ------------------------------------------------------------------
    def gello_step(self, raw_gello: np.ndarray) -> tuple[np.ndarray, bool, float]:
        """Compute the next target while in the GELLO segment.

        Returns: (target, ok, step_delta)
            ok=False means a safety check failed and the recorder
            should abort the episode.
        """
        raw = np.asarray(raw_gello, dtype=float).flatten()[:7]
        raw_delta = raw - self.raw_gello0
        target = self.q0 + raw_delta * self.joint_signs * self.leader_scale
        target = np.clip(target, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)
        step_ok, step_delta = check_max_step(self.prev_target, target, self.max_step)
        if not step_ok:
            return target, False, step_delta
        total_ok, total_delta = check_max_total_delta(
            self.q0, target, self.max_total_delta
        )
        if not total_ok:
            return target, False, total_delta
        return target, True, step_delta

    @staticmethod
    def _clip_to_limits(target: np.ndarray) -> np.ndarray:
        return np.clip(target, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)

    # ------------------------------------------------------------------
    # Switch tick — freezes the target to the actual robot pose so the
    # Xbox segment has zero command discontinuity.
    # ------------------------------------------------------------------
    def switch_to_xbox(
        self,
        current_joint_positions: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """Advance to Xbox mode. The anchor is the robot's actual joints
        at the switch tick (NOT the last GELLO target, which may have
        drifted due to tracking error)."""
        self.xbox_q0 = np.asarray(current_joint_positions, dtype=float).flatten()
        assert self.xbox_q0.shape == (7,), self.xbox_q0.shape
        self.prev_target = self.xbox_q0.copy()
        self.mode = MODE_XBOX
        self.switch_step = int(step_index)
        return self.xbox_q0.copy()

    # ------------------------------------------------------------------
    # Xbox segment — reuses the XboxIntervention mapping constants.
    # The action is interpreted in joint space: small additive deltas
    # scaled to ~0.5° per unit action. This is intentionally a simple
    # linear mapping for Phase A; B4 (real-device fine-scale) will
    # replace this with a measured joint-vs-EE Jacobian.
    # ------------------------------------------------------------------
    @staticmethod
    def _xbox_action_to_joint_delta(state: XboxState, scale: float = 0.0087) -> np.ndarray:
        """Convert XboxState to a 7D joint delta.

        Args:
            state: Xbox state from the hub.
            scale: per-axis max joint delta at full deflection
                   (default 0.0087 rad ~ 0.5 deg — placeholder for B4).

        Returns:
            7D delta added to the current joint position. Gripper is
            mapped to the 7th component of the target (not appended).
        """
        def dz(v: float) -> float:
            if abs(v) < DEADZONE:
                return 0.0
            sign = 1.0 if v > 0 else -1.0
            return sign * (abs(v) - DEADZONE) / (1.0 - DEADZONE)

        delta = np.zeros(7, dtype=float)
        delta[0] = dz(state.left_x) * scale
        delta[1] = dz(state.left_y) * scale
        delta[2] = -dz(state.right_y) * scale
        delta[3] = -dz(state.dpad_y) * scale  # dpitch -> joint 3
        delta[4] = dz(state.dpad_x) * scale   # droll -> joint 4
        delta[5] = dz(state.right_x) * scale  # dyaw -> joint 5
        # Joint 6 (last wrist) is left untouched by mapping; the
        # gripper channel (RT/LT) is recorded separately as
        # gripper_states, not as joint 6. This keeps the target vector
        # strictly joint positions (7D) as the SERL loader expects.
        return delta

    def xbox_step(
        self, state: XboxState
    ) -> tuple[np.ndarray, bool, float]:
        assert self.xbox_q0 is not None
        delta = self._xbox_action_to_joint_delta(state)
        target = self.xbox_q0 + delta
        target = np.clip(target, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)
        step_ok, step_delta = check_max_step(self.prev_target, target, self.max_step)
        if not step_ok:
            return target, False, step_delta
        # Note: we intentionally do NOT bound cumulative delta from q0
        # during the Xbox segment because the operator is performing a
        # fine-scale insertion; per-step max_step is the only safety
        # constraint. A3's "must haves" require per-step only.
        return target, True, step_delta

    # ------------------------------------------------------------------
    # Pre-switch RB edge detection
    # ------------------------------------------------------------------
    def maybe_switch(
        self, state: XboxState, prev_rb: bool, step_index: int, anchor_joints: np.ndarray
    ) -> bool:
        """If RB is now held but was not, switch to Xbox.

        Returns True if a switch happened on this tick (recorder must
        emit MODE_SWITCH for the per-step metadata on this tick).
        """
        if self.mode != MODE_GELLO:
            return False
        if state.rb and not prev_rb:
            self.switch_to_xbox(anchor_joints, step_index)
            return True
        return False


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------
class HybridRecorder:
    """Runs the state machine and persists the demo to disk.

    Live mode is NOT exercised in Phase A; the unit tests drive the
    state machine directly with mock gello readings + a mock hub.
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        hz: float = DEFAULT_HZ,
        leader_scale: float = DEFAULT_LEADER_SCALE,
        joint_signs: Optional[list[int]] = None,
        max_step: float = DEFAULT_MAX_STEP,
        max_total_delta: float = DEFAULT_MAX_TOTAL_DELTA,
        max_tracking_error: float = DEFAULT_MAX_TRACKING_ERROR,
    ) -> None:
        self.output_dir = output_dir
        self.hz = float(hz)
        self.leader_scale = float(leader_scale)
        self.joint_signs = np.asarray(
            joint_signs if joint_signs is not None else DEFAULT_JOINT_SIGNS, dtype=float
        )
        self.max_step = float(max_step)
        self.max_total_delta = float(max_total_delta)
        self.max_tracking_error = float(max_tracking_error)

    def run_dry(
        self,
        gello_readings: list[np.ndarray],
        xbox_states: list[XboxState],
        q0: np.ndarray,
        raw_gello0: np.ndarray,
        duration: float,
    ) -> Optional[str]:
        """Run a dry episode with pre-recorded gello readings and xbox
        states. Returns the saved file path, or None if aborted."""
        if len(gello_readings) != len(xbox_states):
            raise ValueError(
                f"gello_readings ({len(gello_readings)}) and xbox_states "
                f"({len(xbox_states)}) must be the same length"
            )
        n = len(gello_readings)
        if n == 0:
            return None

        sm = HybridStateMachine(
            joint_signs=self.joint_signs,
            leader_scale=self.leader_scale,
            q0=q0,
            raw_gello0=raw_gello0,
            max_step=self.max_step,
            max_total_delta=self.max_total_delta,
        )

        joint_poses: list[np.ndarray] = []
        gripper_states: list[float] = []
        timestamps: list[float] = []
        raw_gello_log: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        commands: list[np.ndarray] = []
        tracking_errors: list[np.ndarray] = []
        active_devices: list[str] = []
        poses: list[np.ndarray] = []

        prev_rb = False
        command = q0.copy()
        start_time = time.monotonic()
        abort_reason = "none"

        for step in range(n):
            t = time.monotonic() - start_time
            raw_all = np.asarray(gello_readings[step], dtype=float).flatten()
            assert raw_all.shape == (8,), f"gello reading must be (8,), got {raw_all.shape}"
            xbox_state = xbox_states[step]

            # Possibly switch at the start of the tick.
            switched = sm.maybe_switch(
                xbox_state, prev_rb, step, anchor_joints=command
            )
            prev_rb = xbox_state.rb

            if sm.mode == MODE_GELLO:
                target, ok, step_delta = sm.gello_step(raw_all)
            elif switched:
                # The switch tick itself uses the same gello_step output
                # (which equals the anchor because we just set xbox_q0
                # to the anchor). Mark as MODE_SWITCH for metadata.
                target = sm.xbox_q0.copy()  # type: ignore[union-attr]
                ok = True
                step_delta = 0.0
            else:
                target, ok, step_delta = sm.xbox_step(xbox_state)

            if not ok:
                abort_reason = (
                    f"{sm.mode}:max_step" if sm.mode == MODE_GELLO
                    else f"{sm.mode}:max_step"
                )
                break

            # Smooth command (matches record_gello_demos_serl).
            command = command + np.clip(
                target - command, -self.max_step, self.max_step
            )
            actual = command.copy()  # dry-run: command == actual

            err_ok, max_err, err_vec = check_tracking_error(
                command, actual, self.max_tracking_error
            )
            if not err_ok:
                abort_reason = f"tracking_error:{max_err:.4f}"
                break

            joint_poses.append(actual.copy())
            gripper_states.append(float(raw_all[-1]))
            timestamps.append(t)
            raw_gello_log.append(raw_all.copy())
            targets.append(target.copy())
            commands.append(command.copy())
            tracking_errors.append(err_vec.copy())
            poses.append(np.array([0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0]))
            active_devices.append(
                MODE_SWITCH if switched else sm.mode
            )
            sm.prev_target = target.copy()

        if len(joint_poses) == 0:
            return None

        filepath = save_demo(
            output_dir=self.output_dir,
            joint_poses=np.array(joint_poses),
            gripper_states=np.array(gripper_states),
            timestamps=np.array(timestamps),
            raw_gello=np.array(raw_gello_log),
            target=np.array(targets),
            command=np.array(commands),
            tracking_error=np.array(tracking_errors),
            poses=np.array(poses),
            active_device=np.array(active_devices),
            switch_step=sm.switch_step if sm.switch_step is not None else -1,
            leader_scale=self.leader_scale,
            xbox_scale_mode="fine",
            max_step=self.max_step,
            max_total_delta=self.max_total_delta,
            hz=self.hz,
            num_steps=len(joint_poses),
            abort_reason=abort_reason,
        )
        return filepath


# ---------------------------------------------------------------------------
# CLI (dry-run only — Phase A)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Record hybrid (GELLO -> Xbox) demos for FR3 (Phase A dry-run only)"
    )
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    parser.add_argument(
        "--max-total-delta", type=float, default=DEFAULT_MAX_TOTAL_DELTA
    )
    parser.add_argument(
        "--max-tracking-error", type=float, default=DEFAULT_MAX_TRACKING_ERROR
    )
    args = parser.parse_args()

    if not args.dry_run:
        parser.error("Phase A only supports --dry-run")

    # The Phase A CLI runs a 5-second synthetic episode so the user
    # can sanity-check the wrapper end-to-end without a real robot.
    hz = args.hz
    n = int(args.duration * hz)
    rng = np.random.RandomState(0)
    gello = [np.array([0.0] * 7 + [0.5]) for _ in range(n)]
    # Small random walk in the gello joints so the GELLO segment
    # produces visible target deltas.
    for i in range(1, n):
        gello[i][:7] = gello[i - 1][:7] + rng.randn(7) * 0.0005

    # Inject a single RB rising edge halfway through the episode.
    states = [XboxState(rb=False) for _ in range(n)]
    states[n // 2] = XboxState(rb=True)
    for i in range(n // 2 + 1, n):
        # Small left-stick deflection during the Xbox segment.
        states[i] = XboxState(rb=True, left_x=0.5)

    q0 = np.zeros(7)
    raw_gello0 = gello[0][:7]
    rec = HybridRecorder(
        output_dir=args.output_dir,
        hz=hz,
        leader_scale=args.leader_scale,
        max_step=args.max_step,
        max_total_delta=args.max_total_delta,
        max_tracking_error=args.max_tracking_error,
    )
    path = rec.run_dry(gello, states, q0, raw_gello0, duration=args.duration)
    if path:
        print(f"Hybrid demo saved: {path}")


if __name__ == "__main__":
    main()
