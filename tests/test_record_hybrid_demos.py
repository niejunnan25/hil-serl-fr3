"""A3 — record_hybrid_demos contract tests.

Mock-only: a synthetic GELLO stream + a synthetic Xbox state stream
drive the state machine. Verifies:
  * the state machine transitions GELLO_FOLLOW -> XBOX_DELTA on the
    RB rising edge and never reverts
  * the switch tick produces no command jump (continuity)
  * safety checks (max_step / max_total_delta) abort the episode
    and mark it as such (not a success sample)
  * the pkl/npz schema is field-equivalent to record_gello_demos_serl
  * per-step active_device and episode-level switch_step are correct
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import record_hybrid_demos  # noqa: E402
from record_hybrid_demos import (  # noqa: E402
    HybridRecorder,
    HybridStateMachine,
    MODE_GELLO,
    MODE_SWITCH,
    MODE_XBOX,
)
from teleop_hub import XboxState  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_gello_stream(n: int, max_delta: float = 0.001) -> list[np.ndarray]:
    """Synthetic GELLO readings with a slow random walk."""
    rng = np.random.RandomState(0)
    out = [np.array([0.0] * 7 + [0.5])]
    for _ in range(1, n):
        out.append(out[-1].copy())
        out[-1][:7] += rng.randn(7) * max_delta
    return out


def make_xbox_stream(n: int, switch_at: Optional[int], deflect: bool) -> list[XboxState]:
    states = [XboxState(rb=False) for _ in range(n)]
    if switch_at is not None:
        states[switch_at] = XboxState(rb=True)
        for i in range(switch_at + 1, n):
            states[i] = XboxState(rb=True, left_x=0.5 if deflect else 0.0)
    return states


# ---------------------------------------------------------------------------
# State machine unit tests
# ---------------------------------------------------------------------------
class TestStateMachine:
    def test_initial_mode_is_gello(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.003,
            max_total_delta=0.03,
        )
        assert sm.mode == MODE_GELLO
        assert sm.switch_step is None

    def test_gello_step_under_budget(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.05,
            max_total_delta=0.5,
        )
        # Use q0 within the FR3 joint limits so the np.clip doesn't
        # slam joints 4/6 to the rail (which happens when q0 = 0
        # because the lower/upper limits for j4/j6 aren't symmetric
        # around zero).
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        sm.q0 = q0
        sm.prev_target = q0.copy()
        target, ok, _ = sm.gello_step(np.array([0.001] * 7 + [0.5]))
        assert ok
        # 0.001 rad * 0.5 leader_scale = 0.0005 rad on joint 1
        assert target[0] == pytest.approx(q0[0] + 0.0005, abs=1e-6)

    def test_gello_step_aborts_on_max_step(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.001,
            max_total_delta=10.0,
        )
        # 0.01 rad on joint 1 -> 0.005 rad after leader_scale > 0.001 max
        target, ok, delta = sm.gello_step(np.array([0.01] + [0.0] * 6 + [0.5]))
        assert not ok
        assert delta > 0.001

    def test_switch_freezes_anchor_to_actual_joints(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.05,
            max_total_delta=0.5,
        )
        anchor = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        sm.switch_to_xbox(anchor, step_index=42)
        assert sm.mode == MODE_XBOX
        assert sm.switch_step == 42
        assert np.allclose(sm.xbox_q0, anchor)
        assert np.allclose(sm.prev_target, anchor)

    def test_xbox_step_with_no_deflection_keeps_anchor(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.05,
            max_total_delta=0.5,
        )
        # Anchor within the FR3 joint limits so np.clip is a no-op.
        anchor = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        sm.switch_to_xbox(anchor, step_index=10)
        # Centered sticks -> no delta.
        target, ok, _ = sm.xbox_step(XboxState(rb=True))
        assert ok
        assert np.allclose(target, anchor, atol=1e-9)

    def test_maybe_switch_only_on_rising_edge(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.05,
            max_total_delta=0.5,
        )
        # First RB press: switch happens.
        switched = sm.maybe_switch(
            XboxState(rb=True), prev_rb=False, step_index=5, anchor_joints=np.zeros(7)
        )
        assert switched is True
        assert sm.mode == MODE_XBOX
        # Subsequent calls (RB still held) do NOT re-switch.
        switched = sm.maybe_switch(
            XboxState(rb=True), prev_rb=True, step_index=6, anchor_joints=np.zeros(7)
        )
        assert switched is False
        # After release, the mode is still Xbox (one-way).
        switched = sm.maybe_switch(
            XboxState(rb=False), prev_rb=False, step_index=7, anchor_joints=np.zeros(7)
        )
        assert switched is False
        assert sm.mode == MODE_XBOX

    def test_pre_switch_xbox_input_is_ignored(self):
        sm = HybridStateMachine(
            np.array([1, -1, 1, 1, 1, -1, 1]),
            leader_scale=0.5,
            q0=np.zeros(7),
            raw_gello0=np.zeros(7),
            max_step=0.05,
            max_total_delta=0.5,
        )
        # Pre-switch, RB is held, sticks deflected. The state machine
        # must NOT advance to Xbox; the recorder's gello_step output
        # is what gets used.
        anchor = np.zeros(7)
        # Simulate: gello_step is called and then maybe_switch sees
        # RB=True on a subsequent call without a rising edge.
        sm.maybe_switch(
            XboxState(rb=True, left_x=1.0),
            prev_rb=True,  # already held -> not a rising edge
            step_index=0,
            anchor_joints=anchor,
        )
        assert sm.mode == MODE_GELLO


# ---------------------------------------------------------------------------
# End-to-end recorder (mock)
# ---------------------------------------------------------------------------
class TestRecorder:
    def test_run_dry_emits_continuous_trajectory(self, tmp_path):
        n = 100
        gello = make_gello_stream(n, max_delta=0.0005)
        # Switch halfway; deflect during the Xbox segment.
        states = make_xbox_stream(n, switch_at=n // 2, deflect=True)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.05,
            max_total_delta=0.5,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=5.0)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        # Field schema parity with record_gello_demos_serl.
        for key in (
            "joint_poses", "gripper_states", "timestamps", "raw_gello",
            "target", "command", "tracking_error", "poses",
        ):
            assert key in data.files, f"missing field: {key}"
        # New metadata fields.
        assert "active_device" in data.files
        assert "switch_step" in data.files
        assert int(data["switch_step"]) == n // 2
        devices = data["active_device"].tolist()
        # First half is all gello; switch tick is "switch"; rest is xbox.
        for i in range(n // 2):
            assert devices[i] == MODE_GELLO
        assert devices[n // 2] == MODE_SWITCH
        for i in range(n // 2 + 1, len(devices)):
            assert devices[i] == MODE_XBOX

    def test_switch_tick_has_no_jump(self, tmp_path):
        """The target at the switch tick must equal the previous target."""
        n = 60
        gello = make_gello_stream(n, max_delta=0.0002)
        states = make_xbox_stream(n, switch_at=30, deflect=False)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.05,
            max_total_delta=0.5,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=3.0)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        targets = data["target"]
        # The switch tick target (index 30) must equal the previous
        # target (index 29) within numerical noise — the command was
        # frozen to the anchor at the switch instant.
        assert np.allclose(targets[30], targets[29], atol=1e-6), (
            "switch tick produced a command jump"
        )

    def test_max_step_aborts_episode(self, tmp_path):
        n = 20
        gello = make_gello_stream(n, max_delta=0.001)
        # Inject a giant step on tick 10 to trip max_step=0.001.
        gello[10][:7] += 0.05
        states = make_xbox_stream(n, switch_at=None, deflect=False)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.001,
            max_total_delta=10.0,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=1.0)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        assert str(data["abort_reason"]).startswith("gello:max_step")
        # The episode must have stopped before the offending tick.
        assert len(data["joint_poses"]) <= 10

    def test_no_switch_episode_is_entirely_gello(self, tmp_path):
        n = 30
        gello = make_gello_stream(n, max_delta=0.0002)
        states = make_xbox_stream(n, switch_at=None, deflect=False)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.05,
            max_total_delta=0.5,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=1.5)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        devices = data["active_device"].tolist()
        assert all(d == MODE_GELLO for d in devices)
        assert int(data["switch_step"]) == -1

    def test_xbox_segment_records_xbox_gripper_not_gello(self, tmp_path):
        """I4: during the Xbox segment the gripper must come from the Xbox
        RT/LT triggers (RT>0.05 -> +1.0 close, LT>0.05 -> -1.0 open, else 0),
        NOT from the GELLO 8th channel (which is held at 0.5 here).

        The GELLO segment must still record the GELLO channel (0.5).
        """
        n = 40
        switch_at = n // 2
        gello = make_gello_stream(n, max_delta=0.0002)  # 8th channel == 0.5
        # Switch halfway; from the switch tick onward hold RT=1.0 (close).
        states = [XboxState(rb=False) for _ in range(n)]
        states[switch_at] = XboxState(rb=True, rt=1.0)
        for i in range(switch_at + 1, n):
            states[i] = XboxState(rb=True, rt=1.0)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.05,
            max_total_delta=0.5,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=2.0)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        grip = data["gripper_states"]
        devices = data["active_device"].tolist()
        # GELLO ticks keep the GELLO channel value (0.5).
        for i in range(switch_at):
            assert devices[i] == MODE_GELLO
            assert grip[i] == pytest.approx(0.5), (
                f"GELLO tick {i} gripper should be the GELLO channel 0.5"
            )
        # Switch tick + Xbox ticks with RT=1.0 -> +1.0 (close), NOT 0.5.
        for i in range(switch_at, n):
            assert grip[i] == pytest.approx(1.0), (
                f"Xbox tick {i} gripper should be +1.0 from RT, got {grip[i]}"
            )

    def test_xbox_segment_gripper_open_and_neutral(self, tmp_path):
        """I4 follow-up: LT>0.05 -> -1.0 (open); no trigger -> 0.0."""
        n = 30
        switch_at = 10
        gello = make_gello_stream(n, max_delta=0.0002)
        states = [XboxState(rb=False) for _ in range(n)]
        states[switch_at] = XboxState(rb=True, lt=1.0)  # open on switch tick
        for i in range(switch_at + 1, n):
            # Neutral triggers -> gripper 0.0.
            states[i] = XboxState(rb=True)
        rec = HybridRecorder(
            output_dir=str(tmp_path),
            hz=20.0,
            leader_scale=0.5,
            max_step=0.05,
            max_total_delta=0.5,
        )
        q0 = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])
        path = rec.run_dry(gello, states, q0=q0, raw_gello0=np.zeros(7), duration=1.5)
        assert path is not None
        data = np.load(path, allow_pickle=True)
        grip = data["gripper_states"]
        # Switch tick had LT=1.0 -> -1.0 (open).
        assert grip[switch_at] == pytest.approx(-1.0)
        # Subsequent Xbox ticks: no trigger -> 0.0.
        for i in range(switch_at + 1, n):
            assert grip[i] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CLI dry-run self-check (I7)
# ---------------------------------------------------------------------------
class TestCLIDryRun:
    def test_cli_dry_run_saves_a_demo(self, tmp_path):
        """I7: the dry-run CLI must actually produce a saved demo, not a
        silent no-op (which happened when q0 = zeros tripped the joint
        limit clip on tick 0)."""
        script = ROOT / "scripts" / "record_hybrid_demos.py"
        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dry-run",
                "--duration",
                "1",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (
            f"CLI exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
        assert "saved" in proc.stdout.lower(), (
            f"expected a 'saved' message, got stdout:\n{proc.stdout}"
        )
        npz_files = list(tmp_path.glob("hybrid_demo_*.npz"))
        assert npz_files, "no .npz demo was written by the dry-run CLI"
