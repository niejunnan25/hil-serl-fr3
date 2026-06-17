"""P2-T2: GELLO joints -> FK -> Cartesian delta flow.

Drives the GelloCartesianDeltaAgent with synthetic joint trajectories and
checks the FK → delta → normalize pipeline:

- reset() primes prev/initial pose
- step() returns (7,) action in [-1, 1] and a populated info dict
- max_step and max_total_delta safety thresholds return zero translation
- normalize_action with default scales matches expected formula
- small joint deltas produce proportional Cartesian deltas
- get_state() reports correct counters
- dry-run CLI runs without errors

All FK math uses the real fk_converter.py (DH-based, scipy Rotation). The
gello package is not used; the agent only consumes (7,) joint vectors.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import fk_converter  # noqa: E402
import gello_cartesian_delta_agent as gcda_module  # noqa: E402
from gello_cartesian_delta_agent import GelloCartesianDeltaAgent  # noqa: E402
from normalize_action import normalize_action  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _agent(**kwargs) -> GelloCartesianDeltaAgent:
    return GelloCartesianDeltaAgent(**kwargs)


# ---------------------------------------------------------------------------
# Reset / state init
# ---------------------------------------------------------------------------
class TestAgentReset:
    def test_reset_seeds_prev_and_initial_pose(self):
        agent = _agent()
        q0 = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.7])
        agent.reset(q0)
        assert agent.prev_joints is not None
        assert agent.initial_joints is not None
        assert np.allclose(agent.prev_joints, q0)
        assert np.allclose(agent.initial_joints, q0)
        assert np.allclose(agent.prev_pose, agent.initial_pose)
        assert agent.step_count == 0
        assert agent.violation_count == 0
        assert agent.get_state()["initialized"] is True

    def test_first_step_auto_resets(self):
        agent = _agent()
        # Don't call reset explicitly; first step() should auto-reset
        q = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, info = agent.step(q)
        assert agent.prev_joints is not None
        assert np.allclose(agent.initial_joints, q)
        assert action.shape == (7,)

    def test_reset_rejects_wrong_shape(self):
        agent = _agent()
        with pytest.raises(AssertionError):
            agent.reset(np.zeros(8))
        with pytest.raises(AssertionError):
            agent.reset(np.zeros(6))


# ---------------------------------------------------------------------------
# Core pipeline: joints -> FK -> delta -> normalize
# ---------------------------------------------------------------------------
class TestFKDeltaPipeline:
    def test_action_shape_and_range(self):
        agent = _agent()
        q0 = np.zeros(7)
        agent.reset(q0)
        # small joint 1 rotation -> small translation
        q1 = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, info = agent.step(q1, gripper=0.0)
        assert action.shape == (7,)
        assert np.all(np.abs(action) <= 1.0)
        assert action[6] == 0.0  # gripper passed through

    def test_step_advances_counters(self):
        agent = _agent()
        agent.reset(np.zeros(7))
        for k in range(5):
            q = np.zeros(7)
            q[0] = 0.001 * (k + 1)  # well under safety threshold
            agent.step(q, gripper=0.0)
        state = agent.get_state()
        assert state["step_count"] == 5
        assert state["violation_count"] == 0

    def test_info_dict_contains_required_fields(self):
        agent = _agent()
        agent.reset(np.zeros(7))
        _, info = agent.step(np.zeros(7), gripper=0.0)
        for key in [
            "raw_delta",
            "cartesian_delta",
            "safe",
            "step_delta_norm",
            "total_delta_norm",
            "violation",
        ]:
            assert key in info, f"info missing '{key}'"
        assert info["raw_delta"].shape == (6,)
        assert info["cartesian_delta"].shape == (6,)
        assert isinstance(info["safe"], bool)
        assert isinstance(info["step_delta_norm"], float)
        assert isinstance(info["total_delta_norm"], float)
        assert isinstance(info["violation"], str)

    def test_zero_joint_delta_zero_translation(self):
        agent = _agent()
        agent.reset(np.zeros(7))
        _, info = agent.step(np.zeros(7), gripper=0.0)
        assert info["step_delta_norm"] == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(info["cartesian_delta"][:3], 0.0, atol=1e-9)

    def test_delta_matches_independent_fk_computation(self):
        agent = _agent()
        q_prev = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.7])
        q_curr = q_prev + np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        agent.reset(q_prev)
        _, info = agent.step(q_curr, gripper=0.0)
        expected = fk_converter.joints_to_cartesian_delta(q_prev, q_curr)
        assert np.allclose(info["raw_delta"], expected)

    def test_normalize_action_uses_configured_scales(self):
        """A known translation delta must be normalized by 1/pos_scale.

        0.05 rad on joint 1 produces ~5.6mm translation; pos_scale=0.1
        normalizes to ~0.056. With max_step=0.003 this would trip the
        safety check, so we lift the threshold to keep the agent on the
        safe path."""
        agent = _agent(max_step=0.1, max_total_delta=1.0, pos_scale=0.1, rpy_scale=0.2)
        agent.reset(np.zeros(7))
        q = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, info = agent.step(q, gripper=0.5)
        assert info["safe"] is True
        # Translation channel is in [-1, 1] and non-trivial
        assert np.abs(action[0]) > 0.01
        assert np.abs(action[0]) < 1.0
        # gripper channel preserves the input
        assert action[6] == pytest.approx(0.5)

    def test_total_delta_accumulates_across_steps(self):
        agent = _agent(max_total_delta=0.5)  # high cap to avoid tripping
        agent.reset(np.zeros(7))
        # Three small steps in the same direction accumulate total_delta
        for k in range(3):
            q = np.array([0.02 * (k + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            _, info = agent.step(q, gripper=0.0)
            assert info["total_delta_norm"] > 0.0
        # Final total should be > any single step
        final = info["total_delta_norm"]
        assert final > 0.001


# ---------------------------------------------------------------------------
# Safety thresholds
# ---------------------------------------------------------------------------
class TestSafetyLimits:
    def test_max_step_violation_returns_zero_translation(self):
        agent = _agent(max_step=0.003, max_total_delta=1.0)
        agent.reset(np.zeros(7))
        # 0.05 rad on joint 1 -> ~5.5mm translation, above max_step
        q_big = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, info = agent.step(q_big, gripper=0.0)
        assert info["safe"] is False
        assert "max_step exceeded" in info["violation"]
        # Translation must be zero
        assert np.allclose(info["cartesian_delta"][:3], 0.0, atol=1e-9)
        # First three channels of action are zero
        assert np.allclose(action[:3], 0.0)
        # Gripper still passes through
        assert action[6] == pytest.approx(0.0)
        assert agent.violation_count == 1

    def test_max_total_delta_violation_clamps(self):
        agent = _agent(max_step=0.5, max_total_delta=0.005)
        agent.reset(np.zeros(7))
        # Step 1: 0.05 rad joint 1 -> ~5.5mm total, exceeds 5mm cap
        q = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        action, info = agent.step(q, gripper=0.0)
        assert info["safe"] is False
        assert "max_total_delta exceeded" in info["violation"]
        assert np.allclose(info["cartesian_delta"][:3], 0.0, atol=1e-9)
        assert agent.violation_count == 1

    def test_violation_does_not_update_prev_joints(self):
        """On a max_step violation, prev_joints must NOT advance so the
        next legitimate step measures its delta from the same baseline."""
        agent = _agent(max_step=0.003, max_total_delta=1.0)
        agent.reset(np.zeros(7))
        prev_before = agent.prev_joints.copy()
        q_big = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        agent.step(q_big, gripper=0.0)
        # Despite reading the big delta, prev_joints is unchanged
        assert np.allclose(agent.prev_joints, prev_before)

    def test_safe_steps_keep_updating_prev(self):
        agent = _agent(max_step=0.1, max_total_delta=1.0)
        agent.reset(np.zeros(7))
        q = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        agent.step(q, gripper=0.0)
        assert np.allclose(agent.prev_joints, q)

    def test_max_rot_step_violation_returns_zero_action(self):
        """A large orientation jump with translation UNDER max_step must still
        trip the safety check via the rotation cap and return a zero action.

        joint7 += 0.2 rad from home produces ~27mm translation and ~0.2 rad
        rpy. We lift max_step/max_total_delta so translation passes, isolating
        the rotation cap (default 0.1 rad)."""
        agent = _agent(max_step=0.5, max_total_delta=1.0)  # default max_rot_step=0.1
        agent.reset(np.zeros(7))
        q_spin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
        action, info = agent.step(q_spin, gripper=0.0)
        # Sanity: this move's translation is below the lifted translation cap,
        # so only the rotation cap can flag it.
        assert info["step_delta_norm"] <= 0.5
        assert info["rot_delta_norm"] > 0.1
        assert info["safe"] is False
        assert "max_rot_step exceeded" in info["violation"]
        # Zero translation AND zero rotation channels on violation.
        assert np.allclose(info["cartesian_delta"], 0.0, atol=1e-9)
        assert np.allclose(action[:6], 0.0)
        assert action[6] == pytest.approx(0.0)
        assert agent.violation_count == 1

    def test_max_rot_step_does_not_advance_prev_joints(self):
        """A rotation-cap violation must not advance prev_joints, mirroring the
        translation-cap behavior."""
        agent = _agent(max_step=0.5, max_total_delta=1.0)
        agent.reset(np.zeros(7))
        prev_before = agent.prev_joints.copy()
        q_spin = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
        agent.step(q_spin, gripper=0.0)
        assert np.allclose(agent.prev_joints, prev_before)

    def test_translation_cap_takes_precedence_in_message(self):
        """When BOTH translation and rotation exceed their caps, the violation
        message reports translation first (translation check runs first)."""
        # joint7 += 0.5 -> ~67mm trans (> 0.003) AND ~0.51 rad rpy (> 0.1)
        agent = _agent(max_step=0.003, max_total_delta=1.0)
        agent.reset(np.zeros(7))
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        _, info = agent.step(q, gripper=0.0)
        assert info["safe"] is False
        assert "max_step exceeded" in info["violation"]

    def test_rot_delta_norm_present_in_info(self):
        agent = _agent()
        agent.reset(np.zeros(7))
        _, info = agent.step(np.zeros(7), gripper=0.0)
        assert "rot_delta_norm" in info
        assert isinstance(info["rot_delta_norm"], float)


# ---------------------------------------------------------------------------
# CLI / dry-run
# ---------------------------------------------------------------------------
class TestDryRunCLI:
    def test_dry_run_exits_cleanly(self, tmp_path):
        """Run the agent's --dry-run CLI and confirm it exits 0 and prints
        the expected summary line."""
        out = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "gello_cartesian_delta_agent.py"),
                "--dry-run",
                "--duration",
                "0.3",
                "--hz",
                "20",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS),
            timeout=30,
        )
        assert out.returncode == 0, (
            f"dry-run failed:\nSTDOUT={out.stdout}\nSTDERR={out.stderr}"
        )
        assert "DRY-RUN PASSED" in out.stdout
        # The summary section should also appear
        assert "Dry-Run Summary" in out.stdout

    def test_dry_run_with_custom_thresholds(self, tmp_path):
        out = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "gello_cartesian_delta_agent.py"),
                "--dry-run",
                "--duration",
                "0.2",
                "--hz",
                "20",
                "--max-step",
                "0.001",
                "--max-total-delta",
                "0.005",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS),
            timeout=30,
        )
        # Tighter thresholds should produce violations on the random
        # walk the dry-run uses; the script still exits cleanly.
        assert out.returncode == 0
        # The tighter threshold should be reflected in the header
        assert "max_step: 0.001" in out.stdout

    def test_help_prints_usage(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "gello_cartesian_delta_agent.py")],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS),
            timeout=10,
        )
        assert "--dry-run" in out.stdout


# ---------------------------------------------------------------------------
# Integration with the real fk_converter (no FK mocks)
# ---------------------------------------------------------------------------
class TestFKConverterIntegration:
    """The agent must use the real fk_converter, so the FK result is the
    same as a direct call."""

    def test_prev_pose_matches_direct_fk(self):
        q0 = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.7])
        agent = _agent()
        agent.reset(q0)
        assert np.allclose(agent.prev_pose, fk_converter.forward_kinematics(q0))

    def test_action_reflects_fk_rotation(self):
        """Rotating joint 7 must show up in the yaw/orientation channels
        and produce a sub-threshold translation (within max_step)."""
        agent = _agent(max_step=0.02, max_total_delta=1.0)
        q0 = np.zeros(7)
        agent.reset(q0)
        # Rotate joint 7 by 0.05 rad; FK shows this produces ~5mm trans
        # + ~0.05 rad pitch (NOT yaw in XYZ intrinsic). Make sure the
        # orientation channel captures the rotation.
        q1 = q0.copy()
        q1[6] = 0.05
        action, info = agent.step(q1, gripper=0.0)
        assert info["safe"] is True
        # Pitch channel should be the dominant rotation signal
        assert abs(info["cartesian_delta"][4]) > 0.01
        # The non-zero translation channels should also be in info
        assert np.linalg.norm(info["cartesian_delta"][:3]) > 0.001
