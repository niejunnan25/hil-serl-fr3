"""v2.2.1 Route E + hybrid arbiter — pure-function tests (TDD).

Route E GELLO following = joint-space anchor + CORRECT FK, commanded as an
absolute EE /pose to the existing serl cartesian_impedance controller:

  q_gello_est = q0_robot + (raw_gello - raw_gello0) * joint_signs * leader_scale
  desired_pose = correct_fk(q_gello_est)              # pinocchio + flange->TCP T_offset
  -> apply_cartesian_delta(currpos, desired-currpos, clamp) -> POST /pose

Anchoring (raw_gello0, q0_robot captured at activation) makes the robot mirror
the GELLO's MOTION 1:1 with no jump on takeover. correct_fk needs pinocchio
(desktop env) so it is validated live, not unit-tested here.

Hybrid arbiter: GELLO mode <-> Xbox mode via an edge-triggered toggle button;
when Xbox is active the GELLO is ignored (no interference), and switching back
to GELLO re-anchors (no jump).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from hybrid_teleop import (  # noqa: E402
    RESET_JOINT_TARGET,
    arbiter_step,
    gello_joint_target,
    home_reached,
)


class TestGelloJointTarget:
    """q_target = q0_robot + (raw - raw0) * joint_signs * leader_scale."""

    def test_at_anchor_returns_robot_baseline(self):
        raw0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0])
        q0 = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
        signs = np.array([1, -1, 1, 1, 1, -1, 1])
        q = gello_joint_target(raw0, raw0, q0, signs, leader_scale=1.0)
        np.testing.assert_allclose(q, q0, atol=1e-12)
        assert q.shape == (7,)

    def test_signed_delta_added_to_baseline(self):
        raw0 = np.zeros(8)
        raw = np.zeros(8)
        raw[0] = 0.10   # joint0 +0.10, sign +1 -> +0.10
        raw[1] = 0.10   # joint1 +0.10, sign -1 -> -0.10
        q0 = np.zeros(7)
        signs = np.array([1, -1, 1, 1, 1, -1, 1])
        q = gello_joint_target(raw, raw0, q0, signs, leader_scale=1.0)
        assert q[0] == pytest.approx(0.10)
        assert q[1] == pytest.approx(-0.10)
        np.testing.assert_allclose(q[2:], np.zeros(5), atol=1e-12)

    def test_leader_scale_scales_delta(self):
        raw0 = np.zeros(8)
        raw = np.zeros(8)
        raw[0] = 0.20
        q0 = np.full(7, 0.5)
        signs = np.ones(7)
        q = gello_joint_target(raw, raw0, q0, signs, leader_scale=0.5)
        assert q[0] == pytest.approx(0.5 + 0.20 * 0.5)  # 0.6

    def test_only_first_7_joints_used(self):
        # raw is 8D (7 joints + gripper); gripper channel must not affect q
        raw0 = np.zeros(8)
        raw = np.zeros(8)
        raw[7] = 5.0  # gripper channel changes
        q = gello_joint_target(raw, raw0, np.zeros(7), np.ones(7), leader_scale=1.0)
        np.testing.assert_allclose(q, np.zeros(7), atol=1e-12)


class TestArbiterStep:
    """Edge-triggered toggle: rising edge flips GELLO<->XBOX; held/released do not."""

    def test_rising_edge_toggles_gello_to_xbox(self):
        mode, switched = arbiter_step("gello", toggle_now=True, toggle_prev=False)
        assert mode == "xbox" and switched is True

    def test_rising_edge_toggles_xbox_to_gello(self):
        mode, switched = arbiter_step("xbox", toggle_now=True, toggle_prev=False)
        assert mode == "gello" and switched is True

    def test_held_does_not_retoggle(self):
        mode, switched = arbiter_step("gello", toggle_now=True, toggle_prev=True)
        assert mode == "gello" and switched is False

    def test_released_does_not_toggle(self):
        mode, switched = arbiter_step("xbox", toggle_now=False, toggle_prev=True)
        assert mode == "xbox" and switched is False

    def test_idle_no_press(self):
        mode, switched = arbiter_step("gello", toggle_now=False, toggle_prev=False)
        assert mode == "gello" and switched is False


class TestHomeReached:
    """home_reached guards reset success (server /jointreset can silently no-op)."""

    def test_at_home_true(self):
        assert home_reached(RESET_JOINT_TARGET) is True

    def test_within_tol_true(self):
        q = np.asarray(RESET_JOINT_TARGET) + np.array([0.1, -0.1, 0.05, 0.0, 0.0, 0.1, -0.05])
        assert home_reached(q, tol=0.15) is True

    def test_far_from_home_false(self):
        # the actual failed-reset q from the live bug
        q = np.array([0.02, -0.001, 0.282, -2.157, -0.336, 1.033, 1.095])
        assert home_reached(q, tol=0.15) is False

    def test_one_joint_over_tol_false(self):
        q = np.asarray(RESET_JOINT_TARGET).copy()
        q[6] += 0.3
        assert home_reached(q, tol=0.15) is False
