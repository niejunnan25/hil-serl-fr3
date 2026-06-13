"""B1a — GELLO joint-space follow -> FK -> absolute /pose contract (pure math).

Tests the proven record_gello_demos_serl follow path extracted into a reusable
module: joint target = q0 + raw_delta*joint_signs*leader_scale (clipped to FR3
limits), per-step max_step abort + cumulative max_total_delta abort, then
forward_kinematics(target) -> absolute [x,y,z,qx,qy,qz,qw] for POST /pose.

No network, no real GELLO, no real robot — pure numpy + FK.
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

from gello_pose_follow import (  # noqa: E402
    DEFAULT_JOINT_SIGNS,
    DEFAULT_LEADER_SCALE,
    FR3_DEFAULT_JOINTS,
    GelloPoseFollower,
    joint_target,
)
from fk_converter import forward_kinematics  # noqa: E402


def _raw(q7, gripper=0.5):
    """Build an 8D GELLO reading from 7 joint values."""
    return np.array(list(q7) + [gripper], dtype=float)


class TestJointTarget:
    def test_zero_delta_returns_q0_within_limits(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        tgt = joint_target(q0, raw0, np.zeros(7), DEFAULT_JOINT_SIGNS, DEFAULT_LEADER_SCALE)
        np.testing.assert_allclose(tgt, q0, atol=1e-9)

    def test_joint_signs_and_scale_applied(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        raw = np.zeros(7)
        raw[1] = 1.0  # leader joint 1 moved +1 rad
        tgt = joint_target(q0, raw0, raw, DEFAULT_JOINT_SIGNS, 0.5)
        # joint_signs[1] = -1, leader_scale 0.5 -> q0[1] + (1.0 * -1 * 0.5)
        assert tgt[1] == pytest.approx(q0[1] - 0.5)


class TestFollowerSafety:
    def test_first_step_pose_is_fk_of_q0(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        f = GelloPoseFollower(q0=q0, raw_gello0=raw0)
        cmd, pose, ok, info = f.step(_raw(np.zeros(7)))
        assert ok
        np.testing.assert_allclose(pose, forward_kinematics(q0), atol=1e-9)
        # pose is a valid absolute pose: position (3) + unit quaternion (4)
        assert pose.shape == (7,)
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.0, abs=1e-6)

    def test_per_step_jump_aborts(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        f = GelloPoseFollower(q0=q0, raw_gello0=raw0, max_step=0.003)
        # A 0.5 rad leader jump on joint 0 -> target delta 0.25 rad >> max_step.
        cmd, pose, ok, info = f.step(_raw(np.array([0.5, 0, 0, 0, 0, 0, 0])))
        assert not ok
        assert "max_step" in info["violation"]

    def test_cumulative_drift_aborts(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        f = GelloPoseFollower(q0=q0, raw_gello0=raw0, max_step=0.01, max_total_delta=0.03)
        ok_last = True
        # Creep joint 0 by 0.005 rad/tick (within max_step) until cumulative
        # exceeds max_total_delta=0.03 -> must abort.
        for i in range(1, 30):
            cmd, pose, ok, info = f.step(_raw(np.array([0.005 * i, 0, 0, 0, 0, 0, 0])))
            ok_last = ok
            if not ok:
                assert "max_total_delta" in info["violation"]
                break
        assert ok_last is False

    def test_every_emitted_pose_is_absolute_unit_quat(self):
        q0 = FR3_DEFAULT_JOINTS.copy()
        raw0 = np.zeros(7)
        f = GelloPoseFollower(q0=q0, raw_gello0=raw0, max_step=0.01)
        for i in range(1, 5):
            cmd, pose, ok, info = f.step(_raw(np.array([0.004 * i, 0, 0, 0, 0, 0, 0])))
            assert ok
            assert pose.shape == (7,)
            assert np.linalg.norm(pose[3:]) == pytest.approx(1.0, abs=1e-6)
