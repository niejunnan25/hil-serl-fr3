"""B1b/B2 — relative-pose teleop math (FK-free).

The in-repo absolute FK is ~50 cm off vs the server's O_T_EE frame (proven live),
so GELLO/Xbox teleop must drive the robot RELATIVELY: read the robot's own
currpos (/getstate) and command currpos + a small Cartesian delta to /pose.

- Xbox: stick -> normalized action -> de-normalized Cartesian delta.
- GELLO: joint delta -> robot Jacobian (from /getstate) -> Cartesian twist.

Both feed apply_cartesian_delta(currpos, dxyz, drotvec, max_step). These pure
functions are unit-tested here; the live /getstate + POST /pose loop is
dry-run validated against the real robot (no motion) separately.
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

from relative_teleop import apply_cartesian_delta, gello_twist  # noqa: E402

# A valid absolute pose: position + scalar-last unit quaternion (identity).
CURR = np.array([0.30, 0.00, 0.48, 0.0, 0.0, 0.0, 1.0])


def _unit(q):
    return float(np.linalg.norm(q))


class TestApplyCartesianDelta:
    def test_zero_delta_is_noop(self):
        nextpos, step = apply_cartesian_delta(CURR, np.zeros(3), np.zeros(3), max_step=0.003)
        np.testing.assert_allclose(nextpos, CURR, atol=1e-9)
        assert step == pytest.approx(0.0)

    def test_small_translation_added_exactly(self):
        nextpos, step = apply_cartesian_delta(CURR, np.array([0.001, 0.0, -0.002]), np.zeros(3), max_step=0.003)
        np.testing.assert_allclose(nextpos[:3], CURR[:3] + np.array([0.001, 0.0, -0.002]), atol=1e-9)
        assert _unit(nextpos[3:]) == pytest.approx(1.0, abs=1e-9)
        assert step == pytest.approx(np.linalg.norm([0.001, 0.0, -0.002]))

    def test_over_max_translation_is_clipped_to_max_step(self):
        # 10 mm requested, cap 3 mm -> applied norm exactly 3 mm, direction kept.
        big = np.array([0.010, 0.0, 0.0])
        nextpos, step = apply_cartesian_delta(CURR, big, np.zeros(3), max_step=0.003)
        applied = nextpos[:3] - CURR[:3]
        assert np.linalg.norm(applied) == pytest.approx(0.003, abs=1e-9)
        assert step == pytest.approx(0.003, abs=1e-9)
        # direction preserved (still +x)
        assert applied[0] > 0 and abs(applied[1]) < 1e-12 and abs(applied[2]) < 1e-12

    def test_rotation_keeps_unit_quaternion_and_rotates(self):
        nextpos, _ = apply_cartesian_delta(CURR, np.zeros(3), np.array([0.0, 0.0, 0.05]), max_step=0.003)
        assert _unit(nextpos[3:]) == pytest.approx(1.0, abs=1e-9)
        # a nonzero rotvec must change the orientation
        assert not np.allclose(nextpos[3:], CURR[3:])


class TestGelloTwist:
    def test_jacobian_maps_joint_delta_to_twist(self):
        # Jacobian with joint0 -> +x linear velocity, joint1 -> +z.
        J = np.zeros((6, 7))
        J[0, 0] = 1.0   # x from joint0
        J[2, 1] = 1.0   # z from joint1
        dq_gello = np.zeros(7)
        dq_gello[0] = 0.01
        dq_gello[1] = 0.02
        signs = np.ones(7)
        dxyz, drot = gello_twist(dq_gello, signs, leader_scale=1.0, jacobian=J)
        assert dxyz[0] == pytest.approx(0.01)
        assert dxyz[2] == pytest.approx(0.02)
        np.testing.assert_allclose(drot, np.zeros(3), atol=1e-12)

    def test_joint_signs_and_leader_scale_applied(self):
        J = np.zeros((6, 7))
        J[1, 3] = 1.0   # y from joint3
        dq_gello = np.zeros(7)
        dq_gello[3] = 0.1
        signs = np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)
        dxyz, _ = gello_twist(dq_gello, signs, leader_scale=0.5, jacobian=J)
        # dq_fr3[3] = 0.1 * -1 * 0.5 = -0.05 -> y = -0.05
        assert dxyz[1] == pytest.approx(-0.05)
