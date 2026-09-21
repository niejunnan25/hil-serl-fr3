"""Xbox-only insert demo recorder pure-function tests."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from teleop_hub import XboxState  # noqa: E402


def _currpos():
    return np.array([0.65, -0.01, 0.15, 0.0, 0.0, 0.0, 1.0], dtype=float)


class TestXboxOnlyCommand:
    def test_released_deadman_holds_pose_and_records_hold_grip_action(self):
        from xbox_demo_recorder import XboxOnlyInsertState, compute_xbox_insert_command

        state = XboxOnlyInsertState(z_target=0.15)
        cmd = compute_xbox_insert_command(
            _currpos(),
            XboxState(left_x=1.0, left_y=-1.0, right_y=1.0, rb=False, rt=1.0, lt=1.0),
            state,
            max_step=0.003,
            dt=0.1,
            hold_grip_action=0.0,
        )

        np.testing.assert_allclose(cmd.nextpos, _currpos())
        np.testing.assert_allclose(cmd.action[:6], np.zeros(6), atol=1e-7)
        assert cmd.action[6] == pytest.approx(0.0)
        assert cmd.gripper_command is None
        assert cmd.param_update is None

    def test_rb_left_stick_moves_xy_without_gripper_command(self):
        from xbox_demo_recorder import XboxOnlyInsertState, compute_xbox_insert_command

        state = XboxOnlyInsertState(z_target=0.15)
        cmd = compute_xbox_insert_command(
            _currpos(),
            XboxState(left_x=1.0, left_y=0.0, rb=True, rt=1.0, lt=1.0),
            state,
            max_step=0.003,
            dt=0.1,
            hold_grip_action=0.0,
        )

        expected = _currpos()
        expected[1] += 0.003
        np.testing.assert_allclose(cmd.nextpos[:3], expected[:3], atol=1e-9)
        assert cmd.action[1] == pytest.approx(0.003 / 0.015)
        assert cmd.action[6] == pytest.approx(0.0)
        assert cmd.gripper_command is None

    def test_insert_button_uses_insert_force_cap_and_downward_target(self):
        from hybrid_teleop import XBOX_INSERT_CLIP, XBOX_INSERT_REACH
        from xbox_demo_recorder import XboxOnlyInsertState, compute_xbox_insert_command

        state = XboxOnlyInsertState(z_target=0.15)
        cmd = compute_xbox_insert_command(
            _currpos(),
            XboxState(rb=True, a=True),
            state,
            max_step=0.003,
            dt=0.1,
            hold_grip_action=0.0,
        )

        assert cmd.param_update == {
            "translational_clip_z": XBOX_INSERT_CLIP,
            "translational_clip_neg_z": XBOX_INSERT_CLIP,
        }
        assert cmd.nextpos[2] == pytest.approx(_currpos()[2] - XBOX_INSERT_REACH)
        assert cmd.action[6] == pytest.approx(0.0)

    def test_insert_release_restores_hold_force_cap(self):
        from hybrid_teleop import XBOX_HOLD_CLIP
        from xbox_demo_recorder import XboxOnlyInsertState, compute_xbox_insert_command

        state = XboxOnlyInsertState(z_target=0.15, insert_prev=True)
        cmd = compute_xbox_insert_command(
            _currpos(),
            XboxState(rb=True, a=False),
            state,
            max_step=0.003,
            dt=0.1,
            hold_grip_action=0.0,
        )

        assert cmd.param_update == {
            "translational_clip_z": XBOX_HOLD_CLIP,
            "translational_clip_neg_z": XBOX_HOLD_CLIP,
        }
        assert cmd.gripper_command is None

