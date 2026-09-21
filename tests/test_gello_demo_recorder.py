"""v2.2.1 — GELLO full-process demo recorder pure-function tests (TDD).

The recorder drives the FR3 via the relative-pose GELLO path (reusing
relative_teleop) and records, per tick, the full /getstate, both ZED frames,
the action, and timestamps. On stop it assembles the SERL transition format
that the project contract (sim/data/contract.py) + replay buffer expect:

  observations{state(25,)f32, side_policy/wrist_1/side_classifier (3,128,128)u8 RGB}
  next_observations{...}, actions(7,)f32 in [-1,1], rewards, masks, dones.

The pure assembly/normalization is unit-tested here; the live loop (threaded
ZED capture + /getstate + POST /pose) is validated against the real robot.
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

from gello_demo_recorder import (  # noqa: E402
    obs_state,
    normalize_action,
    gripper_action_for_recording,
    image_to_obs,
    build_transitions,
    ACTION_SCALE,
)


class TestObsState:
    """25D = tcp_pose(7)+tcp_vel(6)+tcp_force(3)+tcp_torque(3)+gripper(6), contract order."""

    def test_concatenates_25d_in_contract_order(self):
        pose = np.arange(7, dtype=float)
        vel = np.arange(7, 13, dtype=float)
        force = np.array([100.0, 101.0, 102.0])
        torque = np.array([200.0, 201.0, 202.0])
        grip = 0.55
        s = obs_state(pose, vel, force, torque, grip)
        assert s.shape == (25,)
        assert s.dtype == np.float32
        np.testing.assert_allclose(s[0:7], pose)
        np.testing.assert_allclose(s[7:13], vel)
        np.testing.assert_allclose(s[13:16], force)
        np.testing.assert_allclose(s[16:19], torque)
        np.testing.assert_allclose(s[19:25], np.full(6, grip))  # gripper scalar tiled 6x

    def test_gripper_scalar_or_array_both_ok(self):
        s = obs_state(np.zeros(7), np.zeros(6), np.zeros(3), np.zeros(3), np.array([0.08]))
        np.testing.assert_allclose(s[19:25], np.full(6, 0.08))


class TestNormalizeAction:
    """[dx,dy,dz,droll,dpitch,dyaw,grip], xyz/0.015, rpy/0.1, clip [-1,1]."""

    def test_translation_scaled_by_action_scale(self):
        a = normalize_action(np.array([0.015, 0.0, -0.015]), np.zeros(3), -1.0)
        assert a.dtype == np.float32
        assert a[0] == pytest.approx(1.0)
        assert a[2] == pytest.approx(-1.0)

    def test_rotation_scaled_by_action_scale(self):
        a = normalize_action(np.zeros(3), np.array([0.1, 0.0, -0.05]), -1.0)
        assert a[3] == pytest.approx(1.0)
        assert a[5] == pytest.approx(-0.5)

    def test_over_range_is_clipped_to_unit(self):
        a = normalize_action(np.array([0.030, 0.0, 0.0]), np.zeros(3), 5.0)
        assert a[0] == pytest.approx(1.0)  # 0.03/0.015 = 2 -> clip 1
        assert a[6] == pytest.approx(1.0)  # gripper 5 -> clip 1

    def test_gripper_passthrough_both_signs(self):
        assert normalize_action(np.zeros(3), np.zeros(3), 1.0)[6] == pytest.approx(1.0)
        assert normalize_action(np.zeros(3), np.zeros(3), -1.0)[6] == pytest.approx(-1.0)

    def test_hold_grip_action_for_insert_only_demos(self):
        assert gripper_action_for_recording(True, hold_grip_action=0.0) == pytest.approx(0.0)
        assert gripper_action_for_recording(False, hold_grip_action=0.0) == pytest.approx(0.0)
        assert gripper_action_for_recording(True, hold_grip_action=None) == pytest.approx(-1.0)
        assert gripper_action_for_recording(False, hold_grip_action=None) == pytest.approx(1.0)

    def test_action_scale_constant_matches_contract(self):
        np.testing.assert_allclose(
            np.asarray(ACTION_SCALE, float), [0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0]
        )


class TestImageToObs:
    """ZED BGR HWC uint8 -> (3,128,128) uint8 CHW RGB (B<->R swapped by default)."""

    def test_shape_dtype_chw(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = image_to_obs(frame, size=128)
        assert out.shape == (3, 128, 128)
        assert out.dtype == np.uint8

    def test_bgr_to_rgb_swaps_channels_by_default(self):
        # ZED gives BGR: channel 0 = Blue. After BGR->RGB, Blue must land in out[2].
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[..., 0] = 255  # B in BGR input
        out = image_to_obs(frame, size=128)  # default bgr_to_rgb=True
        assert out[2].mean() > 250  # Blue -> RGB channel 2
        assert out[0].mean() < 5    # RGB channel 0 (R) empty
        assert out[1].mean() < 5

    def test_channel_order_preserved_when_disabled(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[..., 0] = 255
        out = image_to_obs(frame, size=128, bgr_to_rgb=False)
        assert out[0].mean() > 250  # channel preserved
        assert out[2].mean() < 5

    def test_drops_alpha_and_green_invariant_under_swap(self):
        frame = np.zeros((720, 1280, 4), dtype=np.uint8)  # BGRA
        frame[..., 1] = 255  # G (middle channel, invariant under B<->R swap)
        out = image_to_obs(frame, size=128)
        assert out.shape == (3, 128, 128)
        assert out[1].mean() > 250


class TestBuildTransitions:
    """N+1 obs + N actions -> N transitions; last done=True/mask=0."""

    @staticmethod
    def _obs(v):
        return {
            "state": np.full(25, v, np.float32),
            "side_policy": np.full((3, 4, 4), v, np.uint8),
            "wrist_1": np.full((3, 4, 4), v, np.uint8),
            "side_classifier": np.full((3, 4, 4), v, np.uint8),
        }

    def test_pairs_obs_and_marks_last_done(self):
        obs_list = [self._obs(0), self._obs(1), self._obs(2)]
        action_list = [np.zeros(7, np.float32), np.ones(7, np.float32)]
        trs = build_transitions(obs_list, action_list)
        assert len(trs) == 2
        np.testing.assert_allclose(trs[0]["observations"]["state"], obs_list[0]["state"])
        np.testing.assert_allclose(trs[0]["next_observations"]["state"], obs_list[1]["state"])
        np.testing.assert_allclose(trs[0]["actions"], action_list[0])
        assert trs[0]["dones"] is False
        assert trs[0]["masks"] == pytest.approx(1.0)
        assert trs[1]["dones"] is True
        assert trs[1]["masks"] == pytest.approx(0.0)
        assert trs[0]["rewards"] == pytest.approx(0.0)

    def test_all_obs_keys_present_in_both(self):
        trs = build_transitions([self._obs(0), self._obs(1)], [np.zeros(7, np.float32)])
        for k in ("state", "side_policy", "wrist_1", "side_classifier"):
            assert k in trs[0]["observations"]
            assert k in trs[0]["next_observations"]

    def test_actions_cast_to_float32(self):
        trs = build_transitions([self._obs(0), self._obs(1)], [[0, 0, 0, 0, 0, 0, 1]])
        assert trs[0]["actions"].dtype == np.float32
        assert trs[0]["actions"].shape == (7,)
