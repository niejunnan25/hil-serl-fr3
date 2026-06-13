"""B2a — Xbox deadzone / range / trigger calibration harness.

Pure estimators (testable without hardware) + a thin hub sampler. The CLI
captures real numbers once a handle is plugged into fr3-desktop-ts; here we
test the estimation logic with synthetic XboxState sequences + a mock hub.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from teleop_hub import MockJoystickBackend, TeleopDeviceHub, XboxState  # noqa: E402
from xbox_calibrate import (  # noqa: E402
    DeviceUnavailable,
    build_calibration,
    collect_samples,
    estimate_axis_ranges,
    estimate_deadzone,
    estimate_trigger_thresholds,
)

STICK_AXES = ("left_x", "left_y", "right_x", "right_y")


def _rest(noise):
    return [XboxState(left_x=n, left_y=-n, right_x=n / 2, right_y=-n / 2) for n in noise]


class TestDeadzone:
    def test_deadzone_exceeds_observed_rest_noise(self):
        rest = _rest([0.01, -0.03, 0.04, -0.02, 0.03])
        dz = estimate_deadzone(rest)
        # max observed |axis| at rest is 0.04 -> deadzone must sit above it.
        assert dz > 0.04
        assert dz <= 0.3  # capped, never absurd

    def test_deadzone_floor_when_device_is_quiet(self):
        rest = _rest([0.0, 0.001, -0.002, 0.0])
        dz = estimate_deadzone(rest, floor=0.05)
        assert dz == pytest.approx(0.05)  # clamped up to the floor


class TestRanges:
    def test_axis_ranges_capture_min_max(self):
        samples = [
            XboxState(left_x=-0.9, right_y=0.8),
            XboxState(left_x=0.95, right_y=-0.85),
            XboxState(left_x=0.0, right_y=0.0),
        ]
        ranges = estimate_axis_ranges(samples)
        assert ranges["left_x"][0] == pytest.approx(-0.9)
        assert ranges["left_x"][1] == pytest.approx(0.95)
        assert ranges["right_y"][0] == pytest.approx(-0.85)
        assert ranges["right_y"][1] == pytest.approx(0.8)


class TestTriggers:
    def test_trigger_threshold_between_rest_and_pressed(self):
        rest = [XboxState(rt=0.0, lt=0.02), XboxState(rt=0.01, lt=0.0)]
        pressed = [XboxState(rt=0.95, lt=0.9), XboxState(rt=0.8, lt=0.85)]
        rt_thr, lt_thr = estimate_trigger_thresholds(rest, pressed)
        assert 0.02 < rt_thr < 0.8
        assert 0.02 < lt_thr < 0.85


class TestBuildCalibration:
    def test_calibration_has_expected_shape(self):
        rest = _rest([0.01, -0.02, 0.03])
        rng = [XboxState(left_x=-1.0, right_x=1.0, rt=1.0, lt=1.0), XboxState()]
        cal = build_calibration(rest_samples=rest, range_samples=rng)
        assert set(cal) >= {"deadzone", "axis_ranges", "rt_threshold", "lt_threshold", "n_rest", "n_range"}
        assert cal["deadzone"] > 0
        for ax in STICK_AXES:
            assert ax in cal["axis_ranges"]


class TestHubSampler:
    def test_collect_polls_hub_n_times(self):
        backend = MockJoystickBackend()
        backend.set_state(XboxState(left_x=0.1))
        hub = TeleopDeviceHub(backend="mock", mock_backend=backend)
        samples = collect_samples(hub, n=5, sleep_s=0.0)
        assert len(samples) == 5
        assert backend.read_count == 5
        assert all(s.left_x == pytest.approx(0.1) for s in samples)

    def test_no_device_raises_device_unavailable(self):
        backend = MockJoystickBackend()
        backend.available = False  # simulate unplugged
        hub = TeleopDeviceHub(backend="mock", mock_backend=backend)
        with pytest.raises(DeviceUnavailable):
            collect_samples(hub, n=3, sleep_s=0.0)
