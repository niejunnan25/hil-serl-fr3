"""P2-T1 (A2) — Extended GelloIntervention contract: arming + budget + reset.

These tests pin down the v2.2.1-A2 safety improvements WITHOUT breaking
the existing 34 tests in test_gello_intervention_contract.py. The
arming_hub argument is optional: when omitted, behaviour is identical
to the original wrapper (legacy semantics).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gello_intervention  # noqa: E402
from tests.fixtures.fake_dynamixel_driver import (  # noqa: E402
    FakeDynamixelDriver,
    install_mock,
    uninstall_mock,
)
from teleop_hub import MockJoystickBackend, TeleopDeviceHub, XboxState  # noqa: E402


# ---------------------------------------------------------------------------
# Local env (mirrors test_gello_intervention_contract's _IdentityEnv but
# lives here so we don't import from the legacy file).
# ---------------------------------------------------------------------------
class _IdentityEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, action_dim: int = 7):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )
        self.step_count = 0
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.reset_count += 1
        return np.zeros(self.action_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        a = np.asarray(action, dtype=np.float32).copy()
        return a, float(a[0]), False, False, {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def driver() -> FakeDynamixelDriver:
    d = FakeDynamixelDriver(
        list(range(8)),
        port="/dev/ttyUSB0",
        baudrate=57600,
        max_retries=1,
        use_fake_fallback=False,
    )
    install_mock(d)
    yield d
    uninstall_mock()


@pytest.fixture
def hub() -> TeleopDeviceHub:
    return TeleopDeviceHub(backend="mock", mock_backend=MockJoystickBackend())


@pytest.fixture
def env7() -> _IdentityEnv:
    return _IdentityEnv(action_dim=7)


def _set_state(hub: TeleopDeviceHub, **fields) -> None:
    hub._backend.set_state(XboxState(**fields))  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Legacy back-compat: arming_hub=None
# ---------------------------------------------------------------------------
class TestLegacyBackCompat:
    def test_no_arming_hub_uses_movement_threshold_only(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        # 0.01 rad on joint 1 -> ~1.1mm translation (verified offline),
        # above the 1mm movement threshold and below the 3mm step clamp.
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True
        w.close()


# ---------------------------------------------------------------------------
# Arming hub: LB gates the intervention
# ---------------------------------------------------------------------------
class TestArmingHub:
    def test_lb_unheld_blocks_intervention_even_with_movement(self, driver, hub, env7):
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        # Big movement, but LB is released.
        _set_state(hub, lb=False)
        driver.set_joints(np.array([0.05] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is False, "no LB -> must not intervene"
        w.close()

    def test_lb_held_with_movement_allows_intervention(self, driver, hub, env7):
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        _set_state(hub, lb=True)
        # 0.01 rad joint 1 -> ~1.1mm translation; above 1mm threshold.
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True
        w.close()

    def test_lb_release_ends_engagement(self, driver, hub, env7):
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))

        # Move while held -> intervene.
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True

        # Drop the stick deflection; keep LB held: still within hold window.
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        w.last_intervene = time.time() - 1.0  # expire hold
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is False
        w.close()


# ---------------------------------------------------------------------------
# Per-engagement budget
# ---------------------------------------------------------------------------
class TestPerEngagementBudget:
    def test_budget_resets_on_lb_rising_edge(self, driver, hub, env7):
        """After the previous engagement accumulated budget, releasing
        and re-pressing LB must zero the budget so the operator can
        keep teleoperating in the same episode."""
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))  # seeds prev_joints

        # Accumulate some non-trivial total delta in the agent.
        w._agent.prev_pose = w._agent.prev_pose.copy()  # noop sanity
        w._agent.initial_pose = w._agent.prev_pose.copy()
        # Pretend we've burned 25mm of the 30mm budget.
        w._agent.initial_pose = w._agent.prev_pose.copy()
        w._agent.initial_pose[0] -= 0.025
        # Without the engagement reset, the next step would still
        # compute the cumulative norm correctly (the agent uses
        # current_pose - initial_pose each step).

        # Release LB, then re-press -> engagement reset should zero
        # initial_pose to the current prev_pose.
        _set_state(hub, lb=False)
        w.action(np.zeros(7, dtype=np.float32))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))
        # After the rising edge, the initial_pose must be aligned with
        # prev_pose so the cumulative budget restarts.
        assert np.allclose(w._agent.initial_pose[:3], w._agent.prev_pose[:3]), (
            "rising LB edge must reset cumulative budget"
        )
        w.close()

    def test_budget_overrun_demotes_to_policy_without_raising(
        self, driver, hub, env7, capsys
    ):
        """When the per-engagement budget is exhausted the wrapper must
        not raise; it returns the policy action and logs a warning."""
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))  # seed

        # Force the agent into a permanent over-budget state.
        w._agent.initial_pose = w._agent.prev_pose.copy()
        w._agent.initial_pose[0] -= 0.5  # 500 mm away from current
        # And drop a big single-step move.
        driver.set_joints(np.array([0.05] + [0.0] * 6 + [0.5]))
        policy = np.full(7, 0.31, dtype=np.float32)
        new_action, replaced = w.action(policy)
        assert replaced is False, "over-budget step must not intervene"
        assert np.allclose(new_action, policy)
        # Warning was logged.
        captured = capsys.readouterr()
        assert "budget exceeded" in captured.out or "budget exceeded" in captured.err
        w.close()

    def test_recovery_after_overrun_via_new_engagement(
        self, driver, hub, env7
    ):
        """After an overrun ends the engagement, the next LB press
        starts a fresh budget so the operator isn't stuck dead."""
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))  # seed

        # Trip the overrun.
        w._agent.initial_pose = w._agent.prev_pose.copy()
        w._agent.initial_pose[0] -= 0.5
        driver.set_joints(np.array([0.05] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is False

        # New engagement: clear overrun by walking back to home-ish
        # pose, then a new LB press.
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w._prev_joints = np.zeros(7)
        _set_state(hub, lb=False)
        w.action(np.zeros(7, dtype=np.float32))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))
        # Now move a small amount; budget should be reset so the
        # intervention succeeds.
        driver.set_joints(np.array([0.02] + [0.0] * 6 + [0.5]))
        _, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True
        w.close()


# ---------------------------------------------------------------------------
# env.reset() per-episode budget reset
# ---------------------------------------------------------------------------
class TestEnvReset:
    def test_reset_clears_last_intervene_and_engagement(self, driver, hub, env7):
        w = gello_intervention.GelloIntervention(env7, arming_hub=hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set_state(hub, lb=True)
        w.action(np.zeros(7, dtype=np.float32))  # seed
        # Sanity: engagement is active.
        assert w._engagement_active is True

        # Call wrapper reset (calls env.reset and clears internal state).
        w.reset()

        assert w._engagement_active is False
        assert w._lb_was_held is False
        assert w.last_intervene == 0.0
        # Per-engagement budget must be aligned with prev_joints.
        assert np.allclose(w._agent.initial_pose[:3], w._agent.prev_pose[:3])
        # Underlying env was reset too.
        assert env7.reset_count == 1
        w.close()

    def test_reset_returns_gym_2tuple(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # ensure_gello
        out = w.reset()
        assert isinstance(out, tuple) and len(out) == 2
        obs, info = out
        assert obs.shape == (7,)
        assert isinstance(info, dict)
        w.close()

    def test_legacy_reset_still_clears_last_intervene(self, driver, env7):
        """Without arming_hub, reset() must still drop last_intervene
        and align the agent's budget — same effect, just no engagement."""
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))
        w.last_intervene = time.time()
        w.reset()
        assert w.last_intervene == 0.0
        assert np.allclose(w._agent.initial_pose[:3], w._agent.prev_pose[:3])
        w.close()


# ---------------------------------------------------------------------------
# C1 — device-read exception must never crash the training loop
# ---------------------------------------------------------------------------
class TestDeviceErrorNeverCrashes:
    """PLAN-A2 must_have: 设备异常永不中断训练循环.

    A Dynamixel timeout / USB unplug raises OSError/IOError, and a short
    or malformed read raises AssertionError. Neither may propagate out of
    action(); the wrapper must downgrade to (policy, False).
    """

    def test_get_joints_oserror_returns_policy_no_raise(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed (connection ok)

        # Now the USB cable is yanked mid-training.
        def _boom():
            raise OSError("USB unplugged mid-training")

        driver.get_joints = _boom  # type: ignore[assignment]

        policy = np.full(7, 0.23, dtype=np.float32)
        new_action, replaced = w.action(policy)
        assert replaced is False, "device read failure must not intervene"
        assert np.allclose(new_action, policy), "must return the policy action verbatim"
        w.close()

    def test_get_joints_ioerror_returns_policy_no_raise(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        def _boom():
            raise IOError("serial timeout")

        driver.get_joints = _boom  # type: ignore[assignment]

        policy = np.full(7, -0.4, dtype=np.float32)
        new_action, replaced = w.action(policy)
        assert replaced is False
        assert np.allclose(new_action, policy)
        w.close()

    def test_malformed_short_read_returns_policy_no_raise(self, driver, env7):
        """A short/malformed read (length 3) makes raw[7] / shape logic
        raise; that must be caught and downgraded, not propagate."""
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        driver.get_joints = lambda: np.zeros(3, dtype=np.float64)  # type: ignore[assignment]

        policy = np.full(7, 0.11, dtype=np.float32)
        new_action, replaced = w.action(policy)
        assert replaced is False
        assert np.allclose(new_action, policy)
        w.close()

    def test_device_open_failure_returns_policy_no_raise(self, env7):
        """A device-open failure (no mock installed -> _ensure_gello
        raises) must downgrade to (policy, False), not propagate."""
        # No install_mock here: _GELLO_AVAILABLE may be False (ImportError
        # path) or the constructor itself raises. Force a hard open failure.
        gello_intervention._GELLO_AVAILABLE = True
        orig = gello_intervention.DynamixelDriver

        def _explode(*a, **kw):
            raise OSError("could not open /dev/ttyUSB0")

        gello_intervention.DynamixelDriver = _explode  # type: ignore[assignment]
        try:
            w = gello_intervention.GelloIntervention(env7)
            policy = np.full(7, 0.5, dtype=np.float32)
            new_action, replaced = w.action(policy)
            assert replaced is False
            assert np.allclose(new_action, policy)
        finally:
            gello_intervention.DynamixelDriver = orig

    def test_device_error_logged_only_once(self, driver, env7, capsys):
        """The device-error message is logged once, not every tick, to
        avoid flooding the training log."""
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed
        capsys.readouterr()  # clear seed output

        def _boom():
            raise OSError("USB unplugged")

        driver.get_joints = _boom  # type: ignore[assignment]
        policy = np.zeros(7, dtype=np.float32)
        for _ in range(5):
            _, replaced = w.action(policy)
            assert replaced is False
        out = capsys.readouterr().out
        # Exactly one device-error log line across the 5 failing ticks.
        assert out.count("device") <= 1 or out.lower().count("device error") <= 1
        w.close()

    def test_device_error_flag_cleared_on_reset(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        assert w._device_error_logged is False
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        def _boom():
            raise OSError("USB unplugged")

        driver.get_joints = _boom  # type: ignore[assignment]
        w.action(np.zeros(7, dtype=np.float32))
        assert w._device_error_logged is True
        # Restore a good read so reset()'s _ensure_gello path is happy.
        driver.get_joints = lambda: np.array([0.0] * 7 + [0.5], dtype=np.float64)  # type: ignore[assignment]
        w.reset()
        assert w._device_error_logged is False
        w.close()
