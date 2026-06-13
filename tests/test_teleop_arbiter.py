"""P2-T7 / A2 — TeleopArbiter contract tests.

Truth table (DECISIONS.md #6):
    RB held          -> xbox
    only LB held     -> gello
    both held        -> xbox (dual-input tie-break)
    neither held     -> none

Drives the GELLO and Xbox wrappers with mock backends, fully synthetic
inputs, and verifies:
  * the correct action vector is emitted
  * info["intervene_device"] matches the priority table
  * info["intervene_action"] is set when device != none, unset otherwise
  * the env's policy action is never silently dropped
  * 6D vs 7D action spaces work for both
  * device-unavailable degradation yields device=none
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gello_intervention  # noqa: E402
import xbox_intervention  # noqa: E402
import teleop_arbiter  # noqa: E402
import teleop_hub  # noqa: E402

from gello_intervention import GelloIntervention  # noqa: E402
from xbox_intervention import XboxIntervention  # noqa: E402
from teleop_arbiter import (  # noqa: E402
    DEVICE_GELLO,
    DEVICE_NONE,
    DEVICE_XBOX,
    TeleopArbiter,
)
from teleop_hub import (  # noqa: E402
    MockJoystickBackend,
    TeleopDeviceHub,
    XboxState,
)
from tests.fixtures.fake_dynamixel_driver import (  # noqa: E402
    FakeDynamixelDriver,
    install_mock,
    uninstall_mock,
)


# ---------------------------------------------------------------------------
# Test env
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
        self.last_action: np.ndarray | None = None
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        self.step_count = 0
        return np.zeros(self.action_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.step_count += 1
        return self.last_action.copy(), float(self.last_action[0]), False, False, {}


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
def xbox_backend() -> MockJoystickBackend:
    return MockJoystickBackend()


@pytest.fixture
def hub(xbox_backend: MockJoystickBackend) -> TeleopDeviceHub:
    return TeleopDeviceHub(backend="mock", mock_backend=xbox_backend)


def _set(hub: TeleopDeviceHub, **fields) -> None:
    hub._backend.set_state(XboxState(**fields))  # type: ignore[attr-defined]


def _arbiter(driver, hub, env_action_dim=7):
    env = _IdentityEnv(action_dim=env_action_dim)
    gello = GelloIntervention(env, arming_hub=hub)
    xbox = XboxIntervention(env, hub)
    arb = TeleopArbiter(env, gello, xbox, hub)
    return arb, env, gello, xbox


# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------
class TestTruthTable:
    def test_neither_held_returns_policy(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        # Seed gello with a movement so its action() produces a real
        # expert candidate (otherwise the action is just zero).
        gello.action(np.zeros(7, dtype=np.float32))
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))
        # No RB, no LB.
        _set(hub)
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_NONE
        assert np.allclose(out, policy)
        # step() must not annotate intervene_action
        _, _, _, _, info = arb.step(policy)
        assert info["intervene_device"] == DEVICE_NONE
        assert "intervene_action" not in info

    def test_only_lb_held_returns_gello(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))  # seed prev
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        _set(hub, lb=True)
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_GELLO
        # Output must NOT equal the policy (gello overrode it).
        assert not np.allclose(out, policy)
        # step() annotates both keys.
        _, _, _, _, info = arb.step(policy)
        assert info["intervene_device"] == DEVICE_GELLO
        assert "intervene_action" in info

    def test_only_rb_held_returns_xbox(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))  # seed
        driver.set_joints(np.array([0.0] * 7 + [0.5]))  # no movement
        # Deflect the left stick and hold RB.
        _set(hub, rb=True, left_x=1.0)
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_XBOX
        assert not np.allclose(out, policy)
        _, _, _, _, info = arb.step(policy)
        assert info["intervene_device"] == DEVICE_XBOX
        assert "intervene_action" in info

    def test_both_held_returns_xbox(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        # Both held: xbox wins.
        _set(hub, rb=True, lb=True, left_x=1.0)
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_XBOX
        _, _, _, _, info = arb.step(policy)
        assert info["intervene_device"] == DEVICE_XBOX


# ---------------------------------------------------------------------------
# Downgrade / never-crash
# ---------------------------------------------------------------------------
class TestDowngrade:
    def test_gello_budget_overrun_falls_back_to_policy(self, driver, hub, capsys):
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, lb=True)
        gello.action(np.zeros(7, dtype=np.float32))  # seed

        # Force an over-budget state.
        gello._agent.initial_pose = gello._agent.prev_pose.copy()
        gello._agent.initial_pose[0] -= 0.5
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        policy = np.full(7, 0.31, dtype=np.float32)
        out, device = arb.action(policy)
        # Wrapper downgrades to policy; device label is gello (it was
        # the only armed device) but the action is the policy one.
        assert device == DEVICE_GELLO
        assert np.allclose(out, policy)
        captured = capsys.readouterr()
        assert "budget exceeded" in captured.out

    def test_unavailable_hub_passes_policy_through(self, driver, env7_stub=None):
        # Build a hub with an empty mock state and the wrapper stack.
        # A zeroed XboxState has rb=False and lb=False, so device=none.
        hub = TeleopDeviceHub(backend="mock", mock_backend=MockJoystickBackend())
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))  # seed
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_NONE
        assert np.allclose(out, policy)


# ---------------------------------------------------------------------------
# 6D vs 7D action space
# ---------------------------------------------------------------------------
class TestActionSpace:
    def test_7d_env_arbiter_emits_7d_xbox(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub, env_action_dim=7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, rb=True, left_x=1.0)
        out, device = arb.action(np.zeros(7, dtype=np.float32))
        assert device == DEVICE_XBOX
        assert out.shape == (7,)

    def test_6d_env_arbiter_emits_6d(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub, env_action_dim=6)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(6, dtype=np.float32))
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, rb=True, left_x=1.0)
        out, device = arb.action(np.zeros(6, dtype=np.float32))
        assert device == DEVICE_XBOX
        assert out.shape == (6,)
