"""P2-T7 / A1 — XboxIntervention contract tests.

Mirrors the structure of test_gello_intervention_contract.py so the two
wrappers are demonstrably isomorphic at the action() / step() boundary.
Covers: deadman, deadzone, scale toggle, gripper mapping, 6D vs 7D
trimming, info key compatibility.
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

import xbox_intervention  # noqa: E402
import teleop_hub  # noqa: E402
from teleop_hub import (  # noqa: E402
    MockJoystickBackend,
    TeleopDeviceHub,
    XboxState,
)
from xbox_intervention import (  # noqa: E402
    DEADZONE,
    SCALE_COARSE,
    SCALE_FINE,
    XboxIntervention,
)


# ---------------------------------------------------------------------------
# Fixtures
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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return np.zeros(self.action_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        obs = self.last_action.copy()
        rew = float(self.last_action[0])
        return obs, rew, False, False, {}


@pytest.fixture
def backend() -> MockJoystickBackend:
    return MockJoystickBackend()


@pytest.fixture
def hub(backend: MockJoystickBackend) -> TeleopDeviceHub:
    return TeleopDeviceHub(backend="mock", mock_backend=backend)


@pytest.fixture
def env7() -> _IdentityEnv:
    return _IdentityEnv(action_dim=7)


@pytest.fixture
def env6() -> _IdentityEnv:
    return _IdentityEnv(action_dim=6)


# ---------------------------------------------------------------------------
# Spacemouse/Gello isomorphism
# ---------------------------------------------------------------------------
class TestXboxInterventionContract:
    def test_inherits_gym_action_wrapper(self, hub, env7):
        w = XboxIntervention(env7, hub)
        assert isinstance(w, gym.ActionWrapper)

    def test_action_returns_tuple_of_ndarray_and_bool(self, hub, env7):
        w = XboxIntervention(env7, hub)
        out = w.action(np.zeros(7, dtype=np.float32))
        assert isinstance(out, tuple) and len(out) == 2
        expert, replaced = out
        assert isinstance(expert, np.ndarray) and isinstance(replaced, bool)

    def test_step_returns_gym_5tuple_and_sets_intervene_action(self, hub, env7):
        w = XboxIntervention(env7, hub)
        backend_set = hub._backend  # type: ignore[attr-defined]
        backend_set.set_state(XboxState(left_x=0.5, rb=True))
        obs, rew, done, truncated, info = w.step(np.full(7, 0.42, dtype=np.float32))
        assert obs.shape == (7,)
        assert isinstance(done, bool) and isinstance(truncated, bool)
        assert "intervene_action" in info
        # The expert action must differ from the policy at non-zero axes.
        assert not np.allclose(info["intervene_action"], np.full(7, 0.42, dtype=np.float32))

    def test_step_does_not_set_intervene_action_when_no_replacement(self, hub, env7):
        w = XboxIntervention(env7, hub)
        obs, _, _, _, info = w.step(np.full(7, 0.42, dtype=np.float32))
        assert "intervene_action" not in info


# ---------------------------------------------------------------------------
# Deadman semantics
# ---------------------------------------------------------------------------
class TestDeadman:
    def test_rb_unheld_passes_policy_unchanged_even_if_sticks_deflected(
        self, hub, env7
    ):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(left_x=1.0, left_y=-1.0, rb=False))  # type: ignore[attr-defined]
        policy = np.full(7, 0.11, dtype=np.float32)
        out_action, replaced = w.action(policy)
        assert replaced is False
        assert np.allclose(out_action, policy)

    def test_rb_held_with_centered_sticks_zeroes_action_but_still_intervenes(
        self, hub, env7
    ):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(rb=True))  # type: ignore[attr-defined]
        policy = np.full(7, 0.11, dtype=np.float32)
        out_action, replaced = w.action(policy)
        assert replaced is True
        assert np.allclose(out_action, np.zeros(7, dtype=np.float32), atol=1e-6)
        # step() annotates info
        _, _, _, _, info = w.step(policy)
        assert "intervene_action" in info


# ---------------------------------------------------------------------------
# Mapping & deadzone
# ---------------------------------------------------------------------------
class TestMapping:
    def test_left_stick_maps_to_dx_dy(self, hub, env7):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(left_x=1.0, left_y=1.0, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        # Fine scale, full deflection, deadzone removed.
        assert a[0] == pytest.approx(SCALE_FINE, abs=1e-6)
        assert a[1] == pytest.approx(SCALE_FINE, abs=1e-6)

    def test_right_stick_maps_to_dz_dyaw(self, hub, env7):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(right_x=-1.0, right_y=1.0, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[2] == pytest.approx(-SCALE_FINE, abs=1e-6)  # -right_y
        assert a[5] == pytest.approx(-SCALE_FINE, abs=1e-6)

    def test_dpad_maps_to_dpitch_droll(self, hub, env7):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(dpad_x=1.0, dpad_y=1.0, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[3] == pytest.approx(SCALE_FINE, abs=1e-6)   # droll from dpad_x
        assert a[4] == pytest.approx(-SCALE_FINE, abs=1e-6)  # -dpad_y -> dpitch

    def test_deadzone_suppresses_small_axes(self, hub, env7):
        w = XboxIntervention(env7, hub, deadzone=DEADZONE)
        hub._backend.set_state(XboxState(left_x=0.10, rb=True))  # < DEADZONE=0.15
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[0] == 0.0
        # Above deadzone -> non-zero.
        hub._backend.set_state(XboxState(left_x=0.5, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[0] > 0.0

    def test_gripper_rt_closes_lt_opens(self, hub, env7):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(rt=1.0, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[6] == 1.0
        hub._backend.set_state(XboxState(lt=1.0, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[6] == -1.0
        hub._backend.set_state(XboxState(rb=True))  # both released -> 0
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert a[6] == 0.0

    def test_action_is_clipped_to_unit_range(self, hub, env7):
        w = XboxIntervention(env7, hub)
        # Deflection past the unit interval must still produce action in [-1, 1].
        hub._backend.set_state(XboxState(  # type: ignore[attr-defined]
            left_x=1.0, left_y=1.0, right_x=1.0, right_y=1.0,
            dpad_x=1.0, dpad_y=1.0, rt=1.0, rb=True
        ))
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert np.all(np.abs(a) <= 1.0 + 1e-6)

    def test_y_button_toggles_scale(self, hub, env7):
        w = XboxIntervention(env7, hub)
        assert w.scale_mode == "fine"
        assert w.scale == SCALE_FINE
        # Y unpressed -> pressed (edge).
        hub._backend.set_state(XboxState(y=True, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert w.scale_mode == "coarse"
        assert w.scale == SCALE_COARSE
        # Y still held -> no double-flip on the next call.
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert w.scale_mode == "coarse"
        # Release and re-press -> back to fine.
        hub._backend.set_state(XboxState(rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert w.scale_mode == "coarse"
        hub._backend.set_state(XboxState(y=True, rb=True))  # type: ignore[attr-defined]
        a, _ = w.action(np.zeros(7, dtype=np.float32))
        assert w.scale_mode == "fine"


# ---------------------------------------------------------------------------
# Action-space compatibility
# ---------------------------------------------------------------------------
class TestActionSpace:
    def test_7d_env_returns_7d_expert(self, hub, env7):
        w = XboxIntervention(env7, hub)
        hub._backend.set_state(XboxState(left_x=1.0, rb=True))  # type: ignore[attr-defined]
        a, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True
        assert a.shape == (7,)

    def test_6d_env_drops_gripper_and_disables_gripper_enabled(self, hub, env6):
        w = XboxIntervention(env6, hub)
        assert w.gripper_enabled is False
        hub._backend.set_state(XboxState(left_x=1.0, rt=1.0, rb=True))  # type: ignore[attr-defined]
        a, replaced = w.action(np.zeros(6, dtype=np.float32))
        assert replaced is True
        assert a.shape == (6,)

    def test_action_indices_replaces_only_subset(self, hub, env7):
        w = XboxIntervention(env7, hub, action_indices=np.array([0, 1, 2]))
        hub._backend.set_state(XboxState(left_x=1.0, left_y=1.0, rb=True))  # type: ignore[attr-defined]
        policy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        out_action, replaced = w.action(policy)
        assert replaced is True
        # Channels 3..6 untouched.
        assert np.allclose(out_action[3:], policy[3:])
        # Channels 0..2 come from the expert.
        assert out_action[0] == pytest.approx(SCALE_FINE, abs=1e-6)
        assert out_action[1] == pytest.approx(SCALE_FINE, abs=1e-6)


# ---------------------------------------------------------------------------
# Pygame-unavailable / hub-unavailable degradation
# ---------------------------------------------------------------------------
class TestDegradation:
    def test_unavailable_hub_yields_zero_state_but_wrapper_remains_safe(
        self, env7
    ):
        # Simulate "hub reports not available". A zeroed XboxState has
        # rb=False, so action() must NOT intervene and must pass the
        # policy through unchanged. This proves the wrapper degrades
        # safely when the device is gone.
        from teleop_hub import MockJoystickBackend
        mock = MockJoystickBackend()
        mock.available = False
        # Default XboxState has every bool False -> rb is False.
        hub_offline = TeleopDeviceHub(backend="mock", mock_backend=mock)
        w = XboxIntervention(env7, hub_offline)
        policy = np.full(7, 0.42, dtype=np.float32)
        a, replaced = w.action(policy)
        assert replaced is False
        assert np.allclose(a, policy)
