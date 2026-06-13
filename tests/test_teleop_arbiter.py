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


class _CountEnv(_IdentityEnv):
    """Identity env that counts how many times ``reset()`` is invoked on
    the *base* env. Used to assert that one ``arbiter.reset()`` triggers
    exactly one base-env reset (I2 — no double homing on real hardware)."""

    def __init__(self, action_dim: int = 7):
        super().__init__(action_dim=action_dim)
        self.reset_count = 0

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1
        return super().reset(seed=seed, options=options)


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


def _arbiter(driver, hub, env_action_dim=7, env_cls=_IdentityEnv):
    env = env_cls(action_dim=env_action_dim)
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
        # I3: the wrapper downgraded (replaced=False). LB was held but
        # GELLO produced NO genuine intervention, so the arbiter must
        # label the step as DEVICE_NONE — labelling it gello here would
        # write the policy's own action into the HIL-SERL intervention
        # buffer as if it were a human correction (data poisoning).
        assert device == DEVICE_NONE
        assert np.allclose(out, policy)
        captured = capsys.readouterr()
        assert "budget exceeded" in captured.out

    def test_gello_downgrade_does_not_set_intervene_action(self, driver, hub):
        """I3 regression: a downgraded GELLO step must NOT annotate
        info["intervene_action"] — otherwise step() poisons the buffer
        with the policy action mislabelled as an intervention."""
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, lb=True)
        gello.action(np.zeros(7, dtype=np.float32))  # seed

        # Force an over-budget state -> wrapper returns replaced=False.
        gello._agent.initial_pose = gello._agent.prev_pose.copy()
        gello._agent.initial_pose[0] -= 0.5
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        policy = np.full(7, 0.31, dtype=np.float32)
        _, _, _, _, info = arb.step(policy)
        assert info["intervene_device"] == DEVICE_NONE
        assert "intervene_action" not in info
        # The env still received the (unaltered) policy action.
        assert np.allclose(env.last_action, policy)

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


# ---------------------------------------------------------------------------
# I2 — reset() resets the shared base env EXACTLY once
# ---------------------------------------------------------------------------
class TestResetOnce:
    def test_arbiter_reset_resets_base_env_exactly_once(self, driver, hub):
        # gello, xbox and the arbiter all wrap the SAME base env. A naive
        # arbiter.reset() that calls gello.reset() (which resets the env)
        # AND env.reset() again homes the real robot twice per episode.
        arb, env, gello, xbox = _arbiter(driver, hub, env_cls=_CountEnv)
        assert env.reset_count == 0
        arb.reset()
        assert env.reset_count == 1

    def test_arbiter_reset_returns_base_obs(self, driver, hub):
        arb, env, gello, xbox = _arbiter(driver, hub, env_cls=_CountEnv)
        obs, info = arb.reset()
        assert obs.shape == (7,)
        assert isinstance(info, dict)

    def test_arbiter_reset_clears_xbox_transient_state(self, driver, hub):
        # reset() must clear the xbox wrapper's cached last_state WITHOUT
        # an extra env.reset() side-effect.
        arb, env, gello, xbox = _arbiter(driver, hub, env_cls=_CountEnv)
        _set(hub, rb=True, left_x=1.0)
        xbox.action(np.zeros(7, dtype=np.float32))
        assert xbox.last_state.rb is True
        arb.reset()
        assert xbox.last_state.rb is False
        assert env.reset_count == 1


# ---------------------------------------------------------------------------
# A2-F3 — engagement-while-xbox-controls / resume re-seed
# ---------------------------------------------------------------------------
class TestEngagementWhileXbox:
    def test_both_held_routes_to_xbox_regression(self, driver, hub):
        # Explicit RB+LB guard: even though LB would arm GELLO, RB wins.
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))  # seed
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, rb=True, lb=True, left_x=1.0)
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        assert device == DEVICE_XBOX
        assert not np.allclose(out, policy)

    def test_lb_engagement_tracked_while_xbox_controls(self, driver, hub):
        """LB rising-edge engagement must register even on a tick where
        RB (xbox) is the active device, so GELLO is already engaged when
        RB releases — not waiting for a fresh LB rising edge."""
        arb, env, gello, xbox = _arbiter(driver, hub)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        gello.action(np.zeros(7, dtype=np.float32))  # seed prev (LB low)
        gello._engagement_active = False
        gello._lb_was_held = False

        # First tick: RB + LB both held -> xbox controls, but the LB
        # rising edge must still arm GELLO's engagement.
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, rb=True, lb=True, left_x=1.0)
        out, device = arb.action(np.full(7, 0.42, dtype=np.float32))
        assert device == DEVICE_XBOX
        assert gello._engagement_active is True

    def test_gello_resume_after_xbox_stretch_does_not_refuse(self, driver, hub):
        """A2-F3: LB held throughout. RB held for a stretch (xbox active)
        during which the GELLO leader physically moves. When RB releases
        (LB still held), the FIRST gello tick must NOT spuriously refuse
        from a stale anchor — the arbiter re-seeds the agent to the
        current leader pose so the resumed step delta is small."""
        arb, env, gello, xbox = _arbiter(driver, hub)
        # Establish a genuine GELLO engagement with a real, safe move
        # (0.01 rad on joint 0 -> ~1.1mm translation: above the 1mm move
        # threshold, below the 3mm max_step clamp).
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        _set(hub, lb=True)
        arb.action(np.zeros(7, dtype=np.float32))  # seed prev at home
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        out, device = arb.action(np.zeros(7, dtype=np.float32))
        assert device == DEVICE_GELLO  # genuine engagement, no refusal
        assert gello._agent.violation_count == 0

        # ---- Xbox stretch: RB + LB held; xbox controls for a few ticks.
        # GELLO's agent anchor is now FROZEN at the pre-stretch pose.
        _set(hub, rb=True, lb=True, left_x=1.0)
        for _ in range(3):
            # The GELLO leader keeps drifting far while xbox drives.
            cur = driver.get_joints()
            cur[0] += 0.05  # large joint drift -> would blow max_step
            driver.set_joints(cur)
            out, device = arb.action(np.zeros(7, dtype=np.float32))
            assert device == DEVICE_XBOX

        violations_before_resume = gello._agent.violation_count

        # ---- RB released, LB still held: GELLO resumes. The leader sits
        # at a pose far from the stale anchor. Without a re-seed the first
        # tick computes a huge step delta -> max_step refusal -> the
        # operator's first real motion is silently dropped.
        _set(hub, lb=True)  # RB released, LB still held
        policy = np.full(7, 0.42, dtype=np.float32)
        out, device = arb.action(policy)
        # The first resumed gello tick must NOT register a spurious
        # max_step refusal from the stale anchor.
        assert gello._agent.violation_count == violations_before_resume

        # And a small legitimate nudge on the NEXT tick is a clean,
        # non-refused intervention that overrides the policy.
        cur = driver.get_joints()
        cur[0] += 0.01  # ~1.1mm translation: a genuine, safe move
        driver.set_joints(cur)
        out, device = arb.action(policy)
        assert device == DEVICE_GELLO
        assert not np.allclose(out, policy)
        assert gello._agent.violation_count == violations_before_resume
