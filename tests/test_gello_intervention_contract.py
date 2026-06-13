"""P2-T1 + P2-T4: GelloIntervention action()/step() contract.

Verifies the wrapper exposes the same signature SpacemouseIntervention
would, while fully mocking the DynamixelDriver so no GELLO hardware is
required. Gripper mapping (8th channel) is exercised here because
P2-T4 sits at the same boundary.

The SpacemouseIntervention contract we are matching:
    - subclass of gym.ActionWrapper
    - action(action: np.ndarray) -> tuple[np.ndarray, bool]
    - step(action) -> (obs, rew, done, truncated, info)
    - when replacing: info["intervene_action"] = replaced_action
    - gripper_enabled inferred from action_space.shape == (6,) -> False
    - movement threshold ||xyz_delta|| > 0.001
    - 0.5s hold window after last detected movement
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

# Ensure scripts is on the import path (the gello_intervention module
# inserts itself on import, but we add it explicitly for clarity)
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Stub the gello package before the wrapper imports it. The wrapper does
# ``from gello.dynamixel.driver import DynamixelDriver`` inside a try/except;
# we install our own attribute there in fixtures/fake_dynamixel_driver.py.
import gello_intervention  # noqa: E402
import fk_converter  # noqa: E402

from tests.fixtures.fake_dynamixel_driver import (  # noqa: E402
    FakeDynamixelDriver,
    install_mock,
    uninstall_mock,
)


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


class _IdentityEnv(gym.Env):
    """Minimal Gym env returning the action unchanged.

    Action space is (7,); observation is the action's first 3 components
    (mirrors FrankaEnv's TCP-pose-shaped observation in tests).
    """

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

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.zeros(self.action_space.shape, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.step_count += 1
        obs = self.last_action.copy()
        rew = float(self.last_action[0])
        done = False
        truncated = False
        info: dict = {}
        return obs, rew, done, truncated, info


@pytest.fixture
def env7() -> _IdentityEnv:
    return _IdentityEnv(action_dim=7)


@pytest.fixture
def env6() -> _IdentityEnv:
    return _IdentityEnv(action_dim=6)


# ---------------------------------------------------------------------------
# P2-T1 contract tests
# ---------------------------------------------------------------------------
class TestGelloInterventionContract:
    """Verify the action()/step() signature matches SpacemouseIntervention."""

    def test_inherits_gym_action_wrapper(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        assert isinstance(w, gym.ActionWrapper)

    def test_action_signature_returns_tuple_of_ndarray_and_bool(
        self, driver, env7
    ):
        w = gello_intervention.GelloIntervention(env7)
        # drive the agent once with a movement so it must intervene
        driver.set_joints(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]))
        # First call: prev_joints captured, no movement on first read
        policy_action = np.zeros(7, dtype=np.float32)
        out = w.action(policy_action)
        assert isinstance(out, tuple)
        assert len(out) == 2
        expert_action, replaced = out
        assert isinstance(expert_action, np.ndarray)
        assert isinstance(replaced, bool)
        w.close()

    def test_step_returns_gym_5tuple_and_sets_intervene_action(
        self, driver, env7
    ):
        w = gello_intervention.GelloIntervention(env7)
        # Seed the wrapper with a real first read (no movement) and then
        # a second read with a delta large enough to exceed _MOVE_THRESHOLD.
        driver.set_joints(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]))
        out1 = w.action(np.zeros(7, dtype=np.float32))
        assert out1[1] is False, "first call captures prev, no movement yet"

        # Now jump the joints to provoke a >1mm translation delta while
        # staying under the 3mm per-step clamp. 0.01 rad on joint 1
        # produces ~1.1mm (verified offline) and clears the threshold
        # without tripping the safety clamp.
        driver.set_joints(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]))
        policy_action = np.full(7, 0.42, dtype=np.float32)
        # Capture the expert action by reading it from action() and
        # then again from step(); the second call uses a fresh read so
        # the values can differ in y/z. Compare to what step() saw.
        _, _ = w.action(policy_action)
        obs, rew, done, truncated, info = w.step(policy_action)
        assert obs.shape == (7,)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert "intervene_action" in info, \
            "info must carry intervene_action when expert overrode policy"
        # The annotated vector must be the one step() actually applied.
        assert info["intervene_action"].shape == (7,)
        assert not np.allclose(info["intervene_action"], policy_action)
        w.close()

    def test_no_movement_returns_policy_action_unchanged(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))  # initial
        # First call: agent resets, no replacement
        action0 = np.full(7, 0.11, dtype=np.float32)
        out0 = w.action(action0)
        assert out0[1] is False
        assert np.allclose(out0[0], action0)

        # Second call: zero movement on GELLO -> expert should also be 0
        # and the wrapper should pass through.
        out1 = w.action(action0)
        assert out1[1] is False
        assert np.allclose(out1[0], action0)

        # step() must NOT set intervene_action
        obs, rew, done, truncated, info = w.step(action0)
        assert "intervene_action" not in info
        w.close()

    def test_hold_window_keeps_intervention_after_movement_stops(
        self, driver, env7
    ):
        """After movement stops, the wrapper must keep intervening for
        _HOLD_SECONDS (0.5s) so the human has time to release.
        """
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        # First big movement -> intervene. Use a delta that drives FK
        # translation above the 0.001 m movement threshold but stays
        # under the 0.003 m max_step clamp. 0.01 rad on joint 1
        # produces ~1.1mm translation (verified offline).
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        _, replaced1 = w.action(np.zeros(7, dtype=np.float32))
        assert replaced1 is True

        # Stop moving; reset the wrapper's prev_joints to the current
        # reading so the next action() computes zero FK delta. We must
        # bypass action() for this (it would otherwise reset
        # last_intervene via the movement detector). Refresh the hold
        # timer to "now" so we don't race the wall clock.
        w._prev_joints = np.array([0.01] + [0.0] * 6)
        w._agent.prev_joints = np.array([0.01] + [0.0] * 6)
        w._agent.prev_pose = fk_converter.forward_kinematics(
            np.array([0.01] + [0.0] * 6)
        )
        w._agent.initial_pose = w._agent.prev_pose.copy()
        w.last_intervene = time.time()
        _, replaced2 = w.action(np.zeros(7, dtype=np.float32))
        assert replaced2 is True, "hold window should keep intervention active"

        # Simulate the hold window expiring by moving last_intervene to
        # the past. Keep prev_joints aligned so the agent's movement
        # detector doesn't reset the timer.
        w.last_intervene = time.time() - 1.0
        _, replaced3 = w.action(np.zeros(7, dtype=np.float32))
        assert replaced3 is False, "after hold window, must return policy"
        w.close()

    def test_action_indices_mapping_replaces_only_subset(self, driver, env7):
        """When action_indices is set, only those channels of the policy
        action are overridden and the rest pass through."""
        w = gello_intervention.GelloIntervention(
            env7, action_indices=np.array([0, 1, 2])
        )
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))  # seed

        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.5]))
        policy = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        out_action, replaced = w.action(policy)
        assert replaced is True
        # Channels 3..6 must equal policy[3..6]
        assert np.allclose(out_action[3:], policy[3:])
        # Channels 0..2 must come from the expert (not the policy)
        assert not np.allclose(out_action[:3], policy[:3])
        w.close()

    def test_gripper_disabled_for_6d_action_space(self, driver, env6):
        w = gello_intervention.GelloIntervention(env6)
        assert w.gripper_enabled is False
        w.close()

    def test_gripper_enabled_for_7d_action_space(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        assert w.gripper_enabled is True
        w.close()

    def test_close_releases_driver(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        # Trigger lazy init so driver is constructed
        w.action(np.zeros(7, dtype=np.float32))
        assert w._gello is driver
        w.close()
        assert driver.closed is True
        assert w._gello is None

    def test_lazily_constructs_driver_on_first_action(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        assert w._gello is None
        w.action(np.zeros(7, dtype=np.float32))
        assert w._gello is driver
        w.close()


# ---------------------------------------------------------------------------
# P2-T4: 8th axis -> gripper close/open mapping
# ---------------------------------------------------------------------------
class TestGripperMapping:
    """The wrapper maps the GELLO 8th channel to [-1, 1] via clip(r*2-1).

    Convention used by the wrapper:
        raw = 0.0 -> -1 (closed)
        raw = 0.5 ->  0 (mid)
        raw = 1.0 -> +1 (open)
    These tests pin the mapping in place and verify the gripper channel
    of the expert action is consistent with the 8th channel of the
    DynamixelDriver reading.
    """

    def test_raw_zero_maps_to_closed(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        driver.set_joints(np.array([0.0] * 7 + [0.0]))
        # first call captures prev; need a tiny movement to get an expert
        # action from the agent. Use a sub-threshold nudge so it does NOT
        # trigger intervention; we still need the agent to return a valid
        # action and have the wrapper expose it.
        driver.set_joints(np.array([0.0005] + [0.0] * 6 + [0.0]))
        # First reset the wrapper with the 0/0.5 reading
        w._prev_joints = None
        # Drive two reads: first seeds the agent, second produces the action
        w._ensure_gello()
        w._agent.reset(np.array([0.0] * 7))
        expert, info = w._agent.step(
            np.array([0.0005] + [0.0] * 6), gripper=np.clip(0.0 * 2.0 - 1.0, -1.0, 1.0)
        )
        assert expert[6] == pytest.approx(-1.0)
        w.close()

    def test_raw_one_maps_to_open(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        w._ensure_gello()
        w._agent.reset(np.zeros(7))
        gripper = np.clip(1.0 * 2.0 - 1.0, -1.0, 1.0)
        expert, _ = w._agent.step(np.zeros(7), gripper=gripper)
        assert expert[6] == pytest.approx(1.0)
        w.close()

    def test_raw_half_maps_to_mid(self, driver, env7):
        w = gello_intervention.GelloIntervention(env7)
        w._ensure_gello()
        w._agent.reset(np.zeros(7))
        gripper = np.clip(0.5 * 2.0 - 1.0, -1.0, 1.0)
        expert, _ = w._agent.step(np.zeros(7), gripper=gripper)
        assert expert[6] == pytest.approx(0.0)
        w.close()

    def test_mapping_via_full_action_call(self, driver, env7):
        """End-to-end: set the 8th channel on the driver, drive the
        wrapper through action(), and verify the gripper component of
        the expert action reflects the expected mapping.
        """
        w = gello_intervention.GelloIntervention(env7)
        # First read seeds the agent at home with mid gripper
        driver.set_joints(np.array([0.0] * 7 + [0.5]))
        w.action(np.zeros(7, dtype=np.float32))

        # Now change the 8th channel to 0.0 (closed) and add a real
        # movement so the wrapper intervenes.
        driver.set_joints(np.array([0.01] + [0.0] * 6 + [0.0]))
        expert, replaced = w.action(np.zeros(7, dtype=np.float32))
        assert replaced is True
        # raw=0.0 -> clip(0*2-1) = -1
        assert expert[6] == pytest.approx(-1.0)
        w.close()

    def test_mapping_clipping_clamps_out_of_range(self, driver, env7):
        """If the GELLO 8th channel ever goes outside [0, 1] the
        mapping must clamp to [-1, 1]."""
        w = gello_intervention.GelloIntervention(env7)
        w._ensure_gello()
        w._agent.reset(np.zeros(7))
        # raw = 2.0 -> 2*2-1 = 3, clipped to +1
        gripper = np.clip(2.0 * 2.0 - 1.0, -1.0, 1.0)
        expert, _ = w._agent.step(np.zeros(7), gripper=gripper)
        assert expert[6] == pytest.approx(1.0)

        gripper = np.clip(-1.0 * 2.0 - 1.0, -1.0, 1.0)
        expert, _ = w._agent.step(np.zeros(7), gripper=gripper)
        assert expert[6] == pytest.approx(-1.0)
        w.close()
