"""Actor-side Xbox intervention regression tests.

The actor copy lives under the v2.2.1 planning evidence tree because the live
training repo is on the FR3 desktop. These tests lock the safety contract before
syncing the same patch to that machine.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Item 20: pin the SAME file the live actor runs (runtime-patches online-stability
# copy), not the superseded yaw-fix evidence snapshot. The two debounce/arming
# assertions below are identical for both copies (verified), so retargeting does
# not change which behavior is exercised -- it just stops this test from being the
# last live dependency on the frozen evidence copy.
ACTOR_SCRIPT = (
    ROOT
    / ".planning/2026-06-13-v221-hybrid-teleop/runtime-patches/20260617-online-stability"
    / "scripts/xbox_intervention.py"
)

try:
    import gymnasium as gym
except ModuleNotFoundError:
    gym = types.ModuleType("gymnasium")

    class _EnvBase:
        def reset(self, *, seed=None, options=None):
            return None

    class _ActionWrapper(_EnvBase):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    gym.Env = _EnvBase
    gym.ActionWrapper = _ActionWrapper
    gym.spaces = types.SimpleNamespace(Box=_Box)
    sys.modules["gymnasium"] = gym

_spec = importlib.util.spec_from_file_location("actor_xbox_intervention", ACTOR_SCRIPT)
assert _spec and _spec.loader
actor_xbox = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = actor_xbox
_spec.loader.exec_module(actor_xbox)


class _Env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        return np.asarray(action, dtype=np.float32), 0.0, False, False, {}


class _State:
    def __init__(self, **kwargs):
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.dpad_x = 0.0
        self.dpad_y = 0.0
        self.rb = False
        self.a = False
        self.b = False
        self.x = False
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Hub:
    available = True

    def __init__(self, state):
        self.state = state

    def poll(self):
        return self.state


def _wrapper(state):
    w = actor_xbox.XboxIntervention(_Env())
    w._hub = _Hub(state)
    return w


def test_rb_release_debounce_returns_zero_motion_not_stale_stick_action(monkeypatch):
    w = _wrapper(_State(rb=False, right_y=1.0, a=True))
    w._armed = True
    w._last_intervene_t = 100.0
    monkeypatch.setattr(time, "time", lambda: 100.1)

    action, replaced = w.action(np.full(7, 0.25, dtype=np.float32))

    assert replaced is True
    np.testing.assert_allclose(action, np.zeros(7, dtype=np.float32), atol=1e-7)
    assert w._insert_ticks == 0
    assert w._spiral_px == 0.0
    assert w._spiral_py == 0.0


def test_rb_release_after_hold_window_resumes_policy(monkeypatch):
    w = _wrapper(_State(rb=False, right_y=1.0, a=True))
    w._armed = True
    w._last_intervene_t = 100.0
    monkeypatch.setattr(time, "time", lambda: 101.0)
    policy = np.full(7, 0.25, dtype=np.float32)

    action, replaced = w.action(policy)

    assert replaced is False
    np.testing.assert_allclose(action, policy, atol=1e-7)
