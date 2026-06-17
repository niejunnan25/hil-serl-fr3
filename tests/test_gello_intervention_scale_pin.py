"""Pin GelloIntervention's Cartesian normalization scales to the corrected
module defaults (DEFAULT_POS_SCALE / DEFAULT_RPY_SCALE).

Regression guard for C1-intervention-scale: the intervention wrapper must NOT
re-introduce the stale pre-Tier-2 literals (pos_scale=0.1 / rpy_scale=0.2).
It should track the single source of truth in gello_cartesian_delta_agent so
the live teleop path and the demo-recording path stay consistent.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

_SCRIPTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")
)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import gello_intervention  # noqa: E402
import gello_cartesian_delta_agent as agent_mod  # noqa: E402


class _IdentityEnv(gym.Env):
    """Minimal 7D env so GelloIntervention can be constructed."""

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def test_intervention_agent_uses_corrected_pos_scale():
    w = gello_intervention.GelloIntervention(_IdentityEnv())
    try:
        assert w._agent.pos_scale == pytest.approx(agent_mod.DEFAULT_POS_SCALE)
        # Guard against the stale pre-Tier-2 literal.
        assert w._agent.pos_scale == pytest.approx(0.015)
        assert w._agent.pos_scale != pytest.approx(0.1)
    finally:
        w.close()


def test_intervention_agent_uses_corrected_rpy_scale():
    w = gello_intervention.GelloIntervention(_IdentityEnv())
    try:
        assert w._agent.rpy_scale == pytest.approx(agent_mod.DEFAULT_RPY_SCALE)
        assert w._agent.rpy_scale == pytest.approx(0.1)
        assert w._agent.rpy_scale != pytest.approx(0.2)
    finally:
        w.close()
