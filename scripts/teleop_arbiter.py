"""P2-T7 / A2 — TeleopArbiter: single-source-of-truth dual-device arbitration.

Wraps two intervention wrappers (GELLO + Xbox) so the policy sees exactly
one (or zero) device acting per step, with the priority table from
DECISIONS.md #6:

    RB held          -> Xbox intervenes (GELLO ignored)
    Only LB held     -> GELLO intervenes
    Both held        -> Xbox wins (per decision: same-tick dual -> Xbox)
    Neither held     -> policy action

The arbiter owns the ``TeleopDeviceHub`` so both wrappers read from the
same snapshot. ``info["intervene_device"]`` is always set to one of
``{"xbox", "gello", "none"}`` regardless of which device won.

This module does NOT alter the policy stack (max_step, max_total_delta,
clipping) — those remain owned by the underlying wrappers. The arbiter
is a pure switch: (policy_action, GELLO_action, Xbox_action) -> one
action + device label.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import gymnasium as gym
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from gello_intervention import GelloIntervention
from xbox_intervention import XboxIntervention
from teleop_hub import TeleopDeviceHub

DEVICE_NONE = "none"
DEVICE_GELLO = "gello"
DEVICE_XBOX = "xbox"


class TeleopArbiter(gym.ActionWrapper):
    """Dual-device arbitration wrapper.

    Parameters
    ----------
    env : gym.Env
        Underlying environment.
    gello : GelloIntervention
        GELLO wrapper (constructed with the same ``hub``).
    xbox : XboxIntervention
        Xbox wrapper (constructed with the same ``hub``).
    hub : TeleopDeviceHub
        The shared hub. Must be the same instance the wrappers use.
    """

    def __init__(
        self,
        env: gym.Env,
        gello: GelloIntervention,
        xbox: XboxIntervention,
        hub: TeleopDeviceHub,
    ):
        super().__init__(env)
        self.gello = gello
        self.xbox = xbox
        self.hub = hub

    # ------------------------------------------------------------------
    # Priority table (DECISIONS.md #6)
    # ------------------------------------------------------------------
    @staticmethod
    def _decide(rb: bool, lb: bool) -> str:
        if rb:
            return DEVICE_XBOX
        if lb:
            return DEVICE_GELLO
        return DEVICE_NONE

    def action(self, action: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (action, intervene_device) per the priority table.

        The GELLO and Xbox wrappers are queried for their candidate
        action. We don't pass the policy action to the loser's
        .action() because both wrappers consume the policy action to
        build their index-subset output; the winner's candidate is
        always returned as-is.
        """
        # Read hub once so both wrappers see the same snapshot.
        snapshot = self.hub.poll()
        self.xbox.last_state = snapshot
        self.gello._update_engagement()  # consumes the same LB state

        device = self._decide(rb=snapshot.rb, lb=snapshot.lb)
        if device == DEVICE_XBOX:
            expert, _ = self.xbox.action(action)
            return expert, DEVICE_XBOX
        if device == DEVICE_GELLO:
            expert, _ = self.gello.action(action)
            return expert, DEVICE_GELLO
        return action, DEVICE_NONE

    def step(self, action):
        new_action, device = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_device"] = device
        if device != DEVICE_NONE:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Per-episode budget reset for both wrappers; the arbiter's
        # only state is the device label, so nothing extra to clear.
        self.gello.reset(seed=seed, options=options)
        return self.env.reset(seed=seed, options=options)
