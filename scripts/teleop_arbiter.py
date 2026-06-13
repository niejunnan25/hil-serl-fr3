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
from teleop_hub import TeleopDeviceHub, XboxState

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
        # A2-F3: remember which device drove the previous tick so we can
        # detect a GELLO resume after an Xbox stretch and re-seed the
        # GELLO agent's anchor to the current leader pose (otherwise the
        # first resumed tick computes a huge step delta from a stale
        # anchor and spuriously refuses).
        self._last_active_device = DEVICE_NONE

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

    def _reseed_gello_to_current_pose(self) -> None:
        """A2-F3: re-anchor the GELLO agent to the leader's current pose.

        Called when GELLO becomes the active device after an Xbox stretch.
        During the stretch ``gello.action()`` never ran, so the agent's
        ``prev_joints``/``prev_pose`` are frozen at the pre-stretch pose
        while the physical GELLO leader has moved. Without re-seeding, the
        first resumed tick computes a large step delta from the stale
        anchor and trips the ``max_step`` clamp (zeroed action / spurious
        refusal). Re-seeding makes the first resumed tick measure from the
        leader's actual position, so legitimate operator motion is honored.

        Best-effort and never raises: a device read fault here must not
        crash the loop (mirrors the wrapper's own C1 downgrade policy).
        """
        gello = self.gello
        driver = getattr(gello, "_gello", None)
        agent = getattr(gello, "_agent", None)
        if driver is None or agent is None:
            return
        try:
            raw = np.asarray(driver.get_joints(), dtype=np.float64)
            if raw.shape != (8,):
                return
            agent.reset(raw[:7])
            # Keep the wrapper's cached prev_joints consistent with the
            # agent so engagement edge resets re-anchor here too.
            gello._prev_joints = raw[:7].copy()
        except Exception:
            # Never let a re-seed read fault halt training.
            return

    def action(self, action: np.ndarray) -> tuple[np.ndarray, str]:
        """Return (action, intervene_device) per the priority table.

        The GELLO and Xbox wrappers are queried for their candidate
        action. We don't pass the policy action to the loser's
        .action() because both wrappers consume the policy action to
        build their index-subset output; the winner's candidate is
        always returned as-is.

        ``intervene_device`` is only set to the winning device when that
        wrapper actually *replaced* the policy action. If the winning
        wrapper downgrades (returns ``replaced=False`` — e.g. GELLO budget
        overrun, no movement, or device fault), we return ``DEVICE_NONE``
        so ``step()`` does not annotate ``intervene_action`` with the
        policy's own action (I3 — that would poison the HIL-SERL buffer).
        """
        # Read hub once so both wrappers see the same snapshot.
        snapshot = self.hub.poll()
        self.xbox.last_state = snapshot

        device = self._decide(rb=snapshot.rb, lb=snapshot.lb)

        if device == DEVICE_XBOX:
            # GELLO is NOT active this tick, so its action() won't run and
            # won't update LB engagement. Update it directly so a LB
            # rising edge is registered even while Xbox controls (A2-F3a),
            # and so engagement persists across the Xbox stretch.
            self.gello._update_engagement()
            expert, replaced = self.xbox.action(action)
            self._last_active_device = DEVICE_XBOX
            return (expert, DEVICE_XBOX) if replaced else (action, DEVICE_NONE)

        if device == DEVICE_GELLO:
            # A2-F3b: resuming GELLO after an Xbox stretch -> re-anchor the
            # agent to the current leader pose before gello.action() reads
            # it, so a stale anchor doesn't trigger a spurious max_step
            # refusal on the first resumed tick.
            if self._last_active_device == DEVICE_XBOX:
                self._reseed_gello_to_current_pose()
            # gello.action() owns the single _update_engagement() call on
            # this tick (it calls it internally) — no separate call here,
            # to avoid double-consuming the LB edge.
            expert, replaced = self.gello.action(action)
            self._last_active_device = DEVICE_GELLO
            return (expert, DEVICE_GELLO) if replaced else (action, DEVICE_NONE)

        # Neither held: keep LB engagement bookkeeping current (a falling
        # edge while neither device is active must still end the
        # engagement) and pass the policy action through unchanged.
        self.gello._update_engagement()
        self._last_active_device = DEVICE_NONE
        return action, DEVICE_NONE

    def step(self, action):
        new_action, device = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        info["intervene_device"] = device
        if device != DEVICE_NONE:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # I2: gello, xbox and the arbiter all wrap the SAME base env.
        # self.gello.reset() already calls self.env.reset() exactly once
        # (and owns the per-engagement budget reset), so we must NOT call
        # self.env.reset() again — that double-homes the real robot.
        # Clear the Xbox wrapper's cached transient state WITHOUT an env
        # side-effect, then delegate the single env.reset to gello.
        self.xbox.last_state = XboxState()
        if hasattr(self.xbox, "_y_was_pressed"):
            self.xbox._y_was_pressed = False
        self._last_active_device = DEVICE_NONE
        return self.gello.reset(seed=seed, options=options)
