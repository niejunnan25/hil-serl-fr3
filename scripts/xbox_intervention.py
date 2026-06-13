"""P2-T7 (v2.2.1-A1) — XboxIntervention wrapper.

Drop-in replacement for ``GelloIntervention`` / ``SpacemouseIntervention``:
same ``gym.ActionWrapper`` interface, same ``action(action)->(action, replaced)``
signature, same ``info["intervene_action"]`` annotation. Reads the Xbox
state from a ``TeleopDeviceHub`` instance so the wrapper has zero direct
dependency on pygame.

Intervention condition (DECISIONS.md #5/6):
    * **RB must be held (deadman)** — the deadman gate, never raw stick
      deflection. This is the operator's explicit "I am in control"
      signal. RB held but sticks centered -> zero action but still
      considered intervening (so info/intervene_action reflect the
      takeover).

Mapping (constants centralised so A3 record_hybrid_demos can import
the same numbers — single source of truth):
    * left  stick x/y -> dx, dy
    * right stick y  -> dz
    * right stick x  -> dyaw
    * d-pad          -> dpitch, droll
    * RT / LT        -> gripper close / open  (already in [0,1])
    * Y button       -> scale toggle: fine (default) / coarse
    * deadzone       -> 0.15 on stick axes (raw)
    * all output     -> clip to [-1, 1]^7

6D env (no gripper channel) -> gripper_enabled=False, RT/LT ignored.

Safety semantics:
    * max_step=0.003 m per-step translation cap is enforced *inside* this
      wrapper (REVIEW I1). The de-normalized translation
      ||action[:3] * pos_scale|| is clamped to <= max_step so the 3mm/step
      invariant cannot be bypassed via the Xbox path, independent of any
      downstream env clamp. Direction is preserved (uniform scaling of the
      [dx, dy, dz] triple). Rotation and gripper channels are NOT clamped
      here — rotational/total-delta limits remain owned by the action
      policy stack (the HIL-SERL env).
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

from teleop_hub import TeleopDeviceHub, XboxState

# ---------------------------------------------------------------------------
# Mapping constants — single source of truth (A3 imports these).
# ---------------------------------------------------------------------------
DEADZONE = 0.15          # raw stick axis below this -> 0
SCALE_FINE = 0.3         # default; fine-scale for precision tasks
SCALE_COARSE = 1.0       # Y toggles to coarse; full range
DX_IDX, DY_IDX, DZ_IDX = 0, 1, 2
DROLL_IDX, DPITCH_IDX, DYAW_IDX = 3, 4, 5
GRIPPER_IDX = 6

# Channel count: 7D (with gripper) or 6D (without).
FULL_ACTION_DIM = 7
GRIPPERLESS_ACTION_DIM = 6


class XboxIntervention(gym.ActionWrapper):
    """RB-deadman Xbox intervention wrapper.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    hub : TeleopDeviceHub
        State source. Pass a mock-backend hub from tests.
    scale : float
        Initial stick-to-action gain (default ``SCALE_FINE``).
    deadzone : float
        Stick deadzone (default ``DEADZONE``).
    action_indices : np.ndarray, optional
        Subset of the policy action that the Xbox expert controls. When
        ``None`` (default), the expert overwrites the full vector.
    """

    def __init__(
        self,
        env: gym.Env,
        hub: TeleopDeviceHub,
        scale: float = SCALE_FINE,
        deadzone: float = DEADZONE,
        action_indices: Optional[np.ndarray] = None,
        max_step: float = 0.003,
        pos_scale: float = 0.1,
    ):
        super().__init__(env)
        self.hub = hub
        self.scale = float(scale)
        self.deadzone = float(deadzone)
        self.action_indices = action_indices
        # I1: enforce the per-step translation cap *inside* the teleop
        # toolchain so the 3mm/step invariant cannot be bypassed via the
        # Xbox path, regardless of whether the downstream env clamps.
        # ``pos_scale`` is the env's normalized->metres gain (env pos_scale)
        # and ``max_step`` is the metres-per-step ceiling.
        self.max_step = float(max_step)
        self.pos_scale = float(pos_scale)

        # Gripper detection: matches GelloIntervention / Spacemouse.
        self.gripper_enabled = self.action_space.shape == (FULL_ACTION_DIM,)

        # Last observed state (exposed for tests / recorders).
        self.last_state: XboxState = XboxState()
        # Track scale flips so A3 can record ``xbox_scale_mode``.
        self.scale_mode: str = "fine" if scale == SCALE_FINE else "coarse"

    # ------------------------------------------------------------------
    # Mapping helpers (tested directly via the suite)
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        if abs(value) < deadzone:
            return 0.0
        # Rescale so the deadzone boundary maps to 0, not the original
        # value (otherwise the action jitters near the boundary).
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)

    def _state_to_action(self, s: XboxState) -> np.ndarray:
        """Map XboxState -> 7D action in [-1, 1]^7.

        The wrapper is agnostic to whether the env is 6D or 7D; the
        action() / step() path trims to the env's action space below.
        """
        # raw axes are already in [-1, 1]; apply deadzone then scale.
        left_x = self._apply_deadzone(s.left_x, self.deadzone)
        left_y = self._apply_deadzone(s.left_y, self.deadzone)
        right_x = self._apply_deadzone(s.right_x, self.deadzone)
        right_y = self._apply_deadzone(s.right_y, self.deadzone)
        dpad_x = self._apply_deadzone(s.dpad_x, self.deadzone)
        dpad_y = self._apply_deadzone(s.dpad_y, self.deadzone)

        action = np.zeros(FULL_ACTION_DIM, dtype=np.float32)
        action[DX_IDX] = left_x * self.scale
        action[DY_IDX] = left_y * self.scale
        action[DZ_IDX] = -right_y * self.scale   # right stick up -> +dz feels right
        action[DYAW_IDX] = right_x * self.scale
        action[DPITCH_IDX] = -dpad_y * self.scale
        action[DROLL_IDX] = dpad_x * self.scale

        # Gripper: RT (close) wins over LT (open) when both held.
        if self.gripper_enabled:
            if s.rt > 0.05:
                action[GRIPPER_IDX] = 1.0
            elif s.lt > 0.05:
                action[GRIPPER_IDX] = -1.0
            else:
                action[GRIPPER_IDX] = 0.0

        action = np.clip(action, -1.0, 1.0)

        # I1: cap the per-step *de-normalized* translation. The normalized
        # translation channels [dx, dy, dz] scale to metres by pos_scale;
        # if that exceeds max_step, shrink the translation triple uniformly
        # so the de-normalized norm sits exactly at max_step (direction
        # preserved). Rotation (3..5) and gripper (6) are untouched.
        xyz = action[DX_IDX : DZ_IDX + 1] * self.pos_scale
        denorm = float(np.linalg.norm(xyz))
        if denorm > self.max_step:
            action[DX_IDX : DZ_IDX + 1] *= self.max_step / denorm

        return action

    def _toggle_scale(self, s: XboxState) -> None:
        """Y button edge toggle: fine <-> coarse."""
        # We can't detect an edge without remembering the last frame;
        # record_hybrid_demos (A3) tracks the edge for us. Here we just
        # reflect the *current* button state and flip if it became
        # pressed this call.
        if s.y and not getattr(self, "_y_was_pressed", False):
            if self.scale_mode == "fine":
                self.scale = SCALE_COARSE
                self.scale_mode = "coarse"
            else:
                self.scale = SCALE_FINE
                self.scale_mode = "fine"
        self._y_was_pressed = bool(s.y)

    # ------------------------------------------------------------------
    # Public API — SpacemouseIntervention / GelloIntervention compatible
    # ------------------------------------------------------------------
    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Read the hub, return (expert_action, True) when RB is held."""
        s = self.hub.poll()
        self.last_state = s
        self._toggle_scale(s)

        if not s.rb:
            return action, False

        expert_full = self._state_to_action(s)

        # Trim/expand to the env action dimension.
        if self.action_indices is not None:
            full = np.array(action, dtype=np.float32).copy()
            full[self.action_indices] = expert_full[: len(self.action_indices)]
            return full, True

        if self.action_space.shape == (GRIPPERLESS_ACTION_DIM,):
            return expert_full[:GRIPPERLESS_ACTION_DIM], True
        return expert_full, True

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
