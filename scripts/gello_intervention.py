#!/usr/bin/env python3
"""gello_intervention.py

GELLO-based human intervention wrapper for HIL-SERL training.

Drop-in replacement for SpacemouseIntervention: same gym.ActionWrapper
interface, same action()/step() signature, same info["intervene_action"]
annotation. Uses a GELLO (Dynamixel) device instead of a SpaceMouse.

Data flow:
    GELLO (Dynamixel) -> read() -> raw (8D: 7 joints + 1 gripper)
        |
        v
    GelloCartesianDeltaAgent.step(joints[:7]) -> normalized action (7D)
        |
        v
    movement detection: ||xyz_delta|| > 0.001 AND LB-held (arming)
        |
        v
    action() returns (expert_action, True) or (policy_action, False)

Usage:
    from gello_intervention import GelloIntervention

    env = SomeFrankaEnv(config)
    env = GelloIntervention(env, gello_port="/dev/ttyUSB0", arming_hub=hub)
    obs, rew, done, truncated, info = env.step(policy_action)

Safety (inherited from GelloCartesianDeltaAgent):
    - max_step:        0.003 m per-step Cartesian delta
    - max_total_delta: 0.03 m cumulative from initial pose (per-engagement
                       budget when arming_hub is supplied; reset on every
                       engagement transition; wrapper.reset() also resets
                       the budget so a new episode starts fresh)

v2.2.1-A2 changes (DECISIONS.md #6 — safe for hybrid teleop):
    * ``arming_hub``: when provided, GELLO intervention requires **LB held
      AND** 1mm translation movement (AND semantics). Without arming_hub
      the legacy 1mm-only behavior is preserved (back-compat for the
      existing 34-test contract).
    * Per-engagement budget: every LB-press (rising edge) calls
      ``_agent.reset()`` so the 30mm cumulative budget restarts at zero.
      LB release ends the engagement; the next rising edge starts a new
      one. This prevents mid-episode budget starvation.
    * ``reset()`` override: env.reset() triggers ``_agent.reset()`` and
      clears ``last_intervene`` so a new episode starts with a clean
      budget.
    * Budget overrun downgrade: a ``RuntimeError`` from the agent used to
      propagate out and crash the training loop. We now downgrade to
      "skip this intervention, return policy action, log a warning" and
      end the engagement (next LB press starts a fresh budget).
"""

import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np

# Ensure scripts directory is on the import path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from gello_cartesian_delta_agent import GelloCartesianDeltaAgent
from fk_converter import forward_kinematics, joints_to_cartesian_delta

# Try importing DynamixelDriver; fail gracefully if gello package is missing
try:
    from gello.dynamixel.driver import DynamixelDriver

    _GELLO_AVAILABLE = True
except ImportError:
    _GELLO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_GELLO_PORT = "/dev/ttyUSB0"
DEFAULT_GELLO_BAUDRATE = 57600
DEFAULT_ACTION_INDICES = None  # use full action vector
DEFAULT_SERVER_URL = "http://127.0.0.2:5000/"

# Movement detection threshold (meters, translation norm)
_MOVE_THRESHOLD = 0.001

# Intervention hold time (seconds) — keep intervening for this long after
# the last detected movement, matching SpacemouseIntervention's 0.5s window.
_HOLD_SECONDS = 0.5


# ---------------------------------------------------------------------------
# GelloIntervention
# ---------------------------------------------------------------------------
class GelloIntervention(gym.ActionWrapper):
    """GELLO-based human intervention wrapper.

    Same interface as SpacemouseIntervention:
        action(self, action: np.ndarray) -> tuple[np.ndarray, bool]
        step(self, action) -> (obs, rew, done, truncated, info)

    Args:
        env:               Gym environment to wrap.
        gello_port:        Serial port for GELLO Dynamixel device.
        action_indices:    Optional array of indices into the env action space
                           that the GELLO expert controls.  ``None`` means
                           the expert controls the full action vector.
        server_url:        (reserved) HTTP URL for the robot server.
        arming_hub:        Optional TeleopDeviceHub. When supplied, the
                           wrapper requires LB to be held *in addition to*
                           the 1mm movement threshold before intervening.
                           LB also defines engagement boundaries: a
                           rising edge resets the per-engagement 30mm
                           budget; a falling edge ends the engagement.
    """

    def __init__(
        self,
        env: gym.Env,
        gello_port: str = DEFAULT_GELLO_PORT,
        action_indices: Optional[np.ndarray] = DEFAULT_ACTION_INDICES,
        server_url: str = DEFAULT_SERVER_URL,
        arming_hub: Optional[object] = None,  # TeleopDeviceHub, kept untyped
                                                # to avoid an import cycle.
    ):
        super().__init__(env)

        self.gello_port = gello_port
        self.action_indices = action_indices
        self.server_url = server_url
        self.arming_hub = arming_hub

        # Gripper detection: same heuristic as SpacemouseIntervention
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        # GELLO device
        self._gello = None
        self._initialised = False

        # Cartesian delta agent (safety + FK)
        self._agent = GelloCartesianDeltaAgent(
            max_step=0.003,
            max_total_delta=0.03,
            pos_scale=0.1,
            rpy_scale=0.2,
        )

        # Intervention state
        self.last_intervene: float = 0.0
        self._prev_joints: Optional[np.ndarray] = None

        # Engagement state (A2 — only meaningful when arming_hub is set)
        self._engagement_active: bool = False
        self._lb_was_held: bool = False
        self._budget_overrun_logged: bool = False

    # ------------------------------------------------------------------
    # GELLO device management
    # ------------------------------------------------------------------
    def _ensure_gello(self) -> None:
        """Lazy-initialise the GELLO device on first use."""
        if self._gello is not None:
            return
        if not _GELLO_AVAILABLE:
            raise ImportError(
                "gello package not found. Install gello to use GelloIntervention."
            )
        self._gello = DynamixelDriver(
            list(range(8)),
            port=self.gello_port,
            baudrate=DEFAULT_GELLO_BAUDRATE,
            max_retries=1,
            use_fake_fallback=False,
        )
        # Initial read to verify connection
        raw = np.asarray(self._gello.get_joints(), dtype=np.float64)
        assert raw.shape == (8,), f"Expected (8,) from GELLO, got {raw.shape}"
        self._prev_joints = raw[:7].copy()
        self._agent.reset(self._prev_joints)
        self._initialised = True
        print(f"[GelloIntervention] Connected to {self.gello_port}")

    # ------------------------------------------------------------------
    # Engagement helpers (A2)
    # ------------------------------------------------------------------
    def _lb_held(self) -> bool:
        """Read LB from the arming hub. False if no hub or hub unavailable."""
        if self.arming_hub is None:
            return False
        try:
            return bool(self.arming_hub.poll().lb)
        except Exception:
            return False

    def _update_engagement(self) -> None:
        """Track the LB edge and reset the per-engagement budget on
        every rising edge (LB from released -> held)."""
        lb_now = self._lb_held()
        if lb_now and not self._lb_was_held:
            # Rising edge: new engagement.
            self._engagement_active = True
            if self._prev_joints is not None:
                self._agent.reset(self._prev_joints)
            self._budget_overrun_logged = False
        elif not lb_now and self._lb_was_held:
            # Falling edge: engagement ends.
            self._engagement_active = False
        self._lb_was_held = lb_now

    def _force_end_engagement(self) -> None:
        """Drop the current engagement (e.g. on budget overrun)."""
        self._engagement_active = False
        self._lb_was_held = False
        # Clear the hold window so we don't fire stale interventions.
        self.last_intervene = 0.0

    # ------------------------------------------------------------------
    # action() — same signature as SpacemouseIntervention
    # ------------------------------------------------------------------
    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Read GELLO input and decide whether to override the policy action.

        Args:
            action: Policy action from the RL agent.

        Returns:
            (expert_action, True)  if GELLO movement detected AND (LB held
                                   when arming_hub is set)
            (action,        False) otherwise
        """
        self._ensure_gello()

        # Engagement bookkeeping (LB edge detection / per-engagement reset).
        self._update_engagement()

        # Read GELLO state (8D: 7 joints + 1 gripper)
        raw = np.asarray(self._gello.get_joints(), dtype=np.float64)
        gello_joints = raw[:7]
        raw_gripper = float(raw[7])

        # Map gripper: GELLO 8th joint -> [-1, 1]
        # Convention: lower values = closed (-1), higher = open (+1)
        gripper = np.clip(raw_gripper * 2.0 - 1.0, -1.0, 1.0)

        # Convert joints to Cartesian delta via agent
        try:
            expert_action, info = self._agent.step(gello_joints, gripper=gripper)
        except RuntimeError as exc:
            # The agent raised on a hard safety violation. Downgrade:
            # end the engagement and return the policy action.
            if not self._budget_overrun_logged:
                print(
                    f"[GelloIntervention] safety violation, downgrading to "
                    f"policy action: {exc}"
                )
                self._budget_overrun_logged = True
            self._force_end_engagement()
            return action, False

        # If the agent reported an unsafe step, the action it returned
        # is already zeroed. The original semantics of the wrapper only
        # intervene when the *cartesian delta* exceeds the threshold
        # (max_step); a max_total_delta overrun is a different
        # condition that should not crash training but also should not
        # be treated as a normal intervention. We downgrade silently:
        # end the engagement, return the policy action.
        if not info.get("safe", True):
            violation = info.get("violation", "")
            # Only the cumulative-budget overrun ends the engagement;
            # max_step is a per-tick clamp and the agent has already
            # zeroed the action for this tick. The next LB press will
            # give the operator a fresh budget either way.
            if "max_total_delta" in violation:
                if not self._budget_overrun_logged:
                    print(
                        f"[GelloIntervention] budget exceeded, ending engagement: "
                        f"{violation}"
                    )
                    self._budget_overrun_logged = True
                self._force_end_engagement()
            return action, False

        # Movement detection on raw translation delta
        step_norm = info.get("step_delta_norm", 0.0)

        if step_norm > _MOVE_THRESHOLD:
            self.last_intervene = time.time()

        # Arming gate: when arming_hub is present, LB must be held.
        if self.arming_hub is not None and not self._engagement_active:
            return action, False

        # Hold window: keep intervening for _HOLD_SECONDS after last movement
        if time.time() - self.last_intervene < _HOLD_SECONDS:
            # Apply action_indices mapping if configured
            if self.action_indices is not None:
                full = np.array(action, dtype=np.float32)
                full[self.action_indices] = expert_action[: len(self.action_indices)]
                return full, True
            return expert_action, True

        return action, False

    # ------------------------------------------------------------------
    # step() — same interface as SpacemouseIntervention
    # ------------------------------------------------------------------
    def step(self, action):
        """Step the wrapped env, replacing the action when GELLO intervenes.

        Args:
            action: Policy action from the RL agent.

        Returns:
            (obs, rew, done, truncated, info) with info["intervene_action"]
            set when the expert action replaced the policy action.
        """
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    # ------------------------------------------------------------------
    # reset() — A2: per-episode budget reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the wrapper AND clear intervention state for a new episode.

        Without this, the per-engagement budget would carry across
        episodes and silently disable GELLO intervention late in long
        training runs.
        """
        # Lazy initialise the GELLO device so reset() works before any
        # action() call (matches the previous behaviour of reset()
        # being a no-op until first action).
        if self._gello is None and _GELLO_AVAILABLE:
            try:
                self._ensure_gello()
            except Exception:
                pass

        if self._prev_joints is not None:
            self._agent.reset(self._prev_joints)
        self.last_intervene = 0.0
        self._engagement_active = False
        self._lb_was_held = False
        self._budget_overrun_logged = False
        return self.env.reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Release Dynamixel resources."""
        if self._gello is not None:
            self._gello.close()
            self._gello = None
            self._initialised = False
            print("[GelloIntervention] Disconnected.")
        super().close()
