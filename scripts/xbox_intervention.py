"""xbox_intervention.py — HIL-SERL human takeover via Xbox controller.

Drop-in replacement for GelloIntervention with the SAME contract:
  action(self, policy_action) -> (chosen_action, is_intervening)
  step(self, action) -> (obs, rew, done, truncated, info) with info["intervene_action"]
                        set whenever the human action replaced the policy action.

Takeover trigger = RB dead-man (hold RB to take over; release -> policy resumes),
matching the demo-collection convention. While RB is held the human drives with the
same tuned mapping as hybrid_teleop:
  left stick  -> X/Y (fwd-back / left-right)
  right stick up/down -> Z
  X/B and D-pad rotation are locked by default for insert-only actor training.
  A (with RB) -> straight-down insertion push (+ optional spiral search)
  RT/LT       -> gripper (IGNORED here: hold-grip forces gripper closed downstream)

Reuses teleop_hub.TeleopDeviceHub (pygame Xbox reader) + normalize_action from the
teleop repo. Gripper component is left at 0.0 (no-op) because a downstream hold-grip
override keeps the plug clamped regardless (avoids the demo/env gripper-sign mismatch).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym

# teleop repo holds the tuned Xbox reader (teleop_hub, pygame backend)
_TELEOP_SCRIPTS = os.environ.get("TELEOP_SCRIPTS", os.path.dirname(__file__))
if _TELEOP_SCRIPTS not in sys.path:
    sys.path.insert(0, _TELEOP_SCRIPTS)

# inlined from gello_demo_recorder (avoid that module's heavy import chain)
ACTION_SCALE = np.array([0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0], dtype=float)


def normalize_action(dxyz, drotvec, gripper_action, action_scale=ACTION_SCALE):
    """Cartesian delta + gripper -> normalized 7D action in [-1, 1] (float32)."""
    action_scale = np.asarray(action_scale, dtype=float)
    a = np.empty(7, dtype=float)
    a[:3] = np.asarray(dxyz, dtype=float).reshape(-1)[:3] / action_scale[:3]
    a[3:6] = np.asarray(drotvec, dtype=float).reshape(-1)[:3] / action_scale[3]   # FIX 2026-06-16b: rot scale = action_scale[3]=0.1 (matched in franka_env.step)
    a[6] = float(gripper_action)
    return np.clip(a, -1.0, 1.0).astype(np.float32)

# --- mapping constants (mirror hybrid_teleop, but bounded for online training) ---
XBOX_DZ = 0.08             # stick deadzone (ignore rest drift)
XBOX_ROT_STEP = 0.030      # rad/tick for D-pad roll/pitch + B/X yaw
XBOX_MAX_STEP = 0.010      # m/tick translation at full stick before env norm clamp
XBOX_INSERT_REACH = 0.005  # m straight-down push while A held
XBOX_SPIRAL_R = 0.003     # m max spiral radius while A held
XBOX_SPIRAL_RATE = 0.001  # m/s spiral growth
XBOX_SPIRAL_W = 2.0 * np.pi * 0.7  # rad/s spiral angular speed
XBOX_LOCK_ROTATION = os.environ.get("XBOX_LOCK_ROTATION", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_HOLD_SECONDS = 0.4       # keep intervening this long after RB release (debounce)


def _dz(v: float) -> float:
    return 0.0 if abs(v) < XBOX_DZ else float(v)


class XboxIntervention(gym.ActionWrapper):
    """Xbox-only human intervention wrapper (RB dead-man)."""

    def __init__(self, env, gripper_action_value: float = 0.0):
        super().__init__(env)
        self.gripper_value = float(gripper_action_value)  # hold-grip: no-op (kept closed downstream)
        self._hub = None
        self._warned = False
        self._armed = False   # 开局冻结: 首次按 RB 才解锁策略
        self._last_intervene_t = -1e9
        self._insert_ticks = 0
        self._spiral_px = 0.0
        self._spiral_py = 0.0
        self._dt = 0.1  # actor control period (s); only used for the spiral clock

    def _ensure_hub(self):
        if self._hub is None:
            from teleop_hub import TeleopDeviceHub
            self._hub = TeleopDeviceHub(backend="pygame")
            if not getattr(self._hub, "available", False):
                raise RuntimeError("Xbox controller unavailable; Actor cannot start")
            else:
                print("[XboxIntervention] Xbox hub ready (RB dead-man takeover armed).", flush=True)

    def wait_for_ready(self, operator, recorder):
        if self._armed:
            return
        self._ensure_hub()
        operator.publish("waiting_controller", prompt="按住 Xbox RB，准备开始。等待期间持续录像。")
        while not self._armed:
            recorder.check()
            operator.raise_if_stop()
            state = self._hub.poll()
            if bool(getattr(state, "rb", False)):
                self._armed = True
                self._last_intervene_t = time.time()
                return
            time.sleep(0.05)

    def _xbox_action_components(self, st) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map the Xbox state -> normalized 7D action (gripper = self.gripper_value)."""
        lx, ly = _dz(st.left_x), _dz(st.left_y)
        ry, rxx = _dz(st.right_y), _dz(st.right_x)

        dxy = np.array([-ly, lx]) * XBOX_MAX_STEP   # up/down -> +X/-X ; L/R -> Y
        nrm = float(np.linalg.norm(dxy))
        if nrm > XBOX_MAX_STEP:
            dxy *= XBOX_MAX_STEP / nrm
        dz = -ry * XBOX_MAX_STEP

        sx = sy = 0.0
        if bool(getattr(st, "a", False)):           # A: straight-down push + spiral search (delta)
            self._insert_ticks += 1
            th = self._insert_ticks * self._dt
            rr = min(XBOX_SPIRAL_R, XBOX_SPIRAL_RATE * th)
            ang = XBOX_SPIRAL_W * th
            nx, ny = rr * float(np.cos(ang)), rr * float(np.sin(ang))
            sx, sy = nx - self._spiral_px, ny - self._spiral_py   # spiral DELTA this tick
            self._spiral_px, self._spiral_py = nx, ny
            dz = -XBOX_INSERT_REACH
        else:
            self._insert_ticks = 0
            self._spiral_px = self._spiral_py = 0.0

        dxyz = np.array([dxy[0] + sx, dxy[1] + sy, dz])
        yaw_btn = float(getattr(st, "b", False)) - float(getattr(st, "x", False))  # 换绑: B=+yaw, X=-yaw(原右摇杆左右)
        if XBOX_LOCK_ROTATION:
            drotvec = np.zeros(3, dtype=float)
        else:
            drotvec = np.array([_dz(st.dpad_x), -_dz(st.dpad_y), yaw_btn]) * XBOX_ROT_STEP
        action = normalize_action(dxyz, drotvec, self.gripper_value, ACTION_SCALE)
        return action, dxyz, drotvec

    def action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Return (chosen_action, is_intervening). RB held -> human overrides policy."""
        try:
            self._ensure_hub()
            st = self._hub.poll()   # TeleopDeviceHub.poll() -> XboxState (NOT .read())
        except Exception as e:
            if not self._warned:
                import traceback
                traceback.print_exc()
                print("[XboxIntervention] !!! hub poll FAILED (%s) -> NO RB takeover. Policy runs UNCHECKED."
                      % type(e).__name__, flush=True)
                self._warned = True
            raise RuntimeError("Xbox input failed; Actor paused") from e

        rb = bool(getattr(st, "rb", False))
        now = time.time()
        if rb:
            self._armed = True
            self._last_intervene_t = now
            xbox_action, dxyz, drotvec = self._xbox_action_components(st)
            if now - getattr(self, "_dbg_t", 0.0) > 1.0:
                self._dbg_t = now
                import os as _os
                print(
                    "[XboxDBG] source=rb SDL=%s "
                    "rot_locked=%d "
                    "lx=%+.2f ly=%+.2f rx=%+.2f ry=%+.2f "
                    "dpx=%+.0f dpy=%+.0f a=%d b=%d x=%d "
                    "dxyz=%s drot=%s act_xyz=%s act_rot=%s" % (
                        _os.environ.get("SDL_VIDEODRIVER"),
                        int(XBOX_LOCK_ROTATION),
                        float(getattr(st, "left_x", 0.0)),
                        float(getattr(st, "left_y", 0.0)),
                        float(getattr(st, "right_x", 0.0)),
                        float(getattr(st, "right_y", 0.0)),
                        float(getattr(st, "dpad_x", 0.0)),
                        float(getattr(st, "dpad_y", 0.0)),
                        int(getattr(st, "a", 0)),
                        int(getattr(st, "b", 0)),
                        int(getattr(st, "x", 0)),
                        np.asarray(dxyz).round(4).tolist(),
                        np.asarray(drotvec).round(4).tolist(),
                        np.asarray(xbox_action[:3]).round(3).tolist(),
                        np.asarray(xbox_action[3:6]).round(3).tolist(),
                    ),
                    flush=True,
                )
            return xbox_action, True
        # debounce window: keep intervening (zero motion) briefly after RB release
        if now - self._last_intervene_t < _HOLD_SECONDS:
            self._insert_ticks = 0
            self._spiral_px = self._spiral_py = 0.0
            return np.zeros(7, dtype=np.float32), True
        self._insert_ticks = 0
        self._spiral_px = self._spiral_py = 0.0
        if not self._armed:   # 开局未按过 RB -> 保持零动作不动(不放 policy)
            return np.zeros(7, dtype=np.float32), True
        return action, False

    def step(self, action):
        new_action, intervened = self.action(np.asarray(action))
        obs, rew, done, truncated, info = self.env.step(new_action)
        if intervened:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info


class HoldGripperWrapper(gym.ActionWrapper):
    """Force the gripper action to a no-op so the plug stays clamped throughout the
    whole insert-only episode (hold-grip). Neutralizes BOTH policy and intervention
    gripper output -> sidesteps the demo(+1=closed)/env(+1=open) gripper-sign mismatch
    that would otherwise OPEN the gripper and drop the plug.

    action[6] -> hold_value (default 0.0). The base env gripper command treats |x|<0.5
    as a no-op, so the gripper is never commanded and stays at its closed (~0.567) state.
    Place this directly above the base env (below the intervention wrapper)."""

    def __init__(self, env, hold_value: float = 0.0):
        super().__init__(env)
        self.hold_value = float(hold_value)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).copy()
        if a.shape[-1] >= 7:
            a[6] = self.hold_value
        return a
