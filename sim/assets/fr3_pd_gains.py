"""Canonical PD / actuator gains for the native FR3 (fr3.usd) 'stable_sim' profile.

Single source of truth for the implicit-actuator gains shared by:
  - sim/scenes/plug_scene.py            (_build_fr3_cfg, the production PlugScene)
  - sim/assets/official_fr3_loader.py   (_build_native_fr3_cfg, stable_sim branch)

These are SCALARS only (no isaaclab import) so the module loads in CI and can be
unit-tested without Isaac Sim. Consumers wrap them in ImplicitActuatorCfg.

NOTE: sim/data/gello_replay.py deliberately does NOT use these. It loads the STOCK
Isaac panda_arm_hand.usd and keeps Isaac's stock low-PD panda gains (arm 80/4); its
gains are a different regime for a different USD and must not be unified here.

The 'hardware_kq' low-gain study profile in official_fr3_loader.py is intentionally
kept inline there and is out of scope for this canonical module.
"""
from __future__ import annotations

# --- Arm (fr3_joint1..7) ---------------------------------------------------
# Isaac high-PD convention so the arm tracks commands without sagging.
ARM_STIFFNESS: float = 400.0
ARM_DAMPING: float = 80.0

# Per-joint torque envelope from franka_hardware_left.yaml (j1-4: 87 Nm, j5-7: 12 Nm).
ARM_EFFORT_LIMIT_SIM: dict[str, float] = {
    "fr3_joint1": 87.0,
    "fr3_joint2": 87.0,
    "fr3_joint3": 87.0,
    "fr3_joint4": 87.0,
    "fr3_joint5": 12.0,
    "fr3_joint6": 12.0,
    "fr3_joint7": 12.0,
}

# --- Gripper (fr3_finger_joint.*) -----------------------------------------
GRIPPER_STIFFNESS: float = 2e3
GRIPPER_DAMPING: float = 1e2

__all__ = [
    "ARM_STIFFNESS",
    "ARM_DAMPING",
    "ARM_EFFORT_LIMIT_SIM",
    "GRIPPER_STIFFNESS",
    "GRIPPER_DAMPING",
]
