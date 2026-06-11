#!/usr/bin/env python3
"""plug_scene.py

Plug-insertion task scene for FR3 robot in IsaacLab.

Composes the full simulation scene:
  - FR3 (Franka) robot via local fr3.usd
  - Table surface
  - Plug fixture (rigid body) + socket (static target)
  - Dome lighting (via sim_utils)

Follows DROID's InteractiveScene pattern (phase0_minimal_smoke.py) so that
PhysX tensor API returns torch tensors and ArticulationRootAPI is applied.

Key ordering constraint:
    AppLauncher MUST be created BEFORE any isaaclab / omni imports.

Usage (standalone verification on fr3-desktop-ts):
    source /home/robot/miniconda3/bin/activate isaaclab
    python3 plug_scene.py --headless --steps 10
    python3 plug_scene.py --show-gui --steps 50

When imported as a module (plug_scene_preview.py, training env), the
caller MUST create AppLauncher / SimulationApp before importing this module.
"""

from __future__ import annotations

# ===========================================================================
# MANDATORY: AppLauncher must be created BEFORE any isaaclab imports.
# ===========================================================================
import sys as _sys

if __name__ == "__main__":
    from isaaclab.app import AppLauncher
    _show_gui = "--show-gui" in _sys.argv
    _headless = not _show_gui
    _parser = __import__("argparse").ArgumentParser()
    AppLauncher.add_app_launcher_args(_parser)
    _app_args = _parser.parse_args(["--headless"] if _headless else [])
    _app_launcher = AppLauncher(_app_args)
    _standalone_app = _app_launcher.app

# ===========================================================================
# Now safe to import isaaclab & omni modules
# ===========================================================================
import argparse
import dataclasses
import math
import os
import sys
import time
from typing import Any, Optional

import numpy as np
import torch

# --- isaaclab core ---
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext

# A3: TiledCamera (optional; local may lack isaaclab)
try:
    from isaaclab.sensors import TiledCameraCfg
    HAS_TILED_CAMERA = True
except ImportError:
    HAS_TILED_CAMERA = False
    TiledCameraCfg = None  # type: ignore[assignment,misc]


# ===========================================================================
# Constants
# ===========================================================================
FR3_HOME_JOINTS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])
FR3_USD_PATH = "/home/robot/serl_projects/hil-serl-fr3/sim/assets/fr3.usd"

TABLE_HEIGHT = 0.74              # metres
TABLE_SIZE = (1.2, 0.8, 0.04)   # (x, y, z) metres
TABLE_COLOR = (0.4, 0.26, 0.13) # warm wood brown

PLUG_RADIUS = 0.015             # 1.5 cm
PLUG_HEIGHT = 0.06              # 6 cm
PLUG_COLOR = (0.2, 0.2, 0.8)   # blue

SOCKET_RADIUS = 0.018
SOCKET_DEPTH = 0.04
SOCKET_COLOR = (0.8, 0.8, 0.8)  # light grey

PLUG_POSITION = (-0.15, 0.0, TABLE_HEIGHT + PLUG_HEIGHT / 2.0)
SOCKET_POSITION = (0.15, 0.0, TABLE_HEIGHT + 0.01)


# ===========================================================================
# Scene asset configs (follows DROID build_primitive_scene pattern)
# ===========================================================================
def _build_fr3_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """Build FR3 articulation config using local fr3.usd."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=FR3_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-7]"],
                stiffness=400.0,
                damping=80.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                stiffness=200.0,
                damping=50.0,
            ),
        },
    )


def _build_table_cfg(prim_path: str = "{ENV_REGEX_NS}/Table") -> RigidObjectCfg:
    """Static table surface."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT),
        ),
    )


def _build_plug_cfg(prim_path: str = "{ENV_REGEX_NS}/Plug") -> RigidObjectCfg:
    """Rigid body plug cylinder."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=PLUG_RADIUS,
            height=PLUG_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PLUG_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=PLUG_POSITION,
        ),
    )


def _build_socket_cfg(prim_path: str = "{ENV_REGEX_NS}/Socket") -> RigidObjectCfg:
    """Static socket target."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CylinderCfg(
            radius=SOCKET_RADIUS,
            height=SOCKET_DEPTH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=SOCKET_COLOR),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SOCKET_POSITION,
        ),
    )


# ===========================================================================
# A3: Camera builders (side_policy_cam + wrist_1_cam)
# ===========================================================================
@dataclasses.dataclass
class _StubTiledCamera:
    """Local stand-in for TiledCameraCfg when isaaclab is not installed.

    Allows dataclass instantiation + field-presence test without IsaacLab.
    Desktop env (mainline agent) will replace with real TiledCameraCfg.
    """
    prim_path: str
    position: tuple = (0.0, 0.0, 0.0)
    target: tuple = (0.0, 0.0, 0.0)


def _build_side_policy_cam_cfg(prim_path: str = "{ENV_REGEX_NS}/side_policy_cam") -> Any:
    """Top-down side camera looking at plug area (policy network's primary view).

    Position: (0.5, 0.0, 0.5) above and to the side of the workspace.
    Look-at:  (0.0, 0.0, TABLE_HEIGHT) — plug insertion center.
    Resolution: 128x128 (matches IMAGE_SHAPE in sim/data/contract.py).
    """
    if TiledCameraCfg is None:
        return _StubTiledCamera(
            prim_path=prim_path,
            position=(0.5, 0.0, 0.5),
            target=(0.0, 0.0, TABLE_HEIGHT),
        )
    return TiledCameraCfg(  # type: ignore[call-arg]
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(  # type: ignore[attr-defined]
            pos=(0.5, 0.0, 0.5),
            rot=(0.0, -1.0, 0.0, 0.0),  # 180 deg around Y, look down at table
            convention="world",
        ),
        data_type="rgb",
        spawn=sim_utils.PinholeCameraCfg(  # type: ignore[attr-defined]
            focal_length=24.0,
            focus_distance=1.5,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 5.0),
        ),
        width=128,
        height=128,
    )


def _build_wrist_1_cam_cfg(prim_path: str = "{ENV_REGEX_NS}/wrist_1_cam") -> Any:
    """Wrist-mounted camera on gripper link (policy's secondary view).

    Position: relative to gripper link (offset 0, 0, 0.05).
    Look-at:  downward at plug.
    """
    if TiledCameraCfg is None:
        return _StubTiledCamera(
            prim_path=prim_path,
            position=(0.0, 0.0, 0.05),
            target=(0.0, 0.0, -0.1),
        )
    return TiledCameraCfg(  # type: ignore[call-arg]
        prim_path=prim_path,
        offset=TiledCameraCfg.OffsetCfg(  # type: ignore[attr-defined]
            pos=(0.0, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),  # identity quat
            convention="world",
        ),
        data_type="rgb",
        spawn=sim_utils.PinholeCameraCfg(  # type: ignore[attr-defined]
            focal_length=12.0,
            focus_distance=0.3,
            horizontal_aperture=15.0,
            clipping_range=(0.01, 1.0),
        ),
        width=128,
        height=128,
    )


# ===========================================================================
# PlugSceneCfg — InteractiveScene dataclass (DROID pattern)
# ===========================================================================
@dataclasses.dataclass
class PlugSceneCfg(InteractiveSceneCfg):
    """InteractiveScene config for the plug-insertion task.

    A3 改造 (PLAN-A3):
      + side_policy_cam: top-down side camera (policy primary view)
      + wrist_1_cam:     gripper-mounted camera (policy secondary view)
    sim/data/contract.py 中 side_classifier 是 side_policy 的 alias in sim pkl schema。
    """
    num_envs: int = 1
    env_spacing: float = 2.5
    replicate_physics: bool = True

    robot: Any = dataclasses.field(default_factory=_build_fr3_cfg)
    table: Any = dataclasses.field(default_factory=_build_table_cfg)
    plug: Any = dataclasses.field(default_factory=_build_plug_cfg)
    socket: Any = dataclasses.field(default_factory=_build_socket_cfg)

    # A3: 2 camera sensors (policy image keys)
    side_policy_cam: Any = dataclasses.field(default_factory=_build_side_policy_cam_cfg)
    wrist_1_cam: Any = dataclasses.field(default_factory=_build_wrist_1_cam_cfg)


# ===========================================================================
# PlugScene
# ===========================================================================
class PlugScene:
    """Plug-insertion scene: FR3 robot + table + plug fixture.

    Uses InteractiveScene for proper IsaacLab lifecycle::

        scene = PlugScene(headless=True)
        scene.reset()
        scene.set_joint_positions(q)
        ee = scene.get_ee_pose()
        scene.step(action)
        scene.close()
    """

    def __init__(
        self,
        headless: bool = True,
        device: Optional[str] = None,
        dt: float = 1.0 / 30.0,
        substeps: int = 10,
        randomize: bool = False,
        dr_seed: Optional[int] = None,
    ):
        from sim.data.contract import RANDOMIZE_SEED_DEFAULT
        self._headless = headless
        self._dt = dt
        self._substeps = substeps
        self._device = device  # resolved during _build_scene
        # A8: domain randomization
        self._randomize = randomize
        self._dr_seed = dr_seed if dr_seed is not None else RANDOMIZE_SEED_DEFAULT
        self._dr_rng: Optional[np.random.Generator] = None
        self._dr_samples: dict = {}  # for info() reporting

        # Scene handles
        self._sim: Optional[SimulationContext] = None
        self._scene: Optional[InteractiveScene] = None

        self._built = False

    # ------------------------------------------------------------------
    # Scene construction (DROID InteractiveScene pattern)
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        """Create SimulationContext, compose InteractiveScene, init physics."""
        print("[SCENE] Building plug-insertion scene...")

        sim_cfg = SimulationCfg(dt=self._dt, render_interval=1)
        self._sim = SimulationContext(sim_cfg)

        # Resolve device after SimulationContext is created
        if self._device is None:
            self._device = str(self._sim.device)

        # Build scene via InteractiveSceneCfg
        scene_cfg = PlugSceneCfg(num_envs=1, env_spacing=2.5)
        self._scene = InteractiveScene(scene_cfg)
        print("[SCENE] InteractiveScene composed")

        # Physics initialisation
        self._sim.reset()
        self._scene.update(self._dt)
        print("[SCENE] Scene built successfully.")

    @property
    def robot(self):
        """The FR3 articulation."""
        return self._scene["robot"] if self._scene is not None else None

    @property
    def plug(self):
        """The plug rigid object."""
        return self._scene["plug"] if self._scene is not None else None

    @property
    def socket(self):
        """The socket rigid object."""
        return self._scene["socket"] if self._scene is not None else None

    @property
    def table(self):
        """The table rigid object."""
        return self._scene["table"] if self._scene is not None else None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Build scene if needed, reset physics, move robot to home pose."""
        if not self._built:
            self._build_scene()

        # Move robot to home position
        self.set_joint_positions(FR3_HOME_JOINTS)

        # Additional physics steps to settle the scene
        for _ in range(self._substeps * 3):
            self._sim.step(render=False)
        self._scene.update(self._dt)

        print("[SCENE] Reset to home position.")

    def set_joint_positions(self, q: np.ndarray) -> None:
        """Set FR3 arm joint positions (7 DoF).

        Args:
            q: Joint angles in radians, shape (7,).
        """
        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot not initialised. Call reset() first.")

        q = np.asarray(q, dtype=np.float32).ravel()
        if q.shape[0] != 7:
            raise ValueError(f"Expected 7 joint angles, got {q.shape[0]}")

        device = robot.data.joint_pos.device
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, :7] = torch.from_numpy(q).float().unsqueeze(0).to(device)
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        self._scene.write_data_to_sim()

        # Step simulation to converge
        for _ in range(self._substeps):
            self._sim.step(render=False)
        self._scene.update(self._dt)

    def get_ee_pose(self) -> np.ndarray:
        """Get current end-effector pose as (x, y, z, qx, qy, qz, qw).

        Returns:
            np.ndarray of shape (7,) — position + quaternion (xyzw).
        """
        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot not initialised. Call reset() first.")

        ee_pos = robot.data.body_pos_w[0, -1].cpu().numpy()
        ee_quat_wxyz = robot.data.body_quat_w[0, -1].cpu().numpy()

        # Convert (w,x,y,z) -> (x,y,z,w)
        return np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0],
        ])

    def get_joint_positions(self) -> np.ndarray:
        """Get current FR3 arm joint positions (7 DoF).

        Returns:
            np.ndarray of shape (7,) — joint angles in radians.
        """
        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot not initialised. Call reset() first.")

        return robot.data.joint_pos[0, :7].cpu().numpy()

    def get_plug_pose(self) -> np.ndarray:
        """Get current plug world pose as (x, y, z, qx, qy, qz, qw).

        Returns:
            np.ndarray of shape (7,).
        """
        plug = self.plug
        if plug is None:
            raise RuntimeError("Plug not initialised. Call reset() first.")

        pos = plug.data.root_pos_w[0].cpu().numpy()
        quat_wxyz = plug.data.root_quat_w[0].cpu().numpy()
        return np.array([
            pos[0], pos[1], pos[2],
            quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0],
        ])

    def step(self, action: np.ndarray) -> None:
        """Apply an action and step the simulation.

        Args:
            action: np.ndarray of shape (7,) — target joint angles in radians.
        """
        if self._sim is None:
            raise RuntimeError("Simulation not initialised. Call reset() first.")

        action = np.asarray(action, dtype=np.float32).ravel()
        self.set_joint_positions(action)

    def close(self) -> None:
        """Shut down the simulation context."""
        if self._sim is not None:
            # SimulationContext has no close(); use AppLauncher cleanup
            try:
                from omni.isaac.core.utils.stage import close_stage
                close_stage()
            except Exception:
                pass
            self._built = False
            print("[SCENE] Simulation closed.")

    # ------------------------------------------------------------------
    # A8: domain randomization (per PLAN-A8)
    # ------------------------------------------------------------------
    def _init_dr_rng(self) -> None:
        """Initialize DR RNG with the configured seed (called from reset())."""
        if self._randomize and self._dr_rng is None:
            self._dr_rng = np.random.default_rng(self._dr_seed)

    def _randomize_lighting(self, rng: np.random.Generator) -> float:
        """Sample light intensity in [LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX].

        Returns the sampled intensity (scalar float).
        """
        from sim.data.contract import LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX
        return float(rng.uniform(LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX))

    def _randomize_camera_pose(
        self, rng: np.random.Generator,
    ) -> tuple[float, float]:
        """Sample camera yaw/pitch perturbation in ±5° range.

        Returns:
            (yaw_deg, pitch_deg) tuple.
        """
        from sim.data.contract import (
            CAMERA_YAW_RANGE_DEG, CAMERA_PITCH_RANGE_DEG,
        )
        yaw = float(rng.uniform(*CAMERA_YAW_RANGE_DEG))
        pitch = float(rng.uniform(*CAMERA_PITCH_RANGE_DEG))
        return yaw, pitch

    def _randomize_plug_pose(
        self, rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Sample plug xy jitter (±1cm) and rz jitter (±0.1 rad).

        Returns:
            (dx, dy, drz) tuple, all in metres / radians.
        """
        from sim.data.contract import PLUG_XY_JITTER_M, PLUG_RZ_JITTER_RAD
        dx = float(rng.uniform(-PLUG_XY_JITTER_M, PLUG_XY_JITTER_M))
        dy = float(rng.uniform(-PLUG_XY_JITTER_M, PLUG_XY_JITTER_M))
        drz = float(rng.uniform(-PLUG_RZ_JITTER_RAD, PLUG_RZ_JITTER_RAD))
        return dx, dy, drz

    def info_dr_samples(self) -> dict:
        """Return recorded DR sample values (for debug + L3 reporting).

        Returns:
            dict with keys: light_intensity, camera_yaw_deg, camera_pitch_deg,
                            plug_dx, plug_dy, plug_drz.
        """
        return dict(self._dr_samples)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def sim(self) -> Optional[SimulationContext]:
        return self._sim

    @property
    def scene(self) -> Optional[InteractiveScene]:
        return self._scene

    @property
    def device(self) -> str:
        return self._device or "cuda:0"

    @property
    def num_joints(self) -> int:
        return 7

    def info(self) -> dict:
        """Return a summary of the current scene state."""
        return {
            "built": self._built,
            "device": self.device,
            "headless": self._headless,
            "ee_pose": self.get_ee_pose().tolist() if self._built else None,
            "joint_pos": self.get_joint_positions().tolist() if self._built else None,
            "plug_pose": self.get_plug_pose().tolist() if self._built else None,
        }


# ===========================================================================
# Standalone verification
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plug-insertion scene — standalone verification"
    )
    parser.add_argument("--headless", action="store_true", default=True,
                        help="Run headless (default)")
    parser.add_argument("--show-gui", action="store_true",
                        help="Show Isaac Sim GUI (disables headless)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of verification steps (default: 10)")
    args = parser.parse_args()

    headless = not args.show_gui

    print("=" * 60)
    print("  PlugScene — Standalone Verification")
    print("=" * 60)
    print(f"  Headless:    {headless}")
    print(f"  Steps:       {args.steps}")
    print(f"  Device:      {'cuda:0' if torch.cuda.is_available() else 'cpu'}")

    scene = PlugScene(headless=headless)
    scene.reset()

    print(f"\n[TEST] Initial EE pose: {scene.get_ee_pose()}")
    print(f"[TEST] Initial joints:  {scene.get_joint_positions()}")
    print(f"[TEST] Plug pose:       {scene.get_plug_pose()}")

    # Small sinusoidal motion for verification
    t = np.linspace(0, 2 * math.pi, args.steps)
    for step in range(args.steps):
        q = FR3_HOME_JOINTS.copy()
        q[0] += 0.05 * np.sin(t[step])
        q[1] += 0.05 * np.sin(t[step] + 0.3)
        scene.step(q)

        ee = scene.get_ee_pose()
        plug = scene.get_plug_pose()
        print(f"  Step {step:3d} | EE [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]"
              f" | Plug [{plug[0]:.4f}, {plug[1]:.4f}, {plug[2]:.4f}]")

    print(f"\n[TEST] Scene info: {scene.info()}")
    scene.close()
    print("[DONE] PlugScene verification complete.")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
