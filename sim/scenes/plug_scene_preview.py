#!/usr/bin/env python3
"""plug_scene_preview.py

Launch IsaacLab with the plug-insertion scene and optionally replay
GELLO demo trajectories.

Based on DROID's phase3_scene_preview.py structure:
  1. Parse CLI args (--headless / --show-gui / --npz / --replay)
  2. Create AppLauncher (BEFORE any isaaclab imports)
  3. Build PlugScene
  4. If --npz given, replay trajectory; otherwise interactive hold at home

Usage (on fr3-desktop-ts):

    # Show GUI, hold at home position
    source /home/robot/miniconda3/bin/activate isaaclab
    python3 plug_scene_preview.py --show-gui

    # Replay a GELLO demo trajectory in the GUI
    python3 plug_scene_preview.py --show-gui --npz /path/to/demo.npz

    # Headless replay for CI / quick validation
    python3 plug_scene_preview.py --headless --npz /path/to/demo.npz --steps 200

    # Replay with custom start index
    python3 plug_scene_preview.py --show-gui --npz demo.npz --start-idx 50 --steps 100
"""

from __future__ import annotations

# ===========================================================================
# MANDATORY: AppLauncher must be created BEFORE any isaaclab imports
# ===========================================================================
import sys

_show_gui = "--show-gui" in sys.argv
_headless = "--headless" in sys.argv or not _show_gui

from isaaclab.app import AppLauncher
import argparse as _argparse
_parser = _argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
_app_args = _parser.parse_args(["--headless"] if _headless else [])
_app_launcher = AppLauncher(_app_args)
app = _app_launcher.app

# ===========================================================================
# Now safe to import everything
# ===========================================================================
import argparse
import math
import os
import time
import traceback

import numpy as np

try:
    import torch
except ImportError:
    torch = None

# --- Add project paths so `from plug_scene import ...` works ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent of scenes/ so "from scenes.plug_scene" resolves
SIM_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in [SCRIPT_DIR, SIM_DIR]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# --- Import our scene ---
from plug_scene import PlugScene, FR3_HOME_JOINTS

# --- FK converter for validation (optional) ---
HAS_FK = False
FK_SCRIPTS = "/home/robot/serl_projects/hil-serl-fr3/scripts/gello_pipeline"
if os.path.isdir(FK_SCRIPTS) and FK_SCRIPTS not in sys.path:
    sys.path.insert(0, FK_SCRIPTS)
try:
    from fk_converter import forward_kinematics, trajectory_to_poses
    HAS_FK = True
    print("[IMPORT] fk_converter loaded")
except ImportError:
    print("[WARN] fk_converter not available — FK comparison disabled")


# ===========================================================================
# Trajectory loading (reused from isaaclab_franka_sim.py pattern)
# ===========================================================================
def load_npz_trajectory(npz_path: str, start_idx: int, steps: int) -> np.ndarray:
    """Load joint trajectory from GELLO demo .npz file.

    Expects 'joint_poses' field with shape (N, 7).
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    print(f"[DATA] Keys in NPZ: {list(data.keys())}")

    if "joint_poses" not in data:
        raise KeyError(
            f"NPZ must contain 'joint_poses' field. Available: {list(data.keys())}"
        )

    full_traj = np.asarray(data["joint_poses"], dtype=np.float64)
    print(f"[DATA] Full trajectory shape: {full_traj.shape}")

    if full_traj.ndim != 2 or full_traj.shape[1] != 7:
        raise ValueError(f"Expected (N, 7) trajectory, got {full_traj.shape}")

    end_idx = min(start_idx + steps, len(full_traj))
    q_traj = full_traj[start_idx:end_idx]
    print(f"[DATA] Using steps {start_idx}:{end_idx} ({len(q_traj)} frames)")
    return q_traj


def generate_synthetic_trajectory(steps: int) -> np.ndarray:
    """Generate smooth sinusoidal joint trajectory around FR3 home position."""
    t = np.linspace(0, 2 * math.pi, steps)
    q = np.zeros((steps, 7))
    amp = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    phase = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
    for i in range(7):
        q[:, i] = FR3_HOME_JOINTS[i] + amp[i] * np.sin(t + phase[i])
    return q


# ===========================================================================
# Replay loop
# ===========================================================================
def replay_trajectory(scene: PlugScene, trajectory: np.ndarray) -> dict:
    """Replay a joint trajectory through the scene, collecting EE poses.

    Returns:
        dict with 'ee_poses', 'joint_poses', 'plug_poses', 'elapsed'.
    """
    n_steps = len(trajectory)
    ee_poses = []
    joint_poses = []
    plug_poses = []

    print(f"\n[REPLAY] Replaying {n_steps} steps...")
    start = time.time()

    for step in range(n_steps):
        q_target = trajectory[step]
        scene.step(q_target)

        ee = scene.get_ee_pose()
        joints = scene.get_joint_positions()
        plug = scene.get_plug_pose()

        ee_poses.append(ee.copy())
        joint_poses.append(joints.copy())
        plug_poses.append(plug.copy())

        if step % max(1, n_steps // 20) == 0:
            print(f"  Step {step:4d}/{n_steps}"
                  f" | EE [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]"
                  f" | Plug [{plug[0]:.4f}, {plug[1]:.4f}, {plug[2]:.4f}]")

    elapsed = time.time() - start
    print(f"[REPLAY] Completed in {elapsed:.2f}s ({n_steps / max(elapsed, 1e-6):.1f} steps/s)")

    return {
        "ee_poses": np.array(ee_poses),
        "joint_poses": np.array(joint_poses),
        "plug_poses": np.array(plug_poses),
        "elapsed": elapsed,
    }


# ===========================================================================
# Validation report
# ===========================================================================
def print_report(result: dict, trajectory: np.ndarray, has_fk: bool) -> bool:
    """Print a validation report. Returns True if all checks pass."""
    ee = result["ee_poses"]
    n = len(ee)

    print("\n" + "=" * 60)
    print("  PLUG SCENE PREVIEW — VALIDATION REPORT")
    print("=" * 60)
    print(f"  Steps replayed:   {n}")
    print(f"  Wall time:        {result['elapsed']:.2f}s")

    all_pass = True

    # Check 1: All poses finite
    finite_ok = np.isfinite(ee).all()
    tag = "[PASS]" if finite_ok else "[FAIL]"
    print(f"  {tag} All EE poses finite")
    if not finite_ok:
        all_pass = False

    # Check 2: EE within workspace
    ws_ok = np.all(np.abs(ee[:, :3]) < 1.5)
    tag = "[PASS]" if ws_ok else "[FAIL]"
    print(f"  {tag} EE within 1.5m workspace envelope")
    if not ws_ok:
        all_pass = False

    # Check 3: Quaternion norms
    q_norms = np.linalg.norm(ee[:, 3:7], axis=1)
    norm_ok = np.allclose(q_norms, 1.0, atol=1e-4)
    tag = "[PASS]" if norm_ok else "[FAIL]"
    print(f"  {tag} Quaternion norms ≈ 1.0 (min={q_norms.min():.6f})")
    if not norm_ok:
        all_pass = False

    # Check 4: Plug stayed on or near table
    plug = result["plug_poses"]
    plug_z = plug[:, 2]
    plug_ok = np.all(plug_z > 0.5)  # plug shouldn't fall through table
    tag = "[PASS]" if plug_ok else "[FAIL]"
    print(f"  {tag} Plug z > 0.5m (min={plug_z.min():.4f})")
    if not plug_ok:
        all_pass = False

    # Check 5: FK comparison (if available)
    if has_fk and n > 0:
        fk_poses = trajectory_to_poses(trajectory[:n])
        pos_err = np.linalg.norm(ee[:, :3] - fk_poses[:, :3], axis=1)
        fk_ok = pos_err.mean() < 0.02  # 2cm tolerance for sim vs FK
        tag = "[PASS]" if fk_ok else "[FAIL]"
        print(f"  {tag} Sim vs FK mean position error < 2cm ({pos_err.mean():.6f}m)")
        if not fk_ok:
            all_pass = False

    # EE workspace range
    print(f"\n  EE workspace (m):")
    for axis, name in enumerate(["X", "Y", "Z"]):
        lo, hi = ee[:, axis].min(), ee[:, axis].max()
        print(f"    {name}: [{lo:.4f}, {hi:.4f}]")

    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n  Overall: {overall}")
    print("=" * 60)
    return all_pass


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plug scene preview — IsaacLab GUI with optional trajectory replay"
    )
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run headless (no GUI)")
    parser.add_argument("--show-gui", action="store_true", default=False,
                        help="Show Isaac Sim GUI")
    parser.add_argument("--npz", type=str, default=None,
                        help="Path to GELLO demo .npz (must contain 'joint_poses')")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting index in trajectory (default: 0)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Max replay steps (default: 200)")
    parser.add_argument("--hold-home", action="store_true", default=False,
                        help="Hold at home position (no trajectory, useful for GUI inspection)")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0,
                        help="Simulation dt (default: 1/60)")
    parser.add_argument("--substeps", type=int, default=10,
                        help="Physics substeps per control step (default: 10)")
    args = parser.parse_args()

    headless = args.headless and not args.show_gui

    print("=" * 60)
    print("  Plug Scene Preview — FR3 Plug-Insertion")
    print("=" * 60)
    print(f"  Headless:    {headless}")
    print(f"  Show GUI:    {args.show_gui}")
    print(f"  NPZ file:    {args.npz or '(none)'}")
    print(f"  Steps:       {args.steps}")
    print(f"  Hold home:   {args.hold_home}")
    print(f"  FK available:{HAS_FK}")

    # ------------------------------------------------------------------
    # Build scene
    # ------------------------------------------------------------------
    scene = PlugScene(
        headless=headless,
        dt=args.dt,
        substeps=args.substeps,
    )
    scene.reset()

    print(f"\n[INIT] EE pose:  {scene.get_ee_pose()}")
    print(f"[INIT] Joints:   {scene.get_joint_positions()}")
    print(f"[INIT] Plug pose:{scene.get_plug_pose()}")

    # ------------------------------------------------------------------
    # Load or generate trajectory
    # ------------------------------------------------------------------
    if args.hold_home:
        print("\n[HOLD] Holding at home position. Press Ctrl+C to exit.")
        if not headless:
            try:
                while True:
                    scene.step(FR3_HOME_JOINTS)
            except KeyboardInterrupt:
                print("\n[HOLD] Interrupted.")
        else:
            for _ in range(100):
                scene.step(FR3_HOME_JOINTS)
            print("[HOLD] Done (headless hold).")

    elif args.npz:
        print(f"\n[REPLAY] Loading trajectory from {args.npz}")
        trajectory = load_npz_trajectory(args.npz, args.start_idx, args.steps)
        result = replay_trajectory(scene, trajectory)
        print_report(result, trajectory, HAS_FK)

    else:
        print(f"\n[REPLAY] No NPZ provided — using synthetic trajectory ({args.steps} steps)")
        trajectory = generate_synthetic_trajectory(args.steps)
        result = replay_trajectory(scene, trajectory)
        print_report(result, trajectory, HAS_FK)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    scene.close()
    print("\n[DONE] Plug scene preview complete.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
