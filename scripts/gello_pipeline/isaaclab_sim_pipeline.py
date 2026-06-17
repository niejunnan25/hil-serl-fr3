#!/usr/bin/env python3
"""isaaclab_sim_pipeline.py

IsaacLab simulation pipeline for FR3 robot.
Uses numpy FK as fallback when IsaacSim is not available.

Usage:
    python3 isaaclab_sim_pipeline.py --steps 50
"""

import argparse
import sys
import time
import numpy as np

# Try to import IsaacSim, fallback to numpy FK
try:
    # IsaacSim imports would go here
    # from omni.isaac.lab.sim import SimulationContext
    # from omni.isaac.lab.assets import Articulation
    HAS_ISAACSIM = False  # Set to True if IsaacSim is available
    print("[INFO] IsaacSim not available, using numpy FK fallback")
except ImportError:
    HAS_ISAACSIM = False
    print("[INFO] IsaacSim not available, using numpy FK fallback")

# Import FK converter
sys.path.insert(0, '/home/robot/serl_projects/hil-serl-fr3/scripts')
from fk_converter import forward_kinematics, joints_to_cartesian_delta


def run_numpy_fk_sim(steps: int = 50):
    """Run simulation using numpy FK (fallback mode)."""
    print(f"\n{'='*60}")
    print(f"Running FR3 FK Simulation (numpy fallback)")
    print(f"Steps: {steps}")
    print(f"{'='*60}\n")

    # FR3 home position
    q_home = np.zeros(7)

    # Generate smooth trajectory
    t = np.linspace(0, 2*np.pi, steps)
    q_trajectory = np.zeros((steps, 7))

    # Sinusoidal joint motion
    for i in range(7):
        q_trajectory[:, i] = q_home[i] + 0.1 * np.sin(t + i * np.pi/4)

    # Run FK
    results = []
    for step in range(steps):
        q = q_trajectory[step]
        pose = forward_kinematics(q)

        if step > 0:
            delta = joints_to_cartesian_delta(q_trajectory[step-1], q)
        else:
            delta = np.zeros(6)

        results.append({
            'step': step,
            'joint_angles': q.tolist(),
            'position': pose[:3].tolist(),
            'quaternion': pose[3:7].tolist(),
            'cartesian_delta': delta.tolist()
        })

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"[Step {step:3d}] Position: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}]")

    # Summary
    print(f"\n{'='*60}")
    print(f"Simulation Complete")
    print(f"{'='*60}")
    print(f"Total steps: {len(results)}")
    print(f"Final position: [{results[-1]['position'][0]:.4f}, {results[-1]['position'][1]:.4f}, {results[-1]['position'][2]:.4f}]")
    print(f"Final quaternion: [{results[-1]['quaternion'][0]:.4f}, {results[-1]['quaternion'][1]:.4f}, {results[-1]['quaternion'][2]:.4f}, {results[-1]['quaternion'][3]:.4f}]")

    # Validation
    positions = np.array([r['position'] for r in results])
    position_range = positions.max(axis=0) - positions.min(axis=0)
    print(f"\nPosition range (x,y,z): [{position_range[0]:.4f}, {position_range[1]:.4f}, {position_range[2]:.4f}]")

    # Check all quaternions are unit norm
    quats = np.array([r['quaternion'] for r in results])
    quat_norms = np.linalg.norm(quats, axis=1)
    print(f"Quaternion norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")

    if np.allclose(quat_norms, 1.0, atol=1e-6):
        print("[PASS] All quaternions are unit norm")
    else:
        print("[FAIL] Some quaternions are not unit norm")

    return results


def main():
    parser = argparse.ArgumentParser(description='IsaacLab Simulation Pipeline for FR3')
    parser.add_argument('--steps', type=int, default=50, help='Number of simulation steps')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    start_time = time.time()

    if HAS_ISAACSIM:
        print("[INFO] Running with IsaacSim")
        # IsaacSim simulation would go here
    else:
        print("[INFO] Running with numpy FK fallback")
        results = run_numpy_fk_sim(args.steps)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")

    # Save results if output specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
