#!/usr/bin/env python3
"""fk_converter.py

FR3 (Franka Emika) Forward Kinematics converter.

Converts joint-space trajectories to Cartesian-space deltas using
DH-parameter-based FK with scipy.spatial.transform.Rotation for
proper quaternion/rotation handling and SERL-compatible rotvec deltas.

The FR3 kinematic chain is identical to the Panda (7 revolute joints).
DH parameters are taken from the Franka Emika documentation / URDF and use
the modified/Craig convention.

Usage:
    from fk_converter import joints_to_cartesian_delta, forward_kinematics

    # Single frame FK -> [x, y, z, qx, qy, qz, qw]
    pose = forward_kinematics(q)

    # Cartesian delta between two joint states -> [dx, dy, dz, rx, ry, rz]
    delta = joints_to_cartesian_delta(q_prev, q_curr)
"""

import numpy as np
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# FR3 DH Parameters (modified/Craig DH convention)
# Source: franka_description FR3 kinematics.yaml / official Franka docs
#
# Joint | a_{i-1} (m) | d_i (m)  | alpha_{i-1} (rad) | theta_offset (rad)
# ------|------------|-------------------|-----------|--------------------
#  1    | 0            | 0.333    | 0                 | 0
#  2    | 0            | 0        | -pi/2             | 0
#  3    | 0            | 0.316    | pi/2              | 0
#  4    | 0.0825       | 0        | pi/2              | 0
#  5    | -0.0825      | 0.384    | -pi/2             | 0
#  6    | 0            | 0        | pi/2              | 0
#  7    | 0.088        | 0        | pi/2              | 0
# Flange | 0          | 0.107    | 0                 | 0
# TCP    | 0          | 0.1034   | 0                 | -pi/4  (Franka hand F_T_EE)
# ---------------------------------------------------------------------------

_N_JOINTS = 7

# Modified DH parameters per joint: [a_{i-1}, d_i, alpha_{i-1}, theta_offset]
_FR3_DH = np.array([
    [0.0,      0.333,   0.0,        0.0],
    [0.0,      0.0,    -np.pi / 2,  0.0],
    [0.0,      0.316,   np.pi / 2,  0.0],
    [0.0825,   0.0,     np.pi / 2,  0.0],
   [-0.0825,   0.384,  -np.pi / 2,  0.0],
    [0.0,      0.0,     np.pi / 2,  0.0],
    [0.088,    0.0,     np.pi / 2,  0.0],
], dtype=np.float64)

# Fixed transform from joint 7 frame to link8 flange, then Franka-hand TCP.
_FLANGE_D = 0.107
_TCP_D = 0.1034
_TCP_RZ = -np.pi / 4


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Compute the 4x4 homogeneous transformation matrix from DH parameters.

    Modified/Craig DH convention:
        T = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [ct,      -st,      0.0,    a],
        [st * ca,  ct * ca, -sa,   -sa * d],
        [st * sa,  ct * sa,  ca,    ca * d],
        [0.0,      0.0,      0.0,   1.0],
    ], dtype=np.float64)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Compute end-effector pose from joint angles using FK.

    Args:
        q: Joint angles (7,) in radians.

    Returns:
        pose: (7,) array [x, y, z, qx, qy, qz, qw]
              - position (meters)
              - orientation as unit quaternion (scipy convention: scalar-last)
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    assert q.shape == (7,), f"Expected (7,) joint angles, got {q.shape}"

    # Build cumulative transform through the kinematic chain
    T = np.eye(4, dtype=np.float64)
    for i in range(_N_JOINTS):
        a_i, d_i, alpha_i, offset_i = _FR3_DH[i]
        theta_i = q[i] + offset_i
        T = T @ _dh_transform(a_i, alpha_i, d_i, theta_i)

    flange = np.eye(4, dtype=np.float64)
    flange[2, 3] = _FLANGE_D
    T = T @ flange

    c45 = np.cos(_TCP_RZ)
    s45 = np.sin(_TCP_RZ)
    T_tcp = np.array([
        [c45, -s45, 0.0, 0.0],
        [s45,  c45, 0.0, 0.0],
        [0.0,  0.0, 1.0, _TCP_D],
        [0.0,  0.0, 0.0, 1.0],
    ], dtype=np.float64)
    T = T @ T_tcp

    # Extract position
    position = T[:3, 3]

    # Extract rotation via scipy (handles singularity internally)
    R = T[:3, :3]
    quat = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]

    return np.concatenate([position, quat]).astype(np.float64)


def joints_to_cartesian_delta(
    q_prev: np.ndarray,
    q_curr: np.ndarray,
) -> np.ndarray:
    """Compute Cartesian-space delta between two joint configurations.

    Uses scipy Rotation for robust rotation delta computation.
    Returns rotation-vector deltas, matching SERL/franka_env action semantics.

    Args:
        q_prev: Previous joint angles (7,) in radians.
        q_curr: Current joint angles (7,) in radians.

    Returns:
        delta: (6,) array [dx, dy, dz, rx, ry, rz]
               - translation delta in meters
               - orientation delta as an axis-angle rotation vector in radians
    """
    q_prev = np.asarray(q_prev, dtype=np.float64).flatten()
    q_curr = np.asarray(q_curr, dtype=np.float64).flatten()
    assert q_prev.shape == (7,) and q_curr.shape == (7,), \
        f"Expected (7,) joint angles, got {q_prev.shape} and {q_curr.shape}"

    # FK both poses
    pose_prev = forward_kinematics(q_prev)
    pose_curr = forward_kinematics(q_curr)

    # Position delta: simple subtraction
    d_pos = pose_curr[:3] - pose_prev[:3]

    # Rotation delta: R_delta = R_curr @ R_prev^T
    # This gives the relative rotation from prev frame to curr frame.
    R_prev = Rotation.from_quat(pose_prev[3:7])
    R_curr = Rotation.from_quat(pose_curr[3:7])

    # scipy handles the composition; result is well-defined
    R_delta = R_curr * R_prev.inv()

    d_rotvec = R_delta.as_rotvec()

    return np.concatenate([d_pos, d_rotvec]).astype(np.float64)


# ---------------------------------------------------------------------------
# Convenience: batch FK for a trajectory
# ---------------------------------------------------------------------------
def trajectory_to_poses(q_trajectory: np.ndarray) -> np.ndarray:
    """Compute FK (quaternion pose) for every frame in a trajectory.

    Args:
        q_trajectory: (N, 7) joint angles in radians.

    Returns:
        poses: (N, 7) [x, y, z, qx, qy, qz, qw] per frame.
    """
    q_trajectory = np.asarray(q_trajectory, dtype=np.float64)
    N = q_trajectory.shape[0]
    poses = np.zeros((N, 7), dtype=np.float64)
    for i in range(N):
        poses[i] = forward_kinematics(q_trajectory[i])
    return poses


def trajectory_to_cartesian_deltas(q_trajectory: np.ndarray) -> np.ndarray:
    """Compute consecutive Cartesian deltas for a full trajectory.

    The first frame's delta is zero (no previous frame).

    Args:
        q_trajectory: (N, 7) joint angles in radians.

    Returns:
        deltas: (N, 6) [dx, dy, dz, rx, ry, rz]
                deltas[0] = [0, 0, 0, 0, 0, 0]
    """
    q_trajectory = np.asarray(q_trajectory, dtype=np.float64)
    N = q_trajectory.shape[0]
    deltas = np.zeros((N, 6), dtype=np.float64)
    for i in range(1, N):
        deltas[i] = joints_to_cartesian_delta(q_trajectory[i - 1], q_trajectory[i])
    return deltas


# ---------------------------------------------------------------------------
# Self-test / validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FR3 FK Converter Self-Test (scipy-based)")
    print("=" * 60)

    # --- Test 1: Home position (all joints at zero) ---
    q_home = np.zeros(7)
    pose_home = forward_kinematics(q_home)
    print(f"\n[Test 1] Home pose (q=[0]*7):")
    print(f"  position : {pose_home[:3]}")
    print(f"  quaternion: {pose_home[3:7]}")

    # Home TCP pose for the modified/Craig DH convention:
    # position ~ [0.088, 0.0, 0.8226].
    # Quaternion must always be unit norm.
    quat_home = pose_home[3:7]
    quat_norm = np.linalg.norm(quat_home)
    print(f"  ||quat|| = {quat_norm:.10f}  (should be ~1.0)")
    assert abs(quat_norm - 1.0) < 1e-8, "Quaternion should be unit norm"
    # Position should be finite and non-zero
    assert np.isfinite(pose_home[:3]).all(), "Position must be finite"
    assert np.linalg.norm(pose_home[:3]) > 0.1, "Position should be non-trivial"
    print("  PASSED")

    # --- Test 2: Small joint perturbation -> small Cartesian delta ---
    q2 = q_home.copy()
    q2[0] = 0.01  # small rotation on joint 1
    delta = joints_to_cartesian_delta(q_home, q2)
    print(f"\n[Test 2] Delta from joint 1 +0.01 rad:")
    print(f"  [dx, dy, dz, rx, ry, rz] = {delta}")
    delta_norm = np.linalg.norm(delta)
    print(f"  ||delta|| = {delta_norm:.6f}")
    assert delta_norm < 0.5, f"Delta should be small for small joint change, got {delta_norm}"

    # --- Test 3: Symmetry / identity delta ---
    delta_zero = joints_to_cartesian_delta(q_home, q_home)
    print(f"\n[Test 3] Identity delta (q_prev == q_curr):")
    print(f"  delta = {delta_zero}")
    assert np.allclose(delta_zero, 0.0, atol=1e-12), "Identity delta should be zero"
    print("  PASSED (all zeros)")

    # --- Test 4: Non-trivial known pose ---
    # A reachable configuration (modified from Franka docs)
    q_test = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.7])
    pose_test = forward_kinematics(q_test)
    quat_test = pose_test[3:7]
    quat_test_norm = np.linalg.norm(quat_test)
    print(f"\n[Test 4] Non-trivial pose (q=[0, -0.3, 0, -2, 0, 1.5, 0.7]):")
    print(f"  position : {pose_test[:3]}")
    print(f"  quaternion: {pose_test[3:7]}")
    print(f"  ||quat|| = {quat_test_norm:.10f}")
    assert abs(quat_test_norm - 1.0) < 1e-8, "Quaternion must be unit norm"
    assert np.isfinite(pose_test).all(), "All values must be finite"
    print("  PASSED")

    # --- Test 5: Batch trajectory ---
    N = 100
    rng = np.random.RandomState(42)
    q_traj = rng.randn(N, 7) * 0.01
    poses = trajectory_to_poses(q_traj)
    deltas = trajectory_to_cartesian_deltas(q_traj)

    print(f"\n[Test 5] Batch trajectory ({N} frames):")
    print(f"  poses  shape: {poses.shape}")
    print(f"  deltas shape: {deltas.shape}")
    assert poses.shape == (N, 7), f"Expected (N, 7), got {poses.shape}"
    assert deltas.shape == (N, 6), f"Expected (N, 6), got {deltas.shape}"
    assert np.allclose(deltas[0], 0.0), "First delta should be zero"

    # All quaternions should be unit norm
    quat_norms = np.linalg.norm(poses[:, 3:7], axis=1)
    assert np.allclose(quat_norms, 1.0, atol=1e-8), \
        f"All quaternion norms should be 1.0, min={quat_norms.min()}, max={quat_norms.max()}"
    print("  PASSED")

    # --- Test 6: Large rotation -- singularity handling ---
    # pitch = pi/2 is near gimbal lock for XYZ Euler
    q_large = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, 0.0])
    delta_large = joints_to_cartesian_delta(q_home, q_large)
    print(f"\n[Test 6] Large rotation (gimbal lock region):")
    print(f"  delta = {delta_large}")
    assert np.isfinite(delta_large).all(), "Delta should be finite even near singularity"
    print("  PASSED (no NaN/Inf near singularity)")

    print("\n" + "=" * 60)
    print("All tests PASSED")
    print("=" * 60)
