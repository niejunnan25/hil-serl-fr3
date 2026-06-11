#!/usr/bin/env python3
"""gello_replay.py

在 IsaacLab sim 中回放 GELLO demo 轨迹，记录 sim observation (图像+状态)，
输出 SERL pkl 格式数据集。

设计依据:
  - isaaclab_franka_sim.py: InteractiveScene + Franka USD 加载模式
  - fk_converter.py: 关节→笛卡尔 delta 转换
  - convert_to_pkl.py: SERL pkl transition 格式规范

工作流:
  1. 加载 GELLO demo npz (joint_poses, gripper_states, timestamps)
  2. fk_converter 做 FK 转换 (关节 → 笛卡尔 delta)
  3. IsaacLab sim 中逐帧回放关节轨迹
  4. 每帧记录 sim observation:
     - pixels: 相机 RGB (3, 128, 128) uint8 CHW
     - state:  25D SERL state per STATE_KEYS_ORDERED
       (tcp_pose 7 + tcp_vel 6 + tcp_force 3 + tcp_torque 3 + gripper 6 = 25)
  5. 构建 SERL pkl transitions list

A2: 8D 旧实现 (joint 7 + gripper 1) → 25D 新实现 (per sim/data/contract.py)
    force/torque 暂填 0 (A9 之后接 contact sensor)。

用法 (on fr3-desktop-ts):
    source /home/robot/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
    python gello_replay.py --npz /tmp/gello_demos/demo_20260610_120000.npz
    python gello_replay.py --npz /tmp/gello_demos/demo_20260610_120000.npz \\
        --output /tmp/sim_replays/demo_sim.pkl --sub-steps 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from typing import Any, Optional

import numpy as np

# ===========================================================================
# IsaacSim app (MUST be created before isaaclab/omni imports)
# ===========================================================================
try:
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    print("[WARN] IsaacSim not available, will use pure FK mode")

# ===========================================================================
# isaaclab imports (after SimulationApp)
# ===========================================================================
if HAS_SIM:
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        HAS_ISAACLAB = True
        print("[IMPORT] isaaclab loaded")
    except ImportError as e:
        HAS_ISAACLAB = False
        print(f"[WARN] isaaclab unavailable: {e}")
else:
    HAS_ISAACLAB = False

# ===========================================================================
# Project imports (tolerate missing fk_converter/normalize_action in CI/dev)
# ===========================================================================
# A4 deviation: wrap fk_converter / normalize_action imports in try/except.
# /home/robot/... only exists on fr3-desktop-ts, so a vanilla dev box
# cannot import this module otherwise — and that would block every
# constant test below. Surfaces None on failure; the actual replay_*()
# entry points then raise a clear error when invoked.
#
# A2 deviation: also search the local repo's scripts/ dir (where
# fk_converter.py and normalize_action.py live in this checkout),
# so replay_pure_fk() can actually run for unit tests on dev boxes
# that don't have the /home/robot/... path.
import pathlib
GELLO_PIPELINE_CANDIDATES = [
    "/home/robot/serl_projects/hil-serl-fr3/scripts/gello_pipeline",
    str(pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"),
]
for _p in GELLO_PIPELINE_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from fk_converter import trajectory_to_cartesian_deltas, trajectory_to_poses
    from normalize_action import normalize_action
    _GELLO_PIPELINE_OK = True
except ImportError as _e:
    trajectory_to_cartesian_deltas = None
    trajectory_to_poses = None
    normalize_action = None
    _GELLO_PIPELINE_OK = False
    print(f"[WARN] gello_pipeline modules unavailable: {_e}")

# ===========================================================================
# A4: scale values come from sim/data/contract.py (single source of truth).
# 旧 0.1 / 0.2 hardcode 是 pre-v2.1 spec 残留；A4 改用 contract.ACTION_SCALE.
# ===========================================================================
from sim.data.contract import ACTION_SCALE

DEFAULT_POS_SCALE     = ACTION_SCALE[0]   # 0.015  (dx, dy, dz)
DEFAULT_RPY_SCALE     = ACTION_SCALE[3]   # 0.1    (droll, dpitch, dyaw)
DEFAULT_GRIPPER_SCALE = ACTION_SCALE[6]   # 1.0    (gripper)

FRANKA_USD = "/home/robot/plug_insertion_sim/assets/panda_arm_hand.usd"
FR3_HOME_JOINTS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])

IMAGE_H, IMAGE_W, IMAGE_C = 128, 128, 3


# ===========================================================================
# NPZ loader
# ===========================================================================
def load_gello_npz(npz_path: str) -> dict[str, np.ndarray]:
    """加载 GELLO demo npz 文件。

    Returns:
        dict with keys: joint_poses (N,7), gripper_states (N,), timestamps (N,)
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    required = ["joint_poses", "gripper_states", "timestamps"]
    for key in required:
        if key not in data:
            raise KeyError(f"NPZ missing '{key}'. Available: {list(data.keys())}")

    joint_poses = np.asarray(data["joint_poses"], dtype=np.float64)
    gripper_states = np.asarray(data["gripper_states"], dtype=np.float64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)

    N = len(joint_poses)
    assert N >= 2, f"Need >=2 frames, got {N}"
    assert joint_poses.shape == (N, 7), f"Expected (N,7), got {joint_poses.shape}"

    print(f"[DATA] Loaded {npz_path}: {N} frames")
    return {
        "joint_poses": joint_poses,
        "gripper_states": gripper_states,
        "timestamps": timestamps,
    }


# ===========================================================================
# IsaacLab scene builder
# ===========================================================================
def build_scene() -> tuple:
    """构建 IsaacLab InteractiveScene with Franka robot.

    Returns:
        (sim, scene, device) tuple
    """
    device = "cuda:0" if (torch and torch.cuda.is_available()) else "cpu"
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=device)
    sim = SimulationContext(sim_cfg)

    robot_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRANKA_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "fr3_joint1": FR3_HOME_JOINTS[0],
                "fr3_joint2": FR3_HOME_JOINTS[1],
                "fr3_joint3": FR3_HOME_JOINTS[2],
                "fr3_joint4": FR3_HOME_JOINTS[3],
                "fr3_joint5": FR3_HOME_JOINTS[4],
                "fr3_joint6": FR3_HOME_JOINTS[5],
                "fr3_joint7": FR3_HOME_JOINTS[6],
                "fr3_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_cfg

    print("[SIM] Building InteractiveScene...", flush=True)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())
    print("[SIM] Scene ready", flush=True)

    return sim, scene, device


# ===========================================================================
# Observation capture
# ===========================================================================
def capture_observation(
    scene: Any,
    joint_poses: np.ndarray,
    gripper_states: np.ndarray,
    step_idx: int,
    device: str,
    prev_tcp_pose: Optional[np.ndarray] = None,
    dt: float = 1.0 / 30.0,
) -> dict[str, np.ndarray]:
    """从 sim 中读取 observation，产 25D state (per sim/data/contract.py)。

    State ordering (must match STATE_KEYS_ORDERED):
      tcp_pose(7) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper_pose(6) = 25D
      (gripper tiled 6× to match wrapper.py 25D distribution; arith 7+6+3+3+1=20
      conflicts with wrapper.py 25D — see contract.py docstring for resolution.)

    Args:
        scene: InteractiveScene (or None in pure-FK mode)
        joint_poses: 全部关节轨迹 (N, 7)
        gripper_states: 全部夹爪状态 (N,)
        step_idx: 当前帧索引
        device: torch device
        prev_tcp_pose: 上一帧 tcp_pose (7D pos+quat-xyzw); 第一次调用传 None
        dt: 时间步长 (s)

    Returns:
        {"state": (25,) float32, "pixels": (3, 128, 128) uint8,
         "tcp_pose_out": (7,) 用于下一次调用 prev_tcp_pose}
    """
    from sim.data.contract import STATE_DIMS, STATE_KEYS_ORDERED

    # 1) 读 sim 中实际关节角
    if scene is not None and HAS_ISAACLAB:
        q_actual = scene["robot"].data.joint_pos[0, :7].cpu().numpy().astype(np.float64)
    else:
        q_actual = np.asarray(joint_poses[step_idx], dtype=np.float64)

    # 2) tcp_pose: pos(3) + quat_xyzw(4) = 7D (first 7 of the 25D)
    from sim.kinematics.fr3_fk import fk_ee_pose
    T_ee = fk_ee_pose(q_actual)  # 4x4 transform
    tcp_pos = T_ee[:3, 3]
    tcp_quat = _rotmat_to_quat_xyzw(T_ee[:3, :3])
    tcp_pose = np.concatenate([tcp_pos, tcp_quat]).astype(np.float64)  # (7,)

    # 3) tcp_vel: 数值差分 pos(3) + angular placeholder(3) = 6D
    if prev_tcp_pose is None:
        tcp_vel = np.zeros(6, dtype=np.float64)
    else:
        pos_diff = (tcp_pose[:3] - prev_tcp_pose[:3]) / max(dt, 1e-6)
        tcp_vel = np.concatenate([pos_diff, np.zeros(3)]).astype(np.float64)  # (6,)

    # 4) tcp_force / tcp_torque: sim 接触力需要 robot contact sensor API (A9 之后实接)
    #    A2 阶段: hardcode zeros
    tcp_force = np.zeros(3, dtype=np.float64)
    tcp_torque = np.zeros(3, dtype=np.float64)

    # 5) gripper_pose: 第 8 轴位置 (1D scalar)  — 25D spec requires a
    #    1D gripper value; mainline 25D distribution is
    #    tcp_pose(7) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper(6) = 25
    #    per wrapper.py:23-25 comment. To match 25D exactly, we tile the
    #    gripper scalar 6 times (this is the A2 hard-freeze reconciliation;
    #    see VERIFY.md and contract.py docstring for the arith 20 vs 25 conflict).
    gripper_scalar = float(gripper_states[step_idx])
    gripper_pose = np.full(6, gripper_scalar, dtype=np.float64)  # (6,) → 25D total

    # 6) 按 STATE_KEYS_ORDERED 拼接 → 25D
    state_parts = {
        "tcp_pose": tcp_pose,
        "tcp_vel": tcp_vel,
        "tcp_force": tcp_force,
        "tcp_torque": tcp_torque,
        "gripper_pose": gripper_pose,
    }
    state = np.concatenate([state_parts[k] for k in STATE_KEYS_ORDERED]).astype(np.float32)
    assert state.shape == (STATE_DIMS,), f"state shape {state.shape} != ({STATE_DIMS},)"

    # 7) 图像
    pixels = _capture_camera_rgb(scene) if (scene is not None and HAS_ISAACLAB) \
             else np.zeros((IMAGE_C, IMAGE_H, IMAGE_W), dtype=np.uint8)

    return {"state": state, "pixels": pixels, "tcp_pose_out": tcp_pose}


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (xyzw) — branchless, no scipy.

    Uses the trace method: stable for most cases; the last-branch case picks
    the largest diagonal element to maximize numerical precision.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def _capture_camera_rgb(scene: Any) -> np.ndarray:
    """从 scene 中读取 camera RGB，若无 camera 则返回占位。"""
    try:
        # 检查 scene 中是否有 camera sensor
        if hasattr(scene, "keys") and "camera" in scene.keys():
            camera_data = scene["camera"].data
            rgb = camera_data.output["rgb"]  # (B, H, W, 3) torch.uint8
            img = rgb[0]  # (H, W, 3)
            img = img.permute(2, 0, 1).contiguous()  # (3, H, W)
            if img.shape != (IMAGE_C, IMAGE_H, IMAGE_W):
                import torch.nn.functional as F
                img_b = img.unsqueeze(0).float()
                img_b = F.interpolate(
                    img_b, size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False
                )
                img = img_b.squeeze(0).to(torch.uint8)
            return img.cpu().numpy()
    except Exception:
        pass

    # 无 camera: 返回占位图像
    return np.zeros((IMAGE_C, IMAGE_H, IMAGE_W), dtype=np.uint8)


# ===========================================================================
# Core replay
# ===========================================================================
def replay_in_sim(
    demo: dict[str, np.ndarray],
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    sub_steps: int = 10,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """在 IsaacLab sim 中回放 GELLO demo 并记录 observation。

    Args:
        demo: load_gello_npz() 返回的 dict
        pos_scale: xyz 归一化分母
        rpy_scale: rpy 归一化分母
        sub_steps: 每帧 sim 子步数 (越大收敛越准, 越慢)
        max_frames: 最大回放帧数 (None=全部)

    Returns:
        SERL transitions list
    """
    joint_poses = demo["joint_poses"]
    gripper_states = demo["gripper_states"]
    N = len(joint_poses)
    if max_frames is not None:
        N = min(N, max_frames)
        joint_poses = joint_poses[:N]
        gripper_states = gripper_states[:N]

    print(f"\n[REPLAY] {N} frames, sub_steps={sub_steps}")
    print(f"[REPLAY] pos_scale={pos_scale}, rpy_scale={rpy_scale}")

    # FK: 计算 Cartesian deltas
    print("[REPLAY] Computing Cartesian deltas via FK...")
    cartesian_deltas = trajectory_to_cartesian_deltas(joint_poses)
    print(f"[REPLAY] Deltas shape: {cartesian_deltas.shape}")

    # 构建 scene
    sim, scene, device = build_scene()

    # 构建 transitions
    transitions = []
    # A4: 7D from contract (per ACTION_SCALE 顺序 dx/dy/dz/droll/dpitch/dyaw/gripper)
    action_scale = list(ACTION_SCALE)
    start_time = time.time()
    # A2: 25D state needs prev_tcp_pose for tcp_vel numerical differentiation
    prev_tcp_pose = None

    for step in range(N):
        q_target = joint_poses[step]

        # 设置关节目标
        joint_pos = scene["robot"].data.joint_pos.clone()
        t_target = torch.tensor(q_target, dtype=torch.float32).unsqueeze(0).to(device)
        joint_pos[:, :7] = t_target
        scene["robot"].set_joint_position_target(joint_pos)

        # step sim
        for _ in range(sub_steps):
            sim.step()
        scene.update(sim.get_physics_dt())

        # 记录 observation (A2: 25D state via capture_observation with prev_tcp_pose)
        obs = capture_observation(
            scene, joint_poses, gripper_states, step, device,
            prev_tcp_pose=prev_tcp_pose, dt=1.0 / 30.0,
        )
        prev_tcp_pose = obs["tcp_pose_out"]

        # 构建 action (归一化)
        action = normalize_action(
            cartesian_deltas[step], action_scale, float(gripper_states[step])
        )
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # 过滤零动作
        if np.linalg.norm(action) <= 0.0:
            continue

        done = (step == N - 1)
        mask = np.float32(1.0 - float(done))

        # next_obs: 下一帧或最后一帧自身 (A2: 25D; pass current prev_tcp_pose)
        next_step = min(step + 1, N - 1)
        next_obs = capture_observation(
            scene, joint_poses, gripper_states, next_step, device,
            prev_tcp_pose=prev_tcp_pose, dt=1.0 / 30.0,
        )

        transition = {
            "observations": {
                "state": obs["state"].copy(),
                "pixels": obs["pixels"].copy(),
            },
            "next_observations": {
                "state": next_obs["state"].copy(),
                "pixels": next_obs["pixels"].copy(),
            },
            "actions": action.copy(),
            "rewards": np.float32(0.0),  # reward 由 plug_reward_labeler.py 后处理
            "masks": mask,
            "dones": done,
        }
        transitions.append(transition)

        if step % max(1, N // 10) == 0:
            elapsed = time.time() - start_time
            ee = scene["robot"].data.body_pos_w[0, -1].cpu().numpy()
            print(
                f"  Step {step:4d}/{N} | {elapsed:.1f}s | "
                f"EE: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]"
            )

    elapsed = time.time() - start_time
    print(f"[REPLAY] Done: {len(transitions)} transitions in {elapsed:.1f}s")

    return transitions


# ===========================================================================
# Pure FK replay (no IsaacSim)
# ===========================================================================
def _local_trajectory_to_cartesian_deltas(joint_poses: np.ndarray) -> np.ndarray:
    """Pure-Python fallback: sim/kinematics/fr3_fk-based cartesian deltas.

    Used only when the external scripts/fk_converter.py (with scipy Rotation
    for singularity-safe Euler deltas) is unavailable on this dev box. The
    in-worktree sim/kinematics/fr3_fk.py gives a 4x4 transform per frame; we
    approximate the rotation delta with a simple R_curr @ R_prev^T → euler
    decomposition using the same _rotmat_to_quat_xyzw helper used in
    capture_observation. Good enough for unit tests, not for sim deployment.
    """
    from sim.kinematics.fr3_fk import fk_ee_pose
    N = len(joint_poses)
    deltas = np.zeros((N, 6), dtype=np.float64)
    prev_pos = None
    prev_quat = None
    for i in range(N):
        T_curr = fk_ee_pose(joint_poses[i])
        pos_curr = T_curr[:3, 3]
        quat_curr = _rotmat_to_quat_xyzw(T_curr[:3, :3])
        if prev_pos is not None:
            d_pos = pos_curr - prev_pos
            # R_delta = R_curr @ R_prev^T; use quat math: q_delta = q_curr * q_prev^{-1}
            # For unit quat, inverse = conjugate
            px, py, pz, pw = prev_quat
            cx, cy, cz, cw = quat_curr
            # q_curr * q_prev_conj (Hamilton product, xyzw)
            qdx = cw * px + cx * pw + cy * pz - cz * py
            qdy = cw * py - cx * pz + cy * pw + cz * px
            qdz = cw * pz + cx * py - cy * px + cz * pw
            qdw = cw * pw - cx * px - cy * py - cz * pz
            d_euler = _quat_xyzw_to_euler_xyz(np.array([qdx, qdy, qdz, qdw]))
            deltas[i, :3] = d_pos
            deltas[i, 3:6] = d_euler
        prev_pos = pos_curr
        prev_quat = quat_curr
    return deltas


def _quat_xyzw_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (xyzw) to XYZ intrinsic Euler angles (radians).

    Pure-Python, no scipy; uses standard XYZ Tait-Bryan formulas.
    """
    qx, qy, qz, qw = q
    # roll (X)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (Y)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (Z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def replay_pure_fk(
    demo: dict[str, np.ndarray],
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """纯 FK 模式: 用 fk_converter 生成 transition, 无 sim 图像。

    用于无 GPU 的开发环境或快速验证。

    A2: produces 25D state via capture_observation() (calls it with
    scene=None and prev_tcp_pose threaded through the loop).
    """
    joint_poses = demo["joint_poses"]
    gripper_states = demo["gripper_states"]
    N = len(joint_poses)
    if max_frames is not None:
        N = min(N, max_frames)
        joint_poses = joint_poses[:N]
        gripper_states = gripper_states[:N]

    print(f"\n[PURE FK] {N} frames")
    if trajectory_to_cartesian_deltas is not None:
        cartesian_deltas = trajectory_to_cartesian_deltas(joint_poses)
    else:
        # A2 fallback: use in-worktree sim.kinematics.fr3_fk when the
        # external scripts/fk_converter is unavailable (dev box / CI).
        print("[PURE FK] gello_pipeline missing; using sim.kinematics.fr3_fk fallback")
        cartesian_deltas = _local_trajectory_to_cartesian_deltas(joint_poses)

    # A4: 7D from contract (per ACTION_SCALE 顺序 dx/dy/dz/droll/dpitch/dyaw/gripper)
    action_scale = list(ACTION_SCALE)
    pixels_placeholder = np.zeros((IMAGE_C, IMAGE_H, IMAGE_W), dtype=np.uint8)

    transitions = []
    # A2: thread prev_tcp_pose through capture_observation() for tcp_vel
    prev_tcp_pose = None
    for i in range(N):
        # A2: produce 25D state via capture_observation (same code path as sim)
        obs = capture_observation(
            scene=None, joint_poses=joint_poses, gripper_states=gripper_states,
            step_idx=i, device="cpu", prev_tcp_pose=prev_tcp_pose, dt=1.0 / 30.0,
        )
        prev_tcp_pose = obs["tcp_pose_out"]
        next_obs = capture_observation(
            scene=None, joint_poses=joint_poses, gripper_states=gripper_states,
            step_idx=min(i + 1, N - 1), device="cpu",
            prev_tcp_pose=prev_tcp_pose, dt=1.0 / 30.0,
        )

        if normalize_action is None:
            # Dev box: synthesize 7D action = cartesian_delta / scale, clip [-1, 1]
            action = np.concatenate([
                cartesian_deltas[i] / np.array(action_scale[:6], dtype=np.float64),
                [float(gripper_states[i])],
            ]).astype(np.float32)
        else:
            action = normalize_action(
                cartesian_deltas[i], action_scale, float(gripper_states[i])
            )
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if np.linalg.norm(action) <= 0.0:
            continue

        done = (i == N - 1)
        transition = {
            "observations": {
                "state": obs["state"].copy(),
                "pixels": pixels_placeholder.copy(),
            },
            "next_observations": {
                "state": next_obs["state"].copy(),
                "pixels": pixels_placeholder.copy(),
            },
            "actions": action.copy(),
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0 - float(done)),
            "dones": done,
        }
        transitions.append(transition)

    print(f"[PURE FK] {len(transitions)} transitions")
    return transitions


# ===========================================================================
# Save
# ===========================================================================
def save_transitions(transitions: list[dict], output_path: str, overwrite: bool = False) -> str:
    """保存 transitions 为 SERL pkl 文件。"""
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"File exists: {output_path}. Use --overwrite")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[SAVE] {output_path} ({size_kb:.1f} KB, {len(transitions)} transitions)")
    return output_path


# ===========================================================================
# Validation
# ===========================================================================
def validate_output(pkl_path: str) -> bool:
    """验证输出 pkl 文件。"""
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)

    if not transitions:
        print("[FAIL] Empty transitions")
        return False

    t0 = transitions[0]
    ok = True

    # 检查 keys
    for key in ["observations", "next_observations", "actions", "rewards", "masks", "dones"]:
        if key not in t0:
            print(f"[FAIL] Missing key: {key}")
            ok = False

    # 检查 shape/dtype
    state_dim = t0["observations"]["state"].shape
    pixels_shape = t0["observations"]["pixels"].shape
    action_dim = t0["actions"].shape

    print(f"[CHECK] state: {state_dim}, pixels: {pixels_shape}, actions: {action_dim}")
    print(f"[CHECK] state dtype: {t0['observations']['state'].dtype}")
    print(f"[CHECK] pixels dtype: {t0['observations']['pixels'].dtype}")

    # A2: state dim must match sim/data/contract.STATE_DIMS (25D)
    from sim.data.contract import STATE_DIMS
    if state_dim != (STATE_DIMS,):
        print(f"[FAIL] state shape {state_dim} != ({STATE_DIMS},)")
        ok = False
    if pixels_shape != (IMAGE_C, IMAGE_H, IMAGE_W):
        print(f"[FAIL] pixels shape {pixels_shape} != ({IMAGE_C},{IMAGE_H},{IMAGE_W})")
        ok = False
    if action_dim != (7,):
        print(f"[FAIL] action shape {action_dim} != (7,)")
        ok = False

    # 检查 action range
    all_actions = np.array([t["actions"] for t in transitions])
    a_min, a_max = float(all_actions.min()), float(all_actions.max())
    in_range = a_min >= -1.01 and a_max <= 1.01
    print(f"[CHECK] actions range: [{a_min:.4f}, {a_max:.4f}] {'OK' if in_range else 'FAIL'}")
    if not in_range:
        ok = False

    # masks == 1 - dones
    all_masks = np.array([t["masks"] for t in transitions])
    all_dones = np.array([t["dones"] for t in transitions])
    masks_ok = np.allclose(all_masks, 1.0 - all_dones.astype(np.float32))
    print(f"[CHECK] masks == 1-dones: {'OK' if masks_ok else 'FAIL'}")
    if not masks_ok:
        ok = False

    print(f"\n{'VALIDATION PASSED' if ok else 'VALIDATION FAILED'}")
    return ok


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="GELLO replay in IsaacLab sim")
    parser.add_argument("--npz", required=True, help="GELLO demo npz path")
    parser.add_argument("--output", default=None, help="Output pkl path")
    parser.add_argument("--pos-scale", type=float, default=DEFAULT_POS_SCALE)
    parser.add_argument("--rpy-scale", type=float, default=DEFAULT_RPY_SCALE)
    parser.add_argument("--sub-steps", type=int, default=10, help="Sim sub-steps per frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Max replay frames")
    parser.add_argument("--pure-fk", action="store_true", help="Use pure FK mode (no sim)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    # 输出路径
    if args.output is None:
        basename = os.path.splitext(os.path.basename(args.npz))[0]
        args.output = os.path.join(os.path.dirname(args.npz), f"{basename}_sim.pkl")

    print("=" * 60)
    print("  GELLO Replay → SERL pkl")
    print("=" * 60)
    print(f"  NPZ:    {args.npz}")
    print(f"  Output: {args.output}")
    print(f"  IsaacLab: {HAS_ISAACLAB}")
    print(f"  Pure FK:  {args.pure_fk}")

    # 加载 demo
    demo = load_gello_npz(args.npz)

    # 回放
    if args.pure_fk or not HAS_ISAACLAB:
        if not args.pure_fk:
            print("[WARN] IsaacLab unavailable, falling back to pure FK")
        transitions = replay_pure_fk(
            demo,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            max_frames=args.max_frames,
        )
    else:
        transitions = replay_in_sim(
            demo,
            pos_scale=args.pos_scale,
            rpy_scale=args.rpy_scale,
            sub_steps=args.sub_steps,
            max_frames=args.max_frames,
        )

    # 保存
    save_transitions(transitions, args.output, overwrite=args.overwrite)

    # 验证
    if not args.no_verify:
        validate_output(args.output)

    print("\nDone.")

    if HAS_SIM:
        app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if HAS_SIM:
            app.close()
        os._exit(1)
