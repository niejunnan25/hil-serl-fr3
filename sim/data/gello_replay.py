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
     - state:  关节位置 (7,) + 夹爪 (1,) = (8,) float32
  5. 构建 SERL pkl transitions list

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
# Project imports
# ===========================================================================
GELLO_PIPELINE = "/home/robot/serl_projects/hil-serl-fr3/scripts/gello_pipeline"
for _p in [GELLO_PIPELINE]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from fk_converter import trajectory_to_cartesian_deltas, trajectory_to_poses
from normalize_action import normalize_action

# ===========================================================================
# Constants
# ===========================================================================
FRANKA_USD = "/home/robot/plug_insertion_sim/assets/panda_arm_hand.usd"
FR3_HOME_JOINTS = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])

DEFAULT_POS_SCALE = 0.1
DEFAULT_RPY_SCALE = 0.2
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
) -> dict[str, np.ndarray]:
    """从 sim 中读取 observation。

    Args:
        scene: InteractiveScene
        joint_poses: 全部关节轨迹 (N, 7)
        gripper_states: 全部夹爪状态 (N,)
        step_idx: 当前帧索引
        device: torch device

    Returns:
        {"state": (8,) float32, "pixels": (3, 128, 128) uint8}
    """
    # 读取 sim 中的实际关节位置
    q_actual = scene["robot"].data.joint_pos[0, :7].cpu().numpy().astype(np.float32)
    gripper = np.float32(gripper_states[step_idx])

    state = np.concatenate([q_actual, [gripper]]).astype(np.float32)

    # 图像: 尝试从 sim camera 读取; 若无 camera 则返回占位
    pixels = _capture_camera_rgb(scene)

    return {"state": state, "pixels": pixels}


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
    action_scale = [pos_scale, rpy_scale, 0.0]
    start_time = time.time()

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

        # 记录 observation
        obs = capture_observation(scene, joint_poses, gripper_states, step, device)

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

        # next_obs: 下一帧或最后一帧自身
        next_step = min(step + 1, N - 1)
        next_obs = capture_observation(scene, joint_poses, gripper_states, next_step, device)

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
def replay_pure_fk(
    demo: dict[str, np.ndarray],
    pos_scale: float = DEFAULT_POS_SCALE,
    rpy_scale: float = DEFAULT_RPY_SCALE,
    max_frames: Optional[int] = None,
) -> list[dict]:
    """纯 FK 模式: 用 fk_converter 生成 transition, 无 sim 图像。

    用于无 GPU 的开发环境或快速验证。
    """
    joint_poses = demo["joint_poses"]
    gripper_states = demo["gripper_states"]
    N = len(joint_poses)
    if max_frames is not None:
        N = min(N, max_frames)
        joint_poses = joint_poses[:N]
        gripper_states = gripper_states[:N]

    print(f"\n[PURE FK] {N} frames")
    cartesian_deltas = trajectory_to_cartesian_deltas(joint_poses)

    action_scale = [pos_scale, rpy_scale, 0.0]
    states = np.zeros((N, 8), dtype=np.float32)
    states[:, :7] = joint_poses.astype(np.float32)
    states[:, 7] = gripper_states.astype(np.float32)
    pixels_placeholder = np.zeros((IMAGE_C, IMAGE_H, IMAGE_W), dtype=np.uint8)

    transitions = []
    for i in range(N):
        action = normalize_action(
            cartesian_deltas[i], action_scale, float(gripper_states[i])
        )
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if np.linalg.norm(action) <= 0.0:
            continue

        done = (i == N - 1)
        transition = {
            "observations": {
                "state": states[i].copy(),
                "pixels": pixels_placeholder.copy(),
            },
            "next_observations": {
                "state": states[min(i + 1, N - 1)].copy(),
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

    if state_dim != (8,):
        print(f"[FAIL] state shape {state_dim} != (8,)")
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
