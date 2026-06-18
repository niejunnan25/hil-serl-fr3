"""
plug_insertion 任务配置
---
插插头任务：机械臂将插头对准插座并插入。
使用双 ZED 相机 + 机械臂状态作为观测，SAC + ResNet-10 训练。

相机:
  - wrist_1:        ZED serial 13132609, 腕部视角
  - side_policy:    ZED serial 36276705, 侧面视角 (policy)
  - side_classifier: 同 side_policy 相机，不同裁剪区域 (reward classifier)
"""
import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from experiments.config import DefaultTrainingConfig
from experiments.plug_insertion.env import PlugInsertionEnv
from experiments.plug_insertion.wrapper import GripperPenaltyWrapper
# [2026-06-16] Phase C = Xbox-only 在线介入；GelloIntervention 未用(死代码)已禁用
# from scripts.gello_intervention import GelloIntervention
from scripts.xbox_intervention import XboxIntervention, HoldGripperWrapper

REWARD_RELZ_STATE_INDEX = 6


def _load_classifier_adaptive(checkpoint_path, image_keys, key):
    """加载 reward classifier，自动检测 PyTorch (.pt) 或 JAX/Orbax 格式。

    优先尝试 JAX/Orbax (SERL 原生)，若失败则尝试 PyTorch。
    """
    import os

    # 1) 尝试 JAX/Orbax (SERL 原生路径)
    pt_file = os.path.join(checkpoint_path, "reward_classifier.pt")
    try:
        from serl_launcher.networks.reward_classifier import load_classifier_func
        # build a sample obs (shape only; CNN params are leading-dim-independent) so
        # load_classifier_func can re-create the architecture before restoring the checkpoint.
        sample = {k: np.zeros((1, 128, 128, 3), dtype=np.uint8) for k in image_keys}
        classifier_fn = load_classifier_func(
            key=key,
            sample=sample,
            image_keys=image_keys,
            checkpoint_path=checkpoint_path,
        )
        print(f"  [OK] Loaded JAX/Orbax classifier from {checkpoint_path}")
        return classifier_fn
    except Exception as e:
        # 2) 回退: PyTorch .pt checkpoint
        if os.path.isfile(pt_file):
            print(f"  [INFO] JAX load failed ({e}), trying PyTorch: {pt_file}")
            return _load_pytorch_classifier(pt_file, image_keys)
        raise RuntimeError(
            f"Failed to load classifier from {checkpoint_path}. "
            f"JAX error: {e}. No PyTorch .pt fallback found."
        ) from e


def _load_pytorch_classifier(pt_path, image_keys):
    """加载 PyTorch .pt 格式的 reward classifier，返回 JAX 兼容的推理函数。"""
    import torch
    import numpy as np

    # 导入 PyTorch 模型定义
    from scripts.train_reward_classifier import RewardClassifier, IMAGE_SIZE
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    model = RewardClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 图像预处理 (与训练一致)
    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def classifier_fn(obs):
        """从观测中提取图像，运行 PyTorch 分类器，返回 logit。"""
        import jax.numpy as jnp

        # 从 obs 中提取 classifier 图像 (取第一个 image_key)
        img_key = image_keys[0] if image_keys else "side_classifier"
        if img_key in obs:
            img = np.array(obs[img_key])
        else:
            # 回退: 尝试 pixels
            img = np.array(obs.get("pixels", np.zeros((3, 128, 128), dtype=np.uint8)))

        # CHW -> HWC for PIL, 如果需要
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)

        # 预处理
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img.astype(np.uint8))
        tensor = transform(pil_img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            logit = model(tensor).item()

        return jnp.array(logit)

    print(f"  [OK] Loaded PyTorch classifier from {pt_path}")
    return classifier_fn


# ============================================================
# 环境配置：相机、初始位姿、动作缩放、安全限位等
# ============================================================
class EnvConfig(DefaultEnvConfig):
    """插插头任务的环境物理参数。"""

    # -- 远程控制器地址（serl_franka_controllers HTTP API） --
    SERVER_URL: str = "http://172.16.0.1:5000/"

    # -- 相机配置（ZED 相机，2 台物理设备） --
    # wrist_1: 腕部 ZED   serial 13132609
    # side:    外部 ZED   serial 36276705  (policy + classifier 共用)
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "13132609",
            "dim": (1280, 720),
            "exposure": 10500,
        },
        "side_policy": {
            "serial_number": "36276705",
            "dim": (1280, 720),
            "exposure": 13000,
        },
        "side_classifier": {
            # 复用 side_policy 相机（同机位分类视角）
            "serial_number": "36276705",
            "dim": (1280, 720),
            "exposure": 13000,
        },
    }

    # -- 图像裁剪区域（ZED 1280x720 分辨率） --
    # Option A (2026-06-16): demos were recorded FULL-FRAME (gello_demo_recorder.image_to_obs
    # does no crop), so the live env must also be full-frame to match train==deploy. Crop
    # DISABLED. Original (insertion-zoom) crops kept here for a future re-collect-with-crop path:
    #   "wrist_1":         lambda img: img[100:500, 300:900],
    #   "side_policy":     lambda img: img[200:500, 350:750],
    #   "side_classifier": lambda img: img[250:400, 475:625],
    IMAGE_CROP = {}

    # -- 目标位姿 [x, y, z, roll, pitch, yaw] --
    # CALIBRATED 2026-06-16 from the 24-demo consistent cluster (demos 6-30, socket-moved
    # first-5 + y-outlier dropped). xyz = median seated world pose; orientation euler is the
    # NUMERICAL INVERSE of euler_2_quat for the seated quat [0.039,0.046,0.006,0.998] (xyzw)
    # — euler_2_quat is NOT the inverse of quat_2_euler, so these are euler_2_quat-domain euler.
    # !! JOG-CONFIRM ON THE REAL ROBOT BEFORE ANY LIVE RUN !!
    TARGET_POSE = np.array([
        0.6743,         # x  (IQR 9mm)
        -0.0074,        # y  (IQR 9mm)
        0.1355,         # z  seated (IQR 27mm)
        3.12632,        # roll  (FIX 20260616 gripper-down: euler_2_quat -> xyzw [0.9985,0.035,0.041,0.006])
        0.08094,        # pitch
        0.07049,        # yaw
    ])

    # -- 复位位姿 = demo insertion-start (above the socket, ~5.7cm up, ~5cm back in x) --
    # Set explicitly (NOT TARGET+offset) so the live RelativeFrame reset matches the
    # insert-only converted demos' reset frame (seated rel-z ≈ -0.06).
    RESET_POSE = np.array([
        0.6259,         # x
        -0.0161,        # y
        0.1925,         # z  (= seated + 0.057)
        3.12632,        # roll  (FIX 20260616 gripper-down; was 180deg-flipped gripper-up [0.018,-0.016,0.009,1.0])
        0.08094,        # pitch
        0.07049,        # yaw
    ])

    # -- 动作缩放因子 [xyz, xyz, xyz, rpy, rpy, rpy, gripper]（SERL 官方 7D 参数） --
    ACTION_SCALE = np.array([0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0])

    # -- 随机化设置 --
    # Phase C real-robot runs need a fixed reset frame. Randomizing xy/yaw makes
    # operator-guided insertion harder to debug and masks reset-settle drift.
    RANDOM_RESET = False
    DISPLAY_IMAGE = False
    RANDOM_XY_RANGE = 0.01    # 初始位置 XY 随机扰动范围 (m)
    RANDOM_RZ_RANGE = 0.1     # 初始姿态 yaw 随机扰动范围 (rad)

    # -- 安全位姿限位 --
    # CALIBRATED 2026-06-16: position box = insertion-segment span across the 24-demo cluster
    # (x[0.589,0.694] y[-0.029,0.032] z[0.059,0.237]) + ~0.03 m margin; z_high leaves room for
    # the hold-grip extraction lift. Orientation limits set WIDE (±π) so they never bind (the
    # EE stays near-upright during insertion; position is the collision concern, not orientation).
    # !! JOG-CONFIRM THE BOX ON THE REAL ROBOT BEFORE ANY LIVE RUN !!
    ABS_POSE_LIMIT_HIGH = np.array([0.724,  0.062, 0.267,  np.pi,  np.pi,  np.pi])
    ABS_POSE_LIMIT_LOW  = np.array([0.559, -0.059, 0.039, -np.pi, -np.pi, -np.pi])

    # -- 力控刚度参数：合规模式（操作阶段，允许轻微接触） --
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0,
        "nullspace_stiffness": 0.2,   # REVERT 20260616: 25 放大了腕部奇异点 j5~0 的 nullspace-pinv 尖峰->乱颤; 对齐 teach/默认 0.2
    }

    # -- 精密模式参数（接近/插入阶段，更严格的刚度） --
    PRECISION_PARAM = {
        "translational_stiffness": 2500,
        "translational_damping": 100,
        "rotational_stiffness": 200,
        "rotational_damping": 10,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.008,
        "translational_clip_y": 0.008,
        "translational_clip_z": 0.0072,
        "translational_clip_neg_x": 0.008,
        "translational_clip_neg_y": 0.008,
        "translational_clip_neg_z": 0.0072,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.0,
        "nullspace_stiffness": 0.2,   # REVERT 20260616: 25 放大了腕部奇异点 j5~0 的 nullspace-pinv 尖峰->乱颤; 对齐 teach/默认 0.2
    }

    # raised from 150: demo insertion segments are 226-1009 steps (median 365); 150 was too
    # short for the insert+search phase. 500 allows search without runaway. Tune as needed.
    MAX_EPISODE_LENGTH = 500


# ============================================================
# 训练配置：SAC 超参数、观测键、环境工厂方法
# ============================================================
class TrainConfig(DefaultTrainingConfig):
    """HIL-SERL 训练管线配置。"""

    # -- 观测键定义（2 相机，无 wrist_2） --
    image_keys      = ["side_policy", "wrist_1"]
    # classifier uses the WRIST view (full-frame, Option A): seated-vs-approach pixel
    # separability is ~53 on wrist_1 vs only ~5 on the full-frame side view (the plug/socket
    # is a tiny part of the wide side view). 2026-06-16, data-driven from the 24-demo cluster.
    classifier_keys = ["side_classifier"]  # FIX 2026-06-16: wrist_1 classifier-blind (sep +0.007) vs side_classifier (sep +0.810); empirically verified
    proprio_keys    = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]

    # -- SAC 训练超参数 --
    checkpoint_period = 2000
    cta_ratio         = 2        # critic-to-actor update ratio
    random_steps      = 0        # 纯随机探索步数
    discount          = 0.98     # 折扣因子
    buffer_period     = 1000     # 每 N 步 dump buffer 到磁盘
    encoder_type      = "resnet-pretrained"  # 冻结预训练 ResNet-10
    setup_mode        = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        构建插插头任务的 Gym 环境管线。

        Args:
            fake_env:   True 时创建虚拟环境（learner 端不连接真实机器人）
            save_video: 是否录制视频用于调试
            classifier: 是否加载奖励分类器
        Returns:
            gym.Env 包装后的环境实例
        """
        # 1) 基础环境：连接机器人或创建 fake env
        env = PlugInsertionEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )

        # hold-grip: 全程钳住插头(中和夹爪动作,避免 demo/env 夹爪符号不一致而开爪丢插头)
        env = HoldGripperWrapper(env)

        # 2) Xbox 人工干预(仅真实环境;按住 RB 死手接管)
        if not fake_env:
            env = XboxIntervention(env)

        # 3) 相对坐标系变换
        env = RelativeFrame(env)

        # 4) 四元数 -> 欧拉角
        env = Quat2EulerWrapper(env)

        # 5) SERL 标准观测包装
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # 6) Chunking 包装（obs horizon=1, 无 action chunking）
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 7) 奖励分类器（插入成功判定）
        if classifier:
            classifier_fn = _load_classifier_adaptive(
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
                image_keys=self.classifier_keys,
                key=jax.random.PRNGKey(0),
            )

            _reward_dbg = {"t": 0}

            def reward_func(obs):
                """结合分类器与位姿判定是否插入成功。"""
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # RelativeFrame tcp_pose z is reset-relative, not world z. It is NEGATIVE once
                # the plug descends; the 24-demo cluster seats at rel-z ≈ -0.062.
                # SERLObsWrapper flattens gymnasium.spaces.Dict; gymnasium.spaces.Dict sorts keys
                # alphabetically: gripper_pose(1), tcp_force(3), tcp_pose(6), ...
                # After ChunkingWrapper(obs_horizon=1), rel-z is obs["state"][0, 6].
                # Extract SCALARS (classifier_fn returns a (1,) array; int(ndim=1) crashes).
                cls = float(jnp.reshape(sigmoid(classifier_fn(obs)), (-1,))[0])
                relz = float(obs["state"][0, REWARD_RELZ_STATE_INDEX])  # RelativeFrame z vs RESET
                reward = int((cls > 0.7) and (relz < -0.05))  # CALIBRATED 2026-06-16
                _reward_dbg["t"] += 1
                if reward or _reward_dbg["t"] % 25 == 0:
                    print(f"[reward_dbg] cls={cls:.3f} relz={relz:.4f} reward={reward}", flush=True)
                return reward

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)

        # 8) 夹爪惩罚（鼓励减少不必要的开合动作）
        env = GripperPenaltyWrapper(env, penalty=-0.02)

        return env
