"""sim/data/contract.py — sim 内部 hardcode 常量（与 live real-side 兼容的契约）。

Single source of truth for:
  - ACTION_SCALE    : 7D action 各维的归一化分母
  - STATE_KEYS_ORDERED : 19D live SERL flat state 的 ordered 拼接键
  - STATE_DIMS / STATE_DTYPE
  - IMAGE_SHAPE / IMAGE_DTYPE
  - POLICY_IMAGE_KEYS / CLASSIFIER_IMAGE_KEYS / CLASSIFIER_ALIAS_OF
  - IMAGE_KEY_ALIAS_MAP / VALID_PKL_IMAGE_KEYS   (A3 added)
  - TRANSITION_KEYS
  - MAX_EPISODE_LENGTH

禁止: 任何 import 自 real-side 路径 (参见 L1 hard isolation gate)。
本文件必须保持零外部 import（除标准 typing），以满足 L1 硬隔离 gate。

CONTRACT_VERSION 在 schema 变更时必须 bump。
"""
from __future__ import annotations

CONTRACT_VERSION = "v2.1-live-serl19-2026-06-19"

# 7D action 归一化分母（与 real-side ACTION_SCALE 一致；sim 端 hardcode，不 import）
# 顺序: (dx, dy, dz, droll, dpitch, dyaw, gripper)
ACTION_SCALE = (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0)

# FR3 7-DoF 臂 home/reset 关节构型（弧度）——sim 端 replay/reset 起始位姿的
# single source of truth。plug_scene.FR3_HOME_JOINTS 与 gello_replay.FR3_HOME_JOINTS
# 必须 import 此常量（用 np.array(...) 包装）。
# 存为纯 tuple（不引入 numpy）以满足本模块 L1 零外部 import 约束。
#
# 注意：此 'bent' 位姿 (TCP @ z≈0.3855, 经 sim/kinematics/fr3_fk.fk_ee_pose 验证)
# 与 official_fr3_loader._FR3_DEFAULT_HOME 的 USD spawn 位姿
# [0,0,0,-pi/2,0,pi/2,0] (TCP @ z≈0.5211, 上游 RobotEnv.reset_joints)
# 是两个不同的位姿、不同用途：前者是逐次 reset/replay 下发的命令位姿，
# 后者是 articulation 的 USD 初始 spawn 位姿。二者当前 DISAGREE 且各自合理，
# 故此处只统一 sim 端 replay home，不覆盖 loader 的 spawn 位姿。
FR3_HOME_JOINTS = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)

# 19D live SERL flat state — gymnasium.spaces.Dict flatten order.
# Source of truth: experiments/plug_insertion/config.py / state_layout.py
#   proprio_keys insertion order:
#     ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")
#   gymnasium Dict flatten order:
#     ("gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel")
#   dims:
#     gripper_pose(1) + tcp_force(3) + tcp_pose(6 pos+euler)
#     + tcp_torque(3) + tcp_vel(6) = 19.
STATE_KEYS_ORDERED = ("gripper_pose", "tcp_force", "tcp_pose", "tcp_torque", "tcp_vel")
STATE_DIMS = 19
STATE_DTYPE = "float32"

STATE_KEY_DIMS = (1, 3, 6, 3, 6)

# Flat slice boundaries for producers/validators that need explicit indexing.
STATE_GRIPPER_SLICE = slice(0, 1)
STATE_TCP_FORCE_SLICE = slice(1, 4)
STATE_TCP_POSE_SLICE = slice(4, 10)
STATE_TCP_TORQUE_SLICE = slice(10, 13)
STATE_TCP_VEL_SLICE = slice(13, 19)

# Image spec (CHW)
IMAGE_SHAPE = (3, 128, 128)
IMAGE_DTYPE = "uint8"

# Image keys (3 keys total in pkl schema; classifier 是 policy 的 alias in sim)
POLICY_IMAGE_KEYS = ("side_policy", "wrist_1")
CLASSIFIER_IMAGE_KEYS = ("side_classifier",)
# CLASSIFIER_ALIAS_OF: tuple of (alias_key, source_key)
CLASSIFIER_ALIAS_OF = (("side_classifier", "side_policy"),)

# A3: sim pkl image schema 必须包含的 3 键 (A10 verify_sim_data.py assert 用)
VALID_PKL_IMAGE_KEYS = ("side_policy", "wrist_1", "side_classifier")

# A3: alias map: alias_key -> source_key
# sim 用单一 TiledCamera (side_policy), 把 side_classifier 复用为 side_policy 的数据副本
IMAGE_KEY_ALIAS_MAP = {
    "side_classifier": "side_policy",
}

# Episode
# Align with experiments/plug_insertion/config.py:MAX_EPISODE_LENGTH = 300
# (raised from 150: demo insertion segments are 226-1009 steps, median 365;
#  150 was too short for the insert+search phase, 300 allows search without runaway).
MAX_EPISODE_LENGTH = 300

# Transition dict top-level keys
TRANSITION_KEYS = (
    "observations", "next_observations", "actions",
    "rewards", "masks", "dones",
)
# observations 子键: "state" + images.{side_policy, wrist_1, side_classifier}
# A10 verify_sim_data.py 必须 assert 此三键 schema

# ===========================================================================
# A8: domain randomization ranges
# ===========================================================================
# NOTE: Per codex review #3 (DR scope 拆解), 本 fork 只做 light/camera/plug 随机化;
# background / texture / sky randomization 留待 v2.2 或 mainline Phase 6 退出时补。
# PLAN-A8 限定范围如下:

# Light intensity range (lumens 或 IsaacLab intensity unit, sim 端 hardcode)
LIGHT_INTENSITY_MIN = 800.0
LIGHT_INTENSITY_MAX = 1200.0

# Camera yaw/pitch range (degrees, ±5° perturbation)
CAMERA_YAW_RANGE_DEG = (-5.0, 5.0)
CAMERA_PITCH_RANGE_DEG = (-5.0, 5.0)

# Plug initial pose jitter
PLUG_XY_JITTER_M = 0.01      # 1 cm in xy plane
PLUG_RZ_JITTER_RAD = 0.1     # ~5.7° around z axis

# Default RNG seed (复现性, A8 单元测试用)
RANDOMIZE_SEED_DEFAULT = 20260611

# ===========================================================================
# A7: plug insertion detection thresholds (align with mainline 8mm/2mm/5°)
# ===========================================================================
# Source of truth: ~/.planning/hil-serl-plug/evidence/sim-scene/insertion_detector.py
#   - insertion_depth_threshold = 0.008 (8mm)
#   - xy_tolerance              = 0.002 (2mm)
#   - angle_tolerance_deg       = 5.0   (5°)
# plug_reward_labeler.py 仍保留内部副本以避免 import 循环; contract 仅为 single-source documentation.
INSERTION_DEPTH_THRESHOLD = 0.008   # 8mm — 深度阈值
XY_TOLERANCE = 0.002               # 2mm — XY 对齐容差
ANGLE_TOLERANCE_DEG = 5.0          # 5°  — 角度对齐容差

# ===========================================================================
# A9: failure scenario ranges
# ===========================================================================
# 4 类失败: mis_alignment / angle_offset / insufficient_force / drop
# 全部产出 reward=0 帧, 与 real positive 混合做 classifier training (A12)

# (1) mis_alignment: plug 起始 xy 偏移量
FAILURE_MISALIGNMENT_XY_M = 0.03     # 3 cm

# (2) angle_offset:  z 轴旋转偏移
FAILURE_ANGLE_OFFSET_DEG = 10.0

# (3) insufficient_force: 最后 N 帧提前 close_gripper (action gripper = 1.0)
FAILURE_INSUFFICIENT_FORCE_FRAMES = 10

# (4) drop: 中段 release_gripper (action gripper = 0.0)
FAILURE_DROP_FRAME_RATIO = 0.5       # 50% 中段帧 release

# 4 类 class 名称 (字符串 enum)
FAILURE_CLASSES = (
    "mis_alignment", "angle_offset", "insufficient_force", "drop",
)

# 全部 failure 输出的 reward (与 real-side 1.0 区分)
FAILURE_REWARD = 0.0
