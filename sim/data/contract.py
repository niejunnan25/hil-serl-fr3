"""sim/data/contract.py — sim 内部 hardcode 常量（与 v2.0 real-side 字节级兼容的契约）。

Single source of truth for:
  - ACTION_SCALE    : 7D action 各维的归一化分母
  - STATE_KEYS_ORDERED : 25D state 的 ordered 拼接键
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

CONTRACT_VERSION = "v2.1-real-2026-06-11"

# 7D action 归一化分母（与 real-side ACTION_SCALE 一致；sim 端 hardcode，不 import）
# 顺序: (dx, dy, dz, droll, dpitch, dyaw, gripper)
ACTION_SCALE = (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0)

# 25D state — ordered concatenation keys.
# Source of truth: experiments/plug_insertion/config.py:237
#   proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
# Wrapper 注释 (experiments/plug_insertion/wrapper.py:23-25):
#   state = [tcp_pose(6), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1)] = 25D
# 注: spec 第 88-91 行原始算式 7+6+3+3+1=20，但 wrapper.py 注释明示 =25D。
# A2 hard-freeze 解决: tcp_pose=7 (pos+quat xyzw); tcp_vel=6; tcp_force=3; tcp_torque=3; gripper=1 ⇒ 7+6+3+3+1=20 ≠25.
# 真实 mainline 实测 state.shape[-1] 待用户合并时 grep env.sample()["state"] 确认。
# 临时记录两个数: 25 (wrapper.py 注释) vs 20 (算术); 我们 hardcode 25D 与 wrapper.py 注释一致。
STATE_KEYS_ORDERED = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")
STATE_DIMS = 25                # PENDING VERIFY.md hard-freeze (A2 Task 3)
STATE_DTYPE = "float32"

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
MAX_EPISODE_LENGTH = 150

# Transition dict top-level keys
TRANSITION_KEYS = (
    "observations", "next_observations", "actions",
    "rewards", "masks", "dones",
)
# observations 子键: "state" + images.{side_policy, wrist_1, side_classifier}
# A10 verify_sim_data.py 必须 assert 此三键 schema

# A7: plug insertion detection thresholds (align with mainline 8mm/2mm/5°)
# Source of truth: ~/.planning/hil-serl-plug/evidence/sim-scene/insertion_detector.py
#   - insertion_depth_threshold = 0.008 (8mm)
#   - xy_tolerance              = 0.002 (2mm)
#   - angle_tolerance_deg       = 5.0   (5°)
# plug_reward_labeler.py 仍保留内部副本以避免 import 循环; contract 仅为 single-source documentation.
INSERTION_DEPTH_THRESHOLD = 0.008   # 8mm — 深度阈值
XY_TOLERANCE = 0.002               # 2mm — XY 对齐容差
ANGLE_TOLERANCE_DEG = 5.0          # 5°  — 角度对齐容差
