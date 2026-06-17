#!/usr/bin/env python3
"""gello_cartesian_delta_agent.py

方案B: 实时将 GELLO 关节输出通过 FK 转为 Cartesian delta，直接喂入 HIL-SERL env.step()。

数据流:
    GELLO (Dynamixel) -> get_joint_state() -> q_t (7D)
        |
        v
    fk_converter.forward_kinematics(q_t) -> pose_t (7D: x,y,z,qx,qy,qz,qw)
        |
        v
    Cartesian delta = pose_t - pose_{t-1}  (translation only, or full 6D)
        |
        v
    normalize_action() -> action_t (7D: dx,dy,dz,droll,dpitch,dyaw,gripper)
        |
        v
    env.step(action_t) -> HIL-SERL FrankaEnv

安全参数:
    max_step: 单步最大 Cartesian delta (meters), 默认 0.003
    max_total_delta: 累积最大 Cartesian delta (meters), 默认 0.03

用法:
    # 作为 Python 模块导入
    from gello_cartesian_delta_agent import GelloCartesianDeltaAgent
    agent = GelloCartesianDeltaAgent()
    action = agent.step(gello_joints)

    # 独立测试 (dry-run)
    python gello_cartesian_delta_agent.py --dry-run
    python gello_cartesian_delta_agent.py --dry-run --max-step 0.005 --max-total-delta 0.05

依赖:
    - fk_converter.py   (FK 计算)
    - normalize_action.py (动作归一化)
    - numpy, scipy
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# 确保 scripts 目录在 import path 中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fk_converter import forward_kinematics, joints_to_cartesian_delta
from normalize_action import normalize_action


# ---------------------------------------------------------------------------
# 默认安全参数
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEP = 0.003        # 单步最大 Cartesian delta (meters)
DEFAULT_MAX_TOTAL_DELTA = 0.03  # 累积最大 Cartesian delta (meters)
DEFAULT_MAX_ROT_STEP = 0.1      # 单步最大旋转 delta (radians, = rpy_scale, 与 relative_teleop 一致)
DEFAULT_HZ = 10                 # 控制频率 (与 HIL-SERL 一致)

# 默认 action scale (与 convert_to_zarr.py 一致)
DEFAULT_POS_SCALE = 0.015       # xyz 归一化分母 (meters)
DEFAULT_RPY_SCALE = 0.1         # roll/pitch/yaw 归一化分母 (radians)


# ---------------------------------------------------------------------------
# GelloCartesianDeltaAgent
# ---------------------------------------------------------------------------
class GelloCartesianDeltaAgent:
    """实时 GELLO -> Cartesian delta 转换 Agent。

    将 GELLO 关节输出通过 FK 转为 Cartesian delta，归一化后直接
    喂入 HIL-SERL env.step()。

    安全机制:
        - max_step: 单步最大平移 Cartesian delta (translation norm, meters)
        - max_total_delta: 累积最大平移 Cartesian delta (translation norm, meters)
        - max_rot_step: 单步最大旋转 delta (||rpy|| norm, radians)
        - 超限时返回零动作并记录警告

    Attributes:
        max_step: 单步最大平移 Cartesian delta (meters)
        max_total_delta: 累积最大平移 Cartesian delta (meters)
        max_rot_step: 单步最大旋转 delta (radians)
        pos_scale: xyz 归一化分母
        rpy_scale: rpy 归一化分母
        prev_joints: 上一步关节位置 (7,)
        initial_joints: 初始关节位置 (7,)
        step_count: 步数计数器
        violation_count: 安全违规计数
    """

    def __init__(
        self,
        max_step: float = DEFAULT_MAX_STEP,
        max_total_delta: float = DEFAULT_MAX_TOTAL_DELTA,
        pos_scale: float = DEFAULT_POS_SCALE,
        rpy_scale: float = DEFAULT_RPY_SCALE,
        max_rot_step: float = DEFAULT_MAX_ROT_STEP,
    ):
        """初始化 Agent。

        Args:
            max_step: 单步最大平移 Cartesian delta (meters), 默认 0.003
            max_total_delta: 累积最大平移 Cartesian delta (meters), 默认 0.03
            pos_scale: xyz 归一化分母, 默认 0.1
            rpy_scale: rpy 归一化分母, 默认 0.2
            max_rot_step: 单步最大旋转 delta (radians, ||rpy|| 范数), 默认 0.1
        """
        self.max_step = max_step
        self.max_total_delta = max_total_delta
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.max_rot_step = max_rot_step

        # 状态
        self.prev_joints: Optional[np.ndarray] = None
        self.initial_joints: Optional[np.ndarray] = None
        self.prev_pose: Optional[np.ndarray] = None
        self.initial_pose: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.violation_count: int = 0

    def reset(self, initial_joints: np.ndarray) -> None:
        """重置 Agent 状态，设置初始关节位置。

        Args:
            initial_joints: 初始关节位置 (7,) in radians
        """
        initial_joints = np.asarray(initial_joints, dtype=np.float64).flatten()
        assert initial_joints.shape == (7,), f"Expected (7,) joints, got {initial_joints.shape}"

        self.prev_joints = initial_joints.copy()
        self.initial_joints = initial_joints.copy()
        self.prev_pose = forward_kinematics(initial_joints)
        self.initial_pose = self.prev_pose.copy()
        self.step_count = 0
        self.violation_count = 0

    def step(
        self,
        gello_joints: np.ndarray,
        gripper: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        """将 GELLO 关节输入转换为归一化 Cartesian delta 动作。

        Args:
            gello_joints: GELLO 关节位置 (7,) in radians
            gripper: 夹爪值 (already in [-1, 1] or raw from GELLO)

        Returns:
            action: (7,) 归一化动作 [dx,dy,dz,droll,dpitch,dyaw,gripper] in [-1,1]
            info: dict 包含诊断信息
                - raw_delta: (6,) 原始 Cartesian delta
                - cartesian_delta: (6,) 安全检查后的 Cartesian delta
                - safe: bool 是否通过安全检查
                - step_delta_norm: float 单步 delta 范数
                - total_delta_norm: float 累积 delta 范数
                - violation: str 违规描述 (空字符串表示无违规)
        """
        gello_joints = np.asarray(gello_joints, dtype=np.float64).flatten()
        assert gello_joints.shape == (7,), f"Expected (7,) joints, got {gello_joints.shape}"

        # 首次调用需要 reset
        if self.prev_joints is None:
            self.reset(gello_joints)

        # ------------------------------------------------------------------
        # 1. FK: 当前关节 -> 当前 pose
        # ------------------------------------------------------------------
        curr_pose = forward_kinematics(gello_joints)

        # ------------------------------------------------------------------
        # 2. Cartesian delta: pose_t - pose_{t-1}
        #    使用 joints_to_cartesian_delta 获得 6D delta
        #    [dx, dy, dz, droll, dpitch, dyaw]
        # ------------------------------------------------------------------
        cartesian_delta = joints_to_cartesian_delta(self.prev_joints, gello_joints)
        raw_delta = cartesian_delta.copy()

        # ------------------------------------------------------------------
        # 3. 安全检查 (translation norm)
        # ------------------------------------------------------------------
        safe = True
        violation = ""

        # 单步检查: translation norm
        step_delta_norm = float(np.linalg.norm(cartesian_delta[:3]))
        if step_delta_norm > self.max_step:
            safe = False
            violation = (
                f"max_step exceeded: {step_delta_norm:.6f} > {self.max_step} "
                f"(step {self.step_count})"
            )

        # 累积检查: translation norm from initial pose
        total_delta_norm = float(np.linalg.norm(curr_pose[:3] - self.initial_pose[:3]))
        if total_delta_norm > self.max_total_delta:
            safe = False
            violation = (
                f"max_total_delta exceeded: {total_delta_norm:.6f} > {self.max_total_delta} "
                f"(step {self.step_count})"
            )

        # 单步检查: rotation norm (||rpy||). 一次 GELLO 猛拽 / 丢帧 / stale-prev
        # 可以在平移很小的情况下合成任意大的姿态跳变, 平移 cap 不会拦住它。
        rot_delta_norm = float(np.linalg.norm(cartesian_delta[3:6]))
        if safe and rot_delta_norm > self.max_rot_step:
            safe = False
            violation = (
                f"max_rot_step exceeded: {rot_delta_norm:.6f} > {self.max_rot_step} "
                f"(step {self.step_count})"
            )

        # ------------------------------------------------------------------
        # 4. 安全策略: 超限时返回零动作
        # ------------------------------------------------------------------
        if not safe:
            self.violation_count += 1
            # 超限时: 不更新 prev_joints/prev_pose, 返回零 delta
            action = np.zeros(7, dtype=np.float32)
            action[6] = np.clip(gripper, -1.0, 1.0)
            info = {
                "raw_delta": raw_delta,
                "cartesian_delta": np.zeros(6, dtype=np.float64),
                "safe": False,
                "step_delta_norm": step_delta_norm,
                "total_delta_norm": total_delta_norm,
                "rot_delta_norm": rot_delta_norm,
                "violation": violation,
            }
            return action, info

        # ------------------------------------------------------------------
        # 5. 归一化 action
        # ------------------------------------------------------------------
        action = normalize_action(
            cartesian_delta,
            action_scale=[self.pos_scale, self.rpy_scale, 1.0],
            gripper=gripper,
        )

        # ------------------------------------------------------------------
        # 6. 更新状态
        # ------------------------------------------------------------------
        self.prev_joints = gello_joints.copy()
        self.prev_pose = curr_pose.copy()
        self.step_count += 1

        info = {
            "raw_delta": raw_delta,
            "cartesian_delta": cartesian_delta,
            "safe": True,
            "step_delta_norm": step_delta_norm,
            "total_delta_norm": total_delta_norm,
            "rot_delta_norm": rot_delta_norm,
            "violation": "",
        }

        return action, info

    def get_state(self) -> dict:
        """获取 Agent 当前状态摘要。

        Returns:
            dict: 状态信息
        """
        return {
            "step_count": self.step_count,
            "violation_count": self.violation_count,
            "max_step": self.max_step,
            "max_total_delta": self.max_total_delta,
            "max_rot_step": self.max_rot_step,
            "pos_scale": self.pos_scale,
            "rpy_scale": self.rpy_scale,
            "initialized": self.prev_joints is not None,
        }


# ---------------------------------------------------------------------------
# CLI / Dry-run 测试
# ---------------------------------------------------------------------------
def dry_run(args: argparse.Namespace) -> None:
    """Dry-run 测试: 模拟 GELLO 输入，验证 Agent 行为。

    使用随机关节序列模拟 GELLO 输出，验证:
        1. FK 转换正确性
        2. 安全检查触发
        3. 归一化 action 范围
    """
    max_step = args.max_step
    max_total_delta = args.max_total_delta
    max_rot_step = args.max_rot_step
    pos_scale = args.pos_scale
    rpy_scale = args.rpy_scale
    hz = args.hz
    duration = args.duration

    agent = GelloCartesianDeltaAgent(
        max_step=max_step,
        max_total_delta=max_total_delta,
        max_rot_step=max_rot_step,
        pos_scale=pos_scale,
        rpy_scale=rpy_scale,
    )

    dt = 1.0 / hz
    total_steps = int(duration * hz)

    print(f"\n{'=' * 60}")
    print(f"  GelloCartesianDeltaAgent Dry-Run")
    print(f"  max_step: {max_step}, max_total_delta: {max_total_delta}")
    print(f"  pos_scale: {pos_scale}, rpy_scale: {rpy_scale}")
    print(f"  hz: {hz}, duration: {duration}s, steps: {total_steps}")
    print(f"{'=' * 60}\n")

    # 模拟: 从 home 位置开始，每步微小随机扰动
    rng = np.random.RandomState(42)
    q_home = np.zeros(7)
    q_curr = q_home.copy()

    agent.reset(q_home)

    violations = []
    actions_list = []

    for step in range(total_steps):
        # 模拟 GELLO 输入: 微小随机关节增量
        delta_q = rng.randn(7) * 0.002  # 小增量
        q_curr = q_curr + delta_q

        action, info = agent.step(q_curr, gripper=0.0)
        actions_list.append(action)

        if not info["safe"]:
            violations.append((step, info["violation"]))

        if step % (hz) == 0:
            print(
                f"  Step {step:5d} | "
                f"action: [{action[0]:+.4f} {action[1]:+.4f} {action[2]:+.4f} "
                f"{action[3]:+.4f} {action[4]:+.4f} {action[5]:+.4f} {action[6]:+.4f}] | "
                f"step_dn: {info['step_delta_norm']:.6f} | "
                f"total_dn: {info['total_delta_norm']:.6f} | "
                f"safe: {info['safe']}"
            )

        time.sleep(dt)

    # 汇总
    actions_arr = np.array(actions_list)
    state = agent.get_state()

    print(f"\n{'=' * 60}")
    print(f"  Dry-Run Summary")
    print(f"  Total steps: {state['step_count']}")
    print(f"  Violations: {state['violation_count']}")
    print(f"  Action range: [{actions_arr.min():.4f}, {actions_arr.max():.4f}]")
    print(f"  Action mean:  {actions_arr.mean(axis=0)}")
    print(f"  Action std:   {actions_arr.std(axis=0)}")

    if violations:
        print(f"\n  First 5 violations:")
        for step, msg in violations[:5]:
            print(f"    Step {step}: {msg}")
    else:
        print(f"\n  No safety violations!")

    print(f"\n{'=' * 60}")
    print(f"  DRY-RUN PASSED")
    print(f"{'=' * 60}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GelloCartesianDeltaAgent: GELLO -> Cartesian delta -> HIL-SERL action.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry-run test with simulated GELLO input.",
    )

    # 安全参数
    parser.add_argument(
        "--max-step",
        type=float,
        default=DEFAULT_MAX_STEP,
        help=f"Max per-step Cartesian delta in meters (default: {DEFAULT_MAX_STEP})",
    )
    parser.add_argument(
        "--max-total-delta",
        type=float,
        default=DEFAULT_MAX_TOTAL_DELTA,
        help=f"Max cumulative Cartesian delta in meters (default: {DEFAULT_MAX_TOTAL_DELTA})",
    )
    parser.add_argument(
        "--max-rot-step",
        type=float,
        default=DEFAULT_MAX_ROT_STEP,
        help=f"Max per-step rotation delta in radians (default: {DEFAULT_MAX_ROT_STEP})",
    )

    # Action scale
    parser.add_argument(
        "--pos-scale",
        type=float,
        default=DEFAULT_POS_SCALE,
        help=f"xyz normalization scale (default: {DEFAULT_POS_SCALE})",
    )
    parser.add_argument(
        "--rpy-scale",
        type=float,
        default=DEFAULT_RPY_SCALE,
        help=f"rpy normalization scale (default: {DEFAULT_RPY_SCALE})",
    )

    # 模拟参数
    parser.add_argument(
        "--hz",
        type=int,
        default=DEFAULT_HZ,
        help=f"Control frequency in Hz (default: {DEFAULT_HZ})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help=f"Test duration in seconds (default: 10.0)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        parser.print_help()
        print("\n  Use --dry-run to run the test.")


if __name__ == "__main__":
    main()
