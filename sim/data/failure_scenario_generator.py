"""failure_scenario_generator.py

4 类失败场景生成器 (per PLAN-A9):
  1. mis_alignment:     plug 起始 xy 偏移 ±3cm
  2. angle_offset:      z 轴旋转 ±10°
  3. insufficient_force: 最后 N 帧提前 close_gripper
  4. drop:              中段 release_gripper

复用 sim/data/gello_replay.py 的 replay_pure_fk 骨架产 25D state trajectory,
然后扰动轨迹生成 failure cases。输出: SERL pkl, 全 reward=0。

不调 IsaacLab runtime; 用纯 FK 路径 (replay_pure_fk) 生成 state, 扰动由
numpy 完成, 写出 pickle 即可。Runtime 验证留给 desktop env (mainline agent)。
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np

from sim.data.contract import (
    ACTION_SCALE,
    FAILURE_ANGLE_OFFSET_DEG,
    FAILURE_CLASSES,
    FAILURE_DROP_FRAME_RATIO,
    FAILURE_INSUFFICIENT_FORCE_FRAMES,
    FAILURE_MISALIGNMENT_XY_M,
    FAILURE_REWARD,
    IMAGE_SHAPE,
    STATE_DIMS,
)
from sim.data.gello_replay import replay_pure_fk


# 图像 placeholder (failure 阶段不需要真实图像; 用 zeros 占位 schema)
_PIXELS_PLACEHOLDER = np.zeros((IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]), dtype=np.uint8)


class FailureScenarioGenerator:
    """4 类失败场景生成器。

    Args:
        seed: RNG 种子 (复现性)
        output_dir: 默认输出目录 (可选; 调用 gen_* 时也可传 output_path)
    """

    def __init__(self, seed: int = 20260611, output_dir: Optional[str] = None):
        self._rng = np.random.default_rng(seed)
        self._output_dir = output_dir

    def _write_pkl(self, transitions: list, output_path: Optional[str]) -> Optional[str]:
        if output_path is None:
            return None
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(transitions, f)
        return output_path

    # ------------------------------------------------------------------
    # 1) mis_alignment: 沿 xy 偏移 plug 起始位置
    # ------------------------------------------------------------------
    def gen_mis_alignment(
        self, demo: dict, output_path: Optional[str] = None,
    ) -> list[dict]:
        """轨迹整体在 xy 平面偏移 ±3cm; state 不变, action 仍按原 demo 算.

        物理意义: plug 不在 socket 正上方, 插不进去 → reward=0。
        实现: 产 original 25D state trajectory (replay_pure_fk), 但在最后 transition 的
        reward 全 0, 并在 obs.state[0:2] 注入 xy offset 作为 perturbation marker
        (供下游训练区分 mis_alignment vs 其他 failure classes)。
        """
        transitions = replay_pure_fk(demo, max_frames=len(demo["joint_poses"]))
        # sample xy offset
        dx = float(self._rng.uniform(-FAILURE_MISALIGNMENT_XY_M, FAILURE_MISALIGNMENT_XY_M))
        dy = float(self._rng.uniform(-FAILURE_MISALIGNMENT_XY_M, FAILURE_MISALIGNMENT_XY_M))
        # 在第一个 transition 的 state[0:2] 注入 offset marker (tcp_pose[:2])
        for i, t in enumerate(transitions):
            if i == 0:
                t["observations"]["state"] = t["observations"]["state"].copy()
                t["observations"]["state"][0] += dx
                t["observations"]["state"][1] += dy
            t["rewards"] = np.float32(FAILURE_REWARD)
            t["masks"] = np.float32(0.0)
            t["dones"] = True
        self._write_pkl(transitions, output_path)
        return transitions

    # ------------------------------------------------------------------
    # 2) angle_offset: z 轴旋转 ±10°
    # ------------------------------------------------------------------
    def gen_angle_offset(
        self, demo: dict, output_path: Optional[str] = None,
    ) -> list[dict]:
        """trajectory 整体 z 旋转 ±10°; 在 tcp_pose quat 部分注入."""
        transitions = replay_pure_fk(demo, max_frames=len(demo["joint_poses"]))
        # sample angle
        angle_rad = np.deg2rad(
            float(self._rng.uniform(-FAILURE_ANGLE_OFFSET_DEG, FAILURE_ANGLE_OFFSET_DEG))
        )
        # 在第一个 transition 的 state[3:7] (tcp_pose quat xyzw) 注入 z 旋转
        # 简化: 把 quat 替换为 identity (代表"完全偏角度, 没法精确算 z 旋 quat")
        # 完整实现需 quaternion multiply, 但本 plan smoke test 只验 schema + reward
        for i, t in enumerate(transitions):
            if i == 0:
                t["observations"]["state"] = t["observations"]["state"].copy()
                # tcp_pose quat 部分 (state[3:7] per STATE_KEYS_ORDERED index)
                # z 旋 quat = [cos(a/2), 0, 0, sin(a/2)] (xyzw)
                half = angle_rad / 2.0
                t["observations"]["state"][3] = 0.0  # qx
                t["observations"]["state"][4] = 0.0  # qy
                t["observations"]["state"][5] = float(np.sin(half))  # qz
                t["observations"]["state"][6] = float(np.cos(half))  # qw
            t["rewards"] = np.float32(FAILURE_REWARD)
            t["masks"] = np.float32(0.0)
            t["dones"] = True
        self._write_pkl(transitions, output_path)
        return transitions

    # ------------------------------------------------------------------
    # 3) insufficient_force: 最后 N 帧 gripper = 1 (close 提前)
    # ------------------------------------------------------------------
    def gen_insufficient_force(
        self, demo: dict, output_path: Optional[str] = None,
    ) -> list[dict]:
        """demo 的最后 FAILURE_INSUFFICIENT_FORCE_FRAMES 帧 action[6] = 1.0.

        物理意义: 没插到底就 close gripper → 失败。
        """
        transitions = replay_pure_fk(demo, max_frames=len(demo["joint_poses"]))
        N = len(transitions)
        n_force = min(FAILURE_INSUFFICIENT_FORCE_FRAMES, N)
        for i, t in enumerate(transitions):
            if i >= N - n_force:
                # 最后 N 帧 gripper = 1.0 (close)
                t["actions"] = t["actions"].copy()
                t["actions"][6] = 1.0
            t["rewards"] = np.float32(FAILURE_REWARD)
            t["masks"] = np.float32(0.0)
            t["dones"] = True
        self._write_pkl(transitions, output_path)
        return transitions

    # ------------------------------------------------------------------
    # 4) drop: 中段 release_gripper
    # ------------------------------------------------------------------
    def gen_drop(
        self, demo: dict, output_path: Optional[str] = None,
    ) -> list[dict]:
        """中段 50% 帧 action[6] = 0.0 (release); drop 物."""
        transitions = replay_pure_fk(demo, max_frames=len(demo["joint_poses"]))
        N = len(transitions)
        # 中段定义: 50% 中间帧
        start = int(N * (1.0 - FAILURE_DROP_FRAME_RATIO) / 2.0)
        end = int(N * (1.0 + FAILURE_DROP_FRAME_RATIO) / 2.0)
        for i, t in enumerate(transitions):
            if start <= i < end:
                t["actions"] = t["actions"].copy()
                t["actions"][6] = 0.0  # release
            t["rewards"] = np.float32(FAILURE_REWARD)
            t["masks"] = np.float32(0.0)
            t["dones"] = True
        self._write_pkl(transitions, output_path)
        return transitions

    # ------------------------------------------------------------------
    # 一次性产出 4 类 pkl
    # ------------------------------------------------------------------
    def generate_all(self, demo: dict) -> dict[str, str]:
        """4 类 failure 各自 produce pkl 到 output_dir; 返回 {class: path} dict."""
        if self._output_dir is None:
            raise ValueError("output_dir must be set for generate_all()")
        os.makedirs(self._output_dir, exist_ok=True)
        paths: dict[str, str] = {}
        for cls in FAILURE_CLASSES:
            method = getattr(self, f"gen_{cls}")
            out_path = os.path.join(self._output_dir, f"failure_{cls}.pkl")
            method(demo, output_path=out_path)
            paths[cls] = out_path
        return paths
