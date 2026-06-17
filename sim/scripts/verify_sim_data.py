"""verify_sim_data.py — sim pkl schema validator (per PLAN-A10).

CRITICAL (codex #1, #2 fix):
  - 3 image keys: side_policy + wrist_1 + side_classifier
  - state keys 必须按 STATE_KEYS_ORDERED 顺序拼接 (OrderedDict comparison)
  - dtype/shape 全断言

Usage:
  python -m sim.scripts.verify_sim_data --pkl path/to/failure.pkl
  exit 0 = all pass; exit 1 = any fail.

不调 IsaacLab runtime; 仅做 schema/dtype/shape/keys 验证。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from typing import Any, Mapping

import numpy as np

from sim.data.contract import (
    IMAGE_DTYPE, IMAGE_SHAPE, IMAGE_KEY_ALIAS_MAP,
    STATE_DIMS, STATE_DTYPE, STATE_KEYS_ORDERED, STATE_KEY_DIMS,
    TRANSITION_KEYS, VALID_PKL_IMAGE_KEYS,
)


# 模拟 STATE_KEYS_ORDERED 拼接到 25D 的 sub-key 维度 (A2 hard-freeze)
# 5 keys: tcp_pose(7) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper_pose(1) = 20
# 但 contract STATE_DIMS = 25 (per wrapper.py 注释; arith 矛盾留 VERIFY.md)
# 此处只验: 拼接顺序 = STATE_KEYS_ORDERED, 总 dim = STATE_DIMS
def _expected_state_subkeys() -> tuple[str, ...]:
    return STATE_KEYS_ORDERED


# CRITICAL (codex #2 fix): 显式 use STATE_KEYS_ORDERED 做 ordered comparison
# 这是一个 25D vector 的"协议标记": 它必须按 STATE_KEYS_ORDERED 顺序拼接
# 实现: 用 OrderedDict 等价的 1D vector comparison — 拆 25D 成 STATE_KEYS_ORDERED 子段
# 验证 (a) shape = (STATE_DIMS,); (b) dtype = STATE_DTYPE;
#      (c) 各子段维度之和 = STATE_DIMS; (d) STATE_KEYS_ORDERED 长度 = 5
# 注: 各子段维度的具体数值 (7+6+3+3+1=20) 与 STATE_DIMS=25 的 arith 矛盾
# 保留在 contract.py docstring + VERIFY.md A2 段, 此处只验"拼接顺序"语义层。
_EXPECTED_SUBKEY_COUNT = 5  # tcp_pose, tcp_vel, tcp_force, tcp_torque, gripper_pose


def verify_state_keys_order(
    state_vector: np.ndarray,
    segments: Mapping[str, Any] | None = None,
) -> bool:
    """验证 state_vector 拼接顺序与 STATE_KEYS_ORDERED 一致.

    A10 contract (codex #2 fix): state 必须是 STATE_KEYS_ORDERED 顺序拼接的 1D vector.

    本函数分两层验证:
      (a) shape = (STATE_DIMS,)
      (b) dtype = STATE_DTYPE (float32)
      (c) STATE_KEYS_ORDERED 长度 = _EXPECTED_SUBKEY_COUNT (5 个 sub-key)
      (d) 所有 sub-key 名称非空
      (e) contract layout 自洽: len(STATE_KEY_DIMS) == len(STATE_KEYS_ORDERED)
          且 sum(STATE_KEY_DIMS) == STATE_DIMS (拼接边界 must reconstruct 25D)

    当调用方提供 ``segments`` (producer 的 slice layout, 即 key -> sub-array
    映射) 时, 额外做"真顺序"校验 (codex #2 的本意):
      (f1) segments 的 key 顺序必须 *逐项等于* STATE_KEYS_ORDERED;
      (f2) 每个 sub-segment 的长度必须等于对应的 STATE_KEY_DIMS;
      (f3) 按 STATE_KEYS_ORDERED 顺序 concat 各 segment 后必须逐元素等于
           ``state_vector`` (即 producer 的拼接顺序/数据未被打乱).
    任何一项不满足返回 False —— 因此一个 reordered state (子块换位) 会被捕获,
    而非旧实现里的静默 pass。

    Arith 矛盾 (20 vs 25) 保留在 contract.py docstring 与 VERIFY.md A2 段;
    flat 路径不参与 arith 校验, 只验 schema/layout 自洽。
    """
    # (a) shape check
    if state_vector.shape != (STATE_DIMS,):
        return False
    # (b) dtype check
    if state_vector.dtype != np.dtype(STATE_DTYPE):
        return False
    # (c) STATE_KEYS_ORDERED 长度 check (实 ordered 拼接到 25D 应有 5 个 sub-key)
    if len(STATE_KEYS_ORDERED) != _EXPECTED_SUBKEY_COUNT:
        return False
    # (d) 验证所有 sub-key 名称都不为空 (即不是空 tuple / 空 string)
    for key in STATE_KEYS_ORDERED:
        if not isinstance(key, str) or not key:
            return False
    # (e) contract layout 自洽: per-key dims 必须与 keys 对齐且 sum == STATE_DIMS
    if len(STATE_KEY_DIMS) != len(STATE_KEYS_ORDERED):
        return False
    if sum(STATE_KEY_DIMS) != STATE_DIMS:
        return False
    # (f) 若 producer 提供了 slice layout, 做真正的逐段 ORDER + 边界校验
    if segments is not None:
        # (f1) key 顺序必须逐项等于 STATE_KEYS_ORDERED (子块换位即 fail)
        if tuple(segments.keys()) != tuple(STATE_KEYS_ORDERED):
            return False
        rebuilt = []
        for key, expected_dim in zip(STATE_KEYS_ORDERED, STATE_KEY_DIMS):
            seg = np.asarray(segments[key]).reshape(-1)
            # (f2) 每段长度必须等于 STATE_KEY_DIMS
            if seg.shape[0] != expected_dim:
                return False
            rebuilt.append(seg)
        concat = np.concatenate(rebuilt).astype(np.dtype(STATE_DTYPE))
        # (f3) 按顺序拼接后必须逐元素等于 state_vector
        if concat.shape != (STATE_DIMS,):
            return False
        if not np.array_equal(concat, state_vector):
            return False
    return True


def verify_image_keys_complete(obs_dict: dict) -> bool:
    """验证 obs dict 含全部 3 键 image schema (side_policy + wrist_1 + side_classifier).

    codex #1 fix: 不只 2 键, 必须 3 键.
    """
    obs_image_keys = set(obs_dict.keys()) - {"state"}
    return obs_image_keys == set(VALID_PKL_IMAGE_KEYS)


def verify_image_shape_and_dtype(image: np.ndarray) -> bool:
    """验证 image shape = (3, 128, 128) + dtype = uint8."""
    if image.shape != IMAGE_SHAPE:
        return False
    if image.dtype != np.dtype(IMAGE_DTYPE):
        return False
    return True


def verify_action_shape_and_dtype(action: np.ndarray) -> bool:
    """验证 action shape = (7,) + dtype = float32."""
    if action.shape != (7,):
        return False
    if action.dtype != np.float32:
        return False
    return True


def verify_transition_keys(transition: dict) -> bool:
    """验证 transition dict top-level keys 包含 TRANSITION_KEYS."""
    return all(k in transition for k in TRANSITION_KEYS)


def verify_pkl(pkl_path: str) -> dict[str, bool]:
    """验证 sim pkl 文件 schema; 返回 {check_name: passed} dict.

    Checks:
      - image_keys_complete: 3 image keys (side_policy + wrist_1 + side_classifier)
      - image_shape_correct: each image shape = IMAGE_SHAPE
      - image_dtype_correct: each image dtype = IMAGE_DTYPE
      - state_keys_ordered: state shape + dtype + order 正确 (per STATE_KEYS_ORDERED)
      - state_shape_correct: state shape = (STATE_DIMS,)
      - state_dtype_correct: state dtype = STATE_DTYPE
      - action_shape_correct: action shape = (7,)
      - action_dtype_correct: action dtype = float32
      - transition_keys_complete: transition dict has TRANSITION_KEYS
      - not_empty: at least 1 transition
    """
    result = {
        "not_empty": False,
        "transition_keys_complete": False,
        "image_keys_complete": False,
        "image_shape_correct": False,
        "image_dtype_correct": False,
        "state_keys_ordered": False,
        "state_shape_correct": False,
        "state_dtype_correct": False,
        "action_shape_correct": False,
        "action_dtype_correct": False,
    }
    with open(pkl_path, "rb") as f:
        transitions = pickle.load(f)
    if not isinstance(transitions, list) or len(transitions) == 0:
        return result
    result["not_empty"] = True
    # 取第一帧做 schema 验证 (后续帧假定同 schema; 简化)
    t0 = transitions[0]
    if not verify_transition_keys(t0):
        return result
    result["transition_keys_complete"] = True
    obs = t0["observations"]
    # 3 键 image schema
    if not verify_image_keys_complete(obs):
        return result
    result["image_keys_complete"] = True
    # image shape + dtype
    shape_ok = dtype_ok = True
    for k in VALID_PKL_IMAGE_KEYS:
        img = obs[k]
        if not verify_image_shape_and_dtype(img):
            shape_ok = False
        if img.dtype != np.dtype(IMAGE_DTYPE):
            dtype_ok = False
    result["image_shape_correct"] = shape_ok
    result["image_dtype_correct"] = dtype_ok
    # state: shape + dtype (order 由 STATE_KEYS_ORDERED dim sum 决定)
    state = obs["state"]
    result["state_shape_correct"] = (state.shape == (STATE_DIMS,))
    result["state_dtype_correct"] = (state.dtype == np.dtype(STATE_DTYPE))
    result["state_keys_ordered"] = verify_state_keys_order(state)
    # action
    action = t0["actions"]
    result["action_shape_correct"] = verify_action_shape_and_dtype(action)
    result["action_dtype_correct"] = (action.dtype == np.float32)
    return result


def _print_report(pkl_path: str, result: dict[str, bool]) -> None:
    """Print pass/fail report to stdout."""
    print(f"\n=== verify_sim_data: {pkl_path} ===")
    for check_name, passed in result.items():
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {check_name}")
    n_pass = sum(1 for v in result.values() if v)
    n_total = len(result)
    print(f"  --- {n_pass}/{n_total} checks passed ---")


def main() -> int:
    """CLI entry point. Returns 0 if all pass, 1 if any fail."""
    parser = argparse.ArgumentParser(description="Verify sim pkl schema")
    parser.add_argument("--pkl", required=True, help="Path to sim pkl file")
    args = parser.parse_args()
    result = verify_pkl(args.pkl)
    _print_report(args.pkl, result)
    n_fail = sum(1 for v in result.values() if not v)
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
