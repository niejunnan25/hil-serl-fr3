#!/usr/bin/env python3
"""load_zarr_to_serl.py

将 convert_to_zarr.py 输出的 zarr 数据加载到 SERL replay buffer。

功能:
  1. 读取 zarr 数据 (observations/state, actions, rewards, dones)
  2. 尝试适配 SERL 的 MemoryEfficientReplayBufferDataStore
  3. 若 SERL 不可用，使用内置 SimpleReplayBuffer 兼容实现
  4. 支持 demo buffer 和 RL buffer 双缓冲区
  5. 验证数据完整性

用法:
  # 加载单个 zarr
  python load_zarr_to_serl.py data.zarr

  # 加载多个 zarr 到 demo buffer
  python load_zarr_to_serl.py demo1.zarr demo2.zarr --buffer-type demo

  # 指定 capacity
  python load_zarr_to_serl.py data.zarr --demo-capacity 50000 --rl-capacity 200000

  # 仅验证不加载
  python load_zarr_to_serl.py data.zarr --verify-only

  # 从 buffer 采样并打印
  python load_zarr_to_serl.py data.zarr --sample 32

依赖:
  - zarr, numpy
  - (可选) SERL serl_robot_infra 包
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


LIVE_SERL_STATE_DIM = 19
LEGACY_8D_STATE_DIM = 8
LEGACY_8D_WARNING = (
    "zarr contains legacy 8D joint-state observations, not the live SERL19 flat state. "
    "Pass allow_legacy_8d=True or --allow-legacy-8d only for explicit offline legacy use."
)

# ---------------------------------------------------------------------------
# 尝试导入 SERL 原生 replay buffer
# ---------------------------------------------------------------------------
SERL_AVAILABLE = False
MemoryEfficientReplayBufferDataStore = None

try:
    from serl.replay_buffer import MemoryEfficientReplayBufferDataStore  # type: ignore

    SERL_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# SimpleReplayBuffer — SERL 不可用时的兼容实现
# ---------------------------------------------------------------------------
class SimpleReplayBuffer:
    """兼容 SERL MemoryEfficientReplayBufferDataStore 接口的简化 replay buffer。

    支持:
      - insert(): 插入单条 transition
      - insert_batch(): 批量插入 transitions
      - sample(): 随机采样返回 (obs, action, reward, done) tuple
      - n_steps 参数: 多步 return 计算
      - 容量限制 (FIFO 覆盖)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 100_000,
        n_steps: int = 1,
        gamma: float = 0.99,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma

        # Pre-allocate ring buffers
        self._observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)

        self._ptr = 0  # write pointer
        self._size = 0  # current number of entries

    @property
    def size(self) -> int:
        return self._size

    def insert(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """插入单条 transition。"""
        idx = self._ptr % self.capacity
        self._observations[idx] = obs
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._dones[idx] = done
        self._ptr += 1
        self._size = min(self._size + 1, self.capacity)

    def insert_batch(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """批量插入 transitions。"""
        N = len(actions)
        for i in range(N):
            self.insert(observations[i], actions[i], rewards[i], dones[i])

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样 batch_size 条 transitions。

        当 n_steps > 1 时, reward 被聚合为 n-step return:
            R_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1}
        done 标记为 n-step 窗口内是否有任何 done。

        Returns:
            (observations, actions, rewards, dones) — 各自 shape:
            obs: (batch_size, obs_dim), action: (batch_size, action_dim),
            reward: (batch_size,), done: (batch_size,)
        """
        if self._size == 0:
            raise ValueError("Buffer 为空, 无法采样")

        indices = np.random.randint(0, self._size, size=batch_size)

        if self.n_steps == 1:
            return (
                self._observations[indices].copy(),
                self._actions[indices].copy(),
                self._rewards[indices].copy(),
                self._dones[indices].copy(),
            )

        # n-step return
        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        actions = np.zeros((batch_size, self.action_dim), dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.bool_)

        for b, idx in enumerate(indices):
            obs[b] = self._observations[idx]
            actions[b] = self._actions[idx]

            # 累积 n-step return
            n_return = 0.0
            any_done = False
            for step in range(self.n_steps):
                s_idx = (idx + step) % self.capacity
                n_return += (self.gamma ** step) * self._rewards[s_idx]
                if self._dones[s_idx]:
                    any_done = True
                    break

            rewards[b] = n_return
            dones[b] = any_done

        return obs, actions, rewards, dones

    def load_from_zarr(self, zarr_path: str) -> Dict[str, Any]:
        """从 zarr 文件加载数据到 buffer。

        Returns:
            dict: 加载统计信息
        """
        import zarr

        root = zarr.open(zarr_path, mode="r")

        observations = root["observations/state"][:]
        actions = root["actions"][:]
        rewards = root["rewards"][:]
        dones = root["dones"][:]

        N = len(actions)
        self.insert_batch(observations, actions, rewards, dones)

        return {
            "path": zarr_path,
            "n_transitions": N,
            "buffer_size": self._size,
            "obs_shape": observations.shape,
            "action_shape": actions.shape,
        }

    def verify_integrity(self) -> Dict[str, Any]:
        """验证 buffer 内数据完整性。

        Returns:
            dict: 验证结果
        """
        if self._size == 0:
            return {"ok": False, "error": "Buffer 为空"}

        obs = self._observations[: self._size]
        acts = self._actions[: self._size]
        rews = self._rewards[: self._size]
        dns = self._dones[: self._size]

        checks = {
            "size": self._size,
            "obs_shape": obs.shape,
            "action_shape": acts.shape,
            "obs_has_nan": bool(np.isnan(obs).any()),
            "obs_has_inf": bool(np.isinf(obs).any()),
            "action_has_nan": bool(np.isnan(acts).any()),
            "action_range": (float(acts.min()), float(acts.max())),
            "reward_range": (float(rews.min()), float(rews.max())),
            "num_dones": int(dns.sum()),
            "n_steps": self.n_steps,
        }

        # 行动范围检查
        action_in_range = acts.min() >= -1.0 - 1e-5 and acts.max() <= 1.0 + 1e-5
        checks["action_in_1_1"] = action_in_range

        checks["ok"] = (
            not checks["obs_has_nan"]
            and not checks["obs_has_inf"]
            and not checks["action_has_nan"]
            and action_in_range
        )

        return checks

    def __repr__(self) -> str:
        return (
            f"SimpleReplayBuffer(obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"capacity={self.capacity}, n_steps={self.n_steps}, size={self._size})"
        )


# ---------------------------------------------------------------------------
# SERL 适配层
# ---------------------------------------------------------------------------
def load_zarr_to_serl_buffer(
    zarr_paths: List[str],
    buffer_type: str = "demo",
    demo_capacity: int = 100_000,
    rl_capacity: int = 500_000,
    n_steps: int = 1,
    gamma: float = 0.99,
    allow_legacy_8d: bool = False,
) -> Dict[str, Any]:
    """将 zarr 文件加载到 replay buffer (SERL 或 SimpleReplayBuffer)。

    Args:
        zarr_paths: zarr 文件路径列表
        buffer_type: "demo" 或 "rl"
        demo_capacity: demo buffer 容量
        rl_capacity: RL buffer 容量
        n_steps: n-step return 参数
        gamma: discount factor

    Returns:
        dict: 包含 buffer 对象和加载统计
    """
    import zarr

    # 探测数据维度
    first_root = zarr.open(zarr_paths[0], mode="r")
    obs_dim = first_root["observations/state"].shape[1]
    action_dim = first_root["actions"].shape[1]
    _require_supported_state_dim(obs_dim, allow_legacy_8d=allow_legacy_8d)

    capacity = demo_capacity if buffer_type == "demo" else rl_capacity

    print(f"\n{'='*60}")
    print(f"Buffer 类型: {buffer_type}")
    print(f"obs_dim: {obs_dim}, action_dim: {action_dim}")
    print(f"capacity: {capacity}, n_steps: {n_steps}")

    # 尝试使用 SERL 原生 buffer
    if SERL_AVAILABLE and MemoryEfficientReplayBufferDataStore is not None:
        print("使用 SERL MemoryEfficientReplayBufferDataStore")
        try:
            buffer = MemoryEfficientReplayBufferDataStore(
                capacity=capacity,
                obs_dim=obs_dim,
                action_dim=action_dim,
            )
            # SERL 的加载方式: 逐文件 insert
            stats = _load_with_serl_buffer(buffer, zarr_paths, obs_dim, action_dim)
            stats["backend"] = "serl"
            return stats
        except Exception as e:
            print(f"SERL buffer 创建/加载失败: {e}")
            print("回退到 SimpleReplayBuffer")

    # 使用 SimpleReplayBuffer
    if not SERL_AVAILABLE:
        print("SERL 不可用, 使用 SimpleReplayBuffer (兼容实现)")
    else:
        print("回退到 SimpleReplayBuffer")

    buffer = SimpleReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=capacity,
        n_steps=n_steps,
        gamma=gamma,
    )

    load_stats = []
    for zarr_path in zarr_paths:
        print(f"\n  Loading: {zarr_path}")
        stat = buffer.load_from_zarr(zarr_path)
        print(f"    加载 {stat['n_transitions']} 条 transitions")
        load_stats.append(stat)

    return {
        "buffer": buffer,
        "backend": "simple",
        "buffer_type": buffer_type,
        "total_transitions": buffer.size,
        "files_loaded": load_stats,
    }


def _load_with_serl_buffer(
    buffer: Any,
    zarr_paths: List[str],
    obs_dim: int,
    action_dim: int,
) -> Dict[str, Any]:
    """用 SERL 原生 buffer 加载 zarr 数据。"""
    import zarr

    load_stats = []
    for zarr_path in zarr_paths:
        print(f"\n  Loading: {zarr_path}")
        root = zarr.open(zarr_path, mode="r")
        observations = root["observations/state"][:]
        actions = root["actions"][:]
        rewards = root["rewards"][:]
        dones = root["dones"][:]
        N = len(actions)

        for i in range(N):
            buffer.insert(
                obs=observations[i],
                action=actions[i],
                reward=float(rewards[i]),
                done=bool(dones[i]),
            )

        print(f"    加载 {N} 条 transitions")
        load_stats.append({
            "path": zarr_path,
            "n_transitions": N,
        })

    return {
        "buffer": buffer,
        "buffer_type": "unknown",
        "total_transitions": sum(s["n_transitions"] for s in load_stats),
        "files_loaded": load_stats,
    }


# ---------------------------------------------------------------------------
# 验证工具
# ---------------------------------------------------------------------------
def verify_zarr_for_buffer(zarr_path: str, allow_legacy_8d: bool = False) -> bool:
    """验证 zarr 文件是否满足 replay buffer 加载条件。

    Args:
        zarr_path: zarr 目录路径

    Returns:
        bool: 验证是否通过
    """
    return _verify_zarr_for_buffer(zarr_path, allow_legacy_8d=allow_legacy_8d)


def _verify_zarr_for_buffer(zarr_path: str, allow_legacy_8d: bool = False) -> bool:
    import zarr

    if not os.path.exists(zarr_path):
        print(f"  [FAIL] 路径不存在: {zarr_path}")
        return False

    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        print(f"  [FAIL] 无法打开 zarr: {e}")
        return False

    ok = True
    print(f"\n  验证 zarr buffer 兼容性: {zarr_path}")
    print(f"  {'='*50}")

    # 1. 检查必需路径
    required = [
        ("observations/state", True),
        ("actions", True),
        ("rewards", True),
        ("dones", True),
    ]
    for path, is_dataset in required:
        try:
            node = root[path]
            if is_dataset:
                print(f"  [OK]   {path}: shape={node.shape}, dtype={node.dtype}")
        except KeyError:
            print(f"  [FAIL] {path}: 缺失")
            ok = False

    if not ok:
        print(f"\n  FAILED: 缺少必需字段")
        return False

    # 2. 加载全部数据做深度检查
    observations = root["observations/state"][:]
    actions = root["actions"][:]
    rewards = root["rewards"][:]
    dones = root["dones"][:]
    N = actions.shape[0]

    # 3. shape 一致性
    for name, arr in [("observations/state", observations), ("rewards", rewards), ("dones", dones)]:
        if arr.shape[0] != N:
            print(f"  [FAIL] {name} N={arr.shape[0]} != actions N={N}")
            ok = False

    # 4. NaN / Inf 检查
    for name, arr in [("observations/state", observations), ("actions", actions), ("rewards", rewards)]:
        has_nan = np.isnan(arr).any()
        has_inf = np.isinf(arr).any()
        if has_nan:
            print(f"  [FAIL] {name} 包含 NaN")
            ok = False
        if has_inf:
            print(f"  [FAIL] {name} 包含 Inf")
            ok = False

    # 5. action 范围
    a_min, a_max = float(actions.min()), float(actions.max())
    in_range = a_min >= -1.0 - 1e-5 and a_max <= 1.0 + 1e-5
    print(f"  [{'OK' if in_range else 'FAIL'}] action range: [{a_min:.4f}, {a_max:.4f}] (expect [-1, 1])")
    if not in_range:
        ok = False

    # 6. state 维度
    state_dim = observations.shape[1]
    if state_dim == LIVE_SERL_STATE_DIM:
        print(f"  [OK] state dim: {state_dim} (live SERL19)")
    elif state_dim == LEGACY_8D_STATE_DIM and allow_legacy_8d:
        print(f"  [OK] state dim: {state_dim} (explicit legacy 8D)")
    elif state_dim == LEGACY_8D_STATE_DIM:
        print(f"  [FAIL] state dim: {state_dim} is legacy 8D; pass allow_legacy_8d=True only for explicit legacy use")
        ok = False
    else:
        print(f"  [FAIL] state dim: {state_dim} (expect live {LIVE_SERL_STATE_DIM}D)")
        ok = False

    # 7. action 维度
    action_dim = actions.shape[1]
    print(f"  [{'OK' if action_dim == 7 else 'WARN'}] action dim: {action_dim} (expected 7 for SERL)")

    # 8. dones 检查
    num_dones = int(dones.sum())
    print(f"  [INFO] num dones: {num_dones}/{N}")
    if num_dones == 0:
        print(f"  [WARN] 没有 done=True 的帧 (可能导致 n-step 计算异常)")

    print(f"\n  {'PASSED' if ok else 'FAILED'}")
    return ok


def create_demo_and_rl_buffers(
    zarr_paths: List[str],
    demo_capacity: int = 100_000,
    rl_capacity: int = 500_000,
    n_steps: int = 1,
    gamma: float = 0.99,
    allow_legacy_8d: bool = False,
) -> Dict[str, Any]:
    """创建 demo + RL 双缓冲区，并将 zarr 数据加载到 demo buffer。

    Args:
        zarr_paths: demo zarr 文件路径列表
        demo_capacity: demo buffer 容量
        rl_capacity: RL buffer 容量
        n_steps: n-step return 参数
        gamma: discount factor

    Returns:
        dict: {"demo": buffer_info, "rl": buffer_info}
    """
    import zarr

    # 探测维度
    first_root = zarr.open(zarr_paths[0], mode="r")
    obs_dim = first_root["observations/state"].shape[1]
    action_dim = first_root["actions"].shape[1]
    _require_supported_state_dim(obs_dim, allow_legacy_8d=allow_legacy_8d)

    print(f"\n{'='*60}")
    print(f"创建双缓冲区")
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  demo capacity={demo_capacity}, rl capacity={rl_capacity}")
    print(f"  n_steps={n_steps}, gamma={gamma}")

    # 创建 buffers
    if SERL_AVAILABLE and MemoryEfficientReplayBufferDataStore is not None:
        print("  使用 SERL MemoryEfficientReplayBufferDataStore")
        try:
            demo_buffer = MemoryEfficientReplayBufferDataStore(
                capacity=demo_capacity, obs_dim=obs_dim, action_dim=action_dim,
            )
            rl_buffer = MemoryEfficientReplayBufferDataStore(
                capacity=rl_capacity, obs_dim=obs_dim, action_dim=action_dim,
            )
            backend = "serl"
        except Exception as e:
            print(f"  SERL 创建失败: {e}, 回退到 SimpleReplayBuffer")
            demo_buffer = SimpleReplayBuffer(obs_dim, action_dim, demo_capacity, n_steps, gamma)
            rl_buffer = SimpleReplayBuffer(obs_dim, action_dim, rl_capacity, n_steps, gamma)
            backend = "simple"
    else:
        if not SERL_AVAILABLE:
            print("  SERL 不可用, 使用 SimpleReplayBuffer")
        demo_buffer = SimpleReplayBuffer(obs_dim, action_dim, demo_capacity, n_steps, gamma)
        rl_buffer = SimpleReplayBuffer(obs_dim, action_dim, rl_capacity, n_steps, gamma)
        backend = "simple"

    # 加载 zarr 数据到 demo buffer
    total_loaded = 0
    for zarr_path in zarr_paths:
        print(f"\n  Loading demo data: {zarr_path}")
        if backend == "serl":
            root = zarr.open(zarr_path, mode="r")
            observations = root["observations/state"][:]
            actions = root["actions"][:]
            rewards = root["rewards"][:]
            dones = root["dones"][:]
            N = len(actions)
            for i in range(N):
                demo_buffer.insert(
                    obs=observations[i], action=actions[i],
                    reward=float(rewards[i]), done=bool(dones[i]),
                )
            total_loaded += N
        else:
            stat = demo_buffer.load_from_zarr(zarr_path)
            total_loaded += stat["n_transitions"]
        print(f"    加载 {total_loaded if backend == 'serl' else stat['n_transitions']} 条 transitions")

    print(f"\n  Demo buffer: {total_loaded} transitions loaded")
    print(f"  RL buffer: 0 transitions (空, 等待 RL 采样)")

    return {
        "demo": {
            "buffer": demo_buffer,
            "backend": backend,
            "size": total_loaded if backend == "serl" else demo_buffer.size,
        },
        "rl": {
            "buffer": rl_buffer,
            "backend": backend,
            "size": 0,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load zarr data into SERL replay buffer (or compatible SimpleReplayBuffer).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "zarr_paths",
        nargs="+",
        type=str,
        help="One or more zarr file paths to load",
    )
    parser.add_argument(
        "--buffer-type",
        type=str,
        choices=["demo", "rl", "both"],
        default="both",
        help="Buffer type to create (default: both — demo + RL dual buffers)",
    )
    parser.add_argument(
        "--demo-capacity",
        type=int,
        default=100_000,
        help="Demo buffer capacity (default: 100000)",
    )
    parser.add_argument(
        "--rl-capacity",
        type=int,
        default=500_000,
        help="RL buffer capacity (default: 500000)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1,
        help="N-step return parameter (default: 1)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for n-step return (default: 0.99)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅验证 zarr 文件, 不加载到 buffer",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="加载后从 buffer 采样 N 条并打印",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式, 减少输出",
    )
    parser.add_argument(
        "--allow-legacy-8d",
        action="store_true",
        help="acknowledge that input zarr uses legacy 8D joint-state observations, not live SERL19",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 验证所有 zarr 路径
    for path in args.zarr_paths:
        if not os.path.exists(path):
            print(f"[ERROR] zarr 路径不存在: {path}")
            sys.exit(1)

    # --verify-only 模式
    if args.verify_only:
        all_ok = True
        for path in args.zarr_paths:
            ok = _verify_zarr_for_buffer(path, allow_legacy_8d=args.allow_legacy_8d)
            if not ok:
                all_ok = False

        if len(args.zarr_paths) > 1:
            print(f"\n{'='*60}")
            print(f"总结: {'全部通过' if all_ok else '存在失败'}")

        sys.exit(0 if all_ok else 1)

    # 先验证
    if not args.quiet:
        for path in args.zarr_paths:
            _verify_zarr_for_buffer(path, allow_legacy_8d=args.allow_legacy_8d)

    # 加载
    t0 = time.time()

    if args.buffer_type == "both":
        buffers = create_demo_and_rl_buffers(
            zarr_paths=args.zarr_paths,
            demo_capacity=args.demo_capacity,
            rl_capacity=args.rl_capacity,
            n_steps=args.n_steps,
            gamma=args.gamma,
            allow_legacy_8d=args.allow_legacy_8d,
        )
        demo_info = buffers["demo"]
        rl_info = buffers["rl"]

        print(f"\n{'='*60}")
        print(f"双缓冲区创建完成")
        print(f"  Demo buffer: {demo_info['size']} transitions (backend={demo_info['backend']})")
        print(f"  RL buffer:   {rl_info['size']} transitions (backend={rl_info['backend']})")
        print(f"  耗时: {time.time() - t0:.2f}s")

        # 验证 demo buffer
        if isinstance(demo_info["buffer"], SimpleReplayBuffer) and not args.quiet:
            print(f"\n  Demo buffer 数据完整性:")
            integrity = demo_info["buffer"].verify_integrity()
            for k, v in integrity.items():
                print(f"    {k}: {v}")

        # 采样
        if args.sample and demo_info["size"] > 0:
            _print_sample(demo_info["buffer"], args.sample, "demo")

    else:
        result = load_zarr_to_serl_buffer(
            zarr_paths=args.zarr_paths,
            buffer_type=args.buffer_type,
            demo_capacity=args.demo_capacity,
            rl_capacity=args.rl_capacity,
            n_steps=args.n_steps,
            gamma=args.gamma,
            allow_legacy_8d=args.allow_legacy_8d,
        )

        buffer = result["buffer"]
        print(f"\n{'='*60}")
        print(f"加载完成")
        print(f"  Buffer: {result['total_transitions']} transitions")
        print(f"  Backend: {result['backend']}")
        print(f"  耗时: {time.time() - t0:.2f}s")

        if isinstance(buffer, SimpleReplayBuffer) and not args.quiet:
            integrity = buffer.verify_integrity()
            print(f"\n  数据完整性:")
            for k, v in integrity.items():
                print(f"    {k}: {v}")

        if args.sample and result["total_transitions"] > 0:
            _print_sample(buffer, args.sample, args.buffer_type)

    print("\nDone.")


def _print_sample(buffer: Any, batch_size: int, label: str) -> None:
    """从 buffer 采样并打印结果。"""
    if isinstance(buffer, SimpleReplayBuffer):
        obs, actions, rewards, dones = buffer.sample(batch_size)
    else:
        # SERL buffer 可能有不同的 API
        print(f"  [WARN] SERL buffer 的 sample API 未确认, 跳过采样")
        return

    n = min(batch_size, len(obs))
    print(f"\n  [{label}] Sample ({n} transitions):")
    print(f"    obs[0]:     {obs[0]}")
    print(f"    action[0]:  {actions[0]}")
    print(f"    reward[0]:  {rewards[0]:.4f}")
    print(f"    done[0]:    {dones[0]}")
    print(f"    obs shape:     {obs.shape}")
    print(f"    action shape:  {actions.shape}")
    print(f"    reward shape:  {rewards.shape}")
    print(f"    done shape:    {dones.shape}")
    print(f"    reward range:  [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"    action range:  [{actions.min():.4f}, {actions.max():.4f}]")


# ---------------------------------------------------------------------------
# 导出接口 (供其他脚本 import)
# ---------------------------------------------------------------------------
__all__ = [
    "SimpleReplayBuffer",
    "load_zarr_to_serl_buffer",
    "create_demo_and_rl_buffers",
    "verify_zarr_for_buffer",
]


def _require_supported_state_dim(obs_dim: int, *, allow_legacy_8d: bool) -> None:
    if obs_dim == LIVE_SERL_STATE_DIM:
        return
    if obs_dim == LEGACY_8D_STATE_DIM and allow_legacy_8d:
        return
    if obs_dim == LEGACY_8D_STATE_DIM:
        raise RuntimeError(LEGACY_8D_WARNING)
    raise RuntimeError(f"Unsupported zarr observation state dim {obs_dim}; expected live {LIVE_SERL_STATE_DIM}D")


if __name__ == "__main__":
    main()
