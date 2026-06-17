#!/usr/bin/env python3
"""run_learner.py

HIL-SERL Learner 入口 — 运行在 GPU 机 (zktitan)。

Learner 职责:
  1. 接收 Actor 发来的 transitions (通过 agentlace)
  2. 维护 replay buffer (demo + RL)
  3. 运行 SAC 更新 (critic + actor)
  4. 定期 checkpoint
  5. 将更新后的 policy 权重发送给 Actor

用法:
  python -m experiments.plug_insertion.run_learner
  python -m experiments.plug_insertion.run_learner --resume_from 50000

环境变量:
  LEARNER_IP       监听地址 (默认: 0.0.0.0)
  LEARNER_PORT     监听端口 (默认: 50051)
  ACTOR_IP         Actor 机器 IP (默认: 10.192.4.136)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.plug_insertion.config import TrainConfig


def build_learner(config: TrainConfig, env: gym.Env, seed: int = 42):
    """构建 SAC learner agent。"""
    from serl_launcher.agents.continuous.sac_sacred import SACAgent
    from serl_launcher.utils.launcher import make_sac_agent
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

    num_devices = len(jax.local_devices())
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # Build agent
    agent = make_sac_agent(
        seed=seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=config.image_keys,
        encoder_type=config.encoder_type,
        discount=config.discount,
        cta_ratio=config.cta_ratio,
        setup_mode=config.setup_mode,
    )
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    # Build replay buffer
    rl_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=500_000,
        image_keys=config.image_keys,
    )

    return agent, rl_buffer, sharding, num_devices


def load_demo_data(config: TrainConfig, env: gym.Env, demo_capacity: int = 100_000):
    """加载 demo 数据到 replay buffer。"""
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=demo_capacity,
        image_keys=config.image_keys,
    )

    # 查找 demo pkl 文件
    demo_dir = PROJECT_ROOT / "data" / "demos"
    if not demo_dir.exists():
        print(f"  No demo directory found at {demo_dir}, skipping demo loading")
        return demo_buffer

    import pickle
    import glob

    pkl_files = sorted(glob.glob(str(demo_dir / "**/*.pkl"), recursive=True))
    print(f"  Found {len(pkl_files)} demo pkl files")

    total_inserted = 0
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                transitions = pickle.load(f)
            if isinstance(transitions, list):
                for t in transitions:
                    demo_buffer.insert(t)
                    total_inserted += 1
        except Exception as e:
            print(f"  [WARN] Failed to load {pkl_file}: {e}")

    print(f"  Demo buffer: {total_inserted} transitions loaded")
    return demo_buffer


def run_learner(args):
    """Learner 主循环。"""
    config = TrainConfig()

    # 创建 fake env 用于构建 agent (learner 不连真机)
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Build learner
    agent, rl_buffer, sharding, num_devices = build_learner(config, env, seed=args.seed)
    print(f"Learner built on {num_devices} device(s)")

    # Load demo data
    demo_buffer = load_demo_data(config, env)
    print(f"Demo buffer size: {len(demo_buffer)}")

    # agentlace: start learner server
    learner_ip = args.learner_ip or os.environ.get("LEARNER_IP", "0.0.0.0")
    learner_port = int(args.learner_port or os.environ.get("LEARNER_PORT", "50051"))
    actor_ip = args.actor_ip or os.environ.get("ACTOR_IP", "10.192.4.136")

    try:
        from agentlace.learner import LearnerServer
        learner_server = LearnerServer(
            agent=agent,
            learner_ip=learner_ip,
            learner_port=learner_port,
            actor_ip=actor_ip,
        )
        print(f"LearnerServer listening on {learner_ip}:{learner_port}")
        print(f"  Actor expected at: {actor_ip}")
    except ImportError:
        print("[WARN] agentlace not available, running in offline mode")
        learner_server = None

    # Resume from checkpoint
    start_step = 0
    if args.resume_from:
        from orbax.checkpoint import PyTreeCheckpointer
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", f"actor_{args.resume_from}")
        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpointer = PyTreeCheckpointer()
            restored = checkpointer.restore(ckpt_path)
            agent = restored["agent"]
            start_step = int(args.resume_from)
            print(f"  Resumed at step {start_step}")
        else:
            print(f"  [WARN] Checkpoint not found: {ckpt_path}, starting fresh")

    # Training loop
    steps = start_step
    updates = 0
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Replay buffer iterator
    replay_iterator = rl_buffer.get_iterator(
        sample_args={"batch_size": 256, "pack_obs_and_next_obs": False},
        device=sharding.replicate(),
    )

    print(f"\nStarting learner training loop...")
    print(f"  Checkpoint period: {config.checkpoint_period}")
    print(f"  Target steps: infinite (until interrupted)")

    while True:
        # Receive transition from actor
        if learner_server is not None:
            transition = learner_server.receive_transition()
            if transition is not None:
                rl_buffer.insert(transition)
                steps += 1
        else:
            # Offline mode: just wait
            time.sleep(0.01)
            steps += 1

        # SAC update
        if len(rl_buffer) > 256:
            batch = next(replay_iterator)
            agent, info = agent.update(batch)
            updates += 1

            if updates % 100 == 0:
                print(f"  Step {steps} | Update {updates} | "
                      f"actor_loss={info.get('actor_loss', 0):.4f} | "
                      f"critic_loss={info.get('critic_loss', 0):.4f}")

        # Checkpoint
        if steps > 0 and steps % config.checkpoint_period == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"actor_{steps}")
            print(f"  Saving checkpoint: {ckpt_path}")
            from orbax.checkpoint import PyTreeCheckpointer
            checkpointer = PyTreeCheckpointer()
            checkpointer.save(ckpt_path, {"agent": agent, "step": steps})

            # Send updated agent to actor
            if learner_server is not None:
                learner_server.sync_agent(agent)
                print(f"  Synced agent to actor at step {steps}")

        # Send updated agent periodically
        if learner_server is not None and steps % config.buffer_period == 0:
            learner_server.sync_agent(agent)


def main():
    parser = argparse.ArgumentParser(
        description="HIL-SERL Learner — plug_insertion experiment"
    )
    parser.add_argument("--resume_from", type=int, default=None, help="Resume from checkpoint step")
    parser.add_argument("--learner_ip", type=str, default=None, help="Learner listen IP")
    parser.add_argument("--learner_port", type=str, default=None, help="Learner listen port")
    parser.add_argument("--actor_ip", type=str, default=None, help="Actor machine IP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_learner(args)


if __name__ == "__main__":
    main()
