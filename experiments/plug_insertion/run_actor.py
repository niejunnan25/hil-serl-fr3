#!/usr/bin/env python3
"""run_actor.py

HIL-SERL Actor 入口 — 运行在机器人控制机 (fr3-desktop-ts)。

Actor 职责:
  1. 连接真实/仿真环境
  2. 使用当前 SAC policy 收集 transitions
  3. 通过 agentlace 将 transitions 发送给 Learner
  4. 接收 Learner 更新后的 policy 权重
  5. 支持 GELLO 人工干预 (human-in-the-loop)

训练模式:
  python -m experiments.plug_insertion.run_actor

评估模式:
  python -m experiments.plug_insertion.run_actor --eval_mode \
      --eval_checkpoint_step 50000 --eval_n_trajs 20

环境变量:
  LEARNER_IP       Learner 机器 IP (默认: 10.192.4.249)
  LEARNER_PORT     Learner gRPC 端口 (默认: 50051)
  SERL_FRANKA_PORT franka_server 端口 (默认: 5000)
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

from experiments.plug_insertion.config import TrainConfig, EnvConfig


def _info_success(info: dict) -> bool:
    """Return terminal success from either wrapper spelling."""
    return bool(info.get("succeed", info.get("success", False)))


def _resolve_eval_checkpoint_path(args) -> str:
    if args.eval_checkpoint_path:
        ckpt_path = args.eval_checkpoint_path
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        return ckpt_path
    if args.eval_checkpoint_step is not None:
        return os.path.join(PROJECT_ROOT, "checkpoints", f"actor_{args.eval_checkpoint_step}")
    raise ValueError("eval checkpoint requires --eval_checkpoint_path or --eval_checkpoint_step")


def build_agent(config: TrainConfig, env: gym.Env, seed: int = 42):
    """构建 SAC actor agent。

    使用 SERL launcher 构建 SAC agent，匹配 TrainConfig 中的超参数。
    """
    from serl_launcher.agents.continuous.sac_sacred import SACAgent
    from serl_launcher.utils.launcher import make_sac_agent

    num_devices = len(jax.local_devices())
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

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

    return agent, sharding, num_devices


def run_training_actor(args):
    """训练模式: Actor 与 Learner 协同训练。"""
    config = TrainConfig()
    env = config.get_environment(
        fake_env=args.fake_env,
        save_video=args.save_video,
        classifier=args.classifier,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Build agent
    agent, sharding, num_devices = build_agent(config, env, seed=args.seed)
    print(f"Agent built on {num_devices} device(s)")

    # agentlace: connect to learner
    learner_ip = args.learner_ip or os.environ.get("LEARNER_IP", "10.192.4.249")
    learner_port = int(args.learner_port or os.environ.get("LEARNER_PORT", "50051"))

    try:
        from agentlace.actor import ActorServer
        actor_server = ActorServer(
            agent=agent,
            learner_ip=learner_ip,
            learner_port=learner_port,
        )
        print(f"ActorServer connected to {learner_ip}:{learner_port}")
    except ImportError:
        print("[WARN] agentlace not available, running in standalone mode")
        actor_server = None

    # Training loop
    steps = 0
    episode = 0

    print(f"\nStarting training actor...")
    print(f"  Random steps: {config.random_steps}")
    print(f"  Checkpoint period: {config.checkpoint_period}")
    print(f"  Buffer period: {config.buffer_period}")

    while True:
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            # Select action
            if steps < config.random_steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(
                    observations=obs,
                    seed=jax.random.PRNGKey(steps),
                    deterministic=False,
                )
                action = np.array(action)[0]

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Insert transition to buffer
            transition = {
                "observations": obs,
                "next_observations": next_obs,
                "actions": action,
                "rewards": float(reward),
                "dones": bool(done),
            }

            if actor_server is not None:
                actor_server.send_transition(transition)

            obs = next_obs
            episode_reward += float(reward)
            steps += 1
            episode_steps += 1

            # Periodic sync with learner
            if actor_server is not None and steps % config.buffer_period == 0:
                agent = actor_server.sync_agent()

        episode += 1
        print(f"Episode {episode}: reward={episode_reward:.3f}, steps={episode_steps}, "
              f"total_steps={steps}")


def run_eval_mode(args):
    """评估模式: 加载 checkpoint 并评估成功率。"""
    from orbax.checkpoint import PyTreeCheckpointer

    config = TrainConfig()
    env = config.get_environment(
        fake_env=args.fake_env,
        save_video=True,
        classifier=args.classifier,
    )

    # Load checkpoint
    ckpt_path = _resolve_eval_checkpoint_path(args)
    if args.eval_checkpoint_path is None and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", str(args.eval_checkpoint_step))

    print(f"Loading checkpoint: {ckpt_path}")
    checkpointer = PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_path)

    n_trajs = args.eval_n_trajs or 10
    successes = 0

    for ep in range(n_trajs):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = restored["actor"].sample(
                observations=obs,
                seed=jax.random.PRNGKey(ep),
                argmax=True,
            )
            action = np.array(action)[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

        success = _info_success(info)
        if success:
            successes += 1
        print(f"  Episode {ep + 1}/{n_trajs}: reward={ep_reward:.3f} "
              f"{'SUCCESS' if success else 'FAIL'}")

    print(f"\nEval results: {successes}/{n_trajs} "
          f"({successes / n_trajs:.1%} success rate)")


def main():
    parser = argparse.ArgumentParser(
        description="HIL-SERL Actor — plug_insertion experiment"
    )
    parser.add_argument("--eval_mode", action="store_true", help="Run evaluation mode")
    parser.add_argument("--eval_checkpoint_path", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--eval_checkpoint_step", type=int, default=None, help="Checkpoint step for eval")
    parser.add_argument("--eval_n_trajs", type=int, default=None, help="Number of eval trajectories")
    parser.add_argument("--learner_ip", type=str, default=None, help="Learner IP address")
    parser.add_argument("--learner_port", type=str, default=None, help="Learner port")
    parser.add_argument("--fake_env", action="store_true", help="Use fake environment")
    parser.add_argument("--save_video", action="store_true", help="Save video during training")
    parser.add_argument("--classifier", action="store_true", default=True, help="Use reward classifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.eval_mode:
        run_eval_mode(args)
    else:
        run_training_actor(args)


if __name__ == "__main__":
    main()
