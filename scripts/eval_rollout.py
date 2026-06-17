#!/usr/bin/env python3
"""
eval_rollout.py — Rollout Evaluation Script
=============================================
Run N autonomous episodes with a trained policy, record per-step data,
and output a JSON success/failure report.

Success criteria (plug insertion):
  - Reward classifier sigmoid > 0.7  AND  z-height < Z_THRESHOLD

Usage:
    python scripts/eval_rollout.py \
        --checkpoint checkpoints/actor_20000 \
        --num_episodes 20 \
        --output_file artifacts/eval_report.json \
        --save_video

Requires: trained policy checkpoint, running franka_server (or sim).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─── Success thresholds ──────────────────────────────────────────────────────
CLASSIFIER_THRESHOLD = 0.7   # sigmoid output threshold
Z_HEIGHT_THRESHOLD = 0.22    # plug must be below this z (m)
MAX_EPISODE_STEPS = 150      # matches EnvConfig.MAX_EPISODE_LENGTH


# ─── Helpers ──────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def load_policy(checkpoint_path: str):
    """
    Load a trained SERL policy from checkpoint.

    Returns:
        policy_fn: callable(obs) -> action
        params:    loaded checkpoint params
    """
    from serl_launcher.utils.launcher import make_sac_agent
    from experiments.plug_insertion.config import TrainConfig

    cfg = TrainConfig()

    print(f"  Loading checkpoint: {checkpoint_path}")
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint (SERL uses Orbax checkpoints)
    from orbax.checkpoint import PyTreeCheckpointer
    checkpointer = PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    # Build agent skeleton for shape inference
    # (The actual loading depends on SERL checkpoint format)
    print("  Checkpoint loaded successfully")
    return restored


def load_classifier(checkpoint_path: str = "classifier_ckpt/"):
    """
    Load the reward classifier function.

    Returns:
        classifier_fn: callable(obs) -> logit (pre-sigmoid)
    """
    from serl_launcher.networks.reward_classifier import load_classifier_func

    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"  Loading classifier from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    classifier_fn = load_classifier_func(
        key=jax.random.PRNGKey(0),
        image_keys=["side_classifier"],
        checkpoint_path=checkpoint_path,
    )
    print("  Classifier loaded successfully")
    return classifier_fn


def create_env(fake_env: bool = False, save_video: bool = False):
    """Create the plug insertion environment with all wrappers."""
    from experiments.plug_insertion.config import TrainConfig

    cfg = TrainConfig()
    env = cfg.get_environment(
        fake_env=fake_env,
        save_video=save_video,
        classifier=True,
    )
    return env


def run_episode(env, policy_fn, classifier_fn, episode_id: int):
    """
    Run a single episode.

    Returns:
        episode_data: dict with steps, per-step data, reward_sum, success, failure_reason
    """
    obs, info = env.reset()
    steps_data = []
    total_reward = 0.0
    success = False
    failure_reason = None

    for step in range(MAX_EPISODE_STEPS):
        # Get action from policy
        action = policy_fn(obs)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Get z-height from state vector
        # state = [tcp_pose(6), tcp_vel(6), tcp_force(3), tcp_torque(3), gripper_pose(1)]
        # tcp_pose = [x, y, z, roll, pitch, yaw]
        z_height = float(obs["state"][0, 2]) if "state" in obs else 0.0

        # Check classifier for success
        classifier_logit = float(classifier_fn(obs))
        classifier_prob = float(sigmoid(classifier_logit))

        # Record step data
        step_info = {
            "step": step,
            "action": action.tolist() if hasattr(action, "tolist") else list(action),
            "reward": float(reward),
            "z_height": z_height,
            "classifier_prob": classifier_prob,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        steps_data.append(step_info)
        total_reward += float(reward)

        # Check success: classifier says success AND z below threshold
        if classifier_prob > CLASSIFIER_THRESHOLD and z_height < Z_HEIGHT_THRESHOLD:
            success = True
            # Continue recording but mark success
            if step >= 5:  # require minimum steps to confirm
                break

        # Check failure conditions
        if terminated and not success:
            failure_reason = "terminated_early"
            break

        if truncated:
            failure_reason = "max_steps_reached"
            break

        obs = next_obs

    if not success and failure_reason is None:
        failure_reason = "max_steps_reached"

    episode_data = {
        "episode_id": episode_id,
        "steps": len(steps_data),
        "reward_sum": round(total_reward, 4),
        "success": success,
        "failure_reason": failure_reason if not success else None,
        "final_z_height": steps_data[-1]["z_height"] if steps_data else None,
        "final_classifier_prob": steps_data[-1]["classifier_prob"] if steps_data else None,
        "steps_data": steps_data,
    }

    return episode_data


def compute_aggregate(episodes: list) -> dict:
    """Compute aggregate statistics across episodes."""
    n = len(episodes)
    if n == 0:
        return {}

    successes = sum(1 for e in episodes if e["success"])
    return {
        "num_episodes": n,
        "success_rate": round(successes / n, 4),
        "successes": successes,
        "failures": n - successes,
        "avg_steps": round(sum(e["steps"] for e in episodes) / n, 2),
        "avg_reward": round(sum(e["reward_sum"] for e in episodes) / n, 4),
        "min_reward": round(min(e["reward_sum"] for e in episodes), 4),
        "max_reward": round(max(e["reward_sum"] for e in episodes), 4),
        "avg_steps_success": round(
            sum(e["steps"] for e in episodes if e["success"]) / max(successes, 1), 2
        ),
        "failure_reasons": {
            reason: sum(1 for e in episodes if e["failure_reason"] == reason)
            for reason in set(e["failure_reason"] for e in episodes if e["failure_reason"])
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy over N rollout episodes."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the policy checkpoint directory",
    )
    parser.add_argument(
        "--classifier_ckpt", type=str, default="classifier_ckpt/",
        help="Path to the reward classifier checkpoint (default: classifier_ckpt/)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output JSON file path (default: artifacts/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Record video during evaluation",
    )
    parser.add_argument(
        "--fake_env", action="store_true",
        help="Use fake environment (no robot, for testing script logic)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set default output file
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "artifacts"
        output_dir.mkdir(exist_ok=True)
        args.output_file = str(output_dir / f"eval_{timestamp}.json")

    print("=" * 60)
    print(" Rollout Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Classifier:   {args.classifier_ckpt}")
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Output:       {args.output_file}")
    print(f"  Save video:   {args.save_video}")
    print(f"  Fake env:     {args.fake_env}")
    print(f"  Seed:         {args.seed}")
    print("=" * 60)

    # Set seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ── Step 1: Load policy ──
    print("\n[1/4] Loading policy...")
    try:
        restored = load_policy(args.checkpoint)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # Build policy function (SERL SAC agent pattern)
    # The restored checkpoint contains actor params; we wrap into a callable
    def policy_fn(obs):
        """Select action using the loaded policy."""
        # SERL actor expects obs dict, returns action
        actions = restored["actor"].sample(
            observations=obs,
            seed=key,
            argmax=True,  # deterministic for eval
        )
        return np.array(actions)[0]

    # ── Step 2: Load classifier ──
    print("\n[2/4] Loading reward classifier...")
    try:
        classifier_fn = load_classifier(args.classifier_ckpt)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # ── Step 3: Create environment ──
    print("\n[3/4] Creating environment...")
    env = create_env(fake_env=args.fake_env, save_video=args.save_video)
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # ── Step 4: Run evaluation episodes ──
    print(f"\n[4/4] Running {args.num_episodes} evaluation episodes...")
    episodes = []
    for ep in range(args.num_episodes):
        print(f"\n  Episode {ep + 1}/{args.num_episodes}:", end=" ")
        ep_data = run_episode(env, policy_fn, classifier_fn, ep)
        episodes.append(ep_data)
        status = "SUCCESS" if ep_data["success"] else f"FAIL ({ep_data['failure_reason']})"
        print(f"{status} | steps={ep_data['steps']} | reward={ep_data['reward_sum']:.3f} | "
              f"z={ep_data['final_z_height']:.4f} | cls={ep_data['final_classifier_prob']:.3f}")

    # ── Compute aggregate stats ──
    aggregate = compute_aggregate(episodes)

    # ── Build report ──
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": os.path.abspath(args.checkpoint),
            "classifier_ckpt": os.path.abspath(args.classifier_ckpt),
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "success_criteria": {
                "classifier_threshold": CLASSIFIER_THRESHOLD,
                "z_height_threshold": Z_HEIGHT_THRESHOLD,
                "min_steps": 5,
            },
        },
        "aggregate": aggregate,
        "episodes": [
            {k: v for k, v in ep.items() if k != "steps_data"}
            for ep in episodes
        ],
    }

    # ── Write report ──
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print(" Evaluation Summary")
    print("=" * 60)
    print(f"  Success rate:    {aggregate['success_rate']:.1%} ({aggregate['successes']}/{aggregate['num_episodes']})")
    print(f"  Avg steps:       {aggregate['avg_steps']}")
    print(f"  Avg reward:      {aggregate['avg_reward']}")
    print(f"  Reward range:    [{aggregate['min_reward']}, {aggregate['max_reward']}]")
    if aggregate["failure_reasons"]:
        print(f"  Failure reasons: {aggregate['failure_reasons']}")
    print(f"\n  Report saved to: {args.output_file}")
    print("=" * 60)

    # Exit code: 0 if success rate >= 50%, 1 otherwise
    if aggregate["success_rate"] >= 0.5:
        print("\n  RESULT: PASS (success rate >= 50%)")
        sys.exit(0)
    else:
        print("\n  RESULT: FAIL (success rate < 50%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
