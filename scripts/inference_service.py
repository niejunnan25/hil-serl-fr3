#!/usr/bin/env python3
"""inference_service.py

SERL 自主推理服务 — 加载训练好的 SAC policy checkpoint，
在真实/仿真环境中运行自主推理循环 (无 GELLO 干预)，
记录每集数据并输出 JSON 成功率报告。

数据流:
  SAC training → checkpoints/actor_XXXXX
  → inference_service.py → stdout JSON report

用法:
  python scripts/inference_service.py \\
      --checkpoint_path checkpoints/actor_20000 \\
      --num_episodes 10

  python scripts/inference_service.py \\
      --checkpoint_path checkpoints/actor_20000 \\
      --num_episodes 20 --max_steps 200 --save_video

依赖: trained SAC policy checkpoint, running franka_server (或 sim).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.plug_insertion.state_layout import tcp_pose_z
from experiments.plug_insertion.success_gate import (
    DEFAULT_CLASSIFIER_THRESHOLD,
    DEFAULT_DEPTH_THRESHOLD,
    DEFAULT_STREAK_REQUIRED,
    SuccessStreak,
    classify_success,
)


# ---------------------------------------------------------------------------
# 成功判定阈值 (与 config.py / eval_rollout.py 一致)
# ---------------------------------------------------------------------------
CLASSIFIER_THRESHOLD = DEFAULT_CLASSIFIER_THRESHOLD
Z_HEIGHT_THRESHOLD = DEFAULT_DEPTH_THRESHOLD
SUCCESS_STREAK_REQUIRED = DEFAULT_STREAK_REQUIRED
DEFAULT_MAX_STEPS = 150      # matches EnvConfig.MAX_EPISODE_LENGTH
DEFAULT_NUM_EPISODES = 10
DEFAULT_SERVER_URL = "http://127.0.0.2:5000/"


# ---------------------------------------------------------------------------
# 安全预检
# ---------------------------------------------------------------------------
def run_safety_prechecks(server_url: str) -> dict:
    """启动前安全预检。

    Returns:
        dict with 'passed' (bool) and 'checks' (list of check results).
    """
    checks = []

    # Check 1: 服务器可达性
    try:
        import urllib.request
        req = urllib.request.Request(
            server_url.rstrip("/") + "/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status_code = resp.getcode()
        checks.append({
            "name": "server_reachable",
            "passed": status_code == 200,
            "detail": f"HTTP {status_code}",
        })
    except Exception as e:
        checks.append({
            "name": "server_reachable",
            "passed": False,
            "detail": f"Connection failed: {e}",
        })

    # Check 2: 环境变量/路径检查 — 支持 PyTorch (.pt) 和 JAX/Orbax (目录) 两种格式
    classifier_ckpt_dir = PROJECT_ROOT / "classifier_ckpt"
    classifier_pt = classifier_ckpt_dir / "reward_classifier.pt"
    classifier_jax = classifier_ckpt_dir / "default"  # Orbax checkpoint 子目录
    classifier_exists = classifier_pt.exists() or classifier_jax.exists() or (
        classifier_ckpt_dir.exists() and any(classifier_ckpt_dir.iterdir())
    )
    ckpt_detail = str(classifier_ckpt_dir)
    if classifier_pt.exists():
        ckpt_detail += " (PyTorch .pt found)"
    elif classifier_jax.exists():
        ckpt_detail += " (JAX/Orbax found)"
    elif classifier_ckpt_dir.exists():
        ckpt_detail += " (directory exists)"
    else:
        ckpt_detail += " (NOT FOUND)"
    checks.append({
        "name": "classifier_checkpoint_exists",
        "passed": classifier_exists,
        "detail": ckpt_detail,
    })

    # Check 3: JAX 设备检查
    try:
        devices = jax.devices()
        checks.append({
            "name": "jax_device_available",
            "passed": len(devices) > 0,
            "detail": f"Devices: {[str(d) for d in devices]}",
        })
    except Exception as e:
        checks.append({
            "name": "jax_device_available",
            "passed": False,
            "detail": str(e),
        })

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_policy(checkpoint_path: str) -> Any:
    """加载 SAC policy checkpoint。

    Returns:
        restored checkpoint dict (含 actor params).
    """
    from orbax.checkpoint import PyTreeCheckpointer

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")

    print(f"  Loading policy checkpoint: {checkpoint_path}")
    checkpointer = PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)
    print("  Policy loaded successfully")
    return restored


def load_classifier(checkpoint_path: str = "classifier_ckpt/"):
    """加载 reward classifier，自动检测 PyTorch (.pt) 或 JAX/Orbax 格式。

    Returns:
        classifier_fn: callable(obs) -> logit (pre-sigmoid).
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")

    print(f"  Loading classifier from: {checkpoint_path}")

    # 1) 尝试 JAX/Orbax (SERL 原生)
    try:
        from serl_launcher.networks.reward_classifier import load_classifier_func
        classifier_fn = load_classifier_func(
            key=jax.random.PRNGKey(0),
            image_keys=["side_classifier"],
            checkpoint_path=checkpoint_path,
        )
        print("  Classifier loaded successfully (JAX/Orbax)")
        return classifier_fn
    except Exception as jax_err:
        # 2) 回退: PyTorch .pt
        pt_path = os.path.join(checkpoint_path, "reward_classifier.pt")
        if os.path.isfile(pt_path):
            print(f"  JAX load failed ({jax_err}), trying PyTorch: {pt_path}")
            return _load_pytorch_classifier_fallback(pt_path)
        raise RuntimeError(
            f"Failed to load classifier from {checkpoint_path}. "
            f"JAX error: {jax_err}. No PyTorch .pt fallback found."
        ) from jax_err


def _load_pytorch_classifier_fallback(pt_path: str):
    """加载 PyTorch .pt classifier 作为 fallback。"""
    import torch
    from scripts.train_reward_classifier import RewardClassifier, IMAGE_SIZE
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    model = RewardClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    def classifier_fn(obs):
        """从观测中提取图像，运行 PyTorch 分类器，返回 logit。"""
        img_key = "side_classifier"
        if img_key in obs:
            img = np.array(obs[img_key])
        else:
            img = np.array(obs.get("pixels", np.zeros((3, 128, 128), dtype=np.uint8)))

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = img.transpose(1, 2, 0)

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img.astype(np.uint8))
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(tensor).item()

        return jnp.array(logit)

    print("  Classifier loaded successfully (PyTorch fallback)")
    return classifier_fn


def create_env(
    fake_env: bool = False,
    save_video: bool = False,
    server_url: Optional[str] = None,
) -> gym.Env:
    """创建 plug_insertion 环境。"""
    from experiments.plug_insertion.config import TrainConfig, EnvConfig

    cfg = TrainConfig()

    # 如果指定了 server_url，覆盖默认值
    if server_url is not None:
        EnvConfig.SERVER_URL = server_url

    env = cfg.get_environment(
        fake_env=fake_env,
        save_video=save_video,
        classifier=True,
    )
    return env


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# 单集推理
# ---------------------------------------------------------------------------
def run_episode(
    env: gym.Env,
    policy_fn,
    classifier_fn,
    episode_id: int,
    max_steps: int,
) -> dict:
    """运行单集推理。

    Returns:
        episode_data dict with observations, actions, rewards, success/failure.
    """
    obs, info = env.reset()
    steps_data = []
    total_reward = 0.0
    success = False
    failure_reason = None
    success_streak = SuccessStreak(
        required=SUCCESS_STREAK_REQUIRED,
        classifier_threshold=CLASSIFIER_THRESHOLD,
        depth_threshold=Z_HEIGHT_THRESHOLD,
    )

    for step in range(max_steps):
        # 策略选择动作
        action = policy_fn(obs)

        # 环境步进
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 获取 z-height from SERLObsWrapper's gymnasium Dict flatten order.
        z_height = tcp_pose_z(obs["state"]) if "state" in obs else 0.0

        # 分类器判定
        classifier_logit = float(classifier_fn(obs))
        classifier_prob = float(sigmoid(classifier_logit))

        # 记录步骤数据
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

        success = success_streak.update(classifier_prob=classifier_prob, depth=z_height)
        if success:
            break

        # 失败判定
        if terminated and not success:
            failure_reason = "terminated_early"
            break

        if truncated:
            failure_reason = "max_steps_reached"
            break

        obs = next_obs

    if not success and failure_reason is None:
        failure_reason = "max_steps_reached"

    return {
        "episode_id": episode_id,
        "steps": len(steps_data),
        "reward_sum": round(total_reward, 4),
        "success": success,
        "failure_reason": failure_reason if not success else None,
        "final_z_height": steps_data[-1]["z_height"] if steps_data else None,
        "final_classifier_prob": steps_data[-1]["classifier_prob"] if steps_data else None,
        "steps_data": steps_data,
    }


def compute_aggregate(episodes: list) -> dict:
    """计算聚合统计。"""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run autonomous inference with a trained SERL SAC policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认推理 (10 episodes)
  python scripts/inference_service.py \\
      --checkpoint_path checkpoints/actor_20000

  # 自定义参数
  python scripts/inference_service.py \\
      --checkpoint_path checkpoints/actor_20000 \\
      --num_episodes 20 --max_steps 200 --save_video

  # 使用仿真环境
  python scripts/inference_service.py \\
      --checkpoint_path checkpoints/actor_20000 \\
      --fake_env --num_episodes 5
        """,
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to the SAC policy checkpoint directory",
    )
    parser.add_argument(
        "--classifier_ckpt", type=str, default="classifier_ckpt/",
        help="Path to the reward classifier checkpoint (default: classifier_ckpt/)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=DEFAULT_NUM_EPISODES,
        help=f"Number of inference episodes (default: {DEFAULT_NUM_EPISODES})",
    )
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--server_url", type=str, default=None,
        help=f"Franka controller server URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--save_video", action="store_true",
        help="Record video during inference",
    )
    parser.add_argument(
        "--fake_env", action="store_true",
        help="Use fake environment (no robot, for testing script logic)",
    )
    parser.add_argument(
        "--skip_safety_check", action="store_true",
        help="Skip safety pre-checks (not recommended for real robot)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output JSON report path (default: artifacts/inference_<timestamp>.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # 路径处理
    project_root = PROJECT_ROOT
    checkpoint_path = args.checkpoint_path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(project_root / checkpoint_path)

    classifier_ckpt = args.classifier_ckpt
    if not os.path.isabs(classifier_ckpt):
        classifier_ckpt = str(project_root / classifier_ckpt)

    # 输出文件
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "artifacts"
        output_dir.mkdir(exist_ok=True)
        output_file = str(output_dir / f"inference_{timestamp}.json")
    else:
        output_file = args.output_file
        if not os.path.isabs(output_file):
            output_file = str(project_root / output_file)

    print("=" * 60)
    print(" SERL Autonomous Inference Service")
    print("=" * 60)
    print(f"  Policy checkpoint:  {checkpoint_path}")
    print(f"  Classifier:         {classifier_ckpt}")
    print(f"  Episodes:           {args.num_episodes}")
    print(f"  Max steps/episode:  {args.max_steps}")
    print(f"  Server URL:         {args.server_url or DEFAULT_SERVER_URL}")
    print(f"  Save video:         {args.save_video}")
    print(f"  Fake env:           {args.fake_env}")
    print(f"  Output:             {output_file}")
    print(f"  Seed:               {args.seed}")
    print("=" * 60)

    # Set seed
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ── Step 0: Safety pre-checks ──
    if not args.fake_env and not args.skip_safety_check:
        print("\n[0/4] Running safety pre-checks...")
        server_url = args.server_url or DEFAULT_SERVER_URL
        safety = run_safety_prechecks(server_url)

        for check in safety["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check['detail']}")

        if not safety["passed"]:
            print("\n  SAFETY CHECK FAILED. Aborting.")
            print("  Use --skip_safety_check to override (NOT recommended for real robot).")
            sys.exit(1)
        print("  All safety checks passed.")
    else:
        safety = {"passed": True, "checks": [], "skipped": True}

    # ── Step 1: Load policy ──
    print("\n[1/4] Loading policy...")
    try:
        restored = load_policy(checkpoint_path)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 构建策略函数 (SERL SAC agent pattern)
    def policy_fn(obs):
        """Select action using the loaded policy (deterministic)."""
        actions = restored["actor"].sample(
            observations=obs,
            seed=key,
            argmax=True,  # 确定性推理
        )
        return np.array(actions)[0]

    # ── Step 2: Load classifier ──
    print("\n[2/4] Loading reward classifier...")
    try:
        classifier_fn = load_classifier(classifier_ckpt)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Create environment ──
    print("\n[3/4] Creating environment...")
    env = create_env(
        fake_env=args.fake_env,
        save_video=args.save_video,
        server_url=args.server_url,
    )
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # ── Step 4: Run inference episodes ──
    print(f"\n[4/4] Running {args.num_episodes} inference episodes...")
    episodes = []
    start_time = time.time()

    for ep in range(args.num_episodes):
        ep_start = time.time()
        print(f"\n  Episode {ep + 1}/{args.num_episodes}:", end=" ")
        ep_data = run_episode(env, policy_fn, classifier_fn, ep, args.max_steps)
        ep_data["duration_s"] = round(time.time() - ep_start, 2)
        episodes.append(ep_data)

        status = "SUCCESS" if ep_data["success"] else f"FAIL ({ep_data['failure_reason']})"
        print(f"{status} | steps={ep_data['steps']} | reward={ep_data['reward_sum']:.3f} | "
              f"z={ep_data['final_z_height']:.4f} | cls={ep_data['final_classifier_prob']:.3f} | "
              f"{ep_data['duration_s']:.1f}s")

    total_duration = round(time.time() - start_time, 2)

    # ── Compute aggregate stats ──
    aggregate = compute_aggregate(episodes)
    aggregate["total_duration_s"] = total_duration
    aggregate["avg_episode_duration_s"] = round(total_duration / max(len(episodes), 1), 2)

    # ── Build report ──
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "checkpoint": os.path.abspath(checkpoint_path),
            "classifier_ckpt": os.path.abspath(classifier_ckpt),
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "server_url": args.server_url or DEFAULT_SERVER_URL,
            "fake_env": args.fake_env,
            "seed": args.seed,
            "success_criteria": {
                "classifier_threshold": CLASSIFIER_THRESHOLD,
                "z_height_threshold": Z_HEIGHT_THRESHOLD,
                "min_steps": 5,
            },
        },
        "safety_precheck": safety,
        "aggregate": aggregate,
        "episodes": [
            {k: v for k, v in ep.items() if k != "steps_data"}
            for ep in episodes
        ],
    }

    # ── Write report ──
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # ── Print summary ──
    print("\n" + "=" * 60)
    print(" Inference Summary")
    print("=" * 60)
    print(f"  Success rate:       {aggregate['success_rate']:.1%} "
          f"({aggregate['successes']}/{aggregate['num_episodes']})")
    print(f"  Avg steps:          {aggregate['avg_steps']}")
    print(f"  Avg reward:         {aggregate['avg_reward']}")
    print(f"  Reward range:       [{aggregate['min_reward']}, {aggregate['max_reward']}]")
    print(f"  Total duration:     {total_duration:.1f}s")
    if aggregate["failure_reasons"]:
        print(f"  Failure reasons:    {aggregate['failure_reasons']}")
    print(f"\n  Report saved to: {output_file}")
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
