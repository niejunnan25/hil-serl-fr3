#!/usr/bin/env python3
"""
e2e_pipeline_test.py — End-to-End Pipeline Verification
=========================================================
Offline verification of the full HIL-SERL pipeline.
No robot connection required — uses synthetic/mock data.

Tests:
  1. Demo PKL format matches SERL expectations
  2. Classifier checkpoint loads and runs inference
  3. Policy checkpoint loads and runs forward pass
  4. Environment creation (fake_env mode)
  5. All imports work (SERL, JAX, experiments)

Exit code: 0 if all tests pass, 1 if any fail.

Usage:
    python scripts/e2e_pipeline_test.py
    python scripts/e2e_pipeline_test.py --demo_pkl artifacts/demos/episode_000.pkl
    python scripts/e2e_pipeline_test.py --verbose
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Test result tracking ─────────────────────────────────────────────────────
RESULTS = []


def record(name: str, passed: bool, detail: str = ""):
    RESULTS.append({"name": name, "passed": passed, "detail": detail})
    status = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Demo PKL format verification
# ═══════════════════════════════════════════════════════════════════════════════
def test_demo_pkl_format(demo_pkl_path: str = None):
    """
    Verify that demo data PKL files match SERL expected format.

    Expected SERL transition format:
      - dict with keys: 'observations', 'actions', 'next_observations', 'rewards', 'dones'
      - observations: dict with 'state' (np.ndarray) and optional image keys
      - actions: np.ndarray of shape (action_dim,)
      - Compatible with MemoryEfficientReplayBufferDataStore
    """
    print("\n── Test 1: Demo PKL Format ──")

    import pickle as pkl
    import glob

    # Find demo files
    if demo_pkl_path and os.path.exists(demo_pkl_path):
        pkl_files = [demo_pkl_path]
    else:
        search_paths = [
            str(PROJECT_ROOT / "artifacts" / "**" / "*.pkl"),
            str(PROJECT_ROOT / "data" / "**" / "*.pkl"),
            str(PROJECT_ROOT / "*.pkl"),
        ]
        pkl_files = []
        for pattern in search_paths:
            pkl_files.extend(glob.glob(pattern, recursive=True))
        pkl_files = [f for f in pkl_files if "transitions" in f or "demo" in f]

    if not pkl_files:
        record("demo_pkl:file_found", False, "No demo PKL files found (provide --demo_pkl)")
        return

    record("demo_pkl:file_found", True, f"Found {len(pkl_files)} file(s)")

    # Load and validate first file
    try:
        with open(pkl_files[0], "rb") as f:
            data = pkl.load(f)
    except Exception as e:
        record("demo_pkl:loadable", False, f"Failed to load: {e}")
        return

    record("demo_pkl:loadable", True, f"Loaded {pkl_files[0]}")

    # Normalize to list of transitions
    transitions = []
    if isinstance(data, list):
        transitions = [t for t in data if isinstance(t, dict)]
    elif isinstance(data, dict):
        transitions = [data]

    if not transitions:
        record("demo_pkl:has_transitions", False, "No dict transitions found")
        return

    record("demo_pkl:has_transitions", True, f"{len(transitions)} transitions")

    # Validate transition structure
    t0 = transitions[0]
    required_keys = ["observations", "actions"]
    missing = [k for k in required_keys if k not in t0]
    if missing:
        record("demo_pkl:required_keys", False, f"Missing: {missing}")
        return

    record("demo_pkl:required_keys", True, f"Keys: {list(t0.keys())}")

    # Validate observations format
    obs = t0["observations"]
    if not isinstance(obs, dict):
        record("demo_pkl:obs_format", False, f"observations is {type(obs).__name__}, expected dict")
        return

    if "state" not in obs:
        record("demo_pkl:has_state", False, "observations missing 'state' key")
        return

    state = obs["state"]
    record("demo_pkl:has_state", True, f"state shape={np.array(state).shape}")

    # Validate actions
    actions = np.array(t0["actions"])
    record("demo_pkl:action_shape", len(actions.shape) >= 1, f"action shape={actions.shape}, dtype={actions.dtype}")

    # Check for non-zero actions
    nonzero = sum(1 for t in transitions if np.linalg.norm(np.array(t["actions"])) > 0)
    record("demo_pkl:nonzero_actions", nonzero > 0, f"{nonzero}/{len(transitions)} non-zero")

    # Check next_observations if present
    has_next = "next_observations" in t0
    record("demo_pkl:has_next_obs", has_next,
           "next_observations present" if has_next else "optional, not found")

    # Check rewards if present
    has_rewards = "rewards" in t0
    record("demo_pkl:has_rewards", has_rewards,
           f"rewards present" if has_rewards else "optional, not found")

    print(f"  Summary: {len(transitions)} transitions, action_dim={actions.shape[0]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Classifier checkpoint inference
# ═══════════════════════════════════════════════════════════════════════════════
def test_classifier_inference(classifier_ckpt: str = "classifier_ckpt/"):
    """
    Load the reward classifier checkpoint and run inference on sample data.

    Verifies:
      - Checkpoint exists and is loadable
      - Network accepts expected input shape
      - Output is a scalar logit
    """
    print("\n── Test 2: Classifier Inference ──")

    ckpt_path = os.path.abspath(classifier_ckpt)
    if not os.path.isdir(ckpt_path):
        # Try relative to project root
        ckpt_path = str(PROJECT_ROOT / classifier_ckpt)

    if not os.path.isdir(ckpt_path):
        record("classifier:ckpt_exists", False, f"Not found: {ckpt_path}")
        return

    record("classifier:ckpt_exists", True, ckpt_path)

    try:
        import jax
        import jax.numpy as jnp
        from serl_launcher.networks.reward_classifier import load_classifier_func
    except ImportError as e:
        record("classifier:imports", False, str(e))
        return

    record("classifier:imports", True, "serl_launcher.networks.reward_classifier")

    # Create synthetic observation matching expected format
    # side_classifier camera: 150x200 crop → typically 128x128 after resize
    sample_obs = {
        "side_classifier": np.random.randint(0, 255, (1, 128, 128, 3), dtype=np.uint8),
        "state": np.random.randn(1, 25).astype(np.float32),
    }

    try:
        classifier_fn = load_classifier_func(
            key=jax.random.PRNGKey(0),
            sample=sample_obs,
            image_keys=["side_classifier"],
            checkpoint_path=ckpt_path,
        )
        record("classifier:load", True, "Classifier function created")
    except Exception as e:
        record("classifier:load", False, str(e))
        return

    # Run inference
    try:
        logit = classifier_fn(sample_obs)
        logit_val = float(logit) if np.isscalar(logit) else float(np.array(logit).flat[0])
        sigmoid_val = 1.0 / (1.0 + np.exp(-logit_val))
        record("classifier:inference", True,
               f"logit={logit_val:.4f}, sigmoid={sigmoid_val:.4f}")
    except Exception as e:
        record("classifier:inference", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Policy checkpoint forward pass
# ═══════════════════════════════════════════════════════════════════════════════
def test_policy_checkpoint(policy_ckpt: str = None):
    """
    Load a policy checkpoint and run one forward pass.

    If no checkpoint provided, build a fresh SAC agent and verify
    that the agent skeleton works.
    """
    print("\n── Test 3: Policy Checkpoint ──")

    try:
        import jax
        import jax.numpy as jnp
        from serl_launcher.agents.continuous.sac import SACAgent
        from serl_launcher.utils.launcher import make_sac_agent
    except ImportError as e:
        record("policy:imports", False, str(e))
        return

    record("policy:imports", True, "SAC agent modules")

    # Build agent skeleton
    from gymnasium import spaces
    obs_space = spaces.Dict({"state": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float32)})
    act_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

    try:
        agent = make_sac_agent(
            seed=42,
            sample_obs=obs_space.sample(),
            sample_action=act_space.sample(),
            image_keys=[],
            encoder_type="mlp",
        )
        record("policy:build_agent", True, "SAC agent built")
    except Exception as e:
        record("policy:build_agent", False, str(e))
        return

    # Count params
    param_count = sum(x.size for x in jax.tree.leaves(agent.state.params))
    record("policy:param_count", True, f"{param_count:,} parameters")

    # Forward pass (deterministic)
    sample_obs = {"state": np.random.randn(1, 25).astype(np.float32)}
    try:
        actions = agent.sample_actions(
            observations=sample_obs,
            seed=jax.random.PRNGKey(0),
            argmax=True,
        )
        actions_np = np.array(actions)
        record("policy:forward_pass", True,
               f"action shape={actions_np.shape}, values={actions_np[0][:3]}...")
    except Exception as e:
        record("policy:forward_pass", False, str(e))
        return

    # If a checkpoint path is given, try loading it
    if policy_ckpt:
        ckpt_path = os.path.abspath(policy_ckpt)
        if not os.path.exists(ckpt_path):
            ckpt_path = str(PROJECT_ROOT / policy_ckpt)

        if os.path.exists(ckpt_path):
            try:
                from orbax.checkpoint import PyTreeCheckpointer
                checkpointer = PyTreeCheckpointer()
                restored = checkpointer.restore(ckpt_path)
                record("policy:load_checkpoint", True, f"Loaded from {ckpt_path}")
            except Exception as e:
                record("policy:load_checkpoint", False, str(e))
        else:
            record("policy:load_checkpoint", False, f"Not found: {ckpt_path}")
    else:
        record("policy:load_checkpoint", True, "Skipped (no --policy_ckpt, skeleton test passed)")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Environment creation (fake_env)
# ═══════════════════════════════════════════════════════════════════════════════
def test_env_creation():
    """
    Verify environment can be created in fake_env mode.

    This tests:
      - PlugInsertionEnv instantiation
      - Wrapper chain assembly
      - Observation/action space definition
    """
    print("\n── Test 4: Environment Creation (fake_env) ──")

    try:
        from experiments.plug_insertion.config import TrainConfig, EnvConfig
    except ImportError as e:
        record("env:imports", False, str(e))
        return

    record("env:imports", True, "TrainConfig, EnvConfig")

    # Verify config fields
    cfg = TrainConfig()
    record("env:image_keys", len(cfg.image_keys) > 0, f"{cfg.image_keys}")
    record("env:proprio_keys", len(cfg.proprio_keys) > 0, f"{cfg.proprio_keys}")

    ecfg = EnvConfig()
    record("env:max_episode_length", ecfg.MAX_EPISODE_LENGTH > 0,
           f"MAX_EPISODE_LENGTH={ecfg.MAX_EPISODE_LENGTH}")

    # Try creating fake environment
    try:
        env = cfg.get_environment(fake_env=True, save_video=False, classifier=False)
        record("env:create_fake", True, f"type={type(env).__name__}")
    except Exception as e:
        record("env:create_fake", False, str(e))
        return

    # Check spaces
    try:
        obs_space = env.observation_space
        act_space = env.action_space
        record("env:obs_space", obs_space is not None, f"{obs_space}")
        record("env:act_space", act_space is not None,
               f"shape={act_space.shape}, dtype={act_space.dtype}")
    except Exception as e:
        record("env:spaces", False, str(e))
        return

    # Try reset
    try:
        obs, info = env.reset()
        record("env:reset", True, f"obs keys={list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    except Exception as e:
        record("env:reset", False, str(e))

    # Try step
    try:
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        record("env:step", True, f"reward={reward:.4f}, terminated={terminated}")
    except Exception as e:
        record("env:step", False, str(e))

    env.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: Import verification
# ═══════════════════════════════════════════════════════════════════════════════
def test_imports():
    """
    Verify all critical imports work.

    Checks:
      - JAX with GPU support
      - SERL launcher modules
      - Experiment modules
      - Gymnasium / Gym
      - Core ML libraries
    """
    print("\n── Test 5: Import Verification ──")

    imports_to_check = [
        ("jax", "import jax; print(f'JAX {jax.__version__}')"),
        ("jax.numpy", "import jax.numpy as jnp"),
        ("jax.devices", "import jax; devs=jax.local_devices(); print(f'{len(devs)} device(s)')"),
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("gymnasium", "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')"),
        ("flax", "import flax; print(f'Flax {flax.__version__}')"),
        ("optax", "import optax; print(f'Optax {optax.__version__}')"),
        ("orbax", "import orbax.checkpoint"),
        ("serl_launcher.agents", "from serl_launcher.agents.continuous.sac import SACAgent"),
        ("serl_launcher.data", "from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore"),
        ("serl_launcher.networks", "from serl_launcher.networks.reward_classifier import load_classifier_func"),
        ("serl_launcher.wrappers", "from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper"),
        ("serl_launcher.utils", "from serl_launcher.utils.launcher import make_sac_agent"),
        ("franka_env", "from franka_env.envs.franka_env import FrankaEnv"),
        ("franka_env.wrappers", "from franka_env.envs.wrappers import Quat2EulerWrapper"),
        ("experiments.config", "from experiments.config import DefaultTrainingConfig"),
        ("experiments.plug_insertion", "from experiments.plug_insertion.config import TrainConfig"),
    ]

    all_ok = True
    for name, import_stmt in imports_to_check:
        try:
            exec(import_stmt)
            record(f"import:{name}", True)
        except Exception as e:
            record(f"import:{name}", False, str(e))
            all_ok = False

    # JAX GPU check
    try:
        import jax
        devices = jax.local_devices()
        gpu_devices = [d for d in devices if "gpu" in d.platform.lower() or "cuda" in d.platform.lower()]
        if gpu_devices:
            record("import:jax_gpu", True, f"{len(gpu_devices)} GPU(s): {[str(d) for d in gpu_devices]}")
        else:
            record("import:jax_gpu", False, f"No GPU found, devices: {[str(d) for d in devices]}")
            all_ok = False
    except Exception as e:
        record("import:jax_gpu", False, str(e))
        all_ok = False

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline verification (offline, no robot)."
    )
    parser.add_argument(
        "--demo_pkl", type=str, default=None,
        help="Path to a demo PKL file for format verification",
    )
    parser.add_argument(
        "--classifier_ckpt", type=str, default="classifier_ckpt/",
        help="Path to classifier checkpoint (default: classifier_ckpt/)",
    )
    parser.add_argument(
        "--policy_ckpt", type=str, default=None,
        help="Path to policy checkpoint (optional, builds fresh agent if absent)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print full tracebacks on failure",
    )
    parser.add_argument(
        "--skip_env", action="store_true",
        help="Skip environment creation test (useful if franka_env not installed)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" HIL-SERL End-to-End Pipeline Test")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Demo PKL:     {args.demo_pkl or 'auto-detect'}")
    print(f"  Classifier:   {args.classifier_ckpt}")
    print(f"  Policy:       {args.policy_ckpt or 'skeleton test'}")
    print("=" * 60)

    start = time.time()

    # Run all tests
    test_imports()

    try:
        test_demo_pkl_format(args.demo_pkl)
    except Exception as e:
        record("demo_pkl:exception", False, str(e))
        if args.verbose:
            traceback.print_exc()

    try:
        test_classifier_inference(args.classifier_ckpt)
    except Exception as e:
        record("classifier:exception", False, str(e))
        if args.verbose:
            traceback.print_exc()

    try:
        test_policy_checkpoint(args.policy_ckpt)
    except Exception as e:
        record("policy:exception", False, str(e))
        if args.verbose:
            traceback.print_exc()

    if not args.skip_env:
        try:
            test_env_creation()
        except Exception as e:
            record("env:exception", False, str(e))
            if args.verbose:
                traceback.print_exc()

    elapsed = time.time() - start

    # ── Summary ──
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = sum(1 for r in RESULTS if not r["passed"])
    total = len(RESULTS)

    print("\n" + "=" * 60)
    print(" Pipeline Test Summary")
    print("=" * 60)
    print(f"  Total:  {total}")
    print(f"  Passed: \033[32m{passed}\033[0m")
    print(f"  Failed: \033[31m{failed}\033[0m")
    print(f"  Time:   {elapsed:.1f}s")

    if failed > 0:
        print(f"\n  Failed tests:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"    - {r['name']}: {r['detail']}")

    print()

    if failed == 0:
        print("  \033[32mRESULT: ALL TESTS PASSED\033[0m")
        sys.exit(0)
    else:
        print("  \033[31mRESULT: SOME TESTS FAILED\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
