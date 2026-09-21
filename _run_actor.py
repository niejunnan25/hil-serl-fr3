#!/usr/bin/env python3

import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
import copy
import pickle as pkl
import sys
import threading
import queue
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

ACTOR_SAFETY_DQ_MAX = float(os.environ.get("ACTOR_SAFETY_DQ_MAX", "0.35"))
ACTOR_SAFETY_FORCE_MAX = float(os.environ.get("ACTOR_SAFETY_FORCE_MAX", "45.0"))
ACTOR_RELZ_ABS_MAX = float(os.environ.get("ACTOR_RELZ_ABS_MAX", "0.35"))
ACTOR_SAFETY_LOG_PERIOD = float(os.environ.get("ACTOR_SAFETY_LOG_PERIOD", "1.0"))
AGENTLACE_PORT = int(os.environ.get("AGENTLACE_PORT", "5588"))
AGENTLACE_BROADCAST_PORT = int(os.environ.get("AGENTLACE_BROADCAST_PORT", "5589"))
SERL_FLAT_RELZ_STATE_INDEX = 6
_LAST_SAFETY_LOG = 0.0


def _make_runtime_trainer_config():
    result = make_trainer_config(
        port_number=AGENTLACE_PORT,
        broadcast_port=AGENTLACE_BROADCAST_PORT,
    )
    if os.environ.get("HILSERL_REWARD_SPEC"):
        from hilserl.episode_commit import REQUEST
        result.request_types = [*result.request_types, REQUEST]
    return result


def _start_initial_network_retry(publish, *, period, grace_seconds):
    """Repeat startup policy publication while a late actor subscribes.

    Agentlace's PUB/SUB channel is deliberately non-sticky.  A short retry
    window bridges the normal gap between the learner opening its sockets and
    the actor finishing model/environment initialization.
    """
    stop = threading.Event()
    period = max(0.1, float(period))
    grace_seconds = max(0.0, float(grace_seconds))

    def run():
        deadline = time.monotonic() + grace_seconds
        while not stop.wait(period):
            if time.monotonic() >= deadline:
                return
            try:
                publish("startup-retry")
            except Exception as exc:
                # A transient socket failure should not take down training.
                print(f"[agentlace] startup policy retry failed: {type(exc).__name__}: {exc}", flush=True)

    thread = threading.Thread(target=run, name="learner-initial-network", daemon=True)
    thread.start()
    return stop, thread


def _as_array(value):
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def _extract_state(obs):
    state = obs.get("state", {}) if isinstance(obs, dict) else {}
    if isinstance(state, dict):
        return state
    arr = _as_array(state)
    return {"state_array": arr}


def _extract_relz(obs):
    if not isinstance(obs, dict):
        return None

    # SERLObsWrapper may flatten the observation state. For the current 19D single-arm
    # state, gymnasium.spaces.Dict sorts keys as gripper_pose, tcp_force, tcp_pose,
    # tcp_torque, tcp_vel, so reset-relative tcp_pose z is index 6. Unknown flat shapes
    # are not stable enough for a physical safety gate, so only trust explicit rel-z keys.
    for key in ("relz", "relative_z", "tcp_rel_z", "reset_rel_z"):
        if key in obs:
            arr = _as_array(obs[key]).reshape(-1)
            if arr.size:
                return float(arr[0])

    state = obs.get("state", {})
    state_arr = _as_array(state)
    if state_arr.size:
        arr = state_arr.reshape(-1)
        if arr.size == 19:
            return float(arr[SERL_FLAT_RELZ_STATE_INDEX])
        return None

    if not isinstance(state, dict):
        return None

    for key in ("relz", "relative_z", "tcp_rel_z", "reset_rel_z"):
        if key in state:
            arr = _as_array(state[key]).reshape(-1)
            if arr.size:
                return float(arr[0])

    # tcp_pose[2] is world z, not reset-relative z; do not infer rel-z from it.
    return None


def _check_runtime_safety(obs, step, raw_state=None):
    global _LAST_SAFETY_LOG
    state = raw_state if raw_state is not None else _extract_state(obs)
    dq = _as_array(state.get("dq", []))
    force = _as_array(state.get("tcp_force", state.get("force", [])))
    dq_max = float(np.max(np.abs(dq))) if dq.size else 0.0
    force_norm = float(np.linalg.norm(force)) if force.size else 0.0
    relz = _extract_relz(obs)

    now = time.time()
    if now - _LAST_SAFETY_LOG >= ACTOR_SAFETY_LOG_PERIOD:
        _LAST_SAFETY_LOG = now
        print(
            f"[actor_safety] step={step} dq_max={dq_max:.4f} "
            f"force_norm={force_norm:.2f} relz={relz if relz is not None else 'NA'}",
            flush=True,
        )

    if dq_max > ACTOR_SAFETY_DQ_MAX:
        raise RuntimeError(f"[actor_safety_fatal] dq_max={dq_max:.4f} > {ACTOR_SAFETY_DQ_MAX}")
    if force_norm > ACTOR_SAFETY_FORCE_MAX:
        raise RuntimeError(f"[actor_safety_fatal] force_norm={force_norm:.2f} > {ACTOR_SAFETY_FORCE_MAX}")
    if relz is not None and abs(relz) > ACTOR_RELZ_ABS_MAX:
        raise RuntimeError(f"[actor_safety_fatal] relz={relz:.4f} outside +/-{ACTOR_RELZ_ABS_MAX}")


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in ("0", "false", "no", "off", "")


def _success_credit_horizon():
    return max(0, int(os.environ.get("ACTOR_SUCCESS_CREDIT_HORIZON", "24")))


def _success_credit_min_action_norm():
    return float(os.environ.get("ACTOR_SUCCESS_CREDIT_MIN_ACTION_NORM", "0.05"))


def _transition_action_norm(transition):
    try:
        return float(np.linalg.norm(np.asarray(transition.get("actions", []), dtype=float)))
    except Exception:
        return 0.0


def _remember_success_credit_transition(tail, transition, is_human):
    """Keep recent non-zero human actions so manual success labels train insertion."""
    if not is_human or _success_credit_horizon() == 0:
        return
    if _transition_action_norm(transition) < _success_credit_min_action_norm():
        return
    tail.append(copy.deepcopy(transition))
    del tail[:-_success_credit_horizon()]


def _make_success_credit_transition(transition):
    credited = copy.deepcopy(transition)
    credited["rewards"] = 1.0
    credited["dones"] = False
    credited["masks"] = 1.0
    infos = dict(credited.get("infos", {}))
    infos["manual_success_credit"] = True
    infos["derived"] = True
    infos["parent_raw_transition_id"] = infos.get("raw_transition_id")
    infos["source_action"] = "human"
    credited["infos"] = infos
    return credited


def _insert_success_credit_tail(
    tail,
    intvn_data_store,
    demo_transitions,
    *,
    data_store=None,
    transitions=None,
):
    """Replay recent human insertion actions into demo data after operator success."""
    if not tail or _success_credit_horizon() == 0:
        return 0
    mirror_online = _env_bool("ACTOR_SUCCESS_CREDIT_ONLINE", False)
    added = 0
    for transition in tail[-_success_credit_horizon():]:
        credited = _make_success_credit_transition(transition)
        intvn_data_store.insert(credited)
        demo_transitions.append(copy.deepcopy(credited))
        if mirror_online and data_store is not None and transitions is not None:
            data_store.insert(credited)
            transitions.append(copy.deepcopy(credited))
        added += 1
    return added

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("resume_checkpoint_step", -1, "Explicit committed Learner checkpoint step to resume; -1 starts fresh.")
flags.DEFINE_boolean("learner_paused", False, "Start with manual Learner pause enabled; resume still requires active collection.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def _next_dump_step(directory, step):
    """Allocate beyond all saved chunks, including earlier Actor attempts."""
    from hilserl.replay_io import next_chunk_index
    return next_chunk_index(directory, step)


def _dump_pending_transitions(ckpt_path, subdir, step, transitions, *, final=False):
    """Best-effort Ctrl-C flush of one actor in-memory transition list."""
    if not ckpt_path or not transitions:
        return None
    from hilserl.replay_io import write_replay_chunk
    try:
        if subdir not in ("buffer", "demo_buffer"):
            raise ValueError(f"Unexpected replay buffer directory: {subdir}")
        out_dir = os.path.join(ckpt_path, subdir)
        os.makedirs(out_dir, exist_ok=True)
        n = _next_dump_step(out_dir, step)
        path = os.path.join(out_dir, f"transitions_{n}.pkl.lz4")
        write_replay_chunk(path, transitions, final=final)
        print(f"[ctrl-c-save] dumped {len(transitions)} transitions -> {path}", flush=True)
        return path
    except Exception as exc:
        print(f"[ctrl-c-save] buffer dump failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _save_training_checkpoint(ckpt_path, state, step, *, final=False):
    """Commit a new model before running the independent retention maintenance."""
    from hilserl.storage_space import array_storage_bytes, require_space
    from hilserl.checkpoint_retention import maintenance_after_save
    # Reserve the uncompressed array payload plus room for checkpoint metadata.
    # Existing committed checkpoints return in _save_final_checkpoint without a write.
    require_space(ckpt_path, pending_bytes=array_storage_bytes(state) + 64 * 2**20,
                  reserve_gib=0 if final else None)
    path = checkpoints.save_checkpoint(
        os.path.abspath(ckpt_path), state, step=step,
        keep=float("inf"), overwrite=False,
    )
    maintenance_after_save(os.path.abspath(ckpt_path), step)
    return path


def _save_final_checkpoint(agent, ckpt_path, step, *, raise_on_error=False):
    """Best-effort final checkpoint save on Ctrl-C / natural end."""
    if not ckpt_path or step is None:
        return None
    step = int(step)
    if step < 0:
        return None
    import json

    def committed(path):
        if os.path.isfile(path):
            return True  # Atomic legacy Flax checkpoint file.
        try:
            with open(os.path.join(path, "_CHECKPOINT_METADATA")) as stream:
                return bool(json.load(stream).get("commit_timestamp_nsecs"))
        except (OSError, ValueError):
            return False
    ckpt_dir = os.path.abspath(ckpt_path)
    final_path = os.path.join(ckpt_dir, f"checkpoint_{step}")
    if os.path.exists(final_path):
        if not committed(final_path):
            raise RuntimeError(f"Existing checkpoint has no committed save receipt: {final_path}")
        print(f"[ctrl-c-save] checkpoint_{step} already exists", flush=True)
        from hilserl.checkpoint_retention import maintenance_after_save
        maintenance_after_save(ckpt_dir, step)
        return final_path
    try:
        state = jax.block_until_ready(agent.state)
        _save_training_checkpoint(ckpt_dir, state, step, final=True)
        if not committed(final_path):
            raise RuntimeError(f"Checkpoint save returned without a commit receipt: {final_path}")
        print(f"[ctrl-c-save] saved final checkpoint -> {final_path}", flush=True)
        return final_path
    except Exception as exc:
        print(f"[ctrl-c-save] final checkpoint failed: {type(exc).__name__}: {exc}", flush=True)
        if raise_on_error:
            raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
        return None


def _validate_training_directory(ckpt_path, resume_step):
    """Validate explicit continuation before allocating the model or touching buffers."""
    import json
    from pathlib import Path
    directory = Path(ckpt_path).resolve()
    if type(resume_step) is not int or resume_step < -1:
        raise ValueError("resume_checkpoint_step must be -1 or a nonnegative integer")
    if resume_step == -1:
        if any(path.name != ".writer.lock" for path in directory.iterdir()):
            raise RuntimeError("Fresh training requires a new empty checkpoint directory")
        return None
    selected = directory / f"checkpoint_{resume_step}"
    if selected.is_file():
        committed = True  # Legacy Flax files are committed with atomic rename.
    else:
        try:
            committed = bool(json.loads((selected / "_CHECKPOINT_METADATA").read_text()).get("commit_timestamp_nsecs"))
        except (OSError, ValueError, AttributeError):
            committed = False
    if not committed:
        raise RuntimeError(f"Selected resume checkpoint has no committed save receipt: {selected}")
    steps = [int(path.name[11:]) for path in directory.glob("checkpoint_*") if path.name[11:].isdigit()]
    if max(steps, default=-1) != resume_step:
        raise RuntimeError("Resume in this directory requires its latest checkpoint; refusing to overwrite later training")
    return str(selected)


def _restore_training_checkpoint(agent, ckpt_path, resume_step):
    """Restore the whole TrainState, including optimizer slots, targets and RNG."""
    selected = _validate_training_directory(ckpt_path, resume_step)
    if selected is None:
        return agent
    try:
        state = checkpoints.restore_checkpoint(os.path.abspath(ckpt_path), agent.state, step=resume_step)
        old_leaves, old_tree = jax.tree_util.tree_flatten(agent.state)
        new_leaves, new_tree = jax.tree_util.tree_flatten(state)
        if old_tree != new_tree or len(old_leaves) != len(new_leaves):
            raise ValueError("TrainState structure differs from the current model/optimizer")
        for old, new in zip(old_leaves, new_leaves):
            if np.shape(old) != np.shape(new):
                raise ValueError(f"TrainState leaf shape changed: {np.shape(old)} -> {np.shape(new)}")
            old_dtype = getattr(old, "dtype", None)
            new_dtype = getattr(new, "dtype", None)
            if old_dtype is not None and (new_dtype is None or str(old_dtype) != str(new_dtype)):
                raise ValueError(f"TrainState leaf dtype changed: {old_dtype} -> {new_dtype}")
        for name in ("params", "target_params", "opt_states", "rng", "step"):
            if not hasattr(state, name):
                raise ValueError(f"Full training state is missing {name}")
        return agent.replace(state=state)
    except Exception as exc:
        raise RuntimeError(f"Cannot resume full training state from {selected}: {exc}") from exc


def _reload_online_buffers(ckpt_path, replay_buffer, demo_buffer, check_stop=lambda: None):
    """Reload by saved-file time; legacy restarted Actors reused low numeric IDs."""
    from pathlib import Path
    from hilserl.replay_io import load_replay_chunk, replay_chunks
    counts = {"buffer": 0, "demo_buffer": 0}
    for subdir, destination in (("buffer", replay_buffer), ("demo_buffer", demo_buffer)):
        directory = Path(ckpt_path) / subdir
        for path in replay_chunks(directory):
            check_stop()
            transitions = load_replay_chunk(path)
            if not isinstance(transitions, (list, tuple)):
                raise RuntimeError(f"Persisted replay chunk must contain a transition list: {path}")
            for transition in transitions:
                check_stop()
                if not isinstance(transition, dict):
                    raise RuntimeError(f"Invalid transition in persisted replay chunk: {path}")
                transition = dict(transition)
                if "grasp_penalty" not in transition:
                    transition["grasp_penalty"] = transition.get("infos", {}).get("grasp_penalty", 0.0)
                destination.insert(transition)
                counts[subdir] += 1
    return counts


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """Training adapter; all device modes share hilserl.episodes.run_episodes."""
    from hilserl.episodes import run_episodes
    from hilserl.storage import RecordingError
    base = env.unwrapped
    recorder, operator = base.recorder, base.operator
    mode = os.environ.get("HILSERL_MODE", "eval" if FLAGS.eval_checkpoint_step else "train")
    # Broadcast updates are best effort, while the startup snapshot carries a
    # server revision.  Keep the revision alongside the immutable agent state
    # so a delayed snapshot cannot roll an actor back to an older policy.
    current_policy = (agent, -1)
    policy_lock = threading.Lock()
    latest_server_revision = -1
    ready = threading.Event()
    client = None
    transitions, demo_transitions, credit_tail = [], [], []
    last_step = -1
    credit_episode = None
    reward_pipeline = None

    def close_client():
        nonlocal client
        if client is None:
            return
        current, client = client, None
        closer = getattr(current, "stop", None)
        if closer is not None:
            try:
                closer()
            except Exception as exc:
                # Preserve the original actor failure; cleanup is best effort.
                print(f"[agentlace] actor client cleanup failed: {type(exc).__name__}: {exc}", flush=True)

    if mode == "eval":
        path = os.path.join(os.path.abspath(FLAGS.checkpoint_path), f"checkpoint_{FLAGS.eval_checkpoint_step}")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        state = checkpoints.restore_checkpoint(os.path.abspath(FLAGS.checkpoint_path), agent.state,
                                               step=FLAGS.eval_checkpoint_step)
        current_policy = (agent.replace(state=state), FLAGS.eval_checkpoint_step)
    elif mode == "train":
        client = TrainerClient("actor_env", FLAGS.ip, _make_runtime_trainer_config(),
                               data_stores={"actor_env": data_store, "actor_env_intvn": intvn_data_store},
                               client_id=os.environ.get("HILSERL_ATTEMPT_ID"),
                               wait_for_server=False, timeout_ms=1000)

        def update_params(params, revision=None):
            nonlocal current_policy, latest_server_revision
            with policy_lock:
                previous, previous_revision = current_policy
                if revision is None:
                    next_revision = previous_revision + 1
                else:
                    try:
                        next_revision = int(revision)
                    except (TypeError, ValueError):
                        next_revision = previous_revision + 1
                    # A reconnect can deliver an old snapshot.  Compare only
                    # against revisions received from the server: broadcasts
                    # predate this metadata and use a local sequence number.
                    if ready.is_set() and next_revision <= latest_server_revision:
                        return
                    latest_server_revision = max(latest_server_revision, next_revision)
                    # Keep the actor's observable revision monotonic even when
                    # an unversioned broadcast arrived between two snapshots.
                    next_revision = max(next_revision, previous_revision)
                current_policy = (
                    previous.replace(state=previous.state.replace(params=params)),
                    next_revision,
                )
            ready.set()

        client.recv_network_callback(update_params)
        timeout = time.monotonic() + float(os.environ.get("ACTOR_LEARNER_PARAMS_TIMEOUT", "60"))
        snapshot_period = max(
            0.1,
            float(os.environ.get("ACTOR_LEARNER_PARAMS_SNAPSHOT_PERIOD", "1.0")),
        )
        snapshot_timeout_ms = max(
            1,
            int(os.environ.get("ACTOR_LEARNER_PARAMS_SNAPSHOT_TIMEOUT_MS", "10000")),
        )
        next_snapshot_poll = 0.0

        def pull_network_snapshot():
            getter = getattr(client, "get_network", None)
            if getter is None:
                return False
            try:
                try:
                    snapshot = getter(timeout_ms=snapshot_timeout_ms)
                except TypeError:
                    # Compatibility with local test doubles and older adapters
                    # that expose get_network() without a timeout keyword.
                    snapshot = getter()
            except Exception as exc:
                print(
                    f"[agentlace] policy snapshot request failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                return False
            if snapshot is None:
                return False
            if isinstance(snapshot, dict) and "params" in snapshot:
                update_params(snapshot["params"], snapshot.get("revision"))
            else:
                # Accept a plain payload from a compatible local adapter.
                update_params(snapshot)
            return True

        operator.publish("starting", prompt="等待 Learner 的策略参数")
        try:
            while not ready.wait(timeout=0.1):
                now = time.monotonic()
                if now >= next_snapshot_poll:
                    pull_network_snapshot()
                    next_snapshot_poll = now + snapshot_period
                if ready.is_set():
                    break
                listener = getattr(client, "broadcast_client", None)
                listener_error = getattr(listener, "last_error", None)
                if listener_error is not None:
                    raise RuntimeError(
                        "Learner policy listener failed: "
                        f"{type(listener_error).__name__}: {listener_error}"
                    )
                recorder.check()
                operator.raise_if_stop()
                if time.monotonic() >= timeout:
                    raise RuntimeError("No learner policy received before first reset")
        except BaseException:
            close_client()
            raise

    def sample(obs, step):
        nonlocal sampling_rng
        if mode == "collect":
            return np.zeros(env.action_space.shape, dtype=np.float32), {"source": "human"}
        with policy_lock:
            snapshot, revision = current_policy
        sampling_rng, key = jax.random.split(sampling_rng)
        action = snapshot.sample_actions(observations=jax.device_put(obs), seed=key, argmax=mode == "eval")
        return np.asarray(jax.device_get(action)), {"revision": revision, "source": "checkpoint" if mode == "eval" else "learner"}

    def flush_buffers(*, final=False):
        for subdir, pending in (("buffer", transitions), ("demo_buffer", demo_transitions)):
            if pending:
                path = _dump_pending_transitions(FLAGS.checkpoint_path, subdir, last_step, pending, final=final)
                if path is None:
                    raise RecordingError(f"Failed to save {subdir}; raw archive is retained")
                pending.clear()

    def emit(transition):
        nonlocal last_step, credit_episode
        if mode != "train":
            return
        if credit_episode != transition["infos"]["episode_id"]:
            credit_tail.clear()
            credit_episode = transition["infos"]["episode_id"]
        last_step = transition["infos"]["step"]
        human = transition["infos"]["source_action"] == "human"
        manual_success = bool(transition["dones"] and transition["infos"]["succeed"])
        data_store.insert(transition)
        transitions.append(copy.deepcopy(transition))
        if human or manual_success:
            intvn_data_store.insert(transition)
            demo_transitions.append(copy.deepcopy(transition))
        if getattr(config, "action_contract", "legacy-hybrid-7d-v1") != "fixed-xyz-v1":
            _remember_success_credit_transition(credit_tail, transition, is_human=human)
            if manual_success:
                _insert_success_credit_tail(credit_tail, intvn_data_store, demo_transitions,
                                            data_store=data_store, transitions=transitions)
        if transition["dones"]:
            credit_tail.clear()
        if last_step > 0 and config.buffer_period > 0 and last_step % config.buffer_period == 0:
            flush_buffers()
        client.update()

    def episode_finished(summary):
        if client:
            client.request("send-stats", {"environment": {"succeed": bool(summary["outcome"]),
                           "episode": {"r": float(summary["outcome"]), "l": summary["steps"],
                                       "intervention_steps": summary["human_steps"]}}})
            flush_buffers()

    try:
        if mode == "train":
            from pathlib import Path
            from hilserl.reward_provider import RewardSpec, RoboMeterClient
            from hilserl.episode_reward import EpisodeRewardPipeline, RewardTransport
            reward_spec = RewardSpec.from_env()
            if reward_spec is not None:
                if reward_spec.gamma != config.discount or config.action_contract != "fixed-xyz-v1":
                    raise ValueError("Reward gamma/action contract differs from Learner")

                def provider_factory():
                    return RoboMeterClient(reward_spec, os.environ["HILSERL_REWARD_URL"],
                        batch_size=int(os.environ["HILSERL_REWARD_BATCH_SIZE"]),
                        timeout=float(os.environ["HILSERL_REWARD_TIMEOUT"]))

                probe = provider_factory()
                try:
                    probe.health()
                finally:
                    probe.close()
                reward_pipeline = EpisodeRewardPipeline(reward_spec,
                    Path(FLAGS.checkpoint_path) / "reward_pending", recorder.directory / "reward/status.json",
                    os.environ["HILSERL_RUN_DIR"], operator.state["attempt_id"], provider_factory,
                    lambda: RewardTransport(FLAGS.ip, AGENTLACE_PORT),
                    timeout=float(os.environ["HILSERL_REWARD_TIMEOUT"]))
                operator.publish(reward_status_path=str(reward_pipeline.status_path))
        if mode == "eval":
            # Fix the reset sampler's sequence for comparable frozen evaluations.
            np.random.seed(FLAGS.seed)
        results = run_episodes(env, sample, recorder, operator, mode=mode,
                                action_contract=getattr(config, "action_contract", "legacy-hybrid-7d-v1"),
                                image_profile=getattr(config, "image_profile", "full-frame128-v1"),
                                eval_seed=FLAGS.seed if mode == "eval" else None,
                                max_episode_steps=int(os.environ.get("HILSERL_MAX_EPISODE_STEPS", "190")),
                                max_total_steps=int(os.environ.get("HILSERL_ACTOR_STEPS", str(config.max_steps))),
                                max_episodes=FLAGS.eval_n_trajs if mode == "eval" else None,
                                emit=emit, on_episode=episode_finished,
                                before_step=lambda obs, step: _check_runtime_safety(obs, step, base.raw_state()["values"]),
                                on_exit=(lambda: flush_buffers(final=True)) if mode == "train" else lambda: None,
                                episode_reward=reward_pipeline)
        if mode == "eval":
            import json
            from hilserl.checkpoint_scores import record_evaluation
            from hilserl.checkpoint_retention import retention_lock
            with open(os.environ["HILSERL_CONFIG_SNAPSHOT"]) as stream:
                snapshot = json.load(stream)
            from pathlib import Path
            metadata_file = Path(os.environ["HILSERL_RUN_DIR"]) / "run.json"
            if metadata_file.is_file():
                snapshot["evaluation_source_sha256"] = json.loads(metadata_file.read_text()).get("source_sha256", {})
            with retention_lock(FLAGS.checkpoint_path):
                report = record_evaluation(FLAGS.checkpoint_path, FLAGS.eval_checkpoint_step,
                    evaluation_run=os.environ["HILSERL_RUN_DIR"], summaries=results,
                    expected_episodes=FLAGS.eval_n_trajs, seed=FLAGS.seed, config_snapshot=snapshot)
            # run_episodes has already closed the raw recorder.
            print(f"[checkpoint-evaluation] {json.dumps(report, ensure_ascii=False)}", flush=True)
        return results
    finally:
        if reward_pipeline is not None:
            reward_pipeline.close()
        close_client()


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None, control=None,
            *, start_step=0, training_gate=None, episode_server=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    step = start_step
    from hilserl.learner_control import LearnerControl
    from hilserl.training_gate import TrainingGate
    control = control or LearnerControl(os.environ.get("HILSERL_RUN_DIR"))
    fixed_xyz = getattr(config, "action_contract", "legacy-hybrid-7d-v1") == "fixed-xyz-v1"
    initial_online_count = replay_buffer.transition_count if fixed_xyz else 0
    completed_groups = 0
    metrics = None
    last_update_info = None
    from hilserl.reward_provider import RewardSpec
    reward_spec = RewardSpec.from_env()
    if reward_spec is not None and episode_server is None:
        from pathlib import Path
        from hilserl.episode_commit import EpisodeCommitServer
        if config.discount != reward_spec.gamma:
            raise ValueError("Learner discount differs from reward contract")
        jax.block_until_ready(agent)
        episode_server = EpisodeCommitServer(Path(FLAGS.checkpoint_path) / "episode_replay",
            str(control.run_dir.resolve()), control.state["attempt_id"], reward_spec,
            replay_buffer, demo_buffer, max_steps=int(os.environ.get("HILSERL_MAX_EPISODE_STEPS", "190")))
        initial_online_count = replay_buffer.transition_count

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        if episode_server is not None:
            from hilserl.episode_commit import REQUEST
            if type == REQUEST:
                return episode_server.handle(payload)
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    print(
        f"[agentlace] server port={AGENTLACE_PORT} "
        f"broadcast_port={AGENTLACE_BROADCAST_PORT}",
        flush=True,
    )
    server = TrainerServer(
        _make_runtime_trainer_config(),
        request_callback=stats_callback,
    )
    if episode_server is not None:
        from hilserl.episode_commit import EpisodeOnlyStore
        server.register_data_store("actor_env", EpisodeOnlyStore())
        server.register_data_store("actor_env_intvn", EpisodeOnlyStore())
    else:
        server.register_data_store("actor_env", replay_buffer)
        server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)
    training_gate = training_gate or TrainingGate(control.run_dir, server.get_data_activity,
        episode_activity=episode_server.snapshot if episode_server is not None else None)

    last_step = start_step - 1
    runtime_error = None
    retry_stop = retry_thread = network_stop = network_thread = None
    try:
        if fixed_xyz:
            from hilserl.training_metrics import LearnerMetrics
            metrics = LearnerMetrics(control.run_dir, control.state["attempt_id"],
                                     cta_ratio=config.cta_ratio, start_step=start_step)
            control.publish(metrics_path=str(metrics.path), action_contract=config.action_contract,
                            restored_online_transitions=initial_online_count)
        # pyzmq sockets are not thread-safe.  The learner loop and the bounded
        # startup retry worker therefore enqueue publications, while one dedicated
        # thread owns the PUB socket. Retry messages use the current generation and
        # cannot overwrite a newer policy queued by training.
        network_queue = queue.Queue()
        network_stop = threading.Event()
        network_state_lock = threading.Lock()
        network_generation = 0
        latest_network = agent.state.params

        def enqueue_network(params, reason=None, *, cache=True):
            nonlocal network_generation, latest_network
            with network_state_lock:
                if cache:
                    network_generation += 1
                    latest_network = params
                generation = network_generation
            network_queue.put((generation, params, reason, cache))

        def network_publisher():
            while not network_stop.is_set():
                try:
                    generation, params, reason, cache = network_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with network_state_lock:
                    if generation < network_generation:
                        continue
                try:
                    if cache:
                        server.publish_network(params)
                    else:
                        server.broadcast_network(params)
                    if reason:
                        print_green(f"sent initial network to actor ({reason})")
                except Exception as exc:
                    # A transient PUB failure must not take down the learner. The
                    # Actor's snapshot endpoint remains the authoritative retry path.
                    print(f"[agentlace] network publish failed: {type(exc).__name__}: {exc}", flush=True)

        network_thread = threading.Thread(
            target=network_publisher, name="learner-network-publisher", daemon=True
        )
        network_thread.start()

        def publish_network(params, reason=None):
            enqueue_network(params, reason, cache=True)

        def publish_initial_network(reason: str):
            with network_state_lock:
                params = latest_network
            # The broadcast channel is not sticky; actors that subscribe after a
            # single publish can miss it. Retries broadcast the latest cached value
            # without creating artificial server revisions.
            enqueue_network(params, reason, cache=reason != "startup-retry")

        publish_initial_network("server-start")
        initial_network_broadcast_period = max(
            0.1, float(os.environ.get("LEARNER_INITIAL_NETWORK_BROADCAST_PERIOD", "2.0"))
        )
        initial_network_grace_seconds = max(
            float(os.environ.get("ACTOR_LEARNER_PARAMS_TIMEOUT", "60")),
            float(os.environ.get("LEARNER_INITIAL_NETWORK_GRACE_SECONDS", "180")),
        )
        retry_stop, retry_thread = _start_initial_network_retry(
            publish_initial_network,
            period=initial_network_broadcast_period,
            grace_seconds=initial_network_grace_seconds,
        )
        paused = False
        last_heartbeat = 0.0
        pause_checkpoint_step = None

        def wait_for_collection(*, reserve=False):
            nonlocal paused, last_heartbeat, pause_checkpoint_step
            while not control.stop_requested():
                decision = training_gate.check(manual_paused=control.activity_paused())
                online_count = replay_buffer.transition_count if fixed_xyz else len(replay_buffer)
                if online_count < config.training_starts and decision["allowed"]:
                    decision.update(allowed=False, pause_code="waiting_replay",
                                    pause_reason=(f"等待真实在线数据：{online_count}/{config.training_starts} 条。"
                                                  if fixed_xyz else "等待 replay 达到启动所需的数据量。"))
                if fixed_xyz:
                    decision["online_transition_count"] = online_count
                fields = {key: value for key, value in decision.items() if key not in {"allowed", "save_checkpoint"}}
                if decision["allowed"]:
                    if reserve and episode_server is not None and not episode_server.begin_update():
                        continue
                    if paused or control.state["phase"] != "training":
                        control.publish("training", step=last_step, next_step=last_step + 1, **fields)
                        last_heartbeat = time.monotonic()
                    paused = False
                    pause_checkpoint_step = None
                    return True
                now = time.monotonic()
                if not paused:
                    paused = True
                    control.publish("paused", step=last_step, next_step=last_step + 1, **fields)
                    publish_network(agent.state.params, "collection-paused")
                    last_heartbeat = now
                elif now - last_heartbeat >= 1.0 or fields["pause_reason"] != control.state.get("pause_reason"):
                    control.publish("paused", step=last_step, next_step=last_step + 1, **fields)
                    last_heartbeat = now
                # Normal episode/reset waits retain RAM and the normal checkpoint
                # schedule. Fault/exit/manual pauses save once, including a fault
                # discovered while already waiting. No retention policy changes.
                if decision.get("save_checkpoint") and last_step >= 0 and pause_checkpoint_step != last_step:
                    try:
                        control.publish("saving_checkpoint", step=last_step, **fields)
                        saved = _save_final_checkpoint(agent, FLAGS.checkpoint_path, last_step, raise_on_error=True)
                        if saved is None:
                            raise RuntimeError("Save returned without a committed checkpoint")
                        control.publish("paused", step=last_step, checkpoint_saved=True,
                                        checkpoint_path=saved, **fields)
                        pause_checkpoint_step = last_step
                    except Exception as exc:
                        message = f"Pause checkpoint save failed: {exc}"
                        control.publish("fault", checkpoint_error=message, error=message)
                        raise RuntimeError(message) from exc
                time.sleep(0.1)
            return False

        # No iterator prefetch or optimizer update starts before the first batch
        # from the identity-matched collecting Actor. Fixed XYZ seeds only demos.
        if not wait_for_collection():
            return
        publish_initial_network("training-start")

        # 50/50 sampling from RLPD, half from demo and half from online experience
        def episode_batches(store):
            # No background prefetch across a reward gap. Transfers and sampling
            # belong to the reserved update group and its GPU completion fence.
            while True:
                yield jax.device_put(store.sample(batch_size=config.batch_size // 2,
                    pack_obs_and_next_obs=True), device=sharding.replicate())

        replay_iterator = episode_batches(replay_buffer) if episode_server is not None else replay_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )
        demo_iterator = episode_batches(demo_buffer) if episode_server is not None else demo_buffer.get_iterator(
            sample_args={
                "batch_size": config.batch_size // 2,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

        # wait till the replay buffer is filled with enough data
        timer = Timer()
    
        if isinstance(agent, SACAgent):
            train_critic_networks_to_update = frozenset({"critic"})
            train_networks_to_update = frozenset({"critic", "actor", "temperature"})
        else:
            train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
            train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

        control.publish("training", step=last_step, next_step=last_step + 1)
        last_heartbeat = time.monotonic()
        for step in tqdm.tqdm(
            range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
        ):
            if not wait_for_collection(reserve=True):
                break
            # run n-1 critic updates and 1 critic + actor update.
            # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
            updating_agent = agent
            for critic_step in range(config.cta_ratio - 1):
                with timer.context("sample_replay_buffer"):
                    batch = next(replay_iterator)
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

                with timer.context("train_critics"):
                    updating_agent, critics_info = updating_agent.update(
                        batch,
                        networks_to_update=train_critic_networks_to_update,
                    )

            with timer.context("train"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
                updating_agent, update_info = updating_agent.update(
                    batch,
                    networks_to_update=train_networks_to_update,
                )
            # Publish/save only complete update groups, even if a later critic
            # call raises while an update group is in progress.
            if fixed_xyz:
                from hilserl.model_health import require_finite_update
                with timer.context("model_health"):
                    require_finite_update(updating_agent.state, update_info)
            completed_groups += 1
            last_update_info = update_info
            if metrics:
                record = metrics.write(step=step, update_groups=completed_groups,
                    unique_online_count=replay_buffer.transition_count - initial_online_count,
                    update_info=update_info, force=completed_groups == 1,
                    extra={"timer": timer.get_average_times(),
                           "online_transitions": replay_buffer.transition_count,
                           "demo_transitions": demo_buffer.transition_count})
                if record and record["invalid_fields"]:
                    raise FloatingPointError("训练指标出现非有限值或无效标量；本组参数未提交："
                                             + ", ".join(record["invalid_fields"]))
            agent = updating_agent
            last_step = step
            if time.monotonic() - last_heartbeat >= 1.0:
                control.publish("training", step=last_step, next_step=last_step + 1)
                last_heartbeat = time.monotonic()
            # publish the updated network
            if step > 0 and step % (config.steps_per_update) == 0:
                agent = jax.block_until_ready(agent)
                publish_network(agent.state.params)

            if step % config.log_period == 0 and wandb_logger:
                wandb_logger.log(update_info, step=step)
                wandb_logger.log({"timer": timer.get_average_times()}, step=step)

            if (
                step > 0
                and config.checkpoint_period
                and step % config.checkpoint_period == 0
            ):
                _save_training_checkpoint(FLAGS.checkpoint_path, agent.state, step)

            if episode_server is not None:
                episode_server.complete_update(lambda: jax.block_until_ready((agent, update_info)), step)


    except KeyboardInterrupt:
        print("\n[ctrl-c-save] learner Ctrl-C received; saving final checkpoint...", flush=True)
    except BaseException as exc:
        runtime_error = str(exc)
        control.publish("fault", error=runtime_error)
        raise
    finally:
        if episode_server is not None:
            episode_server.close(runtime_error)
        # Persist the final state before closing transport resources. A socket
        # cleanup failure must never prevent a model save.
        saved = control.state.get("checkpoint_path") if control.state.get("checkpoint_saved") else None
        save_error = control.state.get("checkpoint_error")
        try:
            if last_step >= 0 and not save_error:
                control.publish("saving_checkpoint", step=last_step, checkpoint_path=FLAGS.checkpoint_path)
                print(f"[learner-stop] saving final checkpoint at step {last_step}", flush=True)
                saved = _save_final_checkpoint(agent, FLAGS.checkpoint_path, last_step)
                if saved is None:
                    raise RuntimeError("Final checkpoint save failed; see Learner log")
        except Exception as exc:
            save_error = str(exc)
            control.publish("fault", checkpoint_error=save_error)
        finally:
            if metrics:
                try:
                    if runtime_error is None:
                        metrics.write(step=last_step, update_groups=completed_groups,
                            unique_online_count=replay_buffer.transition_count - initial_online_count,
                            update_info=last_update_info, force=True)
                except Exception as exc:
                    runtime_error = runtime_error or f"训练诊断保存失败：{exc}"
                    control.publish("fault", error=runtime_error)
                finally:
                    try:
                        metrics.close()
                    except Exception as exc:
                        runtime_error = runtime_error or f"训练诊断关闭失败：{exc}"
                        control.publish("fault", error=runtime_error)
            if not save_error:
                control.publish("fault" if runtime_error else "closing", checkpoint_saved=saved is not None, checkpoint_path=saved)
            if retry_stop is not None:
                retry_stop.set()
                retry_thread.join(timeout=2)
            if network_stop is not None:
                network_stop.set()
                network_thread.join(timeout=2)
            try:
                server.stop()
            except Exception as exc:
                print(f"[agentlace] server cleanup failed: {type(exc).__name__}: {exc}", flush=True)
                control.publish("fault", error=f"Transport cleanup failed: {exc}")
        if save_error:
            raise RuntimeError(save_error)
        if control.state["phase"] != "fault":
            control.publish("stopped", step=last_step)
##############################################################################


def main(_):
    from contextlib import ExitStack
    from pathlib import Path
    from hilserl.control import StopRequested
    from hilserl.processes import device_lease, discover, lease
    resources = {}
    reason = "completed"
    with ExitStack() as lifetime:
        root = Path(__file__).resolve().parent
        if FLAGS.actor:
            mode = os.environ.get("HILSERL_MODE", "eval" if FLAGS.eval_checkpoint_step else "train")
            if mode == "eval":
                from hilserl.checkpoint_retention import open_reader_guard
                inherited = os.environ.get("HILSERL_CHECKPOINT_READER_FD")
                if inherited is None:
                    fd = open_reader_guard(FLAGS.checkpoint_path, FLAGS.eval_checkpoint_step)
                else:
                    fd = int(inherited)
                    expected = Path(FLAGS.checkpoint_path) / ".checkpoint_readers" / f"checkpoint_{FLAGS.eval_checkpoint_step}.lock"
                    actual = os.fstat(fd)
                    named = expected.stat()
                    if fd <= 2 or (actual.st_dev, actual.st_ino) != (named.st_dev, named.st_ino):
                        raise RuntimeError("Invalid inherited checkpoint read guard")
                lifetime.callback(os.close, fd)
            lifetime.enter_context(device_lease(root))
            others = [p for p in discover(root) if p["pid"] != os.getpid() and p["role"] != "learner"]
            if others:
                raise RuntimeError("Another device process is already running; refusing to open cameras")
        elif FLAGS.learner:
            from hilserl.learner_control import LearnerControl
            learner_control = LearnerControl(os.environ.get("HILSERL_RUN_DIR"),
                                             initial_paused=FLAGS.learner_paused)
            resources["learner_control"] = learner_control
            lifetime.enter_context(learner_control.signal_handlers())
            directory = Path(FLAGS.checkpoint_path)
            lifetime.enter_context(lease(directory / ".writer.lock"))
        try:
            if FLAGS.learner:
                _validate_training_directory(FLAGS.checkpoint_path, FLAGS.resume_checkpoint_step)
            if resources.get("learner_control") and resources["learner_control"].stop_requested():
                raise StopRequested("Learner stopped before initialization")
            return _main(_, resources)
        except (KeyboardInterrupt, StopRequested):
            reason = "operator_stop"
        except BaseException as exc:
            reason = str(exc)
            if resources.get("learner_control"):
                resources["learner_control"].publish("fault", error=reason)
            if resources.get("recorder"):
                resources["recorder"].fail(reason)
            if resources.get("operator"):
                resources["operator"].publish("fault", error=reason)
            raise
        finally:
            learner_control = resources.get("learner_control")
            if learner_control and learner_control.state["phase"] not in {"stopped", "fault"}:
                learner_control.publish("stopped", reason=reason)
            try:
                if resources.get("env"):
                    resources["env"].close()
            finally:
                recorder, operator = resources.get("recorder"), resources.get("operator")
                if recorder:
                    recorder.close(reason)
                if operator:
                    if recorder and recorder.error:
                        operator.publish("fault", error=recorder.error)
                    elif operator.state["phase"] not in ("fault", "stopped"):
                        operator.publish("stopped", reason=reason)


def _main(_, resources):
    from hilserl.reward_provider import RewardSpec
    reward_spec = RewardSpec.from_env()
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    learner_control = resources.get("learner_control")

    def check_learner_stop():
        if learner_control and learner_control.stop_requested():
            from hilserl.control import StopRequested
            raise StopRequested("Learner stopped during initialization")

    assert config.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    from hilserl.storage import SessionRecorder
    from hilserl.control import OperatorControl
    from pathlib import Path
    from datetime import datetime
    mode = os.environ.get("HILSERL_MODE", "eval" if FLAGS.eval_checkpoint_step else "train")
    recorder = operator = None
    if FLAGS.actor:
        root = Path(os.environ.get("HILSERL_RECORDING_DIR", str(Path(FLAGS.checkpoint_path) / "recordings" /
                      datetime.now().strftime("%Y%m%d_%H%M%S_%f"))))
        recorder = SessionRecorder(root, metadata={"mode": mode, "checkpoint_path": FLAGS.checkpoint_path,
                                   "action_contract": getattr(config, "action_contract", "legacy-hybrid-7d-v1"),
                                   "image_profile": getattr(config, "image_profile", "full-frame128-v1"),
                                   "config_file": os.environ.get("HILSERL_CONFIG_SNAPSHOT")},
                                   min_free_bytes=int(float(os.environ.get("HILSERL_MIN_FREE_GIB", "8")) * 2**30),
                                   fps=int(os.environ.get("HILSERL_VIDEO_FPS", "30")),
                                   segment_seconds=int(os.environ.get("HILSERL_VIDEO_SEGMENT_SECONDS", "60")))
        operator = OperatorControl(os.environ.get("HILSERL_CONTROL_DIR"))
        resources.update(recorder=recorder, operator=operator)
    env = config.get_environment(fake_env=FLAGS.learner, save_video=False, classifier=(mode != "collect"),
                                 recorder=recorder, operator=operator, mode=mode)
    env = RecordEpisodeStatistics(env)
    resources["env"] = env
    check_learner_stop()
    if operator:
        from hilserl.gripper import GripperControl

        def manual_gripper(operation):
            recorder.event("gripper_requested", operation=operation)
            try:
                result = GripperControl(env.unwrapped.url).command(operation)
            except Exception as exc:
                recorder.event("gripper_result", operation=operation, error=str(exc),
                               result=getattr(exc, "result", {}))
                raise
            recorder.event("gripper_result", result=result)
            return result

        operator.gripper_handler = manual_gripper
        operator.publish(gripper_available=True)
        operator.raise_if_stop()
    if mode == "collect" and FLAGS.actor:
        return actor(None, None, None, env, sampling_rng)

    rng, sampling_rng = jax.random.split(rng)
    
    if config.setup_mode == 'single-arm-fixed-gripper' or config.setup_mode == 'dual-arm-fixed-gripper':   
        from hilserl.image_profile import get_image_profile
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
            encoder_image_size=(tuple(get_image_profile(config.image_profile)["cameras"][config.image_keys[0]]["size"])
                                if config.image_profile != "full-frame128-v1" else None),
        )
        if config.action_contract == "fixed-xyz-v1":
            agent = agent.replace(config={**agent.config, "selective_optimizer_updates": True})
        include_grasp_penalty = False
    elif config.setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == 'dual-arm-learned-gripper':
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    # replicate agent across devices
    check_learner_stop()
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )
    if operator:
        operator.raise_if_stop()

    # A new Learner never resumes implicitly. Explicit continuation restores the
    # full state; evaluation's separately selected policy remains in actor().
    if FLAGS.learner:
        agent = _restore_training_checkpoint(agent, FLAGS.checkpoint_path, FLAGS.resume_checkpoint_step)
        check_learner_stop()
        agent = jax.device_put(agent, sharding.replicate())
        if learner_control:
            learner_control.publish("initializing", resume_checkpoint_step=FLAGS.resume_checkpoint_step,
                                    step=FLAGS.resume_checkpoint_step, next_step=FLAGS.resume_checkpoint_step + 1,
                                    checkpoint_saved=FLAGS.resume_checkpoint_step >= 0,
                                    checkpoint_path=(str(Path(FLAGS.checkpoint_path).resolve() / f"checkpoint_{FLAGS.resume_checkpoint_step}")
                                                     if FLAGS.resume_checkpoint_step >= 0 else None))

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        if config.action_contract == "fixed-xyz-v1":
            from hilserl.learning_replay import ContractReplayStore
            replay_buffer = ContractReplayStore(replay_buffer, image_profile=config.image_profile, reward_spec=reward_spec)
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        if config.action_contract == "fixed-xyz-v1":
            from hilserl.learning_replay import ContractReplayStore
            from hilserl.seed_dataset import iter_seed_transitions
            demo_buffer = ContractReplayStore(demo_buffer, image_profile=config.image_profile, reward_spec=reward_spec)
            seed_digest = os.environ.get("HILSERL_SEED_DATASET_SHA256")
            if not seed_digest:
                raise ValueError("fixed-xyz-v1 requires a pinned seed dataset SHA-256")
            seed_transitions = iter_seed_transitions(os.environ["HILSERL_SEED_DATASET"],
                    expected_action_contract=config.action_contract, expected_manifest_sha256=seed_digest,
                    expected_image_profile=config.image_profile,
                    expected_action_max_z_step=(float(os.environ["HILSERL_ACTION_MAX_Z_STEP"])
                                               if "HILSERL_ACTION_MAX_Z_STEP" in os.environ else None),
                    expected_position_target_mode=os.environ.get("HILSERL_POSITION_TARGET_MODE", "measured-relative-v1"))
            if reward_spec is not None:
                from hilserl.reward_seed import iter_reward_seed
                seed_transitions = iter_reward_seed(seed_transitions, os.environ["HILSERL_REWARD_SEED_CACHE"],
                                                    seed_digest, reward_spec)
            for transition in seed_transitions:
                check_learner_stop()
                demo_buffer.insert(transition)
        else:
            # Historical checkpoints retain their original seeding semantics.
            assert FLAGS.demo_path is not None
            for path in FLAGS.demo_path:
                check_learner_stop()
                with open(path, "rb") as f:
                    for transition in pkl.load(f):
                        if "grasp_penalty" in transition.get("infos", {}):
                            transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                        if "grasp_penalty" not in transition:
                            transition["grasp_penalty"] = 0.0
                        demo_buffer.insert(transition)
            for path in FLAGS.demo_path:
                check_learner_stop()
                with open(path, "rb") as f:
                    for transition in pkl.load(f):
                        if "grasp_penalty" not in transition:
                            transition["grasp_penalty"] = 0.0
                        replay_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size ({config.action_contract}): {len(replay_buffer)}")
        if FLAGS.resume_checkpoint_step >= 0 and reward_spec is None:
            restored_buffers = _reload_online_buffers(FLAGS.checkpoint_path, replay_buffer, demo_buffer,
                                                      check_stop=check_learner_stop)
            print_green(f"restored persisted online chunks: {restored_buffers}")
            if learner_control:
                learner_control.publish(buffer_reload=restored_buffers, buffer_reload_order="saved_file_mtime_ns")

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
            control=learner_control,
            start_step=FLAGS.resume_checkpoint_step + 1,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
