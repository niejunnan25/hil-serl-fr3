"""Independent, comparable evaluation evidence for checkpoint retention.

The caller holds the checkpoint directory's retention lock. This module never
loads a model or touches replay data, and an unranked evaluation has no score.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time


EVAL_PROTOCOL = "policy-only-seeded-reset-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_STEP = re.compile(r"0|[1-9][0-9]*")
_NON_EVAL_FIELDS = frozenset({
    "root", "python", "data_dir", "demo_dir", "seed_dataset_sha256", "seed",
    "batch_size", "replay_buffer_capacity", "cta_ratio", "learner_steps",
    "actor_steps", "training_starts", "steps_per_update", "steps_per_network_update",
    "port", "broadcast_port", "console_port", "min_free_gib",
    "video_segment_seconds", "startup_timeout_seconds", "checkpoint_period",
    "checkpoint_keep", "checkpoint_keep_latest", "checkpoint_retention",
    "log_period", "buffer_period", "publish_period", "network_update_period",
})


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _directory(value):
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or path != path.resolve() or not path.is_dir():
        raise ValueError(f"Expected an existing directory without symlinks: {path}")
    return path


def _read_json(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Evidence must be a regular file without symlinks: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path, value):
    if path.is_symlink() or path.exists() and not path.is_file():
        raise ValueError(f"Evidence must be a regular file without symlinks: {path}")
    payload = _json_bytes(value) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _committed(path):
    if path.is_symlink():
        return False
    if path.is_file():
        # The legacy single-host Flax format commits by atomic rename.
        # Multiprocess siblings need a different reader and are not scored here.
        return path.stat().st_size > 0 and not path.with_name(path.name + "_gda").exists()
    try:
        receipt = _read_json(path / "_CHECKPOINT_METADATA")
        stamp = receipt.get("commit_timestamp_nsecs")
        return type(stamp) is int and stamp > 0
    except (OSError, ValueError, AttributeError):
        return False


def _conditions(config_snapshot, seed, episodes):
    if not isinstance(config_snapshot, dict) or not all(isinstance(key, str) for key in config_snapshot):
        raise ValueError("config_snapshot must be a JSON object with string keys")
    environment = {key: value for key, value in config_snapshot.items()
                   if key not in _NON_EVAL_FIELDS
                   and not key.startswith(("checkpoint_retention_", "checkpoint_keep_", "retention_"))}
    result = dict(eval_protocol=EVAL_PROTOCOL, seed=seed, episodes=episodes, config=environment)
    # Freeze nested values and reject non-JSON/nonfinite configuration data.
    return json.loads(_json_bytes(result))


def _assess(summaries, expected_episodes):
    if not isinstance(summaries, list):
        return "invalid_summaries", []
    normalized, identifiers = [], set()
    for summary in summaries:
        if not isinstance(summary, dict):
            return "invalid_summary", []
        episode_id = summary.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or any(ord(c) < 32 for c in episode_id):
            return "invalid_episode_id", []
        if episode_id in identifiers:
            return "duplicate_episode_id", []
        identifiers.add(episode_id)
        if summary.get("mode") != "eval":
            return "non_eval_episode", []
        if summary.get("complete", True) is not True:
            return "incomplete_episode", []
        if type(summary.get("outcome")) is not int or summary["outcome"] not in (0, 1):
            return "invalid_outcome", []
        if type(summary.get("steps")) is not int or summary["steps"] <= 0:
            return "invalid_steps", []
        if type(summary.get("human_steps")) is not int or summary["human_steps"] < 0:
            return "invalid_human_steps", []
        normalized.append({key: summary[key] for key in ("episode_id", "mode", "outcome", "steps", "human_steps")})
    if expected_episodes < 10:
        return "at_least_10_episodes_required", normalized
    if len(normalized) != expected_episodes:
        return "incomplete_evaluation", normalized
    if any(summary["human_steps"] for summary in normalized):
        return "intervention_present", normalized
    return None, normalized


def _ledger(directory):
    try:
        value = _read_json(directory / "retention_scores.json")
    except FileNotFoundError:
        return dict(schema_version=1, entries={})
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or not isinstance(value.get("entries"), dict)):
        raise ValueError("Invalid checkpoint score ledger")
    return value


def _valid_entry(directory, key, entry):
    """Validate both the ledger and its durable evaluation receipt."""
    try:
        if not isinstance(key, str) or not _STEP.fullmatch(key) or not isinstance(entry, dict):
            return False
        step = entry.get("step")
        if type(step) is not int or step < 0 or str(step) != key:
            return False
        if type(entry.get("schema_version")) is not int or entry["schema_version"] != 1 or entry.get("status") != "ranked":
            return False
        checkpoint = directory / f"checkpoint_{step}"
        if entry.get("checkpoint") != str(checkpoint) or not _committed(checkpoint):
            return False
        evaluation = entry.get("evaluation_run")
        if not isinstance(evaluation, str):
            return False
        run = _directory(evaluation)
        if str(run) != evaluation or run == directory or directory in run.parents:
            return False
        if type(entry.get("unix_ns")) is not int or entry["unix_ns"] <= 0:
            return False
        episodes, successes = entry.get("episodes"), entry.get("successes")
        if type(episodes) is not int or episodes < 10 or type(successes) is not int or not 0 <= successes <= episodes:
            return False
        if type(entry.get("expected_episodes")) is not int or entry["expected_episodes"] != episodes:
            return False
        score = entry.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or score != successes / episodes:
            return False
        conditions = entry.get("conditions")
        if (not isinstance(conditions, dict) or conditions.get("eval_protocol") != EVAL_PROTOCOL
                or type(conditions.get("episodes")) is not int or conditions["episodes"] != episodes
                or type(conditions.get("seed")) is not int or not 0 <= conditions["seed"] < 2**32
                or not isinstance(conditions.get("config"), dict)):
            return False
        digest = entry.get("conditions_sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or _digest(conditions) != digest:
            return False
        reason, normalized = _assess(entry.get("summaries"), episodes)
        if reason is not None or sum(summary["outcome"] for summary in normalized) != successes:
            return False
        return _read_json(run / "evaluation.json") == entry
    except (OSError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def record_evaluation(checkpoint_dir, step, *, evaluation_run, summaries,
                      expected_episodes, seed, config_snapshot):
    """Persist a complete policy-only evaluation, or explain why it is unranked.

Repeated calls for the same evaluation run are idempotent. Conflicting rewrites
are rejected, and retrying an old evaluation never replaces a newer result.
"""
    directory, run = _directory(checkpoint_dir), _directory(evaluation_run)
    if run == directory or directory in run.parents:
        raise ValueError("Evaluation evidence must be outside the checkpoint directory")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if type(expected_episodes) is not int or expected_episodes < 0:
        raise ValueError("expected_episodes must be a nonnegative integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    checkpoint = directory / f"checkpoint_{step}"
    if not _committed(checkpoint):
        raise ValueError(f"Evaluation requires an existing committed checkpoint: {checkpoint}")
    conditions = _conditions(config_snapshot, seed, expected_episodes)
    reason, normalized = _assess(summaries, expected_episodes)
    record = dict(schema_version=1, status="unranked" if reason else "ranked",
                  step=step, checkpoint=str(checkpoint), evaluation_run=str(run),
                  expected_episodes=expected_episodes, conditions=conditions,
                  conditions_sha256=_digest(conditions), summaries=normalized)
    if reason:
        record["reason"] = reason
        record["completed_episodes"] = len(summaries) if isinstance(summaries, list) else 0
    else:
        successes = sum(summary["outcome"] for summary in normalized)
        record.update(successes=successes, episodes=expected_episodes, score=successes / expected_episodes)
    output = run / "evaluation.json"
    try:
        previous = _read_json(output)
    except FileNotFoundError:
        previous = None
    if previous is not None:
        if (not isinstance(previous, dict) or type(previous.get("unix_ns")) is not int
                or previous["unix_ns"] <= 0 or {k: v for k, v in previous.items() if k != "unix_ns"} != record):
            raise ValueError("Evaluation run already has a different result")
        record = previous
    else:
        record["unix_ns"] = time.time_ns()
    ledger = _ledger(directory) if not reason else None
    if previous is None:
        _atomic_json(output, record)
    if ledger is not None:
        key = str(step)
        current = ledger["entries"].get(key)
        if (not _valid_entry(directory, key, current)
                or (record["unix_ns"], record["evaluation_run"]) > (current["unix_ns"], current["evaluation_run"])):
            ledger["entries"][key] = record
            _atomic_json(directory / "retention_scores.json", ledger)
    return record


def ranked_steps(checkpoint_dir, available_steps):
    """Return scored candidates in the newest valid condition group.

Malformed, missing, uncommitted, and mismatched evidence is never a score.
Other condition groups remain in the ledger but are unscored for this ranking.
"""
    directory = _directory(checkpoint_dir)
    available = list(available_steps)
    if any(type(step) is not int or step < 0 for step in available):
        raise ValueError("available_steps must contain nonnegative integers")
    available = set(available)
    try:
        ledger = _ledger(directory)
    except (OSError, ValueError, TypeError):
        return []
    candidates = [entry for key, entry in ledger["entries"].items()
                  if _valid_entry(directory, key, entry)]
    if not candidates:
        return []
    group = max(candidates, key=lambda entry: (entry["unix_ns"], entry["evaluation_run"]))["conditions_sha256"]
    # The latest/pinned reservation may already exclude the newest evaluation
    # from available_steps. Its conditions still define the comparison group.
    candidates = [entry for entry in candidates
                  if entry["conditions_sha256"] == group and entry["step"] in available]
    candidates.sort(key=lambda entry: (Fraction(entry["successes"], entry["episodes"]), entry["step"]), reverse=True)
    return [entry["step"] for entry in candidates]
