"""Episode transaction journal and update/pause barrier in the Learner process.

The existing threaded Agentlace request server calls handle(). The optimizer
uses the same lock to reserve a complete update group. Only complete, versioned
episodes become trainable; a partial insertion latches a fault until restart.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
import threading
import time

from hilserl.reward_provider import digest

REQUEST = "episode-reward-v1"


class EpisodeOnlyStore:
    """Disable the old per-step transport route in episode mode."""
    def batch_insert(self, transitions):
        if transitions:
            raise ValueError("Episode mode accepts only complete reward transactions")


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temp).unlink(missing_ok=True)


def save_pickle(path, value):
    atomic_bytes(path, pickle.dumps(value, protocol=5))


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def actor_is_current(run_dir, attempt):
    from hilserl.training_gate import _read_object, _process_info
    record = _read_object(Path(run_dir) / "actor.json")
    status = _read_object(Path(run_dir) / "control/status.json")
    pid = record.get("pid")
    if (type(pid) is not int or pid <= 0 or record.get("mode") != "train"
            or record.get("role") != "actor" or record.get("attempt_id") != attempt
            or record.get("run_dir") != str(Path(run_dir).resolve())
            or any(status.get(k) != record.get(k) for k in ("pid", "start_time", "attempt_id"))):
        return False
    start, state = _process_info(pid)
    return start == record.get("start_time") and state not in {None, "Z", "X", "x"}


def episode_key(source_attempt, episode_id):
    if not all(isinstance(v, str) and 0 < len(v) <= 512 for v in (source_attempt, episode_id)):
        raise ValueError("Invalid episode identity")
    return digest([source_attempt, episode_id])


def demo_subset(transitions):
    return [t for t in transitions if t["infos"]["source_action"] == "human"
            or bool(t["dones"] and t["infos"]["succeed"])]


def envelope(run_id, source_attempt, episode_id, raw, labeled, spec):
    value = dict(run_id=run_id, source_attempt=source_attempt, episode_id=episode_id,
                 contract_sha256=spec.sha256, raw_digest=digest(raw), transitions=labeled)
    value["payload_digest"] = digest(value)
    return value


def validate_envelope(value, run_id, spec, max_steps):
    from hilserl.learning_replay import validate_transition
    if value.get("run_id") != run_id or value.get("contract_sha256") != spec.sha256:
        raise ValueError("Episode run/reward contract mismatch")
    if value.get("payload_digest") != digest({k: v for k, v in value.items() if k != "payload_digest"}):
        raise ValueError("Episode payload digest mismatch")
    key = episode_key(value["source_attempt"], value["episode_id"])
    transitions = value.get("transitions")
    if not isinstance(transitions, list) or not 1 <= len(transitions) <= max_steps:
        raise ValueError("Invalid complete episode length")
    parent = None
    raw = []
    for i, transition in enumerate(transitions):
        raw_id = validate_transition(transition, spec.image_profile, reward_spec=spec)
        prefix, _, step = raw_id.rpartition("/")
        parent = parent or prefix
        if (prefix != parent or int(step) != i or transition["infos"].get("episode_id") != value["episode_id"]
                or bool(transition["dones"]) != (i == len(transitions) - 1)
                or transition["infos"].get("source_action") not in {"human", "policy"}):
            raise ValueError("Episode order, identity, source or terminal mismatch")
        original = copy.deepcopy(transition)
        original["rewards"] = float(original["infos"].pop("reward")["task_reward"])
        raw.append(original)
    if digest(raw) != value["raw_digest"]:
        raise ValueError("Original task data digest mismatch")
    return key


class EpisodeCommitServer:
    def __init__(self, directory, run_id, generation, spec, online, demos, *, max_steps=190,
                 authorize=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id, self.generation, self.spec = run_id, generation, spec
        self.online, self.demos, self.max_steps = online, demos, max_steps
        self.authorize = authorize or (lambda attempt: actor_is_current(run_id, attempt))
        self.lock = threading.RLock()
        self.inflight = False
        self.gap = None
        self.error = None
        self.closed = False
        self.last_update = -1
        self.committed = {}
        self.released_to = None
        self.last_release = None
        self.online_count = 0
        self.raw_ids = set()
        # RAM is rebuilt from complete durable payloads, including a prepared
        # transaction whose previous process died before returning its ACK.
        for path in sorted(self.directory.glob("*.pkl")):
            with path.open("rb") as stream:
                value = pickle.load(stream)
            key = validate_envelope(value, self.run_id, self.spec, self.max_steps)
            if key in self.committed:
                raise ValueError("Duplicate episode in transaction journal")
            self._check_new_ids(value)
            self._insert(value, key)

    def _check_new_ids(self, value):
        ids = {item["infos"]["raw_transition_id"] for item in value["transitions"]}
        demo_ids = {item["infos"]["raw_transition_id"] for item in demo_subset(value["transitions"])}
        if (self.raw_ids.intersection(ids)
                or getattr(self.online, "existing_ids", lambda ids: set())(ids)
                or getattr(self.demos, "existing_ids", lambda ids: set())(demo_ids)):
            raise ValueError("Raw transitions already exist under another episode identity")

    def _insert(self, value, key):
        transitions = value["transitions"]
        selected = demo_subset(transitions)
        for item in transitions:
            self.online.insert(item)
        for item in selected:
            self.demos.insert(item)
        receipt = dict(episode_key=key, payload_digest=value["payload_digest"],
                       contract_sha256=self.spec.sha256, generation=self.generation,
                       online_count=len(transitions), demo_count=len(selected),
                       sequence=len(self.committed) + 1, committed_unix_ns=time.time_ns())
        save_json(self.directory / f"{key}.receipt.json", receipt)
        self.committed[key] = receipt
        self.online_count += len(transitions)
        self.raw_ids.update(item["infos"]["raw_transition_id"] for item in transitions)
        return receipt

    def snapshot(self):
        with self.lock:
            return dict(generation=self.generation, gap=copy.deepcopy(self.gap),
                        paused=not self.inflight, error=self.error, closed=self.closed,
                        committed_episodes=len(self.committed), online_count=self.online_count,
                        released_to=self.released_to, last_update=self.last_update)

    def begin_update(self):
        with self.lock:
            if self.gap or self.error or self.closed or self.inflight:
                return False
            self.inflight = True
            return True

    def complete_update(self, synchronize, step):
        # Synchronize BEFORE advertising no in-flight GPU work. A gap can be
        # requested while synchronize runs; it will observe inflight=True.
        synchronize()
        with self.lock:
            self.last_update = step
            self.inflight = False

    def close(self, error=None):
        with self.lock:
            self.closed = True
            self.error = error or self.error

    def handle(self, payload):
        try:
            with self.lock:
                if not isinstance(payload, dict) or payload.get("run_id") != self.run_id:
                    raise ValueError("Wrong reward run")
                owner = payload.get("actor_attempt")
                if not self.authorize(owner):
                    raise ValueError("Actor identity is not current")
                if payload.get("contract_sha256") != self.spec.sha256:
                    raise ValueError("Wrong reward version")
                if self.closed or self.error:
                    raise RuntimeError(self.error or "Learner is closing")
                operation = payload.get("operation")
                if operation == "hello":
                    return dict(success=True, **self.snapshot())
                if operation == "begin":
                    source, episode = payload["source_attempt"], payload["episode_id"]
                    key = episode_key(source, episode)
                    gap_id = payload["gap_id"]
                    if not isinstance(gap_id, str) or not gap_id:
                        raise ValueError("Missing gap identity")
                    candidate = dict(id=gap_id, episode_key=key, owner=owner)
                    if self.gap and self.gap != candidate and self.authorize(self.gap["owner"]):
                        raise ValueError("Another live episode owns the reward gap")
                    self.gap = candidate
                    self.released_to = None
                    return dict(success=True, **self.snapshot())
                if payload.get("generation") != self.generation:
                    return dict(success=False, stale_generation=True, generation=self.generation)
                if operation == "release" and self.last_release == (owner, payload.get("gap_id"), payload.get("receipt")):
                    return dict(success=True, generation=self.generation, released=True)
                if not self.gap or self.gap["owner"] != owner or self.gap["id"] != payload.get("gap_id"):
                    raise ValueError("Wrong reward gap")
                if operation == "status":
                    return dict(success=True, **self.snapshot())
                if self.inflight:
                    return dict(success=False, not_ready=True)
                if operation == "commit":
                    value = payload["episode"]
                    key = validate_envelope(value, self.run_id, self.spec, self.max_steps)
                    if key != self.gap["episode_key"]:
                        raise ValueError("Gap and episode disagree")
                    previous = self.committed.get(key)
                    if previous:
                        if previous["payload_digest"] != value["payload_digest"]:
                            raise ValueError("Committed episode content changed")
                        return dict(success=True, receipt=previous)
                    self._check_new_ids(value)
                    path = self.directory / f"{len(self.committed):09d}-{key}.pkl"
                    try:
                        save_pickle(path, value)
                        receipt = self._insert(value, key)
                    except BaseException as exc:
                        self.error = f"Episode insertion failed; restart to rebuild replay: {exc}"
                        raise
                    return dict(success=True, receipt=receipt)
                if operation == "release":
                    receipt = self.committed.get(self.gap["episode_key"])
                    if not receipt or receipt != payload.get("receipt"):
                        raise ValueError("No matching complete episode receipt")
                    self.released_to = owner
                    self.last_release = (owner, self.gap["id"], receipt)
                    self.gap = None
                    return dict(success=True, generation=self.generation, released=True)
                raise ValueError("Unknown reward operation")
        except Exception as exc:
            return dict(success=False, error=f"{type(exc).__name__}: {exc}")
