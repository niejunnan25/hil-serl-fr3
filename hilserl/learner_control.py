"""Cooperative Learner shutdown without JAX or hardware dependencies.

The Learner owns ``run/control/learner_status.json``. The manager owns
``learner_stop.json`` and targets the exact PID, process start time and attempt.
Signals only latch a stop event; the training loop calls ``stop_requested`` at
safe boundaries, then saves its checkpoint before closing worker resources.
``checkpoint_saved`` is a receipt supplied after the save completes, never an
inference from a signal, phase transition or process exit.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid


PROTOCOL_VERSION = 1
PHASES = frozenset({
    "initializing", "training", "paused", "stop_requested", "saving_checkpoint",
    "closing", "stopped", "fault",
})
_IDENTITY_FIELDS = ("pid", "start_time", "attempt_id")
_OWNED_FIELDS = frozenset({
    *_IDENTITY_FIELDS, "protocol_version", "revision", "unix_ns", "monotonic_ns",
    "stop_requested", "stop_request_id", "stop_source", "stop_signal",
    "manual_paused", "activity_request_id",
})


def _stamp():
    return {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}


def _process_start(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _atomic_json(path, value):
    """Use a unique temporary file so repeated requests cannot share a writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_request(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            text = stream.read(65537)
        if len(text) > 65536:
            return None
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError):
        return None


def _valid_identity(value):
    return (
        type(value.get("pid")) is int and value["pid"] > 0
        and isinstance(value.get("start_time"), str) and bool(value["start_time"])
        and isinstance(value.get("attempt_id"), str) and bool(value["attempt_id"])
    )


def _matches_request(value, identity, commands=("stop",)):
    return (
        isinstance(value, dict) and _valid_identity(value)
        and value.get("command") in commands
        and type(value.get("protocol_version", PROTOCOL_VERSION)) is int
        and value.get("protocol_version", PROTOCOL_VERSION) == PROTOCOL_VERSION
        and isinstance(value.get("request_id"), str) and bool(value["request_id"])
        and all(value.get(key) == identity.get(key) for key in _IDENTITY_FIELDS)
    )


def request_stop(run_dir, *, pid, start_time, attempt_id, request_id=None):
    """Write one idempotent stop request; do not send signals or claim a save.

    The caller verifies the live process before calling. An already pending
    request for this identity is returned unchanged, including its request ID.
    Old attempts' requests are replaceable but cannot stop the new Learner.
    """
    identity = dict(pid=pid, start_time=start_time, attempt_id=attempt_id)
    if not _valid_identity(identity):
        raise ValueError("Learner stop requires a PID, process start time and attempt ID")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise ValueError("request_id must be a nonempty string")
    path = Path(run_dir) / "control" / "learner_stop.json"
    previous = _read_request(path)
    if _matches_request(previous, identity):
        return previous
    value = dict(
        protocol_version=PROTOCOL_VERSION, command="stop", **identity,
        request_id=request_id or uuid.uuid4().hex, **_stamp(),
    )
    _atomic_json(path, value)
    return value


def request_activity(run_dir, *, command, pid, start_time, attempt_id, request_id=None):
    """Pause or resume updates cooperatively; resume never bypasses the Actor gate."""
    identity = dict(pid=pid, start_time=start_time, attempt_id=attempt_id)
    if command not in {"pause", "resume"} or not _valid_identity(identity):
        raise ValueError("Learner activity requires pause/resume and an exact live identity")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise ValueError("request_id must be a nonempty string")
    path = Path(run_dir) / "control" / "learner_activity.json"
    previous = _read_request(path)
    if _matches_request(previous, identity, (command,)):
        return previous
    value = dict(protocol_version=PROTOCOL_VERSION, command=command, **identity,
                 request_id=request_id or uuid.uuid4().hex, **_stamp())
    _atomic_json(path, value)
    return value


class LearnerControl:
    def __init__(self, run_dir=None, *, attempt_id=None, pid=None, start_time=None, initial_paused=False):
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.directory = self.run_dir / "control" if self.run_dir is not None else None
        self.status_path = self.directory / "learner_status.json" if self.directory is not None else None
        self.request_path = self.directory / "learner_stop.json" if self.directory is not None else None
        self.activity_path = self.directory / "learner_activity.json" if self.directory is not None else None
        pid = os.getpid() if pid is None else pid
        if type(pid) is not int or pid <= 0:
            raise ValueError("pid must be a positive integer")
        if attempt_id is None:
            attempt_id = os.environ.get("HILSERL_ATTEMPT_ID") or uuid.uuid4().hex
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt_id must be a nonempty string")
        start_time = _process_start(pid) if start_time is None else start_time
        if start_time is not None and (not isinstance(start_time, str) or not start_time):
            raise ValueError("start_time must be a nonempty string when available")
        self._stop_latched = False
        self._stop_signal = None
        self._acknowledged = False
        self.state = dict(
            protocol_version=PROTOCOL_VERSION, pid=pid, start_time=start_time,
            attempt_id=attempt_id, phase="initializing", revision=0, step=None,
            checkpoint_path=None, checkpoint_error=None, checkpoint_saved=False,
            stop_requested=False, stop_request_id=None, stop_source=None, stop_signal=None,
            manual_paused=bool(initial_paused), activity_request_id=None, pause_reason=None,
        )
        self.publish("initializing")

    def publish(self, phase=None, **fields):
        """Publish caller-observed progress; a completed save needs an explicit receipt."""
        if phase is not None and phase not in PHASES:
            raise ValueError(f"Unknown Learner phase: {phase}")
        forbidden = _OWNED_FIELDS.intersection(fields)
        if forbidden:
            raise ValueError(f"Learner identity/protocol fields are owned: {', '.join(sorted(forbidden))}")
        state = dict(self.state)
        if phase is not None:
            state["phase"] = phase
        if phase == "saving_checkpoint":
            state.update(checkpoint_saved=False, checkpoint_error=None)
        if "checkpoint_path" in fields and fields["checkpoint_path"] is not None:
            fields["checkpoint_path"] = str(fields["checkpoint_path"])
        state.update(fields)
        if type(state["checkpoint_saved"]) is not bool:
            raise ValueError("checkpoint_saved must be a boolean save receipt")
        if state.get("checkpoint_error") is not None:
            state["checkpoint_saved"] = False
        if state["checkpoint_saved"] and not state.get("checkpoint_path"):
            raise ValueError("A completed checkpoint receipt requires checkpoint_path")
        if state["phase"] == "saving_checkpoint" and state["checkpoint_saved"]:
            raise ValueError("Cannot report a completed save while saving_checkpoint")
        state.update(revision=self.state["revision"] + 1, **_stamp())
        if self.status_path is not None:
            _atomic_json(self.status_path, state)
        self.state = state
        return dict(state)

    def activity_paused(self):
        if self.activity_path is not None:
            value = _read_request(self.activity_path)
            if (_matches_request(value, self.state, ("pause", "resume"))
                    and value["request_id"] != self.state["activity_request_id"]):
                self.state.update(manual_paused=value["command"] == "pause",
                                  activity_request_id=value["request_id"])
                self.publish()
        return self.state["manual_paused"]

    def stop_requested(self):
        """Latch identity-matched requests and acknowledge them once at a safe boundary."""
        new_request = None
        if self.request_path is not None and self.state["stop_request_id"] is None:
            value = _read_request(self.request_path)
            if _matches_request(value, self.state):
                new_request = value["request_id"]
                self._stop_latched = True
        if not self._stop_latched:
            return False
        if not self._acknowledged or new_request is not None:
            self.state.update(
                stop_requested=True,
                stop_request_id=new_request or self.state["stop_request_id"],
                stop_source="file" if new_request is not None else self.state["stop_source"] or "signal",
                stop_signal=self._stop_signal,
            )
            phase = "stop_requested" if self.state["phase"] in {"initializing", "training", "paused"} else None
            self.publish(phase)
            self._acknowledged = True
        return True

    @contextmanager
    def signal_handlers(self):
        """Temporarily replace SIGINT/SIGTERM; handlers only latch, never raise or do I/O."""
        previous = []

        def stop(signum, _frame):
            if self._stop_signal is None:
                self._stop_signal = int(signum)
            # Assignment only: even Event.set() acquires a lock and can
            # deadlock if a signal interrupts a thread holding that lock.
            self._stop_latched = True

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handler = signal.getsignal(signum)
                signal.signal(signum, stop)
                previous.append((signum, old_handler))
            yield self
        finally:
            for signum, old_handler in reversed(previous):
                signal.signal(signum, old_handler)
