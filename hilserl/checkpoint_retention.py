"""Checkpoint retention operates only on committed, canonical model snapshots.

The caller owns the checkpoint writer lease. This module never treats replay,
recordings, a temporary save or a vaguely similar filename as a checkpoint.
Latest-by-step is a recency policy and makes no claim about policy quality.
"""
from __future__ import annotations

import json
from contextlib import contextmanager, ExitStack
import fcntl
import os
from pathlib import Path
import re
import shutil
import stat
import time


def configured_keep():
    raw = os.environ.get("HILSERL_CHECKPOINT_KEEP", "0")
    if not re.fullmatch(r"0|[1-9][0-9]*", raw):
        raise ValueError("HILSERL_CHECKPOINT_KEEP must be a nonnegative integer")
    return int(raw)


def _identity(path):
    value = path.lstat()
    return [value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns]


def committed(path):
    path = Path(path)
    if path.is_symlink():
        return False
    if Path(str(path) + "_gda").exists():
        return False  # Unsupported legacy multi-process sidecar: preserve both.
    if path.is_file():
        return path.stat().st_size > 0
    try:
        value = json.loads((path / "_CHECKPOINT_METADATA").read_text())
        return type(value.get("commit_timestamp_nsecs")) is int and value["commit_timestamp_nsecs"] > 0
    except (OSError, ValueError, AttributeError):
        return False


def _inventory(path):
    """Refuse links/special files so a retention plan has one confined owner."""
    entries = {}
    paths = [path]
    if path.is_dir():
        for parent, directories, files in os.walk(path, followlinks=False):
            paths.extend(Path(parent) / name for name in directories + files)
    for entry in paths:
        identity = _identity(entry)
        if not (stat.S_ISREG(identity[2]) or stat.S_ISDIR(identity[2])):
            raise ValueError(f"Checkpoint contains a link or special file: {entry}")
        if identity[0] != path.stat().st_dev:
            raise ValueError(f"Checkpoint crosses a filesystem boundary: {entry}")
        entries[str(entry.relative_to(path))] = identity
    return entries


def plan_retention(directory, keep=5, *, ranked_steps=(), protected_steps=()):
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    given = Path(directory).absolute()
    if given.is_symlink() or given != given.resolve():
        raise ValueError("Checkpoint directory must not traverse a symlink")
    directory = given.resolve()
    candidates, ignored = [], []
    for path in directory.iterdir():
        if not re.fullmatch(r"checkpoint_(0|[1-9][0-9]*)", path.name):
            continue
        if not committed(path):
            ignored.append(path.name)
            continue
        inventory = _inventory(path)
        candidates.append(dict(name=path.name, step=int(path.name[11:]), inventory=inventory,
                               bytes=sum(v[3] for v in inventory.values() if stat.S_ISREG(v[2]))))
    candidates.sort(key=lambda entry: entry["step"])
    available = {entry["step"] for entry in candidates}
    ranked = [step for step in ranked_steps if type(step) is int and step in available]
    protected = sorted({step for step in protected_steps if type(step) is int and step in available})
    selected = set(protected)
    if candidates:
        selected.add(candidates[-1]["step"])  # The recovery point uses one slot.
    if len(selected) > keep:
        raise RuntimeError("More checkpoints are in use than the retention budget; cleanup deferred")
    for step in [*ranked, *(entry["step"] for entry in reversed(candidates))]:
        if len(selected) < keep:
            selected.add(step)
    return dict(directory=str(directory), rule="evaluation_then_latest" if ranked else "latest_by_step",
                keep_count=keep, ranked_steps=ranked, protected_steps=protected,
                keep=[entry for entry in candidates if entry["step"] in selected],
                remove=[entry for entry in candidates if entry["step"] not in selected], ignored=sorted(ignored))


def plan_latest(directory, keep=5):
    return plan_retention(directory, keep)


def apply_latest_plan(plan, *, journal=None):
    """Apply an already-reviewed plan, revalidating every model before deletion."""
    current = plan_retention(plan["directory"], plan["keep_count"],
                             ranked_steps=plan.get("ranked_steps",()), protected_steps=plan.get("protected_steps",()))
    if current != plan:
        raise RuntimeError("Checkpoint inventory changed since retention was planned")
    # A journal inside control/ (not checkpoint_*) is kept outside the model scan.
    owned_journal = None
    if journal is None:
        path = Path(plan["directory"]) / "retention.jsonl"
        owned_journal = path.open("a", encoding="utf-8")
        journal = owned_journal
    def event(value):
        journal.write(json.dumps(dict(unix_ns=time.time_ns(), **value), ensure_ascii=False, allow_nan=False) + "\n")
        journal.flush()
        os.fsync(journal.fileno())
    removed = []
    try:
        event(dict(event="planned", plan=plan))
        for entry in plan["remove"]:
            path = Path(plan["directory"]) / entry["name"]
            if _inventory(path) != entry["inventory"] or not committed(path):
                raise RuntimeError(f"Checkpoint changed before deletion: {path}")
            event(dict(event="deleting", directory=plan["directory"], checkpoint=entry["name"]))
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            descriptor = os.open(plan["directory"], os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            removed.append(entry["name"])
            event(dict(event="deleted", directory=plan["directory"], checkpoint=entry["name"], bytes=entry["bytes"]))
        return dict(rule=plan["rule"], kept=[entry["name"] for entry in plan["keep"]],
                    removed=removed, released_logical_bytes=sum(entry["bytes"] for entry in plan["remove"]))
    finally:
        if owned_journal:
            owned_journal.close()


@contextmanager
def retention_lock(directory):
    directory = Path(directory).resolve()
    with (directory / ".retention.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def open_reader_guard(directory, step):
    """Return an inheritable shared lock for an evaluation source checkpoint.

    The launcher passes this descriptor across exec. Closing the parent's copy
    leaves the child lock alive through model loading and the whole evaluation.
    """
    if type(step) is not int or step < 0:
        raise ValueError("Invalid checkpoint step")
    directory = Path(directory).resolve()
    with retention_lock(directory):
        guards = directory / ".checkpoint_readers"
        guards.mkdir(exist_ok=True)
        fd = os.open(guards / f"checkpoint_{step}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            if not committed(directory / f"checkpoint_{step}"):
                raise ValueError("Evaluation source checkpoint is not committed or no longer retained")
            os.set_inheritable(fd, True)
            return fd
        except BaseException:
            os.close(fd)
            raise


@contextmanager
def locked_retention_plan(directory, keep=5):
    """Serialize planners and pin active evaluation sources inside the budget."""
    from hilserl.checkpoint_scores import ranked_steps
    directory = Path(directory).resolve()
    with retention_lock(directory), ExitStack() as guards:
        initial = plan_latest(directory, keep)
        readers = directory / ".checkpoint_readers"
        readers.mkdir(exist_ok=True)
        protected = []
        entries = initial["keep"] + initial["remove"]
        for entry in entries:
            handle = guards.enter_context((readers / (entry["name"] + ".lock")).open("a"))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                protected.append(entry["step"])
        yield plan_retention(directory, keep,
            ranked_steps=ranked_steps(directory, [entry["step"] for entry in entries]), protected_steps=protected)


def retain_after_save(directory, step):
    keep = configured_keep()
    if not keep:
        return None
    with locked_retention_plan(directory, keep) as plan:
        latest = plan["keep"][-1] if plan["keep"] else None
        if latest is None or latest["step"] != step:
            raise RuntimeError("Retention requires the newly committed latest checkpoint")
        if not plan["remove"]:
            return None
        return apply_latest_plan(plan)


def maintenance_after_save(directory, step):
    """A cleanup failure cannot invalidate a successfully committed model save."""
    from hilserl.files import atomic_json
    try:
        result = retain_after_save(directory, step)
        if result:
            atomic_json(Path(directory) / "retention_status.json", dict(status="ok", step=step, **result))
        return result
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"[checkpoint-retention] model saved; cleanup deferred: {message}", flush=True)
        try:
            atomic_json(Path(directory) / "retention_status.json", dict(status="error", step=step, error=message))
        except OSError as status_error:
            print(f"[checkpoint-retention] status write failed: {status_error}", flush=True)
        return dict(status="error", error=message)
