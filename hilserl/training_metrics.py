"""Small, durable, local Learner diagnostics, independent of JAX and wandb.

The caller owns the unique-transition counter: count newly accepted raw
transition identities in this Learner attempt, excluding restored replay,
demonstrations, image-support slots, retransmissions and synthetic copies.
This module deliberately cannot infer that count from a replay buffer length.

``update_groups`` counts fully committed groups in this attempt, starting at
zero even when ``start_step`` resumes a checkpoint. A continuous-critic update
means one optimizer application, not the number of ensemble members (or the
additional grasp critic in a legacy Hybrid agent). Therefore C = groups * CTA.
The cumulative and between-sample C/N values describe observed data reuse, not
an update budget. Their denominator is null when no new online data arrived.

Call ``write`` after every complete update group. Scalars are transferred from
the device only when a sample is due; no tensor tree is retained between calls.
SAC's nested auxiliary keys are preserved as slash-separated names. Values
describe the last update call, not an average over unlogged update groups.
I/O failures propagate to the training owner; nonfinite scalars are recorded as
null with explicit ``invalid_fields`` and ``invalid_reasons``.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import numbers
import os
from pathlib import Path
import re
import time


SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 64 * 1024
MAX_METRICS = 256


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_positive(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _scalar_metrics(update_info):
    """Flatten bounded mappings, checking shape before any device transfer."""
    if not isinstance(update_info, Mapping):
        raise TypeError("update_info must be a mapping of scalar metrics")
    values, invalid = {}, {}
    visited = 0

    def walk(node, path, depth):
        nonlocal visited
        visited += 1
        if visited > MAX_METRICS * 2 or depth > 8:
            raise ValueError("Metric mapping exceeds the bounded size/depth limit")
        if isinstance(node, Mapping):
            for key, value in node.items():
                if not isinstance(key, str) or not key or len(key) > 128:
                    raise ValueError("Metric keys must be nonempty strings of at most 128 characters")
                walk(value, f"{path}/{key}" if path else key, depth + 1)
            return
        if len(values) >= MAX_METRICS or path in values:
            raise ValueError(f"Too many or colliding metric names at {path!r}")
        shape = getattr(node, "shape", None)
        if shape is not None and tuple(shape) != ():
            raise ValueError(f"Metric {path!r} must be scalar; received shape {tuple(shape)}")
        if getattr(getattr(node, "dtype", None), "kind", None) == "c":
            raise TypeError(f"Metric {path!r} must be a real scalar")
        if node is None:
            values[path], invalid[path] = None, "missing"
            return
        if shape is None and not isinstance(node, numbers.Real):
            raise TypeError(f"Metric {path!r} must be a real scalar")
        try:
            value = float(node)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Metric {path!r} must be a real scalar") from exc
        except OverflowError:
            values[path], invalid[path] = None, "overflow"
            return
        if not math.isfinite(value):
            reason = "nan" if math.isnan(value) else ("positive_infinity" if value > 0 else "negative_infinity")
            values[path], invalid[path] = None, reason
        else:
            values[path] = value

    walk(update_info, "", 0)
    return values, invalid


class LearnerMetrics:
    """One exclusive ``run/metrics/learner-<attempt_id>.jsonl`` writer.

    ``write`` returns a persisted sample dictionary or None when not due. This
    lets the owner surface ``invalid_fields`` without guessing from stdout.
    Every sampled record is flushed and fsynced; close adds a final counter-only
    record and closes the file. Reusing an attempt identifier raises instead of
    silently appending incompatible resume windows or overwriting evidence.
    """

    def __init__(self, run_dir, attempt_id, *, cta_ratio, start_step=0,
                 every_steps=50, every_seconds=10.0, clock=time.monotonic):
        if (not isinstance(attempt_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", attempt_id)):
            raise ValueError("attempt_id must be a safe nonempty identifier")
        self.attempt_id = attempt_id
        self.cta_ratio = _integer("cta_ratio", cta_ratio, 1)
        self.start_step = _integer("start_step", start_step)
        self.every_steps = _integer("every_steps", every_steps, 1)
        self.every_seconds = _finite_positive("every_seconds", every_seconds)
        self._clock = clock
        self._started = float(clock())
        if not math.isfinite(self._started):
            raise ValueError("Metrics clock must be finite")
        self._sample_at = self._started
        self._last_now = self._started
        self._sample_groups = self._sample_unique = 0
        self._latest = dict(step=self.start_step - 1, update_groups=0, unique_online_count=0)
        self.last_invalid_fields = ()
        self._closed = self._failed = False
        self.path = Path(run_dir) / "metrics" / f"learner-{attempt_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=8192)
        try:
            self._emit(self._record("attempt_start", self._started))
            # Persist both the file entry and a newly created metrics directory.
            for directory in (self.path.parent, self.path.parent.parent):
                directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            self._stream.close()
            self._closed = True
            raise

    def _now(self):
        value = float(self._clock())
        if not math.isfinite(value) or value < self._last_now:
            raise ValueError("Metrics clock must be finite and monotonic")
        self._last_now = value
        return value

    def _check_open(self):
        if self._closed or self._failed:
            raise RuntimeError("Metrics writer is closed or has a previous I/O failure")

    def should_write(self, *, update_groups, force=False):
        """Optional readiness probe before a caller performs batched device_get."""
        self._check_open()
        groups = _integer("update_groups", update_groups)
        if groups < self._latest["update_groups"]:
            raise ValueError("update_groups cannot decrease within an attempt")
        return bool(force or groups - self._sample_groups >= self.every_steps
                    or self._now() - self._sample_at >= self.every_seconds)

    def _record(self, event, now, *, metrics=None, invalid=None):
        groups = self._latest["update_groups"]
        unique = self._latest["unique_online_count"]
        critic_updates = self.cta_ratio * groups
        delta_unique = unique - self._sample_unique
        delta_critic = self.cta_ratio * (groups - self._sample_groups)
        return dict(
            schema_version=SCHEMA_VERSION, event=event, attempt_id=self.attempt_id,
            pid=os.getpid(), unix_ns=time.time_ns(), start_step=self.start_step,
            elapsed_seconds=now - self._started, counter_scope="learner_attempt",
            restored_replay_included=False, cta_ratio=self.cta_ratio, **self._latest,
            critic_updates=critic_updates,
            critic_utd=critic_updates / unique if unique else None,
            critic_utd_since_sample=delta_critic / delta_unique if delta_unique else None,
            new_online_since_sample=delta_unique, critic_updates_since_sample=delta_critic,
            metrics_scope="last_update_call" if event == "sample" else None,
            metrics=metrics or {}, invalid_fields=sorted(invalid or {}),
            invalid_reasons=invalid or {},
        )

    def _emit(self, record):
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
        if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError("Metric record exceeds 64 KiB limit")
        try:
            self._stream.write(line)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            self._failed = True
            raise

    def write(self, *, step, update_groups, unique_online_count, update_info=None,
              extra=None, force=False):
        """Observe counters on each group; sync/write scalar info only when due.

        Scalar conversion may block for device completion. Call should_write and
        then device_get the small metric tree first if a single explicit transfer
        is preferred. Never pass observations, actions, images or sample batches.
        Optional ``extra`` contains additional scalar diagnostics (for example
        timer averages or buffer transition counts), stored under ``runtime/``.
        """
        self._check_open()
        latest = dict(
            step=_integer("step", step, self.start_step - 1),
            update_groups=_integer("update_groups", update_groups),
            unique_online_count=_integer("unique_online_count", unique_online_count),
        )
        if latest["step"] != self.start_step + latest["update_groups"] - 1:
            raise ValueError("step must identify the last completed update group in this attempt")
        if any(latest[key] < self._latest[key] for key in latest):
            raise ValueError("Step and counters cannot decrease within a Learner attempt")
        now = self._now()
        due = (force or latest["update_groups"] - self._sample_groups >= self.every_steps
               or now - self._sample_at >= self.every_seconds)
        # Do not retain JAX values when this call is not sampled.
        metrics, invalid = _scalar_metrics({} if update_info is None else update_info) if due else ({}, {})
        if due and extra is not None:
            extra_values, extra_invalid = _scalar_metrics(extra)
            extra_values = {f"runtime/{key}": value for key, value in extra_values.items()}
            if metrics.keys() & extra_values.keys() or len(metrics) + len(extra_values) > MAX_METRICS:
                raise ValueError("Additional metrics exceed or collide with the bounded metric names")
            metrics.update(extra_values)
            invalid.update({f"runtime/{key}": reason for key, reason in extra_invalid.items()})
        self._latest = latest
        if not due:
            return None
        record = self._record("sample", now, metrics=metrics, invalid=invalid)
        self._emit(record)
        self._sample_groups = latest["update_groups"]
        self._sample_unique = latest["unique_online_count"]
        self._sample_at = now
        self.last_invalid_fields = tuple(record["invalid_fields"])
        return record

    def close(self):
        if self._closed:
            return
        try:
            if not self._failed:
                self._emit(self._record("attempt_end", self._now()))
        finally:
            self._closed = True
            self._stream.close()

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


def read_metrics_tail(path, *, limit=20, max_bytes=256 * 1024):
    """Read a bounded tail, reporting an incomplete trailing line explicitly.

    Malformed complete records raise. A possibly concurrent partial final write
    is excluded and reported, never presented as a valid sample. This reader
    makes no claim that attempt-local counters cover earlier attempts.
    """
    limit = _integer("limit", limit, 1)
    max_bytes = _integer("max_bytes", max_bytes, 1)
    if limit > 1000 or max_bytes > 4 * 1024 * 1024:
        raise ValueError("Metrics tail limit exceeds the bounded read allowance")
    with Path(path).open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        offset = max(0, size - max_bytes)
        stream.seek(offset)
        data = stream.read(max_bytes)
    bytes_read = len(data)
    incomplete = bool(data and not data.endswith(b"\n"))
    if offset:
        _, separator, data = data.partition(b"\n")
        if not separator:
            data = b""
    lines = data.split(b"\n")
    lines.pop()  # Empty trailing item or the explicitly excluded partial record.
    records = []

    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON token in metrics: {value}")

    def parse_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number in metrics")
        return number

    for line in lines[-limit:]:
        record = json.loads(line, parse_constant=reject_nonfinite, parse_float=parse_float)
        if not isinstance(record, dict):
            raise ValueError("A metrics record must be a JSON object")
        records.append(record)
    return dict(records=records, bytes_read=bytes_read, truncated_head=bool(offset),
                incomplete_last_line=incomplete)
