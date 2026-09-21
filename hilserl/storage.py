"""Durable session archive. Raw steps never include replay-credit duplicates.

NPZ files contain numeric arrays and a JSON tree, loaded with allow_pickle=False.
The raw terminal step is committed before asking an operator for its final label.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np

from hilserl.files import atomic_json, stamp


class RecordingError(RuntimeError):
    pass


def write_step(path: Path, value):
    arrays = {}

    def pack(v):
        if isinstance(v, dict):
            return {str(k): pack(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [pack(x) for x in v]
        if isinstance(v, np.ndarray):
            if v.dtype.hasobject:
                raise TypeError("Object arrays are not supported in raw trajectories")
            key = f"array_{len(arrays)}"
            arrays[key] = v
            return {"__array__": key}
        if isinstance(v, np.generic):
            return v.item()
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        raise TypeError(f"Unsupported trajectory value: {type(v).__name__}")

    arrays["__tree__"] = np.asarray(json.dumps(pack(value), allow_nan=False))
    tmp = path.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_step(path: Path):
    with np.load(path, allow_pickle=False) as data:
        def unpack(v):
            if isinstance(v, dict):
                if set(v) == {"__array__"}:
                    return data[v["__array__"]].copy()
                return {k: unpack(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return v
        return unpack(json.loads(str(data["__tree__"])))


class VideoWriter:
    """Bounded RAM queue feeding CPU H.264; acquisition time remains in the index."""

    def __init__(self, directory, name, *, fps, segment_seconds, fail, capacity=16):
        self.directory = Path(directory) / "video" / name
        self.directory.mkdir(parents=True)
        self.name, self.fps = name, fps
        self.segment_seconds, self.fail = segment_seconds, fail
        self.queue = queue.Queue(maxsize=capacity)
        self.closing = threading.Event()
        self.process = None
        self.frames_written = 0
        self.thread = threading.Thread(target=self._run, name=f"video-{name}", daemon=True)
        self.thread.start()

    def append(self, frame, capture):
        if self.closing.is_set():
            raise RecordingError(f"Video writer {self.name} has stopped")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise RecordingError("Video expects BGR uint8 H x W x 3")
        try:
            self.queue.put_nowait((frame.copy(), dict(capture)))
        except queue.Full as exc:
            raise RecordingError(f"Video writer is too slow: {self.name}") from exc

    def _run(self):
        log = index = None
        try:
            index = (self.directory / "frames.jsonl").open("w", buffering=1)
            log = (self.directory / "encoder.log").open("wb")
            shape = None
            while not self.closing.is_set() or not self.queue.empty():
                try:
                    frame, capture = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if self.process is None:
                        shape = frame.shape
                        height, width = shape[:2]
                        command = [
                            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                            "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "23", "-threads", "2",
                            "-pix_fmt", "yuv420p", "-g", str(self.fps * min(2, self.segment_seconds)),
                            "-bf", "0", "-force_key_frames", f"expr:gte(t,n_forced*{self.segment_seconds})",
                            "-sc_threshold", "0", "-f", "segment", "-segment_time", str(self.segment_seconds),
                            "-reset_timestamps", "1", "-segment_format", "mp4",
                            "-segment_format_options", "movflags=frag_keyframe+empty_moov+default_base_moof",
                            str(self.directory / "%06d.mp4"),
                        ]
                        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                    if frame.shape != shape:
                        raise RecordingError(f"Camera resolution changed: {self.name}")
                    self.process.stdin.write(frame.tobytes())
                    self.process.stdin.flush()
                    capture.update(frame=self.frames_written, fps=self.fps,
                                   segment=self.frames_written // (self.fps * self.segment_seconds))
                    index.write(json.dumps(capture) + "\n")
                    self.frames_written += 1
                    if self.frames_written % max(1, self.fps // 2) == 0:
                        import cv2
                        preview_dir = self.directory.parents[1] / "preview"
                        preview_dir.mkdir(exist_ok=True)
                        preview = cv2.resize(frame, (640, 360))
                        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            temporary = preview_dir / f"{self.name}.partial"
                            temporary.write_bytes(encoded.tobytes())
                            os.replace(temporary, preview_dir / f"{self.name}.jpg")
                finally:
                    self.queue.task_done()
            if self.process:
                self.process.stdin.close()
                if self.process.wait(timeout=15):
                    raise RecordingError(f"Encoder failed: {self.name}")
        except Exception as exc:
            self.fail(f"Video {self.name}: {exc}")
        finally:
            self.closing.set()
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            if index:
                try:
                    index.flush()
                    os.fsync(index.fileno())
                except Exception as exc:
                    self.fail(f"Video index finalization {self.name}: {exc}")
                finally:
                    index.close()
            if log:
                log.close()

    def close(self):
        self.closing.set()
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.fail(f"Encoder shutdown timed out: {self.name}")
            if self.process and self.process.poll() is None:
                self.process.terminate()


class SessionRecorder:
    """One owner for archive state; video producers only append frames or report faults."""

    def __init__(self, directory, metadata=None, *, min_free_bytes=8 * 2**30, fps=30, segment_seconds=60):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "episodes").mkdir()
        self.min_free_bytes = min_free_bytes
        self.fps, self.segment_seconds = fps, segment_seconds
        self.error = None
        self._lock = threading.RLock()
        self._last_space_check = -float("inf")
        self._closed = False
        self.videos = {}
        self._capture_heartbeat = {}
        self._camera_deadlines = {}
        self.episode = None
        self._episode_counter = 0
        self.queue = queue.Queue(maxsize=8)
        self._closing = threading.Event()
        self.manifest = dict(schema_version=1, status="recording", started=stamp(),
                             metadata=metadata or {}, episodes=[], video={}, error=None)
        self._events = (self.directory / "events.jsonl").open("w", buffering=1, encoding="utf-8")
        self._write_manifest()
        self.worker = threading.Thread(target=self._write_steps, name="trajectory-writer", daemon=True)
        self.worker.start()
        try:
            self.check()
        except BaseException:
            self.close("initialization_failed")
            raise

    def _write_manifest(self):
        atomic_json(self.directory / "manifest.json", self.manifest)

    def fail(self, message):
        with self._lock:
            if self.error is None:
                self.error = str(message)
                self.manifest["error"] = self.error

    def check(self):
        if self.error:
            raise RecordingError(self.error)
        for camera, deadline in list(self._camera_deadlines.items()):
            if camera not in self._capture_heartbeat and time.monotonic_ns() > deadline:
                self.fail(f"Camera did not deliver its first frame: {camera}")
                raise RecordingError(self.error)
        for camera, latest in list(self._capture_heartbeat.items()):
            if time.monotonic_ns() - latest > 2_000_000_000:
                self.fail(f"Camera stopped delivering frames: {camera}")
                raise RecordingError(self.error)
        now = time.monotonic()
        if now - self._last_space_check >= 1:
            self._last_space_check = now
            free = shutil.disk_usage(self.directory).free
            if free < self.min_free_bytes:
                self.fail(f"Storage reserve reached: {free / 2**30:.2f} GiB available")
                raise RecordingError(self.error)

    def event(self, kind, **fields):
        with self._lock:
            self._events.write(json.dumps(dict(type=kind, **stamp(), **fields), ensure_ascii=False) + "\n")
            self._events.flush()
            os.fsync(self._events.fileno())

    def register_camera(self, name):
        self._camera_deadlines[name] = time.monotonic_ns() + 5_000_000_000

    def video_frame(self, name, frame, capture):
        if self._closed or self.error:
            return
        try:
            self._capture_heartbeat[name] = capture["monotonic_ns"]
            self.check()
            with self._lock:
                if name not in self.videos:
                    self.videos[name] = VideoWriter(self.directory, name, fps=self.fps,
                                                   segment_seconds=self.segment_seconds, fail=self.fail)
                self.videos[name].append(frame, capture)
        except Exception as exc:
            self.fail(str(exc))

    def start_episode(self, **metadata):
        self.check()
        if self.episode is not None:
            raise RecordingError("Previous episode has not been finalized")
        self._episode_counter += 1
        episode_id = f"{self._episode_counter:06d}"
        (self.directory / "episodes" / episode_id).mkdir()
        self.episode = dict(id=episode_id, status="collecting", started=stamp(),
                            steps=0, committed_steps=0, outcome=None, complete=False, **metadata)
        self._write_episode()
        self.event("episode_start", episode_id=episode_id)
        return episode_id

    def _write_episode(self):
        atomic_json(self.directory / "episodes" / self.episode["id"] / "episode.json", self.episode)

    def append_step(self, value, *, recovery=False):
        if not recovery:
            self.check()
        if self.episode is None or self.episode["status"] != "collecting":
            raise RecordingError("Steps are accepted only inside an active episode")
        step = self.episode["steps"]
        path = self.directory / "episodes" / self.episode["id"] / f"{step:06d}.npz"
        try:
            self.queue.put_nowait((path, copy.deepcopy(value), self.episode))
        except queue.Full as exc:
            self.fail("Trajectory writer is too slow")
            raise RecordingError(self.error) from exc
        self.episode["steps"] += 1

    def _write_steps(self):
        while not self._closing.is_set() or not self.queue.empty():
            try:
                path, value, episode = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                write_step(path, value)
                episode["committed_steps"] += 1
            except Exception as exc:
                self.fail(f"Trajectory write: {exc}")
            finally:
                self.queue.task_done()

    def flush(self):
        deadline = time.monotonic() + 15
        while self.queue.unfinished_tasks:
            if time.monotonic() > deadline:
                self.fail("Trajectory flush timed out")
                break
            time.sleep(0.01)
        self.check()

    def end_collection(self, reason, *, ended=None):
        if self.episode is None:
            return
        # This barrier makes the final executed step durable before operator input.
        self.flush()
        self.episode.update(status="awaiting_label", collection_ended=ended or stamp(), termination_reason=reason)
        self._write_episode()
        self.event("collection_end", episode_id=self.episode["id"], reason=reason)

    def finish_episode(self, outcome=None, *, complete=True, reason=None):
        if self.episode is None:
            return
        requested_complete = complete
        flush_error = None
        try:
            self.flush()
        except RecordingError as exc:
            complete = False
            flush_error = exc
        if outcome not in (None, 0, 1):
            raise ValueError("Outcome must be 0, 1 or unknown")
        complete = bool(complete and outcome is not None and self.episode["steps"] > 0
                        and self.episode["steps"] == self.episode["committed_steps"])
        self.episode.update(status="complete" if complete else "incomplete", complete=complete,
                            outcome=outcome, finalized=stamp())
        if reason:
            self.episode["incomplete_reason"] = reason
        self._write_episode()
        self.manifest["episodes"].append(copy.deepcopy(self.episode))
        self.event("episode_finalized", episode_id=self.episode["id"], outcome=outcome, complete=complete)
        self._write_manifest()
        result = copy.deepcopy(self.episode)
        self.episode = None
        if requested_complete and flush_error is not None:
            raise flush_error
        return result

    def close(self, reason="stopped"):
        if self._closed:
            return
        self._closed = True
        try:
            self.finish_episode(complete=False, reason=self.error or reason)
        finally:
            self._closing.set()
            self.worker.join(timeout=20)
            for video in self.videos.values():
                video.close()
            self.manifest.update(status="fault" if self.error else "closed", ended=stamp(), error=self.error)
            self.manifest["video"] = {name: {"frames": video.frames_written, "fps": video.fps,
                                             "segment_seconds": self.segment_seconds}
                                      for name, video in self.videos.items()}
            try:
                self._write_manifest()
            finally:
                self._events.close()
