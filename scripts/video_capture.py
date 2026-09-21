#!/usr/bin/env python3
"""
video_capture.py – VideoCapture wrapper compatible with SERL's FrankaEnv.

SERL's env expects a ``VideoCapture`` object (like cv2.VideoCapture) that
exposes ``read()`` and ``close()``.  This module wraps a ZEDCapture instance
behind that interface so the rest of the pipeline sees the same abstraction
it would get from SERL's ``RSCapture`` / ``VideoCapture``.

Usage:
    from scripts.zed_capture import ZEDCapture
    from scripts.video_capture import VideoCapture

    cam = ZEDCapture("external", "36276705")
    cap = VideoCapture(cam)

    frame = cap.read()  # BGR uint8
    cap.close()

    # or as a context manager
    with VideoCapture(ZEDCapture("external", "36276705")) as cap:
        frame = cap.read()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import threading
import time

import numpy as np


@runtime_checkable
class _CaptureLike(Protocol):
    """Structural type for anything with read()/close()."""

    def read(self) -> tuple[bool, np.ndarray]: ...
    def close(self) -> None: ...


class VideoCapture:
    """Thin wrapper around ZEDCapture (or any RSCapture-compatible object).

    Parameters
    ----------
    capture : _CaptureLike
        An object exposing ``read() -> (bool, ndarray)`` and ``close()``.
        Typically a :class:`ZEDCapture` instance.
    """

    def __init__(self, capture: _CaptureLike, *, name=None, recorder=None, continuous=False) -> None:
        if not isinstance(capture, _CaptureLike):
            raise TypeError(
                f"capture must implement read()/close(), got {type(capture).__name__}"
            )
        self._cap = capture
        self.name = name or getattr(capture, "name", "camera")
        self.recorder = recorder
        self.continuous = continuous
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self.last_read = None
        if continuous:
            if recorder:
                recorder.register_camera(self.name)
            self._thread = threading.Thread(target=self._capture, name=f"capture-{self.name}", daemon=True)
            self._thread.start()

    def _capture(self):
        from hilserl.storage import stamp
        frame_id = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Camera acquisition failed: {self.name}")
                capture = dict(camera=self.name, capture_id=frame_id, **stamp())
                frame_id += 1
                # The SDK may reuse its buffer; this copy owns the observation pixels.
                frame = frame.copy()
                with self._condition:
                    self._latest = (frame, capture)
                    self._condition.notify_all()
                if self.recorder:
                    self.recorder.video_frame(self.name, frame, capture)
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            if self.recorder:
                self.recorder.fail(f"Camera {self.name}: {exc}")
        finally:
            self._cap.close()

    # ── public API ─────────────────────────────────────────────────────────

    def read(self) -> np.ndarray:
        """Read one frame, returning the bare FRAME (ndarray) to match RSCapture's
        convention -- franka_env.get_im expects a frame, NOT the (ok, frame) tuple
        that ZEDCapture returns."""
        if self.continuous:
            with self._condition:
                self._condition.wait_for(lambda: self._latest is not None or self._error is not None, timeout=5)
                if self._error:
                    raise RuntimeError(f"Camera {self.name}: {self._error}") from self._error
                if self._latest is None:
                    raise RuntimeError(f"No frame received: {self.name}")
                frame, capture = self._latest
                if time.monotonic_ns() - capture["monotonic_ns"] > 2_000_000_000:
                    raise RuntimeError(f"Camera frame is stale: {self.name}")
                self.last_read = dict(capture)
                return frame.copy()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera acquisition failed: {self.name}")
        return frame

    def read_after(self, after_monotonic_ns: int, *, deadline: float, check) -> np.ndarray:
        """Wait for a post-boundary frame without blocking capture or cancellation.

        The deadline is shared by both reset cameras. Timestamps describe SDK
        read completion, not hardware exposure synchronization.
        """
        if not self.continuous:
            raise RuntimeError(f"Fresh reset frames require continuous capture: {self.name}")
        while True:
            # May read robot state; never run this callback under the camera lock.
            check()
            with self._condition:
                if self._error:
                    raise RuntimeError(f"Camera {self.name}: {self._error}") from self._error
                if self._stop.is_set():
                    raise RuntimeError(f"Camera stopped: {self.name}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Timed out waiting for a fresh reset frame: {self.name}")
                if self._latest is not None:
                    frame, capture = self._latest
                    if (capture["monotonic_ns"] > after_monotonic_ns
                            and 0 <= time.monotonic_ns() - capture["monotonic_ns"] <= 2_000_000_000):
                        self.last_read = dict(capture)
                        return frame.copy()
                self._condition.wait(timeout=min(.05, remaining))

    def close(self) -> None:
        """Release camera resources.  Delegates to the inner capture."""
        if self._stop.is_set():
            return
        self._stop.set()
        if self.continuous:
            self._thread.join(timeout=3)
            if self._thread.is_alive() and self.recorder:
                self.recorder.fail(f"Camera shutdown timed out: {self.name}")
        else:
            self._cap.close()

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> VideoCapture:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"VideoCapture(inner={self._cap!r})"
