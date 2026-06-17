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

    ok, frame = cap.read()  # BGR uint8
    cap.close()

    # or as a context manager
    with VideoCapture(ZEDCapture("external", "36276705")) as cap:
        ok, frame = cap.read()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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

    def __init__(self, capture: _CaptureLike) -> None:
        if not isinstance(capture, _CaptureLike):
            raise TypeError(
                f"capture must implement read()/close(), got {type(capture).__name__}"
            )
        self._cap = capture

    # ── public API ─────────────────────────────────────────────────────────

    def read(self) -> np.ndarray:
        """Read one frame, returning the bare FRAME (ndarray) to match RSCapture's
        convention -- franka_env.get_im expects a frame, NOT the (ok, frame) tuple
        that ZEDCapture returns."""
        ok, frame = self._cap.read()
        return frame

    def close(self) -> None:
        """Release camera resources.  Delegates to the inner capture."""
        self._cap.close()

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> VideoCapture:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"VideoCapture(inner={self._cap!r})"
