"""Shared fixtures for the DROID ZED reference test suite.

Every test in ``tests/test_zed_droid_reference.py`` builds its own fixture
locally (no module-level shared state) so failures are isolated. This file
exists so helpers stay in one place and the tests read top-to-bottom.

Key invariant: **no real ZED is ever opened**. All ``pyzed.sl`` access is
satisfied with a fake namespace injected into ``sys.modules`` before the
reference module is imported.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Fake ``pyzed`` / ``pyzed.sl`` namespace.
#
# The reference module does ``import pyzed.sl as sl`` at import time. We
# inject a module that exposes just enough surface for:
#   * ``sl.Camera()`` -> ``FakeCamera``
#   * ``sl.Mat()`` -> ``FakeMat``
#   * ``sl.InitParameters()`` -> ``FakeInitParameters``
#   * ``sl.InputType()`` / ``sl.InputType(path)`` -> ``FakeInputType``
#   * ``sl.VIEW.LEFT`` / ``sl.VIEW.RIGHT`` -> sentinels
#   * ``sl.ERROR_CODE.SUCCESS`` -> 0
#   * ``sl.RESOLUTION.HD1080`` etc. -> string sentinels
# ---------------------------------------------------------------------------
class FakeMat:
    def __init__(self):
        self._data = None

    def get_data(self):
        return self._data


class FakeInputType:
    def __init__(self, device_path: Optional[str] = None):
        self.device_path = device_path
        self.serial_number: Optional[int] = None

    def set_from_serial_number(self, serial_number: int):
        self.serial_number = serial_number


class FakeInitParameters:
    def __init__(self):
        self.camera_resolution = None
        self.camera_fps = 30
        self.input: Optional[FakeInputType] = None


@dataclass
class FakeCamera:
    """Records every call so tests can assert on it."""

    open_args: List[FakeInitParameters] = field(default_factory=list)
    grab_calls: int = 0
    retrieve_calls: List[Any] = field(default_factory=list)
    close_calls: int = 0
    open_error: Any = 0  # 0 == SUCCESS
    grab_returns: List[Any] = field(default_factory=lambda: [0])  # SUCCESS

    def open(self, init_params: FakeInitParameters):
        self.open_args.append(init_params)
        return self.open_error

    def grab(self):
        self.grab_calls += 1
        if not self.grab_returns:
            return 0
        return self.grab_returns.pop(0)

    def retrieve_image(self, mat: FakeMat, view: Any):
        self.retrieve_calls.append((mat, view))
        # Populate a 4x6x4 array so get_data()[:, :, :3] is well-defined.
        import numpy as np
        mat._data = np.zeros((4, 6, 4), dtype=np.uint8)

    def close(self):
        self.close_calls += 1


# Sentinels for VIEW enum
VIEW_LEFT = "VIEW_LEFT"
VIEW_RIGHT = "VIEW_RIGHT"


def build_fake_sl() -> types.ModuleType:
    """Return a fully wired fake ``pyzed.sl`` module."""
    fake_sl = types.ModuleType("pyzed.sl")
    fake_pyzed = types.ModuleType("pyzed")

    fake_sl.Camera = FakeCamera
    fake_sl.Mat = FakeMat
    fake_sl.InitParameters = FakeInitParameters
    fake_sl.InputType = FakeInputType

    class _View:
        LEFT = VIEW_LEFT
        RIGHT = VIEW_RIGHT

    class _ErrCode:
        SUCCESS = 0
        CAMERA_NOT_DETECTED = 1
        INVALID_RESOLUTION = 2
        CAMERA_ALREADY_OPENED = 3

    class _Resolution:
        HD2K = "HD2K"
        HD1080 = "HD1080"
        HD720 = "HD720"
        VGA = "VGA"

    fake_sl.VIEW = _View
    fake_sl.ERROR_CODE = _ErrCode
    fake_sl.RESOLUTION = _Resolution
    fake_pyzed.sl = fake_sl
    return fake_pyzed, fake_sl


# Module-level install: idempotent and side-effect free for non-test code.
def install_fake_sl(register: bool = True) -> Dict[str, Any]:
    """Install the fake ``pyzed`` namespace and return handles for tests.

    Returns a dict with the ``fake_sl`` module and a ``cameras`` list that
    every ``FakeCamera`` instance registers itself with, in order.
    """
    fake_pyzed, fake_sl = build_fake_sl()
    cameras: List[FakeCamera] = []

    # Wrap FakeCamera so every instance is tracked.
    original_camera = fake_sl.Camera

    class _TrackedCamera:
        def __init__(self):
            self._inner = original_camera()
            cameras.append(self._inner)

        def open(self, init_params):
            return self._inner.open(init_params)

        def grab(self):
            return self._inner.grab()

        def retrieve_image(self, mat, view):
            return self._inner.retrieve_image(mat, view)

        def close(self):
            return self._inner.close()

    fake_sl.Camera = _TrackedCamera
    fake_pyzed.sl = fake_sl

    saved_pyzed = sys.modules.get("pyzed")
    saved_pyzed_sl = sys.modules.get("pyzed.sl")
    if register:
        sys.modules["pyzed"] = fake_pyzed
        sys.modules["pyzed.sl"] = fake_sl

    return {
        "fake_sl": fake_sl,
        "fake_pyzed": fake_pyzed,
        "cameras": cameras,
        "restore": lambda: _restore(saved_pyzed, saved_pyzed_sl),
    }


def _restore(saved_pyzed, saved_pyzed_sl):
    if saved_pyzed is None:
        sys.modules.pop("pyzed", None)
    else:
        sys.modules["pyzed"] = saved_pyzed
    if saved_pyzed_sl is None:
        sys.modules.pop("pyzed.sl", None)
    else:
        sys.modules["pyzed.sl"] = saved_pyzed_sl


def import_reference():
    """Import the reference module fresh (no caching across tests)."""
    sys.modules.pop("scripts.zed_droid_reference.zed_camera", None)
    # Allow the tests to be run from either the repo root or the tests/
    # directory; insert the scripts/ parent so the package import resolves.
    import os
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return importlib.import_module("scripts.zed_droid_reference.zed_camera")
