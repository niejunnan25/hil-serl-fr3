"""Mocked DynamixelDriver for GELLO tests.

Used by tests/test_gello_intervention_contract.py and
tests/test_gello_cartesian_delta_agent.py to simulate the 8-channel
GELLO output (7 joints + 1 gripper) without connecting to hardware.

The mock is stateful: set_joints() updates the next reading returned
by get_joints(). By default it returns a home pose, and tests advance
it by step() with joint deltas to simulate teleoperation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


class FakeDynamixelDriver:
    """In-memory mock that satisfies the GelloIntervention contract:

    - __init__(ids, port, baudrate, max_retries, use_fake_fallback)
    - get_joints() -> array-like of shape (8,)
    - close()    -> None

    The mock maintains:
        _joints    : the current 8D reading (7 joints + 1 gripper)
        _history   : list of all readings returned by get_joints()
    """

    def __init__(
        self,
        ids: Iterable[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        max_retries: int = 1,
        use_fake_fallback: bool = False,
    ):
        self.ids = list(ids)
        self.port = port
        self.baudrate = baudrate
        self.max_retries = max_retries
        self.use_fake_fallback = use_fake_fallback
        # Default to a neutral home pose with gripper in the middle
        self._joints = np.zeros(8, dtype=np.float64)
        self._joints[7] = 0.5  # mid-stroke gripper -> mapped to 0.0 (between -1,+1)
        self._history: List[np.ndarray] = []
        self.closed = False
        self.read_count = 0

    # ---- public API matching the real driver ----
    def get_joints(self) -> np.ndarray:
        """Return the current 8D reading (records history)."""
        if self.closed:
            raise RuntimeError("FakeDynamixelDriver.read after close()")
        out = self._joints.copy()
        self._history.append(out)
        self.read_count += 1
        return out

    def close(self) -> None:
        self.closed = True

    # ---- test-only helpers ----
    def set_joints(self, joints: np.ndarray) -> None:
        """Force the next reading to be exactly the given 8D vector."""
        arr = np.asarray(joints, dtype=np.float64).flatten()
        assert arr.shape == (8,), f"FakeDynamixelDriver.set_joints needs (8,), got {arr.shape}"
        self._joints = arr.copy()

    def add_to_joints(self, delta: np.ndarray) -> None:
        """Add a delta (8D) to the current reading (teleoperation helper)."""
        d = np.asarray(delta, dtype=np.float64).flatten()
        assert d.shape == (8,), f"delta must be (8,), got {d.shape}"
        self._joints = self._joints + d

    def set_gripper(self, value: float) -> None:
        """Convenience: set the 8th channel to a specific raw value."""
        self._joints[7] = float(value)

    @property
    def history(self) -> List[np.ndarray]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Module-level singleton management: lets tests swap out the real driver
# import in gello_intervention.py with a FakeDynamixelDriver instance.
# ---------------------------------------------------------------------------
_ACTIVE_MOCK: Optional[FakeDynamixelDriver] = None


def install_mock(driver: FakeDynamixelDriver) -> None:
    """Patch gello_intervention so DynamixelDriver(...) returns `driver`.

    Call from test setup. The wrapper is the only place that constructs
    the driver, so we monkey-patch the module-level reference.

    Also installs a fake `gello.dynamixel.driver` module so scripts
    that import `from gello.dynamixel.driver import DynamixelDriver`
    at the top of their `connect()` (e.g. record_gello_demos_serl.py)
    can run on machines where the real gello package isn't installed.
    """
    global _ACTIVE_MOCK
    _ACTIVE_MOCK = driver
    # Lazy import to avoid loading gym at module import time
    import gello_intervention  # type: ignore

    gello_intervention._GELLO_AVAILABLE = True
    gello_intervention.DynamixelDriver = lambda *a, **kw: driver

    # Install a fake `gello.dynamixel.driver` module so any other
    # script that does `from gello.dynamixel.driver import DynamixelDriver`
    # at call time gets our stub.
    import sys
    import types

    if "gello" not in sys.modules:
        sys.modules["gello"] = types.ModuleType("gello")
    if "gello.dynamixel" not in sys.modules:
        sys.modules["gello.dynamixel"] = types.ModuleType("gello.dynamixel")
    mod = sys.modules["gello.dynamixel"]
    mod.DynamixelDriver = lambda *a, **kw: driver  # type: ignore[attr-defined]
    driver_mod = types.ModuleType("gello.dynamixel.driver")
    driver_mod.DynamixelDriver = lambda *a, **kw: driver  # type: ignore[attr-defined]
    sys.modules["gello.dynamixel.driver"] = driver_mod


def uninstall_mock() -> None:
    """Restore the original gello_intervention imports."""
    global _ACTIVE_MOCK
    _ACTIVE_MOCK = None
    import gello_intervention  # type: ignore

    # Best-effort restore: gello_intervention holds the import inside
    # try/except; flip the flag back to whatever import produced.
    try:
        from gello.dynamixel.driver import DynamixelDriver  # type: ignore

        gello_intervention.DynamixelDriver = DynamixelDriver
        gello_intervention._GELLO_AVAILABLE = True
    except ImportError:
        gello_intervention.DynamixelDriver = None  # type: ignore
        gello_intervention._GELLO_AVAILABLE = False
