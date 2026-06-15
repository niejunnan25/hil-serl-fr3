"""P2-T7 (v2.2.1-A1) — TeleopDeviceHub single-instance device state hub.

The hub owns a single pygame joystick instance and exposes thread-safe
``poll() -> XboxState`` plus ``available: bool``. Tests inject state via
the ``MockJoystickBackend`` so no real hardware is required.

Design points
-------------
* The hub is a process-wide singleton — even if multiple wrappers ask for
  it, only one pygame.Joystick is opened, and only one poll per tick
  returns a coherent snapshot.
* ``backend="pygame"`` (the default) opens the first detected joystick
  on construction; if pygame is missing or no device is plugged in, the
  hub reports ``available=False`` and ``poll()`` returns a zeroed state.
  This is essential for the headless CI environment.
* ``backend="mock"`` routes through a ``MockJoystickBackend`` that holds
  an internal state buffer tests can push to. Used by every test.
"""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

# Ensure scripts/ is on sys.path so the hub is importable both as
# ``scripts.teleop_hub`` and ``teleop_hub`` (test convention).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# pygame import is optional — headless envs (CI, gate runs) lack it.
try:
    import pygame  # type: ignore
    _PYGAME_AVAILABLE = True
except Exception:  # pragma: no cover - exercised via tests
    pygame = None  # type: ignore
    _PYGAME_AVAILABLE = False


@dataclass(frozen=True)
class XboxState:
    """Snapshot of the Xbox controller returned by ``TeleopDeviceHub.poll()``.

    Fields are flat int/float so the wrapper has zero numpy dependency.
    Buttons use bool for clarity; axes are floats in [-1, 1].
    """

    # Left stick (dx, dy)
    left_x: float = 0.0
    left_y: float = 0.0
    # Right stick (dz, dyaw)
    right_x: float = 0.0
    right_y: float = 0.0
    # D-pad (dpitch, droll)
    dpad_x: float = 0.0
    dpad_y: float = 0.0
    # Triggers (RT right=close, LT left=open, in [0,1])
    rt: float = 0.0
    lt: float = 0.0
    # Face / shoulder buttons
    a: bool = False
    b: bool = False
    x: bool = False  # unused today; kept for parity
    y: bool = False
    lb: bool = False
    rb: bool = False
    start: bool = False
    back: bool = False

    def as_dict(self) -> dict:
        return {
            "left_x": self.left_x,
            "left_y": self.left_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "dpad_x": self.dpad_x,
            "dpad_y": self.dpad_y,
            "rt": self.rt,
            "lt": self.lt,
            "a": self.a,
            "b": self.b,
            "x": self.x,
            "y": self.y,
            "lb": self.lb,
            "rb": self.rb,
            "start": self.start,
            "back": self.back,
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class MockJoystickBackend:
    """Test-only backend. ``set_state`` pushes a synthetic XboxState; the
    next ``read()`` returns it. ``available=True`` so the hub considers
    the device live.
    """

    def __init__(self) -> None:
        self._state = XboxState()
        self._lock = threading.Lock()
        self.available = True
        self.read_count = 0

    def set_state(self, state: XboxState) -> None:
        with self._lock:
            self._state = state

    def read(self) -> XboxState:
        with self._lock:
            self.read_count += 1
            return self._state


class _PygameBackend:
    """Real-hardware backend. Lazily initialises pygame.joystick.

    If pygame is unavailable, or no joystick is plugged in,
    ``available=False`` so the hub degrades gracefully and ``poll()``
    returns a zeroed state. This matters because the gate runs in a
    headless environment that has neither.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._joystick = None
        self._lock = threading.Lock()
        self.available = False

        if not _PYGAME_AVAILABLE:
            return
        try:
            # Full pygame.init() (with a headless video driver) is REQUIRED:
            # joystick axis/button state only refreshes when the SDL event
            # subsystem is running and pygame.event.pump() is pumped. With
            # only pygame.joystick.init(), pump() is a no-op and every read
            # returns the rest state (axes 0, no buttons) — the device looks
            # dead even while the operator moves it.
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            count = pygame.joystick.get_count()
            if count <= device_index:
                return
            self._joystick = pygame.joystick.Joystick(device_index)
            self._joystick.init()
            self.available = True
        except Exception:
            # Any pygame init failure (display, drivers, permissions)
            # leaves available=False.
            self._joystick = None
            self.available = False

    def read(self) -> XboxState:
        if self._joystick is None:
            return XboxState()
        with self._lock:
            # Drain pending events without consuming them; pygame requires
            # an event pump for axis state to refresh on some backends.
            try:
                pygame.event.pump()
            except Exception:
                return XboxState()

            def axis(i: int) -> float:
                if i < self._joystick.get_numaxes():
                    return float(self._joystick.get_axis(i))
                return 0.0

            def button(i: int) -> bool:
                if i < self._joystick.get_numbuttons():
                    return bool(self._joystick.get_button(i))
                return False

            def hat(i: int):
                if i < self._joystick.get_numhats():
                    return self._joystick.get_hat(i)
                return (0, 0)

            # Verified live on "Xbox Series X Controller" (SDL2 2.28, /dev/input/js0):
            #   axes: 0=LX 1=LY 2=LT 3=RX 4=RY 5=RT
            #     triggers rest at -1.0, fully pressed +1.0 -> remap to [0,1]
            #   hat0 = D-pad: x in {-1,0,+1} (left/right), y in {-1,0,+1} (down/up)
            #   buttons: 0=A 1=B 2=X 3=Y 4=LB 5=RB 6=View 7=Menu 8=Xbox 9=LS 10=RS
            hx, hy = hat(0)
            return XboxState(
                left_x=axis(0),
                left_y=axis(1),
                right_x=axis(3),
                right_y=axis(4),
                dpad_x=float(hx),
                dpad_y=float(hy),
                rt=(axis(5) + 1.0) / 2.0,
                lt=(axis(2) + 1.0) / 2.0,
                a=button(0),
                b=button(1),
                x=button(2),
                y=button(3),
                lb=button(4),
                rb=button(5),
                back=button(6),
                start=button(7),
            )


# ---------------------------------------------------------------------------
# Singleton / Hub
# ---------------------------------------------------------------------------
_HUB_SINGLETON: Optional["TeleopDeviceHub"] = None
_HUB_SINGLETON_LOCK = threading.Lock()


class TeleopDeviceHub:
    """Process-wide Xbox controller state hub.

    One instance, one backend, one snapshot per ``poll()``. Wrappers read
    the same state simultaneously without racing the underlying pygame
    call.

    Parameters
    ----------
    backend : {"pygame", "mock"}
        "pygame" — real hardware (default; degrades to ``available=False``
        if pygame missing or no device plugged in).
        "mock"   — test-only, requires ``mock_backend`` argument.
    mock_backend : MockJoystickBackend, optional
        Required when ``backend="mock"``; ignored otherwise.
    """

    def __init__(
        self,
        backend: str = "pygame",
        mock_backend: Optional[MockJoystickBackend] = None,
        device_index: int = 0,
    ) -> None:
        if backend == "mock":
            if mock_backend is None:
                raise ValueError("backend='mock' requires mock_backend")
            self._backend = mock_backend
        elif backend == "pygame":
            self._backend = _PygameBackend(device_index=device_index)
        else:
            raise ValueError(f"unknown backend: {backend!r}")
        self._lock = threading.Lock()

    # ----- introspection ---------------------------------------------------
    @property
    def available(self) -> bool:
        return self._backend.available

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__

    # ----- primary API -----------------------------------------------------
    def poll(self) -> XboxState:
        """Return a fresh XboxState snapshot (thread-safe)."""
        with self._lock:
            return self._backend.read()

    def close(self) -> None:
        """Release backend resources. Idempotent."""
        with self._lock:
            close = getattr(self._backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._backend.available = False


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
def get_hub() -> TeleopDeviceHub:
    """Return the process-wide hub (lazy-init pygame backend).

    Tests should not call this — instantiate ``TeleopDeviceHub(backend="mock", ...)``
    directly so each test gets a fresh state.
    """
    global _HUB_SINGLETON
    with _HUB_SINGLETON_LOCK:
        if _HUB_SINGLETON is None:
            _HUB_SINGLETON = TeleopDeviceHub(backend="pygame")
        return _HUB_SINGLETON


def reset_singleton() -> None:
    """Forget the cached hub singleton. Test-only helper."""
    global _HUB_SINGLETON
    with _HUB_SINGLETON_LOCK:
        if _HUB_SINGLETON is not None:
            try:
                _HUB_SINGLETON.close()
            except Exception:
                pass
        _HUB_SINGLETON = None
