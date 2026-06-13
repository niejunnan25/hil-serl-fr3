"""TDD contract tests for TeleopDeviceHub (P2-T7 / A1)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import teleop_hub  # noqa: E402

from teleop_hub import (  # noqa: E402
    MockJoystickBackend,
    TeleopDeviceHub,
    XboxState,
)


@pytest.fixture
def mock_backend() -> MockJoystickBackend:
    return MockJoystickBackend()


@pytest.fixture(autouse=True)
def _reset_singleton():
    teleop_hub.reset_singleton()
    yield
    teleop_hub.reset_singleton()


# ---------------------------------------------------------------------------
# Construction / introspection
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_mock_backend_constructs_and_reports_available(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        assert hub.available is True
        assert hub.backend_name == "MockJoystickBackend"

    def test_pygame_backend_degrades_gracefully_when_no_device(self):
        # No pygame joystick plugged in → available=False, but no exception.
        hub = TeleopDeviceHub(backend="pygame", device_index=999)
        assert hub.available is False
        # Polling must return a zeroed state, never raise.
        s = hub.poll()
        assert s == XboxState()

    def test_mock_backend_requires_explicit_instance(self):
        with pytest.raises(ValueError, match="mock_backend"):
            TeleopDeviceHub(backend="mock")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="unknown backend"):
            TeleopDeviceHub(backend="frobnicate")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Polling semantics
# ---------------------------------------------------------------------------
class TestPolling:
    def test_poll_returns_xbox_state_snapshot(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        s = hub.poll()
        assert isinstance(s, XboxState)
        # Default zeroed state.
        assert s.left_x == 0.0
        assert s.rb is False

    def test_poll_reflects_injected_state(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        mock_backend.set_state(XboxState(left_x=0.5, right_y=-0.3, rb=True, a=True))
        s = hub.poll()
        assert s.left_x == pytest.approx(0.5)
        assert s.right_y == pytest.approx(-0.3)
        assert s.rb is True
        assert s.a is True
        # Other fields still default.
        assert s.lt == 0.0
        assert s.lb is False

    def test_poll_is_thread_safe_under_concurrent_writes(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        errors: list[BaseException] = []

        def writer(i: int) -> None:
            try:
                for k in range(200):
                    mock_backend.set_state(XboxState(left_x=float((i + k) % 7) / 7.0))
            except BaseException as e:  # pragma: no cover - safety net
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(400):
                    _ = hub.poll()
            except BaseException as e:  # pragma: no cover - safety net
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors
        assert mock_backend.read_count >= 400

    def test_as_dict_round_trip(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        mock_backend.set_state(XboxState(left_y=0.25, dpad_x=1.0, y=True))
        d = hub.poll().as_dict()
        assert d["left_y"] == pytest.approx(0.25)
        assert d["dpad_x"] == pytest.approx(1.0)
        assert d["y"] is True
        # Schema is stable: 16 known keys.
        assert set(d.keys()) == {
            "left_x", "left_y", "right_x", "right_y",
            "dpad_x", "dpad_y", "rt", "lt",
            "a", "b", "x", "y", "lb", "rb", "start", "back",
        }


# ---------------------------------------------------------------------------
# Pygame-unavailable / no-device degradation (covered above) plus lifecycle
# ---------------------------------------------------------------------------
class TestLifecycle:
    def test_close_is_idempotent(self, mock_backend):
        hub = TeleopDeviceHub(backend="mock", mock_backend=mock_backend)
        hub.close()
        # Closing again must not raise.
        hub.close()
        assert hub.available is False
