"""Unit tests for the local DROID ZED reference port.

Hard contract: no real ZED is ever opened. Every test injects a fake
``pyzed.sl`` namespace before importing ``zed_camera`` and tears the
injection down at exit so test ordering does not matter.

Each test builds its own fixture (no module-level shared state) so a
failure in one test cannot mask or contaminate another.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from typing import Any, Dict


# Make the fixtures module importable when this file is run directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURES = os.path.abspath(os.path.join(_HERE, os.pardir))
if _FIXTURES not in sys.path:
    sys.path.insert(0, _FIXTURES)

import zed_camera_test_fixtures as fx  # noqa: E402


def _install_env(**overrides: str):
    """Set env vars for the duration of a ``with`` block and restore them."""

    class _Env:
        def __enter__(self):
            self._saved = {k: os.environ.get(k) for k in overrides}
            for k, v in overrides.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, exc_type, exc, tb):
            for k, v in self._saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Env()


def _fresh_reference(fake_sl_setup: Dict[str, Any]):
    """Return a freshly imported reference module wired to the fake sl."""
    # Drop any cached import so a fake ``sl`` is picked up.
    sys.modules.pop("zed_camera", None)
    sys.modules.pop("scripts.zed_droid_reference.zed_camera", None)
    return fx.import_reference()


class _Base(unittest.TestCase):
    """Each test installs a fresh fake sl; tear it down after every test."""

    def setUp(self):
        self._handle = fx.install_fake_sl(register=True)
        self.cameras = self._handle["cameras"]
        self.fake_sl = self._handle["fake_sl"]

    def tearDown(self):
        self._handle["restore"]()


# ---------------------------------------------------------------------------
# 1. Module import contract
# ---------------------------------------------------------------------------
class TestImports(_Base):
    def test_droid_zed_camera_imports(self):
        ref = _fresh_reference(self._handle)
        self.assertTrue(hasattr(ref, "ZedCamera"))
        self.assertTrue(hasattr(ref, "show_live_preview"))
        self.assertTrue(hasattr(ref, "crop_left_right"))
        self.assertTrue(hasattr(ref, "_RESOLUTION_BY_NAME"))


# ---------------------------------------------------------------------------
# 2. Class shape (no construction, no sl access)
# ---------------------------------------------------------------------------
class TestClassShape(_Base):
    def test_droid_zed_camera_class_shape(self):
        ref = _fresh_reference(self._handle)
        cam = ref.ZedCamera.__init__
        # Signature: (self, serial_number=None, device_path=None,
        #             resolution=None, fps=30)
        import inspect

        sig = inspect.signature(cam)
        params = sig.parameters
        self.assertIn("serial_number", params)
        self.assertIn("device_path", params)
        self.assertIn("resolution", params)
        self.assertIn("fps", params)
        # Defaults: serial/device/resolution all default to None/30 for fps.
        self.assertIsNone(params["serial_number"].default)
        self.assertIsNone(params["device_path"].default)
        self.assertIsNone(params["resolution"].default)
        self.assertEqual(params["fps"].default, 30)
        # Methods we expect from the DROID port.
        for name in ("capture_frame", "close", "__del__"):
            self.assertTrue(hasattr(ref.ZedCamera, name), f"missing {name}")


# ---------------------------------------------------------------------------
# 3. Construction paths
# ---------------------------------------------------------------------------
class TestInitSerial(_Base):
    def test_droid_zed_camera_init_with_serial(self):
        ref = _fresh_reference(self._handle)
        cam = ref.ZedCamera(serial_number=13132609)
        # Camera was opened exactly once, with the right input type.
        self.assertEqual(len(self.cameras), 1)
        inner = self.cameras[0]
        self.assertEqual(len(inner.open_args), 1)
        ip = inner.open_args[0]
        self.assertIsNotNone(ip.input)
        self.assertEqual(ip.input.serial_number, 13132609)
        self.assertIsNone(ip.input.device_path)
        # Default resolution/fps are propagated to init params.
        self.assertEqual(ip.camera_resolution, "HD1080")
        self.assertEqual(ip.camera_fps, 30)


class TestInitPath(_Base):
    def test_droid_zed_camera_init_with_path(self):
        ref = _fresh_reference(self._handle)
        cam = ref.ZedCamera(device_path="/dev/video0")
        self.assertEqual(len(self.cameras), 1)
        ip = self.cameras[0].open_args[0]
        self.assertIsNotNone(ip.input)
        self.assertEqual(ip.input.device_path, "/dev/video0")
        self.assertIsNone(ip.input.serial_number)


# ---------------------------------------------------------------------------
# 4. Resolution mapping (no sl required, pure table)
# ---------------------------------------------------------------------------
class TestResolutionMapping(_Base):
    def test_droid_zed_camera_resolution_mapping(self):
        ref = _fresh_reference(self._handle)
        m = ref._RESOLUTION_BY_NAME
        self.assertEqual(set(m), {"HD2K", "HD1080", "HD720", "VGA"})
        # All values resolve to the fake ``sl.RESOLUTION`` enum members.
        self.assertEqual(m["HD2K"], self.fake_sl.RESOLUTION.HD2K)
        self.assertEqual(m["HD1080"], self.fake_sl.RESOLUTION.HD1080)
        self.assertEqual(m["HD720"], self.fake_sl.RESOLUTION.HD720)
        self.assertEqual(m["VGA"], self.fake_sl.RESOLUTION.VGA)


# ---------------------------------------------------------------------------
# 5. Env var overrides (resolution + fps)
# ---------------------------------------------------------------------------
class TestEnvOverrides(_Base):
    def test_droid_zed_camera_env_var_overrides(self):
        ref = _fresh_reference(self._handle)
        # No env -> defaults.
        with _install_env(DROID_ZED_RESOLUTION=None, DROID_ZED_FPS=None):
            self.assertEqual(
                ref._resolve_resolution("DEFAULT_RES"),
                "DEFAULT_RES",
            )
            self.assertEqual(ref._resolve_fps(15), 15)
        # Legal env values are honored and case-insensitive for resolution.
        with _install_env(DROID_ZED_RESOLUTION="hd720", DROID_ZED_FPS="60"):
            self.assertEqual(
                ref._resolve_resolution("DEFAULT_RES"),
                self.fake_sl.RESOLUTION.HD720,
            )
            self.assertEqual(ref._resolve_fps(15), 60)
        # Bad resolution -> ValueError.
        with _install_env(DROID_ZED_RESOLUTION="FOO", DROID_ZED_FPS=None):
            with self.assertRaises(ValueError):
                ref._resolve_resolution("DEFAULT_RES")
        # Bad fps -> ValueError, covers both non-int and <=0 branches.
        with _install_env(DROID_ZED_RESOLUTION=None, DROID_ZED_FPS="zero"):
            with self.assertRaises(ValueError):
                ref._resolve_fps(15)
        with _install_env(DROID_ZED_RESOLUTION=None, DROID_ZED_FPS="0"):
            with self.assertRaises(ValueError):
                ref._resolve_fps(15)


# ---------------------------------------------------------------------------
# 6. ``__main__`` must not NameError
# ---------------------------------------------------------------------------
class TestMainNoNameError(_Base):
    def test_droid_main_no_name_error(self):
        # ``show_live_preview`` blocks indefinitely on ``cv2.waitKey``; rather
        # than spawn a subprocess we just statically assert that the
        # variable name passed to ``show_live_preview`` is actually bound in
        # the ``__main__`` block of the reference module. This catches the
        # exact upstream DROID bug (``zed_wrist_camera`` bound, ``zed_camera``
        # passed) without hanging the test process.
        ref = _fresh_reference(self._handle)
        import ast

        with open(ref.__file__, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        # Locate the ``if __name__ == "__main__":`` block.
        main_block = None
        for node in tree.body:
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"
            ):
                main_block = node
                break
        self.assertIsNotNone(main_block, "no __main__ block found")
        # Collect every name bound in the main block + every name referenced
        # inside a ``show_live_preview(...)`` call.
        bound: set = set()
        referenced: set = set()
        for sub in ast.walk(main_block):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name):
                        bound.add(tgt.id)
            elif isinstance(sub, ast.Call):
                func = sub.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name == "show_live_preview" and sub.args:
                    first = sub.args[0]
                    if isinstance(first, ast.Name):
                        referenced.add(first.id)
        # The upstream bug: bound set is empty for the referenced name.
        self.assertTrue(referenced, "show_live_preview is never called in __main__")
        self.assertTrue(
            referenced <= bound,
            f"show_live_preview uses {referenced} which is not bound in __main__",
        )


# ---------------------------------------------------------------------------
# 7. Static check: no real ZED open at import time
# ---------------------------------------------------------------------------
class TestImportSafety(_Base):
    def test_droid_does_not_open_camera_on_import(self):
        # Before importing the reference, install the fake and snapshot the
        # registered cameras list; importing the module must not append to it.
        ref = _fresh_reference(self._handle)
        # The module import side-effect chain should be: ``try import sl``,
        # then ``_build_resolution_map()``. Neither constructs a Camera.
        self.assertEqual(
            self.cameras,
            [],
            "Importing the reference module must not construct a ZED camera",
        )
        # ``_RESOLUTION_BY_NAME`` is built but no camera exists.
        self.assertIsInstance(ref._RESOLUTION_BY_NAME, dict)


# ---------------------------------------------------------------------------
# 8. ``__del__`` must invoke close (via the underlying sl camera)
# ---------------------------------------------------------------------------
class TestDelCallsClose(_Base):
    def test_droid_close_called_in_del(self):
        ref = _fresh_reference(self._handle)
        cam = ref.ZedCamera(serial_number=1)
        inner = self.cameras[0]
        # Open the camera via the wrapper (already done by __init__).
        self.assertEqual(inner.close_calls, 0)
        # Explicit close() increments the counter.
        cam.close()
        self.assertEqual(inner.close_calls, 1)
        # __del__ on the wrapper should also call close() at least once
        # more. We rebind ``cam`` and drop the only reference.
        cam_local = ref.ZedCamera(serial_number=2)
        # Drop the reference; the wrapper's __del__ will close the camera.
        del cam_local
        # The two cameras created total close_calls: at least 2 (1 explicit,
        # 1 from GC). We allow >2 because CPython GC timing is non-deterministic.
        self.assertGreaterEqual(self.cameras[1].close_calls, 1)
        self.assertGreaterEqual(
            sum(c.close_calls for c in self.cameras), 2
        )


if __name__ == "__main__":
    unittest.main()
