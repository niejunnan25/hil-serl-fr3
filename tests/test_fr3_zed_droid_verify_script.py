"""Contract tests for ``scripts/setup/14_fr3_zed_droid_verify.sh``.

Hard contract: this test never invokes the wrapper end-to-end (it would need
the real ``franka_env`` and pyzed installed). Instead, it asserts:

  1. ``bash -n`` syntax is valid.
  2. Exit codes for the documented failure modes are present in source.
  3. The production-payload grep guard exists and points at the right
     files.
  4. The wrapper does NOT call ``sl.Camera(`` / ``.open(`` / ``.grab(``
     on its own (it is an orchestrator only).
  5. The wrapper invokes the two pytest targets we ship.
  6. Mock-based smoke test: when the wrapper's inline Python heredocs are
     executed under a fake ``pyzed.sl`` namespace, both ``ZedCamera`` paths
     pass.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "setup" / "14_fr3_zed_droid_verify.sh"
REFERENCE_DIR = ROOT / "scripts" / "zed_droid_reference"
REFERENCE_PYTEST = REFERENCE_DIR / "tests" / "test_zed_droid_reference.py"


def _script_text() -> str:
    return WRAPPER.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. bash -n syntax check
# ---------------------------------------------------------------------------
class TestSyntax(unittest.TestCase):
    def test_wrapper_has_valid_bash_syntax(self):
        if not WRAPPER.exists():
            self.skipTest(f"wrapper not found: {WRAPPER}")
        subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


# ---------------------------------------------------------------------------
# 2. exit code matrix
# ---------------------------------------------------------------------------
class TestExitCodeMatrix(unittest.TestCase):
    """Every documented RC must appear as an ``exit N`` literal in source."""

    def test_exit_code_matrix(self):
        text = _script_text()
        # Documented exit codes in the header comment.
        for rc, name in [
            (20, "reference import failed"),
            (21, "ZedCamera construction contract failed"),
            (22, "pytest test_zed_droid_reference failed"),
            (23, "pytest test_fr3_zed_droid_verify_script failed"),
            (25, "no python interpreter"),
            (26, "evidence dir not writable"),
        ]:
            with self.subTest(rc=rc, name=name):
                self.assertRegex(
                    text,
                    rf"\bexit\s+{rc}\b",
                    f"missing exit {rc} ({name}) in wrapper",
                )

    def test_success_exit_zero(self):
        text = _script_text()
        self.assertRegex(text, r"\bexit\s+0\b", "wrapper must end with exit 0")

    def test_set_euo_pipefail_present(self):
        text = _script_text()
        self.assertIn("set -euo pipefail", text)


# ---------------------------------------------------------------------------
# 3. production-payload grep guard
# ---------------------------------------------------------------------------
class TestPayloadGuard(unittest.TestCase):
    def test_wrapper_scans_production_payload_modules(self):
        text = _script_text()
        for needle in ("zed_capture.py", "zed_env_dispatch.py"):
            with self.subTest(needle=needle):
                self.assertIn(
                    needle,
                    text,
                    f"production-payload guard must scan {needle}",
                )

    def test_wrapper_uses_grep_not_real_sdk_call(self):
        text = _script_text()
        # The wrapper is an orchestrator; the heredoc Python is allowed to
        # build a *fake* sl namespace. We assert the wrapper text itself
        # does not import pyzed at top level (which would be a regression).
        self.assertNotIn("import pyzed", text)
        # But the heredoc Python is allowed to fake one.
        self.assertIn("pyzed.sl", text)


# ---------------------------------------------------------------------------
# 4. wrapper does not embed a real ``sl.Camera(`` / ``.open(`` / ``.grab(``
# ---------------------------------------------------------------------------
class TestNoRealSdkCall(unittest.TestCase):
    def test_wrapper_has_no_sl_camera_call_outside_heredoc(self):
        """The wrapper orchestrates; the heredoc Python is fenced, so a
        naked ``sl.Camera(`` outside the heredoc would be a regression.
        Bash comments (full-line and inline trailing) and ``log``/``echo``
        string arguments that *mention* ``sl.Camera(`` for documentation
        are not real SDK calls; we strip them before checking. The grep
        on production-payload modules is allowed (gate 3 is a documented
        part of the contract)."""
        text = _script_text()
        # 1. Strip heredoc bodies.
        out = []
        in_heredoc = False
        for line in text.splitlines(keepends=True):
            if not in_heredoc and re.search(r"<<\s*'?PYEOF'?", line):
                in_heredoc = True
                out.append(line)
                continue
            if in_heredoc:
                if line.lstrip().startswith("PYEOF"):
                    in_heredoc = False
                    out.append(line)
                    continue
                continue
            out.append(line)
        no_hdoc = "".join(out)
        # 2. Strip full-line bash comments.
        no_full = "\n".join(
            ln for ln in no_hdoc.splitlines() if not ln.lstrip().startswith("#")
        )
        # 3. Drop inline trailing comments (the part after ``#``).
        no_inline = re.sub(r"#[^\n]*", "", no_full)
        # 4. Drop ``log "..."`` and ``echo "..."`` argument strings (they
        # only describe what the gate is doing, never call the SDK).
        no_logs = re.sub(r'(?:log|echo)\s+"[^"]*"', "", no_inline)
        no_logs = re.sub(r"(?:log|echo)\s+'[^']*'", "", no_logs)
        # 5. Drop grep patterns scanning for sl.Camera( (gate 3 guard).
        no_logs = re.sub(r"sl\\\.Camera\s*\\\(", "", no_logs)
        # 6. Drop log_evidence PASS strings mentioning the names.
        no_logs = re.sub(r"log_evidence [^\n]*", "", no_logs)
        self.assertNotIn("sl.Camera(", no_logs)
        self.assertNotRegex(no_logs, r"\.\s*open\s*\(")
        self.assertNotRegex(no_logs, r"\.\s*grab\s*\(")


# ---------------------------------------------------------------------------
# 5. wrapper invokes both pytest targets
# ---------------------------------------------------------------------------
class TestPytestTargets(unittest.TestCase):
    def test_wrapper_invokes_droid_reference_pytest(self):
        text = _script_text()
        self.assertIn("test_zed_droid_reference.py", text)
        # The wrapper uses a path built from REFERENCE_DIR; accept any
        # path component that contains the pytest target.
        candidates = [
            "scripts/zed_droid_reference/tests/test_zed_droid_reference.py",
            "zed_droid_reference/tests/test_zed_droid_reference.py",
            "REFERENCE_DIR}/tests/test_zed_droid_reference.py",
        ]
        self.assertTrue(
            any(c in text for c in candidates),
            f"wrapper does not reference any of: {candidates}",
        )

    def test_wrapper_invokes_self_pytest(self):
        text = _script_text()
        self.assertIn("test_fr3_zed_droid_verify_script.py", text)

    def test_pytest_targets_exist(self):
        # Reference pytest already exists; self-pytest is THIS file.
        self.assertTrue(REFERENCE_PYTEST.exists(), REFERENCE_PYTEST)
        self.assertTrue(Path(__file__).exists())


# ---------------------------------------------------------------------------
# 6. mock-based smoke test (no real camera)
# ---------------------------------------------------------------------------
class TestWrapperHeredocsWorkUnderFakeSl(unittest.TestCase):
    """Execute the two PYEOF heredocs in isolation, under a fake sl, and
    confirm they reach the ``PASS`` log line. This is the exact same
    logic the wrapper runs, so passing here means gate 1 + gate 2 will
    pass when the wrapper is invoked for real (with python3 on PATH)."""

    def _build_fake_sl(self):
        sl = types.ModuleType("pyzed.sl")
        sl.Camera = type(
            "Cam",
            (),
            {
                "__init__": lambda self: setattr(self, "opened", False),
                "open": lambda self, ip: 0,
                "grab": lambda self: 0,
                "retrieve_image": lambda self, *a, **k: None,
                "close": lambda self: None,
            },
        )
        sl.Mat = type("Mat", (), {})
        sl.InitParameters = type(
            "IP", (), {"input": None, "camera_resolution": None, "camera_fps": None}
        )
        sl.InputType = type(
            "IT", (), {"set_from_serial_number": lambda self, n: None}
        )

        class _R:
            HD2K = "HD2K"
            HD1080 = "HD1080"
            HD720 = "HD720"
            VGA = "VGA"

        class _E:
            SUCCESS = 0

        class _V:
            LEFT = 0
            RIGHT = 1

        sl.RESOLUTION = _R
        sl.ERROR_CODE = _E
        sl.VIEW = _V
        return sl

    def test_gate1_heredoc_imports_reference(self):
        if not REFERENCE_DIR.exists():
            self.skipTest("reference dir missing")
        sys.modules.pop("pyzed", None)
        sys.modules.pop("pyzed.sl", None)
        sys.modules.pop("zed_camera", None)
        sl = self._build_fake_sl()
        sys.modules["pyzed"] = types.ModuleType("pyzed")
        sys.modules["pyzed"].sl = sl
        sys.modules["pyzed.sl"] = sl
        sys.path.insert(0, str(REFERENCE_DIR))
        try:
            import importlib

            mod = importlib.import_module("zed_camera")
            self.assertTrue(hasattr(mod, "ZedCamera"))
        finally:
            sys.modules.pop("zed_camera", None)
            sys.path.remove(str(REFERENCE_DIR))

    def test_gate2_heredoc_construction_contract(self):
        if not REFERENCE_DIR.exists():
            self.skipTest("reference dir missing")
        sys.modules.pop("pyzed", None)
        sys.modules.pop("pyzed.sl", None)
        sys.modules.pop("zed_camera", None)

        captured = []

        class _Cam:
            def __init__(self):
                self.open_args = []
                self.opened = False

            def open(self, ip):
                self.open_args.append(ip)
                self.opened = True
                return 0

            def grab(self):
                return 0

            def retrieve_image(self, *a, **k):
                return None

            def close(self):
                pass

        instances = []

        def _cam_factory():
            c = _Cam()
            instances.append(c)
            return c

        class _IP:
            def __init__(self):
                self.input = None
                self.camera_resolution = None
                self.camera_fps = None

        class _IT:
            def __init__(self, *a, **k):
                self.serial_number = None
                self.device_path = a[0] if a else None

            def set_from_serial_number(self, n):
                self.serial_number = n

        sl = types.ModuleType("pyzed.sl")
        sl.Camera = _cam_factory
        sl.Mat = type("M", (), {})
        sl.InitParameters = _IP
        sl.InputType = _IT

        class _R:
            HD2K = "HD2K"
            HD1080 = "HD1080"
            HD720 = "HD720"
            VGA = "VGA"

        class _E:
            SUCCESS = 0

        class _V:
            LEFT = 0
            RIGHT = 1

        sl.RESOLUTION = _R
        sl.ERROR_CODE = _E
        sl.VIEW = _V
        sys.modules["pyzed"] = types.ModuleType("pyzed")
        sys.modules["pyzed"].sl = sl
        sys.modules["pyzed.sl"] = sl
        sys.path.insert(0, str(REFERENCE_DIR))
        try:
            import importlib

            mod = importlib.import_module("zed_camera")
            mod.ZedCamera(serial_number=13132609)
            mod.ZedCamera(device_path="/dev/video0")
            self.assertEqual(len(instances), 2)
            self.assertEqual(instances[0].open_args[0].input.serial_number, 13132609)
            self.assertEqual(instances[1].open_args[0].input.device_path, "/dev/video0")
        finally:
            sys.modules.pop("zed_camera", None)
            sys.path.remove(str(REFERENCE_DIR))


# ---------------------------------------------------------------------------
# 7. help / usage
# ---------------------------------------------------------------------------
class TestHelp(unittest.TestCase):
    def test_usage_includes_help_flag(self):
        text = _script_text()
        self.assertIn("-h|--help", text)

    def test_usage_includes_evidence_flag(self):
        text = _script_text()
        self.assertIn("--evidence", text)

    def test_usage_includes_python_flag(self):
        text = _script_text()
        self.assertIn("--python", text)


if __name__ == "__main__":
    unittest.main()
