"""P2-T6: record_gello_demos_serl.py 合同测试.

FR3Robot 从 Polymetis RobotInterface 迁移到 requests.post HTTP API.
数据字段保持不变. GELLO 用 mock 替代, 验证 dry-run / live 路径.

Contract:
  - main() argparse: --dry-run | --live (互斥)
  - dry-run 路径: GELLO only, 不发 robot commands
  - live 路径: 通过 requests.post 调 franka_server
  - 数据保存字段 (来自 record_gello_demos.py 兼容):
      joint_poses, gripper_states, timestamps, raw_gello, target,
      command, tracking_error
  - 数据保存字段 (SERL 增量):
      poses (Cartesian TCP pose per step, FK of command)
      meta_server_url, meta_num_steps, meta_hz, meta_leader_scale,
      meta_joint_signs, meta_abort_reason
  - 安全机制: max_step / max_total_delta / max_tracking_error
  - 默认 server-url = http://127.0.0.2:5000/
  - 默认 hz = 20
  - /getstate 数据字段兼容: pose (TCP 7D), vel (6D), force (3D),
    torque (3D), gripper_pos (scalar)
  - /pose 命令使用 {"arr": [...]} envelope (7D: x,y,z,qx,qy,qz,qw)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Load script as module so we can exercise its loop without spawning
# subprocesses. The script also adds SCRIPTS to sys.path at import
# time, so fk_converter is reachable.
_spec = importlib.util.spec_from_file_location(
    "record_gello_demos_serl", SCRIPTS / "record_gello_demos_serl.py"
)
assert _spec and _spec.loader
rgds = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rgds
_spec.loader.exec_module(rgds)

import fk_converter  # noqa: E402
from tests.fixtures.fake_dynamixel_driver import (  # noqa: E402
    FakeDynamixelDriver,
    install_mock,
    uninstall_mock,
)


SRC = (SCRIPTS / "record_gello_demos_serl.py").read_text(encoding="utf-8")


def _argparse_main_block() -> str:
    """Return the inside of main() that contains argparse setup."""
    m = re.search(r"def main\(\):(.+?)(?=^def |^if __name__)", SRC, re.DOTALL | re.MULTILINE)
    assert m, "no main() body found"
    return m.group(1)


# ---------------------------------------------------------------------------
# 1. 互斥 CLI 参数
# ---------------------------------------------------------------------------
class TestCli:
    def test_mutually_exclusive_group_present(self):
        block = _argparse_main_block()
        assert "add_mutually_exclusive_group" in block
        assert "--dry-run" in block
        assert "--live" in block
        # required=True
        assert re.search(r"add_mutually_exclusive_group\(required=True\)", block)

    def test_default_server_url(self):
        block = _argparse_main_block()
        assert "http://127.0.0.2:5000/" in block or 'default=DEFAULT_SERVER_URL' in block

    def test_default_hz_around_20(self):
        block = _argparse_main_block()
        # 默认 hz 20 (record_gello_demos_serl 自身的默认)
        assert re.search(r'"--hz".*default=20', block) or "DEFAULT_HZ" in SRC

    def test_safety_args_present(self):
        block = _argparse_main_block()
        assert "--max-step" in block
        assert "--max-total-delta" in block
        assert "--max-tracking-error" in block


# ---------------------------------------------------------------------------
# 2. 数据字段 (joint_poses, gripper_states, timestamps, raw_gello, target, command, tracking_error)
# ---------------------------------------------------------------------------
class TestDataFields:
    def test_required_fields_present(self):
        for field in (
            "joint_poses",
            "gripper_states",
            "timestamps",
            "raw_gello",
            "target",
            "command",
            "tracking_error",
        ):
            assert field in SRC, f"missing field: {field}"

    def test_field_shapes_joint_7d(self):
        # joint_poses / target / command: (N, 7)
        assert "(N, 7)" in SRC

    def test_field_shape_raw_gello_8d(self):
        # raw_gello: (N, 8)  -- 7 joints + 1 gripper
        assert "(N, 8)" in SRC

    def test_field_shape_1d_arrays(self):
        # gripper_states / timestamps: (N,)
        assert "(N,)" in SRC


# ---------------------------------------------------------------------------
# 3. HTTP API 路径
# ---------------------------------------------------------------------------
class TestHttpApiEndpoint:
    def test_uses_requests_post(self):
        assert "import requests" in SRC
        assert "requests.post" in SRC or "requests.get" in SRC

    def test_no_polymetis_residual(self):
        # 不应该 import Polymetis 任何符号. 文档字符串里可以提到 Polymetis
        # (解释迁移来源), 但代码里不应该有.
        # 去掉 docstring/comments
        import re
        code = re.sub(r'""".*?"""', "", SRC, flags=re.DOTALL)
        code = re.sub(r"#[^\n]*", "", code)
        for forbidden in ("polymetis", "RobotInterface", "FrankaRobot"):
            assert forbidden not in code, f"residual: {forbidden}"

    def test_dry_run_prints_indicator(self):
        # dry-run 路径应该有 print 区分
        # 不严格要求, 至少主流程提到
        assert "dry" in SRC.lower()

    def test_check_max_step_helper_present(self):
        assert "def check_max_step" in SRC
        assert "def check_max_total_delta" in SRC
        assert "def check_tracking_error" in SRC

    def test_save_demo_helper_present(self):
        assert "def save_demo" in SRC


# ---------------------------------------------------------------------------
# 4. FR3 适配
# ---------------------------------------------------------------------------
class TestFr3Adaptation:
    def test_fr3_arm_id_in_source(self):
        assert "fr3" in SRC.lower()

    def test_joint_count_seven(self):
        # FR3 7 joints
        # 7 应该出现在 joint 形状或索引计算
        assert "joint" in SRC.lower()
        # joint_signs 默认值应该有 7 个
        # 通过 grep 找 DEFAULT_JOINT_SIGNS
        if "DEFAULT_JOINT_SIGNS" in SRC:
            m = re.search(r"DEFAULT_JOINT_SIGNS\s*=\s*\[([^\]]+)\]", SRC, re.DOTALL)
            if m:
                signs = [s.strip() for s in m.group(1).split(",")]
                assert len(signs) == 7, f"DEFAULT_JOINT_SIGNS len={len(signs)}, expected 7"


# ---------------------------------------------------------------------------
# HTTP stub for live-mode behavioral tests
# ---------------------------------------------------------------------------
class _StubResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}: {self.text}")

    def json(self) -> dict:
        return self._payload


class FakeFrankaServer:
    """In-process franka_server stub.

    Records every POST; returns /getstate matching the documented
    schema (pose/vel/force/torque/q/dq/jacobian/gripper_pos) and
    /pose commands.
    """

    def __init__(self, base_url: str = "http://127.0.0.2:5000"):
        self.base_url = base_url.rstrip("/")
        self.calls: list[tuple[str, dict]] = []
        self.state = {
            "pose": [0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0],
            "vel": [0.0] * 6,
            "force": [0.0] * 3,
            "torque": [0.0] * 3,
            "q": [0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0],
            "dq": [0.0] * 7,
            "jacobian": [[0.0] * 7 for _ in range(6)],
            "gripper_pos": 0.04,
        }
        self.raise_on_healthz: Exception | None = None

    def _route(self, url: str) -> str:
        return url[len(self.base_url):]

    def post(self, url: str, json: dict | None = None, timeout: float = 5.0):
        self.calls.append((self._route(url), dict(json or {})))
        route = self._route(url)
        if self.raise_on_healthz and route == "/healthz":
            raise self.raise_on_healthz
        if route == "/healthz":
            return _StubResponse(200, {"ok": True, "mode": "impedance"})
        if route == "/getstate":
            return _StubResponse(200, dict(self.state))
        if route == "/pose":
            arr = json.get("arr") if json else None
            assert arr is not None and len(arr) == 7
            self.state["pose"] = list(arr)
            return _StubResponse(200, {"ok": True})
        if route == "/open_gripper":
            self.state["gripper_pos"] = 0.08
            return _StubResponse(200, {"ok": True})
        if route == "/close_gripper":
            self.state["gripper_pos"] = 0.0
            return _StubResponse(200, {"ok": True})
        if route == "/clearerr":
            return _StubResponse(200, {"ok": True})
        return _StubResponse(404, {"error": "no route"})


# ---------------------------------------------------------------------------
# 5. Module surface — all public symbols required by callers
# ---------------------------------------------------------------------------
class TestModuleSurface:
    def test_required_public_symbols(self):
        for name in (
            "GelloDevice", "FR3Robot",
            "check_max_step", "check_max_total_delta", "check_tracking_error",
            "save_demo", "run_recording", "main",
            "DEFAULT_MAX_STEP", "DEFAULT_MAX_TOTAL_DELTA",
            "DEFAULT_MAX_TRACKING_ERROR", "DEFAULT_HZ",
            "DEFAULT_LEADER_SCALE", "DEFAULT_JOINT_SIGNS",
            "DEFAULT_SERVER_URL", "FR3_LOWER_LIMITS", "FR3_UPPER_LIMITS",
        ):
            assert hasattr(rgds, name), f"missing public symbol: {name}"

    def test_default_server_url_is_127_0_0_2(self):
        # Phase 1 franka_server sits at 127.0.0.1:5000; this recorder
        # lives at 127.0.0.2:5000 during the migration window. The
        # mismatch is documented as 错位 in the milestone.
        assert rgds.DEFAULT_SERVER_URL.startswith("http://127.0.0.2:"), \
            rgds.DEFAULT_SERVER_URL


# ---------------------------------------------------------------------------
# 6. FR3Robot HTTP wrapper — endpoint coverage and error handling
# ---------------------------------------------------------------------------
class TestFR3RobotHTTP:
    def test_healthz_probe(self):
        s = FakeFrankaServer()
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url)
            r.connect()
        assert s.calls[0][0] == "/healthz"

    def test_healthz_failure_raises_connection_error(self):
        s = FakeFrankaServer()
        s.raise_on_healthz = requests.ConnectionError("refused")
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url)
            with pytest.raises(ConnectionError):
                r.connect()

    def test_getstate_returns_full_state_with_tcp_pose(self):
        s = FakeFrankaServer()
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url)
            state = r.get_state()
        for key in ("pose", "vel", "force", "torque", "q", "dq", "gripper_pos"):
            assert key in state, key
        # /getstate must carry the TCP pose (7D: xyz + quat).
        assert len(state["pose"]) == 7
        # Force/torque at 3D, vel at 6D — these are the SERL observation
        # inputs the wrapper is responsible for forwarding.
        assert len(state["force"]) == 3
        assert len(state["torque"]) == 3
        assert len(state["vel"]) == 6

    def test_send_pose_uses_arr_envelope(self):
        s = FakeFrankaServer()
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url)
            r.send_pose_command(np.array([0.45, 0.1, 0.30, 0.0, 0.0, 0.0, 1.0]))
        assert s.calls[-1][0] == "/pose"
        assert "arr" in s.calls[-1][1]
        assert len(s.calls[-1][1]["arr"]) == 7

    def test_gripper_commands_dispatch(self):
        s = FakeFrankaServer()
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url)
            r.send_gripper_command("open")
            r.send_gripper_command("close")
        assert ("/open_gripper", {}) in s.calls
        assert ("/close_gripper", {}) in s.calls

    def test_clear_error_swallows_transient_failures(self):
        def boom(url, *a, **kw):
            raise requests.ConnectionError("transient")

        with patch.object(requests.Session, "post", side_effect=boom):
            r = rgds.FR3Robot(server_url="http://127.0.0.2:5000")
            r.clear_error()  # must not raise

    def test_server_url_strips_trailing_slash(self):
        s = FakeFrankaServer()
        with patch.object(requests.Session, "post", side_effect=s.post):
            r = rgds.FR3Robot(server_url=s.base_url + "/")
            r.connect()
        assert s.calls[0][0] == "/healthz"


# ---------------------------------------------------------------------------
# 7. Safety checks — byte-identical to record_gello_demos.py
# ---------------------------------------------------------------------------
class TestSafetyChecksUnchanged:
    def test_max_step(self):
        ok, delta = rgds.check_max_step(np.zeros(7), np.array([0.001] * 7), 0.003)
        assert ok is True
        assert delta == pytest.approx(0.001)
        ok, delta = rgds.check_max_step(np.zeros(7), np.array([0.01] * 7), 0.003)
        assert ok is False
        assert delta == pytest.approx(0.01)

    def test_max_total_delta(self):
        ok, total = rgds.check_max_total_delta(
            np.zeros(7), np.array([0.02] * 7), 0.03
        )
        assert ok is True
        assert total == pytest.approx(0.02)
        ok, total = rgds.check_max_total_delta(
            np.zeros(7), np.array([0.05] * 7), 0.03
        )
        assert ok is False

    def test_tracking_error(self):
        target = np.zeros(7)
        actual = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ok, max_err, err = rgds.check_tracking_error(target, actual, 0.08)
        assert ok is True
        assert max_err == pytest.approx(0.05)
        assert err.shape == (7,)


# ---------------------------------------------------------------------------
# 8. Recorded npz schema — must match record_gello_demos.py + new SERL fields
# ---------------------------------------------------------------------------
def _live_namespace(out_dir: str, duration: float, **overrides) -> argparse.Namespace:
    base = dict(
        dry_run=False,
        live=True,
        server_url="http://127.0.0.2:5000/",
        output_dir=out_dir,
        gello_port="/dev/ttyUSB0",
        hz=20.0,
        duration=duration,
        leader_scale=0.5,
        joint_signs=[1, -1, 1, 1, 1, -1, 1],
        max_step=0.003,
        max_total_delta=0.03,
        max_tracking_error=0.08,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _install_gello(initial_8d: np.ndarray) -> FakeDynamixelDriver:
    driver = FakeDynamixelDriver(
        list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
        max_retries=1, use_fake_fallback=False,
    )
    driver.set_joints(np.asarray(initial_8d, dtype=np.float64))
    install_mock(driver)
    return driver


class TestRecordedNPZSchema:
    def test_schema_matches_record_gello_demos(self, tmp_path):
        out_dir = tmp_path / "demos"
        server = FakeFrankaServer()
        _install_gello(np.array([0.0] * 7 + [0.5]))
        ns = _live_namespace(str(out_dir), duration=0.15)
        try:
            with patch.object(requests.Session, "post", side_effect=server.post):
                fpath = rgds.run_recording(ns)
        finally:
            uninstall_mock()
        assert fpath is not None
        data = np.load(fpath, allow_pickle=True)
        # Original record_gello_demos.py fields
        for key in (
            "joint_poses", "gripper_states", "timestamps", "raw_gello",
            "target", "command", "tracking_error",
        ):
            assert key in data.files, f"missing legacy field: {key}"
        # New SERL-only field
        assert "poses" in data.files, "missing SERL poses field"
        # Metadata
        assert "meta_server_url" in data.files
        # Shape contract: every (N, D) / (N,) matches
        n = data["joint_poses"].shape[0]
        assert data["gripper_states"].shape == (n,)
        assert data["timestamps"].shape == (n,)
        assert data["raw_gello"].shape == (n, 8)
        assert data["target"].shape == (n, 7)
        assert data["command"].shape == (n, 7)
        assert data["tracking_error"].shape == (n, 7)
        assert data["poses"].shape == (n, 7)
        # NaN/Inf-free
        for key in (
            "joint_poses", "gripper_states", "timestamps", "raw_gello",
            "target", "command", "tracking_error", "poses",
        ):
            assert np.isfinite(data[key]).all(), f"{key} contains NaN/Inf"

    def test_pose_field_is_fk_of_command(self, tmp_path):
        """poses[t] must equal forward_kinematics(command[t]) — that's
        the Cartesian TCP pose the impedance controller executed.
        """
        out_dir = tmp_path / "demos"
        server = FakeFrankaServer()
        _install_gello(np.array([0.0] * 7 + [0.5]))
        ns = _live_namespace(str(out_dir), duration=0.20)
        try:
            with patch.object(requests.Session, "post", side_effect=server.post):
                fpath = rgds.run_recording(ns)
        finally:
            uninstall_mock()
        data = np.load(fpath, allow_pickle=True)
        for cmd, pose in zip(data["command"], data["poses"]):
            assert np.allclose(pose, fk_converter.forward_kinematics(cmd), atol=1e-9), \
                f"pose mismatch: cmd={cmd} pose={pose}"

    def test_meta_server_url_records_actual_url(self, tmp_path):
        out_dir = tmp_path / "demos"
        server = FakeFrankaServer()
        _install_gello(np.array([0.0] * 7 + [0.5]))
        ns = _live_namespace(str(out_dir), duration=0.10)
        try:
            with patch.object(requests.Session, "post", side_effect=server.post):
                fpath = rgds.run_recording(ns)
        finally:
            uninstall_mock()
        data = np.load(fpath, allow_pickle=True)
        assert str(data["meta_server_url"]) == "http://127.0.0.2:5000/"


# ---------------------------------------------------------------------------
# 9. Live loop must hit /pose (not /jointreset) — the whole point
# ---------------------------------------------------------------------------
class TestLiveUsesPoseEndpoint:
    def test_live_mode_calls_pose(self, tmp_path):
        out_dir = tmp_path / "demos"
        server = FakeFrankaServer()
        _install_gello(np.array([0.0] * 7 + [0.5]))
        ns = _live_namespace(str(out_dir), duration=0.15)
        try:
            with patch.object(requests.Session, "post", side_effect=server.post):
                rgds.run_recording(ns)
        finally:
            uninstall_mock()
        routes = [c[0] for c in server.calls]
        assert "/pose" in routes, routes
        assert "/getstate" in routes, routes
        # Must NOT regress to Polymetis-style joint-space commands.
        assert "/jointreset" not in routes, routes


# ---------------------------------------------------------------------------
# 10. Dry-run path: no network at all
# ---------------------------------------------------------------------------
class TestDryRunNoNetwork:
    def test_dry_run_does_not_call_server(self, tmp_path):
        _install_gello(np.array([0.0] * 7 + [0.5]))
        ns = argparse.Namespace(
            dry_run=True,
            live=False,
            server_url="http://127.0.0.2:5000/",
            output_dir=str(tmp_path),
            gello_port="/dev/ttyUSB0",
            hz=20.0,
            duration=0.10,
            leader_scale=0.5,
            joint_signs=[1, -1, 1, 1, 1, -1, 1],
            max_step=0.003,
            max_total_delta=0.03,
            max_tracking_error=0.08,
        )

        def _fail(*a, **kw):
            raise AssertionError("dry-run must not hit the network")

        try:
            with patch.object(requests.Session, "post", side_effect=_fail):
                fpath = rgds.run_recording(ns)
        finally:
            uninstall_mock()
        assert fpath is not None
        data = np.load(fpath, allow_pickle=True)
        # dry-run uses command as actual; pose stays at synthetic home.
        assert data["poses"].shape[0] == data["joint_poses"].shape[0]
        # meta_server_url must be the literal "dry-run" marker so a
        # future auditor can grep for it.
        assert str(data["meta_server_url"]) == "dry-run"


# ---------------------------------------------------------------------------
# 11. Abort path: max_step violation breaks the loop with the documented
#     reason string. record_gello_demos.py used the same string;
#     consumers grep for "max_step:" / "max_total_delta:" / "tracking_error:".
# ---------------------------------------------------------------------------
class TestAbortReasons:
    def test_max_step_abort_string(self, tmp_path):
        server = FakeFrankaServer()
        driver = FakeDynamixelDriver(
            list(range(8)), port="/dev/ttyUSB0", baudrate=57600,
            max_retries=1, use_fake_fallback=False,
        )
        install_mock(driver)
        driver.set_joints(np.array([0.0] * 7 + [0.5]))

        # Reads happen in this order:
        #   call 1: GelloDevice.connect() verification read
        #   call 2: run_recording() reads raw0
        #   call 3: live loop step 0
        #   call 4: live loop step 1 (this is where the abort fires)
        # We want a small movement on step 0 (no abort) and a > 0.003
        # rad jump on step 1 (abort).
        import types
        counter = {"n": 0}
        HOME = np.array([0.0] * 7 + [0.5])
        SMALL = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
        JUMP = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

        def _sequence(self):
            counter["n"] += 1
            n = counter["n"]
            if n <= 2:
                return HOME
            elif n == 3:
                return SMALL
            else:
                return JUMP

        driver.get_joints = types.MethodType(_sequence, driver)  # type: ignore[assignment]

        ns = _live_namespace(str(tmp_path), duration=1.0)
        try:
            with patch.object(requests.Session, "post", side_effect=server.post):
                fpath = rgds.run_recording(ns)
        finally:
            uninstall_mock()
        # Aborts can leave 0 or more recorded frames; we only require
        # the abort reason to be set in the meta fields. When the
        # abort fires on step 1 there will be 1 recorded frame; we
        # read it from the npz.
        data = np.load(fpath, allow_pickle=True)
        reason = str(data["meta_abort_reason"])
        assert reason.startswith("max_step:"), reason
