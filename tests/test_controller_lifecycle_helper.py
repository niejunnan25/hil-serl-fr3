"""Controller recovery helper contracts; no ROS, SSH, HTTP or robot actions.

Set HILSERL_CONTROLLER_LIFECYCLE_HELPER to validate another staged helper copy.
The real recovery function runs against fake process, clock and HTTP boundaries.
"""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest


HELPER = Path(os.environ.get(
    "HILSERL_CONTROLLER_LIFECYCLE_HELPER",
    "/Users/tacyvan/Code/hilserl-controller-recovery-20260912/controller_lifecycle.py",
))


def health(state=10, jacobian=20, gripper=30, **changes):
    return {
        "ready": True,
        "motion_available": True,
        "state_sequence": state,
        "jacobian_sequence": jacobian,
        "gripper_state_sequence": gripper,
        **changes,
    }


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 0.3
        self.sleeps.append(seconds)
        self.now += seconds


class Opener:
    def __init__(self):
        self.values = [{"ready": False}]
        self.calls = []

    def open(self, value, timeout):
        request = urllib.request.Request(value) if isinstance(value, str) else value
        self.calls.append((request.full_url, request.get_method(), request.data, timeout))
        assert self.calls[-1] == ("http://127.0.0.1:5000/health", "GET", None, 2)
        result = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        if callable(result):
            result = result(len(self.calls))
        if isinstance(result, BaseException):
            raise result
        return io.StringIO(json.dumps(result))


class Child:
    pid = 9876

    def __init__(self):
        self.returncode = None
        self.poll_results = [None]

    def poll(self):
        value = self.poll_results.pop(0) if len(self.poll_results) > 1 else self.poll_results[0]
        self.returncode = value
        return value


@pytest.fixture
def rig(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("controller_lifecycle_under_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = tmp_path / "control-project"
    root = base / "src/hil-serl/serl_robot_infra"
    control = base / "artifacts/controller"
    (root / "robot_servers").mkdir(parents=True)
    source = root / "robot_servers/franka_server.py"
    source.write_text("# inert fixture: never executed\n")
    clock, opener, child = Clock(), Opener(), Child()
    item = SimpleNamespace(module=module, base=base, root=root, control=control,
                           source=source, clock=clock, opener=opener, child=child,
                           rows={}, kills=[], launches=[], on_kill=None)

    def table():
        return {pid: dict(row) for pid, row in item.rows.items()}

    def kill(pid, sent_signal):
        item.kills.append((pid, sent_signal))
        if item.on_kill is not None:
            item.on_kill(pid, sent_signal)

    def popen(command, **kwargs):
        item.launches.append((command, kwargs))
        assert command == [
            "/usr/bin/python3", "-u", "robot_servers/franka_server.py",
            "--robot_ip=172.16.0.2", "--gripper_type=Franka", "--flask_url=0.0.0.0",
        ]
        assert kwargs["cwd"] == root
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["start_new_session"] is True
        assert "shell" not in kwargs
        assert kwargs["stdout"].name.startswith(str(control / "service-"))
        return child

    monkeypatch.setattr(module, "BASE", base)
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "CONTROL", control)
    monkeypatch.setattr(module, "time", clock)
    monkeypatch.setattr(module, "OPENER", opener)
    monkeypatch.setattr(module, "process_table", table)
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(
        Popen=popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(module.os, "kill", kill)
    return item


def process(rig, pid=123, parent=1, start="12345", state="S", args=None, cwd=None):
    return {
        "pid": pid, "parent": parent, "start": start, "state": state,
        "args": ["python3", "robot_servers/franka_server.py"] if args is None else args,
        "cwd": str(rig.root if cwd is None else cwd),
    }


def startup_health(rig, *values):
    rig.opener.values = [{"ready": False}, *values]


def receipt(rig):
    return json.loads((rig.control / "latest.json").read_text())


@pytest.mark.parametrize("args", [
    ["/opt/ros/noetic/bin/roscore"],
    ["/opt/ros/noetic/bin/rosmaster"],
    ["/opt/ros/noetic/bin/roslaunch", "franka_control.launch"],
    ["/opt/ros/noetic/lib/franka_control/franka_control_node"],
    ["/opt/ros/noetic/lib/franka_gripper/franka_gripper_node"],
    ["python3", "/another/project/franka_server.py"],
    ["python3", "/another/project/droid/launch.py"],
    ["python3", "-u", "/another/project/droid/launch.py"],
])
def test_foreign_robot_or_ros_process_refuses_recovery_before_any_io(rig, args):
    rig.rows[456] = process(rig, pid=456, args=args, cwd="/another/project")
    with pytest.raises(RuntimeError, match="Foreign robot/ROS process"):
        rig.module.recover()
    assert rig.kills == []
    assert rig.launches == []
    assert rig.opener.calls == []


def test_owned_service_and_transitive_children_are_allowed(rig):
    rig.rows = {
        123: process(rig),
        124: process(rig, pid=124, parent=123, args=["roslaunch", "franka_control.launch"]),
        125: process(rig, pid=125, parent=124, args=["franka_control_node"]),
        126: process(rig, pid=126, parent=124, args=["franka_gripper_node"]),
    }
    rig.on_kill = lambda *_args: rig.rows.clear()
    startup_health(rig, health(), health(11, 21, 31), health(12, 22, 32))
    result = rig.module.recover()
    assert result["phase"] == "ready"
    assert result["previous_pids"] == [123]
    assert rig.kills == [(123, signal.SIGTERM)]
    assert len(rig.launches) == 1


def test_healthy_owned_service_is_reused_without_signal_or_spawn(rig):
    rig.rows[123] = process(rig)
    rig.opener.values = [health()]
    result = rig.module.recover()
    assert result["phase"] == "ready"
    assert result["reused"] is True
    assert result["pid"] == 123
    assert rig.kills == []
    assert rig.launches == []
    assert len(rig.opener.calls) == 1
    assert receipt(rig) == result


def test_multiple_owned_services_are_rejected_without_io(rig):
    rig.rows = {123: process(rig), 124: process(rig, pid=124)}
    with pytest.raises(RuntimeError, match="Multiple HIL-SERL services"):
        rig.module.recover()
    assert rig.kills == rig.launches == rig.opener.calls == []


@pytest.mark.parametrize("replacement", [None, "987654"])
def test_service_identity_changed_before_stop_cannot_signal_or_launch(rig, monkeypatch, replacement):
    original = process(rig)
    tables = [{123: original}, {} if replacement is None else {123: {**original, "start": replacement}}]
    monkeypatch.setattr(rig.module, "process_table", lambda: tables.pop(0))
    with pytest.raises(RuntimeError, match="Service identity changed before stop"):
        rig.module.recover()
    assert rig.kills == []
    assert rig.launches == []


def test_service_stop_timeout_never_force_kills_or_launches_a_duplicate(rig):
    rig.rows[123] = process(rig)
    with pytest.raises(RuntimeError, match="Previous service did not exit"):
        rig.module.recover()
    assert rig.kills == [(123, signal.SIGTERM)]
    assert rig.launches == []
    assert rig.clock.now >= 118
    assert len(rig.opener.calls) == 1


def test_foreign_process_appearing_after_stop_prevents_new_launch(rig):
    rig.rows[123] = process(rig)

    def replace_rows(*_args):
        rig.rows = {999: process(rig, pid=999, args=["roscore"], cwd="/foreign")}

    rig.on_kill = replace_rows
    with pytest.raises(RuntimeError, match="Foreign robot/ROS process"):
        rig.module.recover()
    assert rig.kills == [(123, signal.SIGTERM)]
    assert rig.launches == []


def test_startup_requires_three_advancing_health_samples_and_records_source(rig):
    startup_health(rig, health(2, 20, 200), health(4, 23, 202), health(7, 25, 205))
    result = rig.module.recover()
    assert result["phase"] == "ready"
    assert result["health"] == health(7, 25, 205)
    assert len(rig.opener.calls) == 4
    assert rig.clock.sleeps == [0.3, 0.3]
    assert rig.kills == []
    assert len(rig.launches) == 1
    assert result["source_sha256"] == hashlib.sha256(rig.source.read_bytes()).hexdigest()
    assert receipt(rig) == result


@pytest.mark.parametrize("interruption", [
    health(12, 22, 32, ready=False),
    health(12, 22, 32, motion_available=False),
    health(12, 22, 32, ready="true"),
    health(12, 22, 32, state_sequence=True),
    health(12, 22, 32, jacobian_sequence=22.0),
    health(12, 22, 32, gripper_state_sequence=None),
    OSError("health service unavailable"),
])
def test_unhealthy_or_invalid_sample_restarts_the_three_sample_requirement(rig, interruption):
    startup_health(rig, health(), health(11, 21, 31), interruption,
                   health(13, 23, 33), health(14, 24, 34), health(15, 25, 35))
    result = rig.module.recover()
    assert result["phase"] == "ready"
    assert len(rig.opener.calls) == 7
    assert result["health"] == health(15, 25, 35)


@pytest.mark.parametrize("field", ["state_sequence", "jacobian_sequence", "gripper_state_sequence"])
def test_any_frozen_sequence_prevents_ready_even_when_health_flags_are_true(rig, field):
    startup_health(rig, lambda count: health(count, count + 10, count + 20, **{field: 7}))
    result = rig.module.recover()
    assert result["phase"] == "failed"
    assert result["health"]["ready"] is True
    assert result["health"][field] == 7
    assert rig.clock.now >= 160
    assert len(rig.launches) == 1
    assert rig.kills == []
    assert receipt(rig)["phase"] == "failed"


def test_sequence_regression_resets_progress_instead_of_reusing_earlier_samples(rig):
    startup_health(rig, health(10, 20, 30), health(11, 21, 31), health(9, 22, 32),
                   health(10, 23, 33), health(11, 24, 34))
    result = rig.module.recover()
    assert result["phase"] == "ready"
    assert len(rig.opener.calls) == 6
    assert result["health"] == health(11, 24, 34)


@pytest.mark.parametrize("poll_results, expected_health_calls", [([42], 1), ([None, None, 42], 3)])
def test_child_exit_before_readiness_is_reported_as_failed(rig, poll_results, expected_health_calls):
    rig.child.poll_results = poll_results
    startup_health(rig, health(), health(11, 21, 31), health(12, 22, 32))
    result = rig.module.recover()
    assert result["phase"] == "failed"
    assert "exit=42" in result["error"]
    assert len(rig.opener.calls) == expected_health_calls
    assert len(rig.launches) == 1
    assert rig.kills == []
    assert receipt(rig)["phase"] == "failed"


def test_probe_preserves_nonready_health_json_from_http_error(rig):
    body = {"ready": False, "error": "controller unavailable"}
    rig.opener.values = [urllib.error.HTTPError(
        "http://127.0.0.1:5000/health", 503, "unavailable", {}, io.StringIO(json.dumps(body)))]
    assert rig.module.probe() == body
    assert rig.opener.calls == [("http://127.0.0.1:5000/health", "GET", None, 2)]
    assert rig.kills == rig.launches == []
