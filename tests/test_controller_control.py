"""Controller control contracts with simulated HTTP and SSH; no robot I/O."""
import io
import json
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

import hilserl.controller as controller
from hilserl.controller import ControllerService


def health(**changes):
    return {
        "controller_running": True, "pose_subscribers": 1,
        "state_age_seconds": 0.02, "state_stale": False,
        "jacobian_age_seconds": 0.01, "jacobian_stale": False,
        "gripper_state_age_seconds": 0.03, "gripper_state_stale": False,
        "state_sequence": 31, "jacobian_sequence": 44, "gripper_state_sequence": 20,
        "ready": True, "motion_available": True, "robot_mode_name": "Idle",
        **changes,
    }


class Response:
    def __init__(self, body, status=200):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status = status

    def read(self, size):
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class Opener:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request.full_url, request.get_method(), timeout))
        assert request.full_url == "http://robot.test:5000/health"
        assert request.get_method() == "GET"
        assert request.data is None
        assert timeout == 2.0
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def service(value=None):
    item = ControllerService("http://robot.test:5000/")
    item._opener = Opener(Response(health()) if value is None else value)
    return item


def receipt(**changes):
    return {"phase": "ready", "pid": 1234, "log": "/tmp/controller.log",
            "health": health(), "error": None, **changes}


def ssh(monkeypatch, value=None):
    calls = []
    if value is None:
        value = SimpleNamespace(returncode=0, stdout=json.dumps(receipt()), stderr="")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command == [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "fr3-laptop",
            "python3", "/home/robot/serl_projects/hilserl-fr3-control/controller_lifecycle.py", "recover",
        ]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert 0 < kwargs["timeout"] <= 90
        assert "shell" not in kwargs
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(controller.subprocess, "run", run)
    return calls


def test_construction_and_status_cannot_start_a_controller(monkeypatch):
    monkeypatch.setattr(controller.subprocess, "run", lambda *_a, **_k: pytest.fail("SSH is a mutation"))
    item = service()
    assert item._opener.calls == []
    result = item.status()
    assert result["phase"] == "ready"
    assert result["availability"] == "supported"
    assert result["health"] == health()
    assert item._opener.calls == [("http://robot.test:5000/health", "GET", 2.0)]


def test_requests_disable_proxies_and_redirects(monkeypatch):
    handlers = []
    monkeypatch.setattr(controller.urllib.request, "build_opener", lambda *args: handlers.extend(args))
    ControllerService("http://robot.test:5000/")
    proxy = next(h for h in handlers if isinstance(h, controller.urllib.request.ProxyHandler))
    redirect = next(h for h in handlers if isinstance(h, controller._NoRedirect))
    assert proxy.proxies == {}
    assert redirect.redirect_request(None, None, 302, "Found", {}, "http://other.test/") is None


@pytest.mark.parametrize("field", [
    "controller_running", "pose_subscribers", "state_age_seconds", "state_stale",
    "jacobian_age_seconds", "jacobian_stale", "gripper_state_age_seconds", "gripper_state_stale",
])
def test_missing_health_evidence_is_legacy_and_never_ready(field):
    value = health()
    value.pop(field)
    result = service(Response(value)).status()
    assert result["phase"] == "failed"
    assert result["availability"] == "legacy"
    assert field in result["reason"]


@pytest.mark.parametrize("changes", [
    {"controller_running": False}, {"controller_running": 1},
    {"pose_subscribers": 0}, {"pose_subscribers": None}, {"pose_subscribers": True},
    {"pose_subscribers": "1"}, {"pose_subscribers": -1},
    {"state_stale": True}, {"jacobian_stale": True}, {"gripper_state_stale": True},
    {"state_stale": "false"}, {"jacobian_stale": 0}, {"gripper_state_stale": None},
    {"state_age_seconds": 2.001}, {"jacobian_age_seconds": 2.001},
    {"gripper_state_age_seconds": 2.001}, {"state_age_seconds": -0.1},
    {"jacobian_age_seconds": float("nan")}, {"gripper_state_age_seconds": float("inf")},
    {"state_age_seconds": True}, {"jacobian_age_seconds": "0.1"},
    {"success": False, "error": "unavailable"},
    {"ready": False}, {"motion_available": False}, {"ready": "true"},
    {"motion_available": False, "robot_mode_name": "UserStop"},
    {"motion_available": False, "robot_mode_name": "Guiding"},
    {"motion_available": False, "robot_mode_name": "Reflex"},
])
def test_liveness_without_valid_current_streams_or_motion_mode_cannot_be_ready(changes):
    result = service(Response(health(**changes))).status()
    assert result["phase"] == "failed"
    assert result["reason"]
    assert result["health"] is not None


def test_all_streams_at_two_second_bound_are_accepted():
    value = health(state_age_seconds=2.0, jacobian_age_seconds=2.0, gripper_state_age_seconds=2.0)
    assert service(Response(value)).status()["phase"] == "ready"


def test_aggregate_not_ready_explains_the_actual_stale_stream():
    value = health(ready=False, motion_available=False,
                   gripper_state_stale=True, gripper_state_age_seconds=43.75)
    result = service(Response(value, status=503)).status()
    assert "夹爪反馈" in result["reason"]
    assert "43.8" in result["reason"]


@pytest.mark.parametrize("value", [
    Response(b"OK"), Response([]), Response(b"x" * 65537),
    TimeoutError("health timed out"), urllib.error.URLError("offline"),
])
def test_unavailable_or_invalid_endpoint_never_implies_health(value):
    result = service(value).status()
    assert result["phase"] == "failed"
    assert result["availability"] == "unavailable"


def test_http_503_preserves_health_but_never_reports_ready():
    error = urllib.error.HTTPError("http://robot.test:5000/health", 503, "Unavailable", {},
                                   io.BytesIO(json.dumps(health()).encode()))
    result = service(error).status()
    assert result["phase"] == "failed"
    assert result["http_status"] == 503
    assert result["health"] == health()


def test_legacy_bridge_without_health_route_has_no_endpoint_fallback():
    error = urllib.error.HTTPError("http://robot.test:5000/health", 404, "Not Found", {},
                                   io.BytesIO(b"<html>Not Found</html>"))
    item = service(error)
    result = item.status()
    assert result["phase"] == "failed"
    assert result["availability"] == "legacy"
    assert len(item._opener.calls) == 1


def test_recovery_requires_fixed_helper_ready_receipt_and_a_current_health_read(monkeypatch):
    calls = ssh(monkeypatch)
    item = service()
    result = item.recover()
    assert result["phase"] == "ready"
    assert result["receipt"] == receipt()
    assert result["health"] == health()
    assert result["transport"]["delivery"] == "completed"
    assert result["transport"]["returncode"] == 0
    assert len(calls) == len(item._opener.calls) == 1


def test_receipt_on_last_nonempty_line_can_follow_helper_diagnostics(monkeypatch):
    completed = SimpleNamespace(returncode=0, stdout="Stopping old process\n" + json.dumps(receipt()) + "\n\n", stderr="")
    ssh(monkeypatch, completed)
    assert service().recover()["phase"] == "ready"


@pytest.mark.parametrize("reply", [
    b"", b"Started", b"{}", b"[]", b'{"phase":"ready"}',
    json.dumps(receipt(phase="starting")).encode(),
    json.dumps(receipt(phase="failed", error="controller exited")).encode(),
    json.dumps(receipt(health={})).encode(),
    json.dumps(receipt(health=health(state_stale=True))).encode(),
    json.dumps(receipt(health=health(pose_subscribers=0))).encode(),
    json.dumps(receipt(health=health(motion_available=False))).encode(),
    json.dumps(receipt(error="conflicting failure")).encode(),
])
def test_ssh_zero_and_healthy_http_do_not_override_an_unverified_helper_receipt(monkeypatch, reply):
    calls = ssh(monkeypatch, SimpleNamespace(returncode=0, stdout=reply, stderr=""))
    result = service().recover()
    assert result["phase"] == "failed"
    assert result["reason"]
    assert result["health"] == health()
    assert len(calls) == 1


@pytest.mark.parametrize("value", [
    Response(health(state_stale=True)), Response(health(gripper_state_age_seconds=9.0)),
    Response(health(pose_subscribers=0)), Response(health(ready=False)),
    Response(health(motion_available=False)), Response({}), TimeoutError("HTTP unavailable"),
])
def test_ready_receipt_requires_current_health_after_recovery(monkeypatch, value):
    calls = ssh(monkeypatch)
    result = service(value).recover()
    assert result["phase"] == "failed"
    assert "实时健康检查未通过" in result["reason"]
    assert len(calls) == 1


def test_timeout_keeps_unknown_delivery_even_if_later_http_and_partial_receipt_are_ready(monkeypatch):
    error = subprocess.TimeoutExpired("ssh", 90, output=json.dumps(receipt()).encode(), stderr=b"late exit")
    calls = ssh(monkeypatch, error)
    item = service()
    result = item.recover()
    assert result["phase"] == "failed"
    assert result["transport"] == {
        "delivery": "unknown", "returncode": None, "timed_out": True, "stderr": "late exit",
    }
    assert result["receipt"] == receipt()
    assert result["health"] == health()
    assert "结果未知" in result["reason"]
    assert len(calls) == len(item._opener.calls) == 1


def test_helper_failure_retains_receipt_exit_code_and_error_without_retry(monkeypatch):
    value = receipt(phase="failed", error="FR3 is UserStopped")
    error = subprocess.CalledProcessError(1, "ssh", output=json.dumps(value), stderr="details")
    calls = ssh(monkeypatch, error)
    result = service().recover()
    assert result["phase"] == "failed"
    assert result["receipt"] == value
    assert result["transport"]["returncode"] == 1
    assert result["transport"]["stderr"] == "details"
    assert result["transport"]["delivery"] == "completed"
    assert "UserStopped" in result["reason"]
    assert len(calls) == 1


def test_connection_failure_without_receipt_preserves_unknown_delivery(monkeypatch):
    calls = ssh(monkeypatch, subprocess.CalledProcessError(255, "ssh", stderr="connection closed"))
    result = service().recover()
    assert result["phase"] == "failed"
    assert result["receipt"] is None
    assert result["transport"]["delivery"] == "unknown"
    assert len(calls) == 1


def test_local_ssh_start_failure_is_not_sent(monkeypatch):
    calls = ssh(monkeypatch, FileNotFoundError("ssh unavailable"))
    result = service().recover()
    assert result["phase"] == "failed"
    assert result["transport"]["delivery"] == "not_sent"
    assert len(calls) == 1


def test_a_second_recovery_is_rejected_instead_of_queued(monkeypatch):
    calls = ssh(monkeypatch)
    item = service()
    with item._recovery_lock:
        result = item.recover()
    assert result["phase"] == "failed"
    assert result["transport"]["delivery"] == "not_sent"
    assert calls == []


@pytest.mark.parametrize("url", [
    "file:///tmp/controller", "http://user:pass@robot.test/", "http://robot.test/?command=home",
    "http://robot.test/#home", "http://robot.test:invalid/",
])
def test_invalid_server_url_is_rejected(url):
    with pytest.raises(ValueError):
        ControllerService(url)


@pytest.mark.parametrize("kwargs", [
    {"ssh_host": "other-host"}, {"ssh_host": "-oProxyCommand=sh"},
    {"script": "/tmp/command.py"}, {"script": "/tmp/a; robot home"},
])
def test_no_arbitrary_remote_host_or_command_can_be_substituted(kwargs):
    with pytest.raises(ValueError):
        ControllerService("http://robot.test:5000/", **kwargs)
