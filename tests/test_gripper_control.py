"""Gripper I/O contract tests; every HTTP exchange and wait is simulated."""
import io
import json
from types import SimpleNamespace
import urllib.error
import urllib.parse

import pytest

import hilserl.gripper as gripper
from hilserl.gripper import GripperControl, GripperControlError


class Response:
    def __init__(self, body, status=200):
        self.body = (json.dumps(body) if isinstance(body, dict) else body).encode()
        self.status = status

    def read(self, size):
        return self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ScriptedOpener:
    def __init__(self, exchanges):
        self.exchanges = list(exchanges)
        self.calls = []
        self.last_state = None

    def open(self, request, timeout):
        endpoint = urllib.parse.urlsplit(request.full_url).path.rsplit("/", 1)[-1]
        self.calls.append((endpoint, timeout))
        assert request.get_method() == "POST"
        assert request.data == b"{}"
        assert 0 < timeout <= 2.0
        if self.exchanges:
            expected, value = self.exchanges.pop(0)
            assert endpoint == expected
        else:
            assert endpoint == "getstate", "A device command must never be retried"
            assert self.last_state is not None
            value = self.last_state
        if isinstance(value, Exception):
            raise value
        if endpoint == "getstate" and isinstance(value, dict):
            self.last_state = value
        return Response(value)


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def control(monkeypatch, exchanges, **kwargs):
    clock = Clock()
    monkeypatch.setattr(gripper, "time", SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep))
    item = GripperControl("http://robot.test:5000/", **kwargs)
    item._opener = ScriptedOpener(exchanges)
    return item


def state(position, *, fresh=True, **fields):
    freshness = dict(state_age_seconds=0.01, state_stale=False,
                     gripper_state_age_seconds=0.01, gripper_state_stale=False) if fresh else {}
    return {"gripper_pos": position, **freshness, **fields}


def ack(operation, **fields):
    return {"success": True, "command_acknowledged": True, "operation": operation, **fields}


def endpoints(item):
    return [name for name, _ in item._opener.calls]


def test_construction_does_no_io_and_disables_proxies_and_redirects(monkeypatch):
    handlers = []
    sentinel = object()
    monkeypatch.setattr(gripper.urllib.request, "build_opener", lambda *args: handlers.extend(args) or sentinel)
    item = GripperControl("http://robot.test:5000/")
    assert item._opener is sentinel
    proxy = next(handler for handler in handlers if isinstance(handler, gripper.urllib.request.ProxyHandler))
    assert proxy.proxies == {}
    redirect = next(handler for handler in handlers if isinstance(handler, gripper._NoRedirect))
    assert redirect.redirect_request(None, None, 302, "Found", {}, "http://other.test/") is None


def test_status_uses_normalized_width_without_assuming_sensor_freshness(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.567286, fresh=False))])
    observed = item.status()
    assert observed["gripper_pos"] == pytest.approx(0.567286)
    assert observed["width_m"] == pytest.approx(0.04538288)
    assert observed["width_mm"] == pytest.approx(45.38288)
    assert observed["freshness"] == "unknown"
    assert observed["sensor_timestamp_available"] is False
    assert endpoints(item) == ["getstate"]


def test_status_accepts_server_monotonic_ages_and_ignores_wall_clock_guessing(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.5, state_age_seconds=0.01, state_stale=False,
                                                   gripper_state_age_seconds=0.02, gripper_state_stale=False,
                                                   timestamp=9999999999999))])
    observed = item.status()
    assert observed["freshness"] == "fresh"
    assert observed["gripper_state_age_seconds"] == 0.02


def test_fresh_arm_state_does_not_prove_gripper_sensor_freshness(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.5, fresh=False, state_age_seconds=0.01, state_stale=False))])
    assert item.status()["freshness"] == "unknown"


@pytest.mark.parametrize("fields", [
    {}, {"state_age_seconds": 0.01, "state_stale": False},
    {"gripper_state_age_seconds": 0.01}, {"gripper_state_stale": False},
])
def test_unknown_gripper_freshness_cannot_trigger_any_motion(monkeypatch, fields):
    item = control(monkeypatch, [("getstate", state(0.5, fresh=False, **fields))])
    with pytest.raises(GripperControlError, match="夹爪状态新鲜度未确认") as caught:
        item.command("close")
    assert caught.value.result["before"]["freshness"] == "unknown"
    assert caught.value.result["command_acknowledged"] is False
    assert endpoints(item) == ["getstate"]


@pytest.mark.parametrize("value", [None, True, "0.5", [], float("nan"), float("inf"), -0.1, 1.26])
def test_invalid_reading_cannot_trigger_a_command(monkeypatch, value):
    item = control(monkeypatch, [("getstate", state(value))])
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    assert caught.value.result["command_acknowledged"] is False
    assert endpoints(item) == ["getstate"]


@pytest.mark.parametrize("fields", [
    {"state_stale": True}, {"gripper_state_stale": True},
    {"state_age_seconds": 7200.0}, {"gripper_state_age_seconds": 10.0},
    {"state_age_seconds": -1.0}, {"gripper_state_age_seconds": float("nan")},
    {"state_stale": "false"}, {"success": False, "error": "ROS state stale"},
])
def test_stale_or_failed_state_prevents_device_io(monkeypatch, fields):
    item = control(monkeypatch, [("getstate", state(0.5, **fields))])
    with pytest.raises(GripperControlError) as caught:
        item.command("open")
    assert caught.value.result["command_acknowledged"] is False
    assert endpoints(item) == ["getstate"]


def test_close_with_object_reports_motion_and_partial_width_without_grasp_claim(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.95)), ("close_gripper", ack("close")),
                                ("getstate", state(0.56))])
    result = item.command("close")
    assert result["command_acknowledged"] is True
    assert result["motion_observed"] is True
    assert result["verification"] == "motion_observed"
    assert result["after"]["width_mm"] == pytest.approx(44.8)
    assert result["width_change_mm"] == pytest.approx(-31.2)
    assert "success" not in result and "grasp_success" not in result
    assert result["duration_seconds"] == pytest.approx(2.0)
    assert endpoints(item).count("close_gripper") == 1
    assert all(name in {"getstate", "close_gripper"} for name in endpoints(item))


def test_open_observes_increasing_width_and_uses_legacy_ack(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.3)), ("open_gripper", "Opened"),
                                ("getstate", state(1.0))])
    result = item.command("open")
    assert result["command_acknowledged"] is True
    assert result["motion_observed"] is True
    assert result["width_change_mm"] == pytest.approx(56.0)
    assert endpoints(item).count("open_gripper") == 1


@pytest.mark.parametrize("operation, reply", [("open", "Opened"), ("close", "Closed")])
def test_latched_noop_cannot_be_reported_as_completed(monkeypatch, operation, reply):
    item = control(monkeypatch, [("getstate", state(0.57)), (operation + "_gripper", reply),
                                ("getstate", state(0.57))])
    result = item.command(operation)
    assert result["command_acknowledged"] is True
    assert result["motion_observed"] is False
    assert result["verification"] == "not_verified"
    assert "动作完成未确认" in result["message"]
    assert endpoints(item).count(operation + "_gripper") == 1


@pytest.mark.parametrize("after", [0.6, 0.559])
def test_wrong_direction_and_submillimeter_noise_do_not_verify_close(monkeypatch, after):
    item = control(monkeypatch, [("getstate", state(0.56)), ("close_gripper", "Closed"),
                                ("getstate", state(after))])
    assert item.command("close")["verification"] == "not_verified"


def test_command_timeout_preserves_uncertain_delivery_and_never_retries(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.9)), ("close_gripper", TimeoutError("late ACK")),
                                ("getstate", state(0.5))])
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    result = caught.value.result
    assert result["command_acknowledged"] is None
    assert result["before"]["width_mm"] == pytest.approx(72.0)
    assert result["after"]["width_mm"] == pytest.approx(40.0)
    assert result["verification"] == "not_verified"
    assert endpoints(item) == ["getstate", "close_gripper", "getstate"]


def test_read_timeout_after_ack_keeps_ack_but_does_not_verify(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.9)), ("close_gripper", "Closed"),
                                ("getstate", TimeoutError("no new state"))])
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    assert caught.value.result["command_acknowledged"] is True
    assert caught.value.result["after"] is None
    assert caught.value.result["verification"] == "not_verified"
    assert endpoints(item) == ["getstate", "close_gripper", "getstate"]


def test_stale_after_ack_preserves_the_stale_observation(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.9)), ("close_gripper", "Closed"),
                                ("getstate", state(0.5, gripper_state_stale=True))])
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    assert caught.value.result["command_acknowledged"] is True
    assert caught.value.result["after"]["freshness"] == "stale"
    assert caught.value.result["verification"] == "not_verified"


def test_unknown_freshness_after_ack_cannot_verify_motion(monkeypatch):
    item = control(monkeypatch, [("getstate", state(0.9)), ("close_gripper", "Closed"),
                                ("getstate", state(0.5, fresh=False))])
    with pytest.raises(GripperControlError, match="新鲜度未确认") as caught:
        item.command("close")
    assert caught.value.result["command_acknowledged"] is True
    assert caught.value.result["after"]["freshness"] == "unknown"
    assert caught.value.result["verification"] == "not_verified"
    assert endpoints(item).count("close_gripper") == 1


@pytest.mark.parametrize("reply", [
    "OK", '{"success":true}', ack("open"),
    {"success": False, "command_acknowledged": False, "operation": "close", "error": "ROS unavailable"},
])
def test_http_200_without_matching_ack_cannot_report_command_success(monkeypatch, reply):
    item = control(monkeypatch, [("getstate", state(0.9)), ("close_gripper", reply),
                                ("getstate", state(0.9))])
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    assert caught.value.result["command_acknowledged"] is None
    assert caught.value.result["verification"] == "not_verified"
    assert endpoints(item).count("close_gripper") == 1


@pytest.mark.parametrize("stage", ["getstate", "close_gripper"])
def test_http_503_is_failure_and_never_issues_an_extra_command(monkeypatch, stage):
    error = urllib.error.HTTPError("http://robot.test/" + stage, 503, "Service Unavailable", {},
                                   io.BytesIO(b'{"success":false,"error":"stale"}'))
    exchanges = [(stage, error)] if stage == "getstate" else [
        ("getstate", state(0.9)), (stage, error), ("getstate", state(0.9))]
    item = control(monkeypatch, exchanges)
    with pytest.raises(GripperControlError) as caught:
        item.command("close")
    assert "503" in str(caught.value)
    if stage == "getstate":
        assert caught.value.result["command_acknowledged"] is False
        assert caught.value.result["acknowledgement"] is None
        assert endpoints(item) == ["getstate"]
    else:
        assert caught.value.result["command_acknowledged"] is None
        assert caught.value.result["acknowledgement"]["status_code"] == 503
        assert endpoints(item).count("close_gripper") == 1


@pytest.mark.parametrize("operation", [None, "reset", "home", "toggle", "OPEN", "close_gripper"])
def test_only_explicit_open_close_is_allowed(monkeypatch, operation):
    item = control(monkeypatch, [])
    with pytest.raises(ValueError):
        item.command(operation)
    assert endpoints(item) == []


@pytest.mark.parametrize("url", ["file:///tmp/device", "http://user:pass@robot.test/",
                                "http://robot.test/?endpoint=home", "http://robot.test/#home"])
def test_invalid_server_url_is_rejected(url):
    with pytest.raises(ValueError):
        GripperControl(url)
