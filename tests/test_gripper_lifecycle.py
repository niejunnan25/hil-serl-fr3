"""Manual gripper commands must respect device ownership and Actor phase gates."""
from dataclasses import replace
import json
import time
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.control import OperatorControl, StopRequested, send_command
from hilserl.files import atomic_json
from hilserl.processes import Manager, device_lease


def idle_operator(tmp_path):
    operator = OperatorControl(tmp_path / "operator", terminal=False)
    operator.publish("waiting_reset", gate_id="gate", gripper_available=True)
    operator.gripper_handler = Mock(return_value={"verification": "motion_observed", "command_acknowledged": True})
    return operator


def test_actor_performs_manual_gripper_only_on_same_idle_gate(tmp_path):
    operator = idle_operator(tmp_path)
    item = send_command(operator.directory, "gripper", "close")
    assert operator.poll({"continue"}) is None
    operator.gripper_handler.assert_called_once_with("close")
    assert operator.state["phase"] == "waiting_reset"
    assert operator.state["gripper"]["command_id"] == item["command_id"]
    assert operator.state["gripper"]["verification"] == "motion_observed"


@pytest.mark.parametrize("change", ["phase", "gate", "expiry"])
def test_queued_gripper_cannot_execute_after_gate_change(tmp_path, change):
    operator = idle_operator(tmp_path)
    send_command(operator.directory, "gripper", "open")
    if change == "phase":
        operator.publish("resetting")
    elif change == "gate":
        operator.publish("waiting_reset", gate_id="next")
    else:
        request = next((operator.directory / "commands").iterdir())
        data = json.loads(request.read_text())
        atomic_json(request, dict(data, unix_ns=time.time_ns() - 11_000_000_000))
    operator.poll({"continue"})
    operator.gripper_handler.assert_not_called()
    assert operator.state["gripper"]["verification"] == "rejected"


def test_stop_takes_priority_over_queued_gripper(tmp_path):
    operator = idle_operator(tmp_path)
    send_command(operator.directory, "gripper", "close")
    send_command(operator.directory, "stop")
    with pytest.raises(StopRequested):
        operator.poll({"continue"})
    operator.gripper_handler.assert_not_called()


@pytest.mark.parametrize("phase", ["starting", "resetting", "collecting", "awaiting_label", "fault"])
def test_gripper_rejected_while_actor_is_not_idle(tmp_path, phase):
    operator = idle_operator(tmp_path)
    operator.publish(phase)
    with pytest.raises(ValueError):
        send_command(operator.directory, "gripper", "open")
    operator.gripper_handler.assert_not_called()


def test_gripper_error_is_reported_without_ending_waiting_actor(tmp_path):
    operator = idle_operator(tmp_path)
    operator.gripper_handler.side_effect = RuntimeError("controller is stale")
    send_command(operator.directory, "gripper", "close")
    assert operator.poll({"continue"}) is None
    assert operator.state["phase"] == "waiting_reset"
    assert operator.state["gripper"]["verification"] == "error"
    assert "stale" in operator.state["gripper"]["message"]


def test_idle_manager_uses_device_lease_and_persists_receipt(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path)
    monkeypatch.setattr("hilserl.processes.discover", lambda *args: [])
    command = Mock(return_value={"operation": "close", "verification": "not_verified"})
    monkeypatch.setattr("hilserl.gripper.GripperControl.command", command)
    result = Manager(cfg).gripper("close")
    assert result["verification"] == "not_verified"
    assert json.loads((cfg.control_root / "gripper.json").read_text()) == result
    with device_lease(cfg.root), pytest.raises(RuntimeError, match="占用"):
        Manager(cfg).gripper("open")
    command.assert_called_once_with("close")


def test_manager_routes_live_actor_gripper_through_ipc(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path)
    operator = OperatorControl(tmp_path / "run/control", terminal=False)
    operator.publish("paused", gate_id="gate", gripper_available=True)
    actor = {"role": "train", "run_dir": str(tmp_path / "run"), "pid": operator.state["pid"],
             "start_time": "test", "env": {"HILSERL_ATTEMPT_ID": operator.state["attempt_id"]}}
    operator.publish(start_time="test")
    monkeypatch.setattr("hilserl.processes.process_start", lambda *args: "test")
    monkeypatch.setattr("hilserl.processes.discover", lambda *args: [actor])
    command = Mock(side_effect=AssertionError("Manager must not send Actor device I/O"))
    monkeypatch.setattr("hilserl.gripper.GripperControl.command", command)
    result = Manager(cfg).gripper("open", expected=operator.state)
    assert result["verification"] == "queued"
    assert len(list((operator.directory / "commands").iterdir())) == 1
    command.assert_not_called()


def test_stale_actor_page_cannot_fall_back_to_idle_device_command(tmp_path, monkeypatch):
    monkeypatch.setattr("hilserl.processes.discover", lambda *args: [])
    command = Mock()
    monkeypatch.setattr("hilserl.gripper.GripperControl.command", command)
    with pytest.raises(ValueError, match="旧页面"):
        Manager(replace(Config(), root=tmp_path)).gripper("open", expected={"attempt_id": "old"})
    command.assert_not_called()
