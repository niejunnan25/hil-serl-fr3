"""A stop request is not a checkpoint save, and an exited PID is not a save receipt."""
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.files import atomic_json
from hilserl.learner_control import LearnerControl
from hilserl.processes import Manager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path, min_free_gib=0)
    run = tmp_path / "run"; run.mkdir()
    learner = dict(pid=100, start_time="same", attempt_id="attempt-a", role="learner",
                   run_dir=str(run), checkpoint=str(run / "checkpoints"),
                   env={"HILSERL_ATTEMPT_ID": "attempt-a"})
    atomic_json(run / "learner.json", dict(learner, learner_control_protocol=1))
    alive = [learner]
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: list(alive))
    monkeypatch.setattr("hilserl.processes.process_start", lambda *_: "same")
    control = LearnerControl(run, pid=100, start_time="same", attempt_id="attempt-a")
    control.publish("training", step=7)
    return Manager(cfg), control, alive


def test_modern_stop_is_persistent_idempotent_and_does_not_emit_sigint(setup, monkeypatch):
    manager, control, _ = setup
    kill = Mock(side_effect=AssertionError("modern stop must not use async exceptions"))
    monkeypatch.setattr("hilserl.processes.os.kill", kill)
    first = manager.stop_learner()
    raw = control.request_path.read_bytes()
    second = manager.stop_learner()
    assert first["phase"] == second["phase"] == "stop_requested" and second["repeated"]
    assert control.request_path.read_bytes() == raw
    kill.assert_not_called()
    assert manager.status()["learner_policy"]["state"] == "stop_requested"
    assert manager.status()["learner_runtime"]["phase"] == "training"


def test_status_follows_owner_ack_save_and_close_in_order(setup):
    manager, control, alive = setup
    manager.stop_learner()
    assert control.stop_requested()
    assert manager.status()["learner_runtime"]["stop_requested"] is True
    control.publish("saving_checkpoint", step=7, checkpoint_path=control.run_dir / "checkpoints")
    assert manager.status()["learner_policy"]["state"] == "saving_checkpoint"
    control.publish("closing", checkpoint_path=control.run_dir / "checkpoints/checkpoint_7", checkpoint_saved=True)
    assert manager.status()["learner_policy"]["state"] == "closing"
    control.publish("stopped")
    assert manager.status()["learner_policy"]["state"] == "closing"  # PID still alive.
    alive.clear()
    status = manager.status()
    assert status["launch"]["phase"] == "stopped"
    assert status["learner_runtime"]["checkpoint_saved"] is True
    assert "完成保存" in status["launch"]["reason"]


def test_save_failure_survives_process_exit(setup):
    manager, control, alive = setup
    manager.stop_learner()
    control.publish("fault", checkpoint_error="disk failure")
    alive.clear()
    status = manager.status()
    assert status["launch"]["phase"] == "fault"
    assert "disk failure" in status["launch"]["error"]
    assert status["learner_runtime"]["checkpoint_saved"] is False


def test_stale_learner_status_cannot_ack_new_process(setup):
    manager, control, _ = setup
    manager.stop_learner()
    old = dict(control.state, start_time="previous-process", phase="saving_checkpoint")
    atomic_json(control.status_path, old)
    status = manager.status()
    assert status["learner_runtime"] is None
    assert status["learner_policy"]["state"] == "stop_requested"


def test_restart_cannot_race_a_learner_shutdown(setup):
    manager, control, _ = setup
    manager.stop_learner()
    with pytest.raises(RuntimeError, match="停止并退出"):
        manager.launch()
    assert control.request_path.is_file()
