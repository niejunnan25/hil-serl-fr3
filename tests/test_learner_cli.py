"""CLI contracts with a fake Manager; never start a process or contact a device."""
import json
from types import SimpleNamespace

import pytest

import hilserl.__main__ as cli


@pytest.fixture
def manager(monkeypatch, tmp_path):
    calls = []
    runtime = dict(pid=123, start_time="start-123", attempt_id="learner-one", phase="paused")

    class Manager:
        def __init__(self, config):
            pass

        def launch(self, mode, **kwargs):
            calls.append(("launch", mode, kwargs))
            return {"phase": "accepted"}

        def status(self):
            return {"learner_runtime": runtime}

        def learner_activity(self, operation, *, expected):
            calls.append(("activity", operation, expected))
            return {"command": operation}

    config = SimpleNamespace(python="python", path=lambda value: tmp_path / "absent-python")
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(cli, "Manager", Manager)
    return calls, runtime


def test_train_explicit_resume_and_manual_pause_are_passed_separately(manager, capsys):
    calls, _ = manager
    assert cli.main(["train", "--learner-only", "--resume-checkpoint", "/run/checkpoints/checkpoint_25974", "--learner-paused"]) == 0
    assert calls == [("launch", "train", {"learner_only": True, "resume_checkpoint": "/run/checkpoints/checkpoint_25974", "learner_paused": True})]
    assert json.loads(capsys.readouterr().out)["phase"] == "accepted"


def test_plain_train_keeps_fresh_training_default(manager):
    calls, _ = manager
    assert cli.main(["train", "--learner-only"]) == 0
    assert calls == [("launch", "train", {"learner_only": True, "resume_checkpoint": None, "learner_paused": False})]


@pytest.mark.parametrize("operation", ["pause", "resume"])
def test_activity_uses_current_exact_runtime_identity(manager, operation):
    calls, _ = manager
    assert cli.main(["learner-activity", operation]) == 0
    assert calls == [("activity", operation, {"pid": 123, "start_time": "start-123", "attempt_id": "learner-one"})]


@pytest.mark.parametrize("field,value", [("pid", True), ("pid", None), ("start_time", ""), ("attempt_id", None)])
def test_activity_without_current_identity_never_dispatches(manager, capsys, field, value):
    calls, runtime = manager
    runtime[field] = value
    assert cli.main(["learner-activity", "pause"]) == 1
    assert calls == []
    assert "身份已确认" in capsys.readouterr().err


def test_resume_checkpoint_flag_is_only_a_train_option(manager):
    with pytest.raises(SystemExit) as exc:
        cli.main(["collect", "--resume-checkpoint", "/run/checkpoint_0"])
    assert exc.value.code == 2
