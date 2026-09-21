"""Continuation preserves the run/config and requires an explicit committed save."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.files import atomic_json, read_json
from hilserl.learner_control import LearnerControl
from hilserl.processes import Manager


@pytest.fixture
def saved(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path, data_dir="runs", min_free_gib=0, seed=7,
                  batch_size=128, max_episode_steps=125, classifier_ckpt="relative-classifier")
    manager = Manager(cfg)
    run = manager._new_run("train")
    checkpoint = run / "checkpoints" / "checkpoint_27"
    checkpoint.mkdir(parents=True)
    atomic_json(checkpoint / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1234})
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [])
    monkeypatch.setattr("hilserl.processes.process_start", lambda *_: "live")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: set() if pid is None else {5588, 5589})
    manager.preflight = lambda **kw: {"ok": True}
    manager._spawn = Mock(return_value=SimpleNamespace(pid=100))
    return manager, run, checkpoint, cfg


def test_explicit_resume_reuses_run_checkpoint_and_original_full_config(saved):
    manager, run, checkpoint, cfg = saved
    manager.config = replace(cfg, batch_size=256, max_episode_steps=190, seed=0)
    original = (run / "run.json").read_bytes()
    result = manager.launch(learner_only=True, resume_checkpoint=str(checkpoint), learner_paused=True)
    assert result["run_dir"] == str(run)
    manager._spawn.assert_called_once_with("learner", run, checkpoint.parent, cfg,
                                          resume_step=27, learner_paused=True)
    assert (run / "run.json").read_bytes() == original
    assert len(list(cfg.output_root.iterdir())) == 1


@pytest.mark.parametrize("problem", ["uncommitted", "not_latest", "config_changed", "wrong_mode"])
def test_invalid_continuation_never_spawns(saved, problem):
    manager, run, checkpoint, _ = saved
    if problem == "uncommitted":
        atomic_json(checkpoint / "_CHECKPOINT_METADATA", {})
    elif problem == "not_latest":
        (checkpoint.parent / "checkpoint_28").mkdir()
    elif problem == "config_changed":
        config = read_json(run / "config.json")
        atomic_json(run / "config.json", dict(config, seed=99))
    else:
        record = read_json(run / "run.json")
        atomic_json(run / "run.json", dict(record, mode="eval"))
    with pytest.raises((RuntimeError, ValueError)):
        manager.launch(learner_only=True, resume_checkpoint=str(checkpoint))
    manager._spawn.assert_not_called()
    assert manager._resume_candidate({}) is None


def test_resume_refuses_existing_process_and_implicit_old_directory(saved, monkeypatch):
    manager, _, checkpoint, _ = saved
    with pytest.raises(RuntimeError, match="新训练"):
        manager.launch(learner_only=True, checkpoint=str(checkpoint.parent))
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [{"role": "learner"}])
    with pytest.raises(RuntimeError, match="先结束"):
        manager.launch(learner_only=True, resume_checkpoint=str(checkpoint))
    manager._spawn.assert_not_called()


def test_candidate_points_to_committed_step(saved):
    manager, run, checkpoint, _ = saved
    assert manager.status()["resume_candidate"] == {
        "checkpoint": str(checkpoint), "step": 27, "run_dir": str(run)}


@pytest.fixture
def live(saved, monkeypatch):
    manager, run, checkpoint, cfg = saved
    process = dict(pid=100, start_time="live", role="learner", run_dir=str(run),
                   checkpoint=str(checkpoint.parent), args=["python", "_run_actor.py", "--exp_name=plug_insertion"],
                   env={"HILSERL_ATTEMPT_ID": "new", "HILSERL_CLASSIFIER_CKPT": str(cfg.path(cfg.classifier_ckpt))})
    record = dict(process, attempt_id="new", config_file=str(run / "config.json"), config_sha256=cfg.digest())
    atomic_json(run / "learner.json", record)
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [process])
    control = LearnerControl(run, pid=100, start_time="live", attempt_id="new")
    control.publish("paused", step=27, pause_reason="Actor 已退出，等待新数据。")
    return manager, process, control, cfg


def test_actor_restart_uses_full_learner_snapshot_not_absolute_environment(live):
    manager, process, _, cfg = live
    assert manager._learner_config(process) == cfg
    assert manager._learner_config(process).digest() == cfg.digest()


def test_auto_pause_is_reported_truthfully_and_activity_only_targets_owner(live, monkeypatch):
    manager, process, control, _ = live
    monkeypatch.setattr("hilserl.processes.os.kill", Mock(side_effect=AssertionError("no signals")))
    state = manager.status()
    assert state["learner_policy"] == {"state": "paused", "reason": control.state["pause_reason"]}
    assert state["resume_candidate"] is None
    identity = {key: control.state[key] for key in ("pid", "start_time", "attempt_id")}
    manager.learner_activity("pause", expected=identity)
    assert control.activity_paused()
    manager.learner_activity("resume", expected=identity)
    assert not control.activity_paused()
    request = control.activity_path.read_bytes()
    with pytest.raises(ValueError, match="身份"):
        manager.learner_activity("pause", expected=dict(identity, attempt_id="old"))
    assert control.activity_path.read_bytes() == request


def test_pause_checkpoint_is_not_final_shutdown_and_cannot_override_stop(live):
    manager, process, control, cfg = live
    control.publish("saving_checkpoint", step=27)
    assert manager._learner_config(process) == cfg
    assert "暂停" in manager.status()["learner_policy"]["reason"]
    manager.stop_learner()
    identity = {key: control.state[key] for key in ("pid", "start_time", "attempt_id")}
    with pytest.raises(ValueError, match="停止"):
        manager.learner_activity("resume", expected=identity)


def test_spawn_records_explicit_resume_in_new_attempt(saved, monkeypatch):
    manager, run, checkpoint, cfg = saved
    del manager._spawn
    popen = Mock(return_value=SimpleNamespace(pid=100))
    monkeypatch.setattr("hilserl.processes.subprocess.Popen", popen)
    manager.launch_state = {"request_id": "test"}
    manager._spawn("learner", run, checkpoint.parent, cfg, resume_step=27, learner_paused=True)
    argv = popen.call_args.args[0]
    assert "--resume_checkpoint_step=27" in argv and "--learner_paused" in argv
    record = read_json(run / "learner.json")
    assert record["resume_checkpoint_step"] == 27
    assert record["initial_paused"] is True
    assert Path(record["config_file"]).parent.name == record["attempt_id"]
