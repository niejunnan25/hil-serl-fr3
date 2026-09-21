"""Stopped Actor status follows its identity, independently of the Learner run."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.files import atomic_json, read_json
from hilserl.processes import Manager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    cfg = replace(Config(), root=tmp_path, data_dir="runs", min_free_gib=0)
    cfg.control_root.mkdir(parents=True)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [])
    monkeypatch.setattr("hilserl.processes.os.kill", Mock(side_effect=AssertionError("no signals")))
    return Manager(cfg)


def archived_actor(manager, name, *, pid, start, attempt, phase, **fields):
    run = manager.config.output_root / name
    (run / "control").mkdir(parents=True)
    snapshot = run / "config.json"
    atomic_json(snapshot, {"source": name})
    record = dict(pid=pid, start_time=start, attempt_id=attempt,
                  run_dir=str(run), config_file=str(snapshot))
    atomic_json(run / "actor.json", record)
    atomic_json(run / "control/status.json", dict(record, phase=phase, **fields))
    return record


def select_launch(manager, record, run_dir):
    atomic_json(manager.config.control_root / "launch.json", {
        "phase": "stopped", "run_dir": run_dir,
        "actor_pid": record["pid"], "actor_start": record["start_time"],
    })


@pytest.mark.parametrize("phase,error", [("stopped", None), ("fault", "new drop_plug_fatal")])
def test_stopped_status_uses_actor_identity_after_learner_run_replaces_launch(manager, phase, error):
    old = archived_actor(manager, "old-train", pid=10, start="old", attempt="old",
                         phase="fault", error="old drop_plug_fatal")
    current = archived_actor(manager, "new-collect", pid=20, start="current", attempt="current",
                             phase=phase, error=error, completed_episodes=20)
    select_launch(manager, current, old["run_dir"])
    launch_before = (manager.config.control_root / "launch.json").read_bytes()

    status = manager.status()

    assert status["actor"]["attempt_id"] == "current"
    assert status["actor"]["phase"] == phase
    assert status["actor"]["error"] == error
    assert status["actor"]["completed_episodes"] == 20
    assert status["effective_config"] == {"source": "new-collect"}
    assert (manager.config.control_root / "launch.json").read_bytes() == launch_before


@pytest.mark.parametrize("problem", ["reused_pid", "different_attempt", "missing_start", "no_actor_identity"])
def test_archived_state_without_matching_identity_is_not_displayed(manager, problem):
    record = archived_actor(manager, "run", pid=20, start="current", attempt="current",
                            phase="fault", error="old drop_plug_fatal")
    select_launch(manager, record, record["run_dir"])
    launch_path = manager.config.control_root / "launch.json"
    launch = read_json(launch_path)
    if problem == "reused_pid":
        launch["actor_start"] = "other-process"
    elif problem == "different_attempt":
        status_path = manager.config.output_root / "run/control/status.json"
        atomic_json(status_path, dict(read_json(status_path), attempt_id="other-attempt"))
    elif problem == "missing_start":
        launch.pop("actor_start")
    else:
        launch.pop("actor_start")
        launch.pop("actor_pid")
    atomic_json(launch_path, launch)

    assert manager.status()["actor"] is None


def test_live_actor_keeps_priority_over_archived_launch(manager, monkeypatch):
    old = archived_actor(manager, "old", pid=10, start="old", attempt="old", phase="fault")
    current = archived_actor(manager, "current", pid=20, start="current", attempt="current",
                             phase="waiting_reset", completed_episodes=20)
    select_launch(manager, old, old["run_dir"])
    live = dict(current, role="collect", env={"HILSERL_ATTEMPT_ID": "current",
                                            "HILSERL_CONFIG_SNAPSHOT": current["config_file"]})
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [live])

    status = manager.status()

    assert status["actor"]["attempt_id"] == "current"
    assert status["actor"]["phase"] == "waiting_reset"
    assert status["effective_config"] == {"source": "current"}


def test_stop_learner_and_reconcile_keep_latest_collection_visible(manager, monkeypatch):
    old = archived_actor(manager, "old-train", pid=10, start="old", attempt="old",
                         phase="fault", error="old drop_plug_fatal")
    current = archived_actor(manager, "new-collect", pid=20, start="current", attempt="current",
                             phase="stopped", completed_episodes=20, error=None)
    select_launch(manager, current, current["run_dir"])
    learner = dict(pid=30, start_time="learner", role="learner", run_dir=old["run_dir"],
                   attempt_id="learner", env={"HILSERL_ATTEMPT_ID": "learner"})
    atomic_json(manager.config.output_root / "old-train/learner.json",
                dict(learner, learner_control_protocol=1))
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "learner")

    manager.stop_learner()
    assert read_json(manager.config.control_root / "launch.json")["run_dir"] == old["run_dir"]
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [])
    status = manager.status()

    assert status["launch"]["phase"] == "stopped"
    assert status["actor"]["attempt_id"] == "current"
    assert status["actor"]["completed_episodes"] == 20
    assert status["actor"]["error"] is None
