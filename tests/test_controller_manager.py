"""Controller recovery obeys ownership, launch exclusion, and runtime identity."""
from dataclasses import replace
import json
import os
import time
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.files import atomic_json
from hilserl.processes import Manager, lease, device_lease


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [])
    monkeypatch.setattr("hilserl.processes.process_start", lambda *_: "same")
    return Manager(replace(Config(), root=tmp_path, min_free_gib=0))


def test_recovery_holds_device_and_launch_leases_but_preserves_learner(manager, monkeypatch):
    learner = {"role": "learner", "pid": 7}
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [learner])
    def recover(_self):
        assert manager._controller_info()["phase"] == "recovering"
        with pytest.raises(RuntimeError):
            with lease(manager.config.control_root / "launch.lock"):
                pass
        with pytest.raises(RuntimeError):
            with device_lease(manager.config.root):
                pass
        return {"phase": "ready", "reason": "fresh", "health": {"state_sequence": 3}}
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", recover)
    monkeypatch.setattr("hilserl.processes.os.kill", lambda *_: pytest.fail("Learner must not be signalled"))
    result = manager.recover_controller()
    assert result["phase"] == "ready" and result["message"] == "fresh"
    audit = list((manager.config.control_root / "controller-operations").glob("*.json"))
    assert len(audit) == 1 and json.loads(audit[0].read_text())["phase"] == "ready"


@pytest.mark.parametrize("role", ["train", "eval", "collect", "legacy_device"])
def test_live_device_blocks_controller_restart(manager, monkeypatch, role):
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [{"role": role}])
    recover = Mock()
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", recover)
    with pytest.raises(ValueError, match="停止 Actor"):
        manager.recover_controller()
    recover.assert_not_called()


def test_unrelated_server_configuration_cannot_restart_fr3(manager, monkeypatch):
    manager.config = replace(manager.config, server_url="http://simulator.test:5017/")
    recover = Mock()
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", recover)
    with pytest.raises(ValueError, match="配置"):
        manager.recover_controller()
    recover.assert_not_called()


def test_runtime_transport_failure_is_not_success(manager, monkeypatch):
    result = {"phase": "failed", "reason": "result unknown", "transport": {"delivery": "unknown"}}
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", lambda *_: result)
    response = manager.recover_controller()
    assert response["phase"] == "failed" and response["error"] == "result unknown"
    assert response["transport"]["delivery"] == "unknown"


def test_read_health_cannot_overwrite_recovery_that_started_during_http(manager, monkeypatch):
    def read(_self):
        manager._save_controller({"phase": "recovering", "owner_pid": os.getpid(), "owner_start": "same"})
        return {"phase": "ready", "reason": "previous health"}
    monkeypatch.setattr("hilserl.controller.ControllerService.status", read)
    assert manager.controller_status()["phase"] == "recovering"


def test_expired_health_cannot_remain_ready(manager):
    manager.config.control_root.mkdir(parents=True)
    atomic_json(manager.config.control_root / "controller.json", {
        "phase": "ready", "unix_ns": time.time_ns() - 11_000_000_000,
        "health": {"state_age_seconds": .001}})
    value = manager._controller_info()
    assert value["phase"] == "unknown" and value["health"] is None


def test_lost_recovery_owner_reports_unknown_outcome(manager):
    manager._save_controller({"phase": "recovering", "owner_pid": 999, "owner_start": "old"})
    assert manager._controller_info()["phase"] == "failed"
    assert "结果未知" in manager._controller_info()["error"]


def test_admin_panel_update_does_not_invalidate_live_learner_but_policy_update_does(manager):
    root = manager.config.root
    (root / "hilserl").mkdir()
    (root / "_run_actor.py").write_text("policy-v1")
    (root / "hilserl/processes.py").write_text("panel-v1")
    run = root / "run"; run.mkdir()
    learner = {"pid": 17, "start_time": "same", "role": "learner", "run_dir": str(run)}
    atomic_json(run / "learner.json", dict(learner, source_sha256=manager._source_hashes()))
    (root / "hilserl/processes.py").write_text("panel-v2")
    (root / "hilserl/controller.py").write_text("new admin feature")
    manager._validate_fingerprints(learner=learner)
    (root / "_run_actor.py").write_text("policy-v2")
    with pytest.raises(RuntimeError, match="source fingerprint"):
        manager._validate_fingerprints(learner=learner)
