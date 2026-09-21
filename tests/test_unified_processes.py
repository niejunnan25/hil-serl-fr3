"""Process orchestration checks; every process and network operation is a stub."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hilserl.config import Config, load_config
from hilserl.files import atomic_json
from hilserl.processes import Manager, discover


@pytest.fixture
def cfg(tmp_path):
    return replace(Config(),root=tmp_path,data_dir="runs",min_free_gib=0)


def pair(cfg):
    common={"AGENTLACE_PORT":"5588","AGENTLACE_BROADCAST_PORT":"5589",
            "HILSERL_CLASSIFIER_CKPT":str(cfg.path(cfg.classifier_ckpt))}
    checkpoint=str(cfg.root/"old-checkpoints")
    learner=dict(pid=100,start_time="same",role="learner",args=["python","_run_actor.py","--learner","--exp_name=plug_insertion","--seed=0"],
                 env=dict(common),checkpoint=checkpoint,run_dir=None)
    actor=dict(pid=101,start_time="same",role="train",args=["python","_run_actor.py","--actor","--exp_name=plug_insertion","--seed=0","--ip=127.0.0.1"],
               env=dict(common),checkpoint=checkpoint,run_dir=None)
    return learner,actor


@pytest.mark.parametrize("field,value",[
    ("experiment","other"),("ip","192.0.2.10"),("seed","8"),("port","9999"),
    ("classifier","other-classifier"),("checkpoint","/other-checkpoint")])
def test_running_pair_is_not_reused_with_mismatched_contract(cfg,monkeypatch,field,value):
    learner,actor=pair(cfg)
    if field in {"experiment","ip","seed"}:
        prefix={"experiment":"--exp_name=","ip":"--ip=","seed":"--seed="}[field]
        actor["args"]=[prefix+value if a.startswith(prefix) else a for a in actor["args"]]
    elif field=="port":actor["env"]["AGENTLACE_PORT"]=value
    elif field=="classifier":actor["env"]["HILSERL_CLASSIFIER_CKPT"]=value
    else:actor["checkpoint"]=value
    monkeypatch.setattr("hilserl.processes.discover",lambda root:[learner,actor])
    monkeypatch.setattr("hilserl.processes.process_start",lambda pid:"same")
    monkeypatch.setattr("hilserl.processes.listening_ports",lambda pid=None:{5588,5589})
    with pytest.raises(RuntimeError):
        Manager(cfg).launch()


def test_matching_pair_is_reused_without_spawning(cfg,monkeypatch):
    learner,actor=pair(cfg)
    monkeypatch.setattr("hilserl.processes.discover",lambda root:[learner,actor])
    monkeypatch.setattr("hilserl.processes.process_start",lambda pid:"same")
    monkeypatch.setattr("hilserl.processes.listening_ports",lambda pid=None:{5588,5589})
    spawn=Mock(side_effect=AssertionError("must not spawn"))
    manager=Manager(cfg);manager._spawn=spawn
    assert manager.launch()["reused"]
    assert not spawn.called


def test_config_cannot_inherit_hidden_safety_overrides(cfg,monkeypatch):
    for key in ("FRANKA_SAFETY_FORCE_MAX","FRANKA_CLEARERR_ON_POSE","HILSERL_RESET_CLEAR_Z","ACTOR_RELZ_ABS_MAX"):
        monkeypatch.setenv(key,"999")
    env=cfg.environment("actor",run_dir=cfg.root/"run")
    assert env["FRANKA_SAFETY_FORCE_MAX"]=="45.0"
    assert env["FRANKA_CLEARERR_ON_POSE"]=="0"
    assert env["HILSERL_RESET_CLEAR_Z"]=="0.15"
    assert env["ACTOR_RELZ_ABS_MAX"]=="0.35"


def test_missing_explicit_profile_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path/"misspelled.json")


def test_old_buffers_are_not_an_implicit_resume(tmp_path):
    (tmp_path/"buffer").mkdir()
    (tmp_path/"buffer/transitions_2.pkl").write_bytes(b"historical")
    with pytest.raises(RuntimeError,match="buffers"):
        Manager._fresh_checkpoint(tmp_path)


def test_ready_ports_do_not_bypass_process_identity(cfg,monkeypatch):
    learner,_=pair(cfg);calls=[]
    def identity(pid):
        if pid==100:
            calls.append(pid)
            return "same" if len(calls)==1 else "reused-pid"
        return "owner"
    monkeypatch.setattr("hilserl.processes.discover",lambda root:[learner])
    monkeypatch.setattr("hilserl.processes.process_start",identity)
    monkeypatch.setattr("hilserl.processes.listening_ports",lambda pid=None:{5588,5589})
    manager=Manager(cfg);manager.preflight=lambda **kwargs:{"ok":True}
    manager._spawn=Mock(side_effect=AssertionError("must not spawn"))
    with pytest.raises(RuntimeError,match="身份"):
        manager.launch()
    assert not manager._spawn.called


def test_other_entrypoint_can_cancel_waiting_launch_before_actor_spawn(cfg,monkeypatch):
    learner,_=pair(cfg)
    monkeypatch.setattr("hilserl.processes.discover",lambda root:[learner])
    monkeypatch.setattr("hilserl.processes.process_start",lambda pid:"same")
    monkeypatch.setattr("hilserl.processes.time.sleep",lambda seconds:None)
    second=Manager(cfg)
    def ports(pid=None):
        assert second.command("stop")["phase"]=="cancelling"
        return set()
    monkeypatch.setattr("hilserl.processes.listening_ports",ports)
    manager=Manager(cfg);manager.preflight=lambda **kwargs:{"ok":True}
    manager._spawn=Mock(side_effect=AssertionError("must not spawn"))
    assert manager.launch()["phase"]=="cancelled"
    assert not manager._spawn.called


def test_each_attempt_has_an_immutable_effective_snapshot(cfg,monkeypatch):
    monkeypatch.setattr("hilserl.processes.process_start",lambda pid:"same")
    spawned=[]
    def popen(*args,**kwargs):
        spawned.append(kwargs["env"])
        return SimpleNamespace(pid=123,poll=lambda:None)
    monkeypatch.setattr("hilserl.processes.subprocess.Popen",popen)
    manager=Manager(cfg);manager.launch_state={"request_id":"test"}
    run=manager._new_run("train")
    manager._spawn("actor",run,run/"checkpoints",cfg)
    first=Path(spawned[0]["HILSERL_CONFIG_SNAPSHOT"]);original=first.read_bytes()
    updated=replace(cfg,seed=8,port=7788,broadcast_port=7789)
    manager._spawn("actor",run,run/"checkpoints",updated)
    second=Path(spawned[1]["HILSERL_CONFIG_SNAPSHOT"])
    assert first!=second and first.read_bytes()==original
    assert json.loads(second.read_text())["seed"]==8
    assert spawned[1]["AGENTLACE_PORT"]=="7788"
    assert spawned[0]["HILSERL_ATTEMPT_ID"]!=spawned[1]["HILSERL_ATTEMPT_ID"]


@pytest.mark.parametrize("args",[
    ["python3","-m","experiments.plug_insertion.run_actor","--eval_mode"],
    ["python3","scripts/gello_demo_recorder.py"],
    ["python3","scripts/xbox_demo_recorder.py"],
    ["python3","scripts/hybrid_teleop.py"]])
def test_device_discovery_covers_legacy_tools(cfg,tmp_path,args):
    proc=tmp_path/"proc";entry=proc/"123";entry.mkdir(parents=True)
    (entry/"cwd").symlink_to(cfg.root,target_is_directory=True)
    (entry/"cmdline").write_bytes(("\0".join(args)+"\0").encode())
    (entry/"environ").write_bytes(b"")
    (entry/"stat").write_text("123 (python3) "+" ".join(["S"]+["0"]*18+["start"]+["0"]*2))
    found=discover(cfg.root,proc)
    assert len(found)==1 and found[0]["role"]=="legacy_device"


def test_actor_state_from_a_previous_pid_is_not_used(cfg,monkeypatch):
    _,actor=pair(cfg)
    actor.update(run_dir=str(cfg.root/"run"))
    actor["env"]["HILSERL_ATTEMPT_ID"]="current"
    control=cfg.root/"run/control";control.mkdir(parents=True)
    atomic_json(control/"status.json",dict(pid=101,start_time="old",attempt_id="old",phase="collecting",episode_id="1"))
    monkeypatch.setattr("hilserl.processes.discover",lambda root:[actor])
    monkeypatch.setattr("hilserl.processes.process_start",lambda pid:"same")
    with pytest.raises(ValueError,match="状态"):
        Manager(cfg).command("label",1)
    assert not (control/"commands").exists()


def test_status_reports_policy_handshake_fault_separately_from_learner_process(cfg, monkeypatch):
    learner, actor = pair(cfg)
    manager = Manager(cfg)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})
    monkeypatch.setattr(manager, "_operator_state", lambda process: {
        "phase": "fault", "error": "No learner policy received before first reset"
    })
    status = manager.status()
    assert status["learner_policy"] == {
        "state": "fault", "reason": "Actor 未收到 Learner 策略参数"
    }


def test_faulted_actor_can_be_stopped_after_it_stops_polling_commands(cfg, monkeypatch):
    _, actor = pair(cfg)
    actor["run_dir"] = str(cfg.root / "run")
    actor["env"]["HILSERL_ATTEMPT_ID"] = "attempt"
    manager = Manager(cfg)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr(manager, "_operator_state", lambda process: {
        "phase": "fault", "attempt_id": "attempt", "pid": 101, "start_time": "same"
    })
    kill = Mock()
    monkeypatch.setattr("hilserl.processes.os.kill", kill)
    assert manager.command("stop")["phase"] == "stopping"
    kill.assert_called_once_with(101, signal.SIGINT)


def test_fingerprint_mismatch_prevents_reusing_old_running_pair(cfg, monkeypatch):
    learner, actor = pair(cfg)
    run = cfg.root / "run"
    run.mkdir()
    learner["run_dir"] = actor["run_dir"] = str(run)
    source = Manager(cfg)._source_hashes()
    atomic_json(run / "learner.json", {
        "pid": 100, "start_time": "same", "source_sha256": dict(source, **{"_run_actor.py": "old"}),
        "config_sha256": "same",
    })
    atomic_json(run / "actor.json", {
        "pid": 101, "start_time": "same", "source_sha256": source, "config_sha256": "same",
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})
    with pytest.raises(RuntimeError, match="source fingerprint"):
        Manager(cfg).launch()


def test_config_fingerprint_mismatch_prevents_reusing_pair(cfg, monkeypatch):
    learner, actor = pair(cfg)
    run = cfg.root / "run"
    run.mkdir()
    learner["run_dir"] = actor["run_dir"] = str(run)
    atomic_json(run / "learner.json", {
        "pid": 100, "start_time": "same", "config_sha256": "learner-config",
    })
    atomic_json(run / "actor.json", {
        "pid": 101, "start_time": "same", "config_sha256": "actor-config",
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})
    with pytest.raises(RuntimeError, match="config fingerprint"):
        Manager(cfg).launch()


@pytest.mark.parametrize("phase,expected", [("stopped", "stopped"), ("fault", "degraded")])
def test_status_converges_persisted_launch_when_actor_is_gone(cfg, monkeypatch, phase, expected):
    run = cfg.root / "run"
    (run / "control").mkdir(parents=True)
    cfg.control_root.mkdir(parents=True)
    atomic_json(run / "control/status.json", {
        "pid": 101, "start_time": "same", "attempt_id": "attempt", "phase": phase,
        "reason": "operator_stop" if phase == "stopped" else None,
        "error": "actor failed" if phase == "fault" else None,
    })
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "running", "run_dir": str(run), "actor_pid": 101,
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [])
    status = Manager(cfg).status()
    assert status["launch"]["phase"] == expected
    assert json.loads((cfg.control_root / "launch.json").read_text())["phase"] == expected


def test_status_keeps_learner_only_launch_running_without_actor(cfg, monkeypatch):
    run = cfg.root / "run"
    run.mkdir()
    cfg.control_root.mkdir(parents=True)
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "running", "run_dir": str(run),
        "learner": {"pid": 100, "start_time": "same", "role": "learner"},
    })
    learner, _ = pair(cfg)
    learner["run_dir"] = str(run)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    assert Manager(cfg).status()["launch"]["phase"] == "running"


def test_status_keeps_live_actor_in_handshake_state_before_status_file(cfg, monkeypatch):
    learner, actor = pair(cfg)
    run = cfg.root / "run"
    run.mkdir()
    actor["run_dir"] = str(run)
    cfg.control_root.mkdir(parents=True)
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "running", "run_dir": str(run), "actor_pid": 101,
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    assert Manager(cfg).status()["launch"]["phase"] == "waiting_actor"


def test_launch_waits_for_actor_policy_handshake(cfg, monkeypatch):
    manager = Manager(cfg)
    manager.launch_state = {"request_id": "test"}
    actor = {"pid": 101, "start_time": "same", "run_dir": str(cfg.root / "run")}
    states = iter([
        {"pid": 101, "start_time": "same", "phase": "starting"},
        {"pid": 101, "start_time": "same", "phase": "waiting_reset"},
    ])
    monkeypatch.setattr("hilserl.processes.read_json", lambda path: next(states))
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.time.monotonic", lambda: 0.0)
    monkeypatch.setattr("hilserl.processes.time.sleep", lambda seconds: None)
    assert manager._wait_actor_ready(actor, timeout=5)["phase"] == "waiting_reset"


def test_launch_reports_actor_policy_handshake_fault(cfg, monkeypatch):
    manager = Manager(cfg)
    manager.launch_state = {"request_id": "test"}
    actor = {"pid": 101, "start_time": "same", "run_dir": str(cfg.root / "run")}
    monkeypatch.setattr("hilserl.processes.read_json", lambda path: {
        "pid": 101, "start_time": "same", "phase": "fault", "error": "No learner policy received before first reset",
    })
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    with pytest.raises(RuntimeError, match="策略握手"):
        manager._wait_actor_ready(actor, timeout=5)


def test_stop_learner_records_request_instead_of_claiming_save(cfg, monkeypatch):
    learner, _ = pair(cfg)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    kill = Mock()
    monkeypatch.setattr("hilserl.processes.os.kill", kill)

    result = Manager(cfg).stop_learner()

    assert result["pid"] == 100 and result["phase"] == "stop_requested"
    kill.assert_called_once_with(100, signal.SIGINT)
    launch = json.loads((cfg.control_root / "launch.json").read_text())
    assert launch["phase"] == "stopping_learner"
    assert launch["learner_pid"] == 100


def test_status_converges_finished_learner_stop(cfg, monkeypatch):
    run = cfg.root / "run"
    cfg.control_root.mkdir(parents=True)
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "stopping_learner", "run_dir": str(run), "learner_pid": 100,
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [])

    status = Manager(cfg).status()

    assert status["launch"]["phase"] == "stopped"
    assert status["launch"]["reason"] == "Learner 已退出；最终保存结果请查看日志"


@pytest.mark.parametrize('role,mode', [('actor','train'), ('actor','eval'), ('learner','train')])
def test_new_attempt_applies_current_reset_protocol_without_rewriting_history(cfg, monkeypatch, role, mode):
    monkeypatch.setattr('hilserl.processes.process_start', lambda pid: 'same')
    spawned = []
    monkeypatch.setattr('hilserl.processes.subprocess.Popen',
                        lambda *args, **kw: spawned.append(kw) or SimpleNamespace(pid=321))
    historical = replace(cfg, reset_clear_z=.2095, reset_feedback=False)
    current = replace(cfg, reset_clear_z=.15, reset_feedback=True)
    manager = Manager(current)
    manager.launch_state = {"request_id": "test-reset"}
    run = manager._new_run("train", config=historical)
    (run/"checkpoints").mkdir()
    (run/"checkpoints/checkpoint_0").write_bytes(b"mock checkpoint")
    before = (run/'config.json').read_bytes()
    manager._spawn(role, run, run/'checkpoints', historical, mode=mode)
    env = spawned[0]['env']
    assert env['HILSERL_RESET_CLEAR_Z'] == '0.15'
    assert env['HILSERL_RESET_FEEDBACK'] == '1'
    effective = json.loads(Path(env['HILSERL_CONFIG_SNAPSHOT']).read_text())
    assert effective['reset_clear_z'] == .15 and effective['reset_feedback'] is True
    assert effective['image_profile'] == historical.image_profile
    assert effective['action_contract'] == historical.action_contract
    assert (run/'config.json').read_bytes() == before
