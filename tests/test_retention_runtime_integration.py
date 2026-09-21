"""Behavioral integration around the real save, launcher and evaluation paths."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.processes import Manager
from hilserl.files import read_json
from test_learner_runtime_shutdown import Runtime


def test_new_attempt_uses_current_storage_policy_with_original_model_config(tmp_path,monkeypatch):
    original=replace(Config(),root=tmp_path,data_dir="runs",checkpoint_keep=0,batch_size=256,image_profile="full-frame128-v1")
    current=replace(original,checkpoint_keep=5,batch_size=128,image_profile="insert-roi160-v1",action_contract="fixed-xyz-v1")
    manager=Manager(current);run=manager._new_run("train",config=original)
    manager.launch_state={"request_id":"test"}
    before=(run/"config.json").read_bytes()
    monkeypatch.setattr("hilserl.processes.process_start",lambda *_:"start")
    popen=Mock(return_value=SimpleNamespace(pid=123))
    monkeypatch.setattr("hilserl.processes.subprocess.Popen",popen)
    manager._spawn("learner",run,run/"checkpoints",original,resume_step=4)
    env=popen.call_args.kwargs["env"]
    assert env["HILSERL_CHECKPOINT_KEEP"]=="5" and env["HILSERL_BATCH_SIZE"]=="256"
    assert env["HILSERL_IMAGE_PROFILE"]=="full-frame128-v1"
    snapshot=read_json(Path(env["HILSERL_CONFIG_SNAPSHOT"]))
    assert snapshot["checkpoint_keep"]==5 and snapshot["batch_size"]==256
    assert (run/"config.json").read_bytes()==before


def test_learner_reports_committed_save_even_when_cleanup_is_deferred(tmp_path,monkeypatch):
    import hilserl.checkpoint_retention as retention
    runtime=Runtime(tmp_path,monkeypatch)
    monkeypatch.setenv("HILSERL_CHECKPOINT_KEEP","5")
    monkeypatch.setattr(retention,"retain_after_save",Mock(side_effect=OSError("cleanup failed")))
    runtime.on_update=lambda _:runtime.stop_file()
    runtime.run()
    status=runtime.status()
    assert status["phase"]=="stopped" and status["checkpoint_saved"]
    assert status["checkpoint_error"] is None
    assert read_json(runtime.checkpoint_dir/"retention_status.json")["status"]=="error"


def test_eval_launcher_holds_reader_before_exec_and_releases_parent_copy(tmp_path,monkeypatch):
    from test_checkpoint_retention_policy import model
    cfg=replace(Config(),root=tmp_path,data_dir="runs",checkpoint_keep=5)
    manager=Manager(cfg);run=manager._new_run("eval")
    manager.launch_state={"request_id":"test"}
    directory=tmp_path/"models";directory.mkdir();model(directory,4)
    received=[]
    def spawn(*args,**kwargs):
        fd=kwargs["pass_fds"][0]
        assert int(kwargs["env"]["HILSERL_CHECKPOINT_READER_FD"])==fd
        os.fstat(fd);received.append(fd)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr("hilserl.processes.process_start",lambda *_:"start")
    monkeypatch.setattr("hilserl.processes.subprocess.Popen",spawn)
    manager._spawn("actor",run,directory,cfg,mode="eval",eval_step=4)
    with pytest.raises(OSError):os.fstat(received[0])


def test_actual_actor_records_eval_after_episode_recorder_has_closed(tmp_path,monkeypatch):
    import ast,threading,numpy as np
    from test_checkpoint_retention_policy import model
    import hilserl.episodes as episodes
    directory=tmp_path/"models";directory.mkdir();model(directory,7)
    run=tmp_path/"eval";run.mkdir()
    snapshot=run/"config.json";snapshot.write_text(json.dumps(replace(Config(),root=tmp_path).snapshot()))
    monkeypatch.setenv("HILSERL_MODE","eval")
    monkeypatch.setenv("HILSERL_RUN_DIR",str(run))
    monkeypatch.setenv("HILSERL_CONFIG_SNAPSHOT",str(snapshot))
    summaries=[dict(episode_id=str(i),outcome=int(i<8),steps=10,human_steps=0,mode="eval") for i in range(10)]
    recorder=SimpleNamespace(closed=False)
    def finished(*args,**kwargs):
        assert kwargs["eval_seed"]==0
        recorder.closed=True
        return summaries
    monkeypatch.setattr(episodes,"run_episodes",finished)
    class Agent:
        state={"params":1}
        def replace(self,**kwargs):return self
    source=Path(__file__).resolve().parents[1]/"_run_actor.py"
    tree=ast.parse(source.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="actor"]
    flags=SimpleNamespace(checkpoint_path=str(directory),eval_checkpoint_step=7,eval_n_trajs=10,seed=0)
    namespace=dict(os=os,np=np,threading=threading,FLAGS=flags,
        config=SimpleNamespace(max_steps=1000,action_contract="legacy-hybrid-7d-v1",image_profile="full-frame128-v1"),
        checkpoints=SimpleNamespace(restore_checkpoint=lambda *args,**kwargs:Agent.state))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(source),"exec"),namespace)
    env=SimpleNamespace(unwrapped=SimpleNamespace(recorder=recorder,operator=object()))
    assert namespace["actor"](Agent(),None,None,env,None)==summaries
    assert recorder.closed
    result=read_json(run/"evaluation.json")
    assert result["status"]=="ranked" and result["successes"]==8


def test_eval_reset_retry_keeps_the_same_case_seed(tmp_path):
    import numpy as np
    from hilserl.episodes import run_episodes
    from hilserl.errors import RobotStateUnavailable
    from hilserl.storage import SessionRecorder
    from test_session_recording import FakeEnvironment,ScriptedOperator
    draws=[]
    class Environment(FakeEnvironment):
        def reset(self):
            draws.append(np.random.uniform(size=3))
            if len(draws)==1:
                raise RobotStateUnavailable("synthetic feedback interruption",status_code=503)
            return super().reset()
    recorder=SessionRecorder(tmp_path/"recording",min_free_bytes=0)
    before=np.random.get_state()
    try:
        results=run_episodes(Environment(terminal_at=1,recorder=recorder),lambda *args:(np.zeros(7),{}),
            recorder,ScriptedOperator(outcome=1),mode="eval",eval_seed=9,max_episodes=2)
    finally:np.random.set_state(before)
    assert len(results)==2 and len(draws)==3
    np.testing.assert_array_equal(draws[0],draws[1])
    np.testing.assert_array_equal(draws[2],np.random.RandomState(10).uniform(size=3))
