"""Regression tests for archive boundaries and actual CPU video segmentation."""
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pytest

from hilserl.archive import Archive
from hilserl.control import OperatorControl, StopRequested, send_command
from hilserl.episodes import run_episodes
from hilserl.storage import SessionRecorder, VideoWriter, RecordingError, read_step, stamp
from test_session_recording import FakeEnvironment, ScriptedOperator


def test_late_reset_confirmation_and_stale_ui_labels_are_rejected(tmp_path):
    control=OperatorControl(tmp_path,terminal=False)
    control.publish("waiting_reset",gate_id="first")
    send_command(tmp_path,"continue")
    control.publish("waiting_reset",gate_id="second")
    assert control.poll({"continue"}) is None
    control.publish("collecting",episode_id="1")
    expected={key:control.state.get(key) for key in ("attempt_id","episode_id","phase","gate_id")}
    control.publish("collecting",episode_id="2")
    with pytest.raises(ValueError,match="变化"):
        send_command(tmp_path,"label",1,expected=expected)
    assert control.poll({"label"}) is None


def test_failure_while_waiting_for_verdict_cannot_count_as_success(tmp_path):
    recorder=SessionRecorder(tmp_path/"recording",min_free_bytes=0)
    env=FakeEnvironment(recorder=recorder)
    def waiting(phase):
        if phase=="awaiting_label":
            recorder.fail("injected recording failure")
    emitted=[];completed=[]
    with pytest.raises(RecordingError):
        run_episodes(env,lambda obs,n:(np.zeros(7),{}),recorder,ScriptedOperator(outcome=1,on_wait=waiting),
                     max_episodes=1,emit=emitted.append,on_episode=completed.append)
    assert not completed
    assert not any(item["dones"] for item in emitted)
    meta=json.loads((tmp_path/"recording/episodes/000001/episode.json").read_text())
    assert meta["complete"] is False


def test_zero_step_labels_do_not_complete_an_episode(tmp_path):
    recorder=SessionRecorder(tmp_path/"recording",min_free_bytes=0)
    env=FakeEnvironment(recorder=recorder)
    class Operator(ScriptedOperator):
        waits=0
        def wait(self,phase,*args,**kwargs):
            if phase=="waiting_reset":
                self.waits+=1
                if self.waits>1:
                    raise StopRequested()
            return super().wait(phase,*args,**kwargs)
    completed=run_episodes(env,lambda obs,n:(np.zeros(7),{}),recorder,Operator(label_after=0),max_episodes=1)
    assert not completed and env.steps==0
    assert json.loads((tmp_path/"recording/episodes/000001/episode.json").read_text())["complete"] is False


def test_outer_wrapper_failure_retains_sent_command_as_partial_attempt(tmp_path):
    root=tmp_path/"runs";capture=root/"run/recordings/cap"
    recorder=SessionRecorder(capture,min_free_bytes=0)
    class Environment(FakeEnvironment):
        def step(self,action):
            self.steps+=1
            self._step_commands=[dict(kind="pose",pose=np.arange(7,dtype=np.float32),returned=True)]
            raise RuntimeError("classifier failed after pose")
    env=Environment(recorder=recorder)
    with pytest.raises(RuntimeError,match="classifier"):
        run_episodes(env,lambda obs,n:(np.zeros(7),{}),recorder,ScriptedOperator(),max_episodes=1)
    raw=read_step(capture/"episodes/000001/000000.npz")
    assert raw["complete_transition"] is False and raw["next_observations"] is None
    assert len(raw["controller_commands"])==1
    with pytest.raises(ValueError,match="没有"):
        Archive(root,min_free_bytes=0).export("run/cap")


@pytest.mark.parametrize("first",[True,False])
def test_camera_health_is_checked_without_requesting_an_observation(tmp_path,first):
    recorder=SessionRecorder(tmp_path/"recording",min_free_bytes=0)
    recorder.register_camera("camera")
    if first:
        recorder._camera_deadlines["camera"]=time.monotonic_ns()-1
    else:
        recorder._capture_heartbeat["camera"]=time.monotonic_ns()-3_000_000_000
    with pytest.raises(RecordingError,match="Camera"):
        recorder.check()
    recorder.close()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),reason="ffmpeg required")
def test_odd_duration_segments_have_exact_index_mapping_and_zero_pts(tmp_path):
    errors=[]
    writer=VideoWriter(tmp_path,"camera",fps=30,segment_seconds=3,fail=errors.append,capacity=250)
    frame=np.zeros((32,64,3),np.uint8)
    for index in range(240):
        writer.append(frame,dict(capture_id=index,**stamp()))
    writer.close()
    assert not errors
    counts=[]
    for movie in sorted((tmp_path/"video/camera").glob("*.mp4")):
        data=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
              "-show_entries","frame=best_effort_timestamp_time","-of","json",str(movie)]))
        frames=data["frames"];counts.append(len(frames))
        assert float(frames[0]["best_effort_timestamp_time"])==pytest.approx(0)
    assert counts==[90,90,60]
    frames=[json.loads(line) for line in (tmp_path/"video/camera/frames.jsonl").read_text().splitlines()]
    assert [frames[index]["segment"] for index in (0,89,90,179,180,239)]==[0,0,1,1,2,2]
