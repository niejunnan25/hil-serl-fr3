"""HTTP and export tests use localhost, temporary synthetic files, and no device runtime."""
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import threading
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

from hilserl.config import Config
from hilserl.storage import SessionRecorder
from hilserl.webserver import make_server


@pytest.fixture
def server(tmp_path):
    output = tmp_path/"runs"
    recording = output/"synthetic"/"recordings"/"sample"
    recorder = SessionRecorder(recording, metadata={"synthetic": True}, min_free_bytes=0)
    recorder.start_episode()
    for step, source in enumerate(["human","policy"]):
        recorder.append_step(dict(id=f"synthetic/sample/000001/{step}",episode_id="000001",
            episode_step=step,global_step=step,complete_transition=True,
            observations={"state":np.full((1,19),step,np.float32)},
            next_observations={"state":np.full((1,19),step+1,np.float32)},
            actions=np.zeros(7,np.float32),source_action=source,observed_reward=0.0,info={},
            termination_reason="time_limit" if step else None))
    recorder.end_collection("time_limit")
    recorder.finish_episode(1)
    recorder.close()
    video=recording/"video"/"wrist_1"
    video.mkdir(parents=True)
    (video/"000000.mp4").write_bytes(b"0123456789")
    config=replace(Config(),root=tmp_path,data_dir=str(output),min_free_gib=0)
    web=make_server(config,port=0,offline=True)
    worker=threading.Thread(target=web.serve_forever,daemon=True)
    worker.start()
    url=f"http://127.0.0.1:{web.server_port}"
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url+"/api/status") as response:
        status=json.load(response)
    yield url,opener,status,output
    web.shutdown();web.server_close();worker.join(timeout=2)


def request(server,path,data=None,headers=None):
    url,opener,state,_=server
    body=json.dumps(data).encode() if data is not None else None
    req=urllib.request.Request(url+path,data=body,headers=headers or {})
    return opener.open(req)


def test_offline_console_never_launches_devices_and_rejects_foreign_posts(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,"/api/start",{"mode":"train"})
    assert error.value.code==403
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,"/api/start",{"mode":"train"},{"X-HILSERL-Token":server[2]["token"]})
    assert error.value.code==400
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,"/api/export",{"id":"synthetic/sample"},
                {"X-HILSERL-Token":server[2]["token"],"Origin":"https://elsewhere.example"})
    assert error.value.code==403


def test_invalid_start_request_is_rejected_before_async_dispatch(tmp_path, monkeypatch):
    calls=[]
    monkeypatch.setattr("hilserl.processes.Manager.launch", lambda *args, **kwargs: calls.append(kwargs))
    config=replace(Config(),root=tmp_path,data_dir=str(tmp_path/"runs"),min_free_gib=0)
    web=make_server(config,port=0,offline=False)
    worker=threading.Thread(target=web.serve_forever,daemon=True)
    worker.start()
    url=f"http://127.0.0.1:{web.server_port}"
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url+"/api/status") as response:
            status=json.load(response)
        live_server=(url,opener,status,config.output_root)
        for data in [
            {"mode":"unknown"},
            {"mode":"eval","checkpoint":""},
            {"mode":"eval","checkpoint":"/tmp/weights","eval_episodes":0},
            {"mode":"eval","checkpoint":"/tmp/weights","eval_episodes":1.5},
            {"mode":"collect","learner_only":True},
            {"mode":"train","unexpected":True},
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                request(live_server,"/api/start",data,{"X-HILSERL-Token":status["token"]})
            assert error.value.code==400
            assert json.load(error.value)["error"]
        assert calls==[]
    finally:
        web.shutdown();web.server_close();worker.join(timeout=2)


def test_archive_video_byte_ranges_and_unknown_files(server):
    with request(server,"/api/video?id=synthetic%2Fsample&camera=wrist_1&file=000000.mp4",
                 headers={"Range":"bytes=2-5"}) as response:
        assert response.status==206
        assert response.headers["Content-Range"]=="bytes 2-5/10"
        assert response.read()==b"2345"
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,"/api/recording?id=../../etc")
    assert error.value.code==400


@pytest.mark.parametrize("format",["npz","serl"])
def test_export_filters_preserve_step_indices_next_obs_and_episode_label(server,format):
    headers={"X-HILSERL-Token":server[2]["token"]}
    with request(server,"/api/export",{"id":"synthetic/sample","source":"human","outcome":"success","format":format},headers) as response:
        exported=json.load(response)
    assert exported["steps"]==1
    with request(server,"/api/download?file="+exported["file"]) as response:
        archive=zipfile.ZipFile(io.BytesIO(response.read()))
        manifest=json.loads(archive.read("manifest.json"))
        episode=manifest["episodes"][0]
        assert episode["selected_step_indices"]==[0]
        assert episode["is_complete_episode_selection"] is False
        assert episode["outcome"]==1
        assert manifest["videos_included"] is False
        if format=="serl":
            transitions=pickle.loads(archive.read("episodes/000001/transitions.pkl"))
            assert transitions[0]["dones"] is False
            np.testing.assert_array_equal(transitions[0]["next_observations"]["state"],np.ones((1,19)))


def test_wrong_outcome_does_not_create_empty_export(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,"/api/export",{"id":"synthetic/sample","outcome":"failure"},
                {"X-HILSERL-Token":server[2]["token"]})
    assert error.value.code==400
    assert not list((server[3]/"exports").glob("*.zip"))


def test_logs_retain_multiple_actor_attempts(server):
    """A retry must not hide the first failed Actor handshake from the UI."""
    output = server[3]
    run = output / "attempted-run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    (logs / "actor_first.log").write_text("No learner policy received before first reset")
    (logs / "actor_second.log").write_text("waiting_reset")
    control = output.parent / "artifacts" / "control"
    control.mkdir(parents=True)
    (control / "launch.json").write_text(json.dumps({"phase": "stopped", "run_dir": str(run)}))

    with request(server, "/api/logs") as response:
        records = json.load(response)

    roles = {record["role"] for record in records}
    assert "actor · attempt first" in roles
    assert "actor · attempt second" in roles
    assert any("No learner policy" in record["text"] for record in records)


def test_offline_gripper_posts_never_reach_device(server, monkeypatch):
    calls = []
    monkeypatch.setattr("hilserl.processes.Manager.gripper", lambda *a, **k: calls.append(k))
    for operation in ("open", "close", "status"):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, "/api/gripper", {"operation": operation}, {"X-HILSERL-Token": server[2]["token"]})
        assert error.value.code == 400
    assert calls == []


def test_gripper_http_dispatch_preserves_operation_and_actor_fence(tmp_path, monkeypatch):
    calls = []
    def command(self, operation, *, expected):
        calls.append((operation, expected))
        return {"operation": operation, "verification": "queued"}
    monkeypatch.setattr("hilserl.processes.Manager.gripper", command)
    config = replace(Config(), root=tmp_path, min_free_gib=0)
    web = make_server(config, port=0)
    worker = threading.Thread(target=web.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{web.server_port}"
    try:
        with opener.open(url + "/api/status") as response:
            state = json.load(response)
        live = (url, opener, state, config.output_root)
        expected = {"attempt_id": "actor-a", "phase": "paused", "gate_id": "gate-a"}
        with request(live, "/api/gripper", {"operation": "close", "expected": expected},
                     {"X-HILSERL-Token": state["token"]}) as response:
            assert json.load(response)["verification"] == "queued"
        assert calls == [("close", expected)]
        with pytest.raises(urllib.error.HTTPError) as error:
            request(live, "/api/gripper", {"operation": "close"})
        assert error.value.code == 403 and len(calls) == 1
    finally:
        web.shutdown(); web.server_close(); worker.join(timeout=2)


def test_offline_controller_actions_and_custom_commands_are_rejected(server, monkeypatch):
    calls = []
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", lambda *_: calls.append("ssh"))
    for route in ("status", "recover"):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, "/api/controller/" + route, {}, {"X-HILSERL-Token": server[2]["token"]})
        assert error.value.code == 400
    assert calls == []


def test_controller_recovery_http_excludes_launch_and_gripper_until_verified(tmp_path, monkeypatch):
    import time
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr("hilserl.processes.process_start", lambda *_: "same")
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [])
    monkeypatch.setattr("hilserl.controller.ControllerService.status", lambda *_: {"phase": "ready", "reason": "fresh"})
    def recover(_):
        entered.set()
        assert release.wait(3)
        return {"phase": "ready", "reason": "fresh after recovery"}
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", recover)
    config = replace(Config(), root=tmp_path, min_free_gib=0)
    web = make_server(config, port=0, monitor_controller=True)
    worker = threading.Thread(target=web.serve_forever, daemon=True); worker.start()
    url = f"http://127.0.0.1:{web.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/api/status") as response: state = json.load(response)
        live = (url, opener, state, config.output_root)
        headers = {"X-HILSERL-Token": state["token"]}
        with request(live, "/api/controller/recover", {}, headers) as response:
            assert json.load(response)["phase"] == "starting"
        assert entered.wait(1)
        with request(live, "/api/status") as response:
            assert json.load(response)["controller"]["phase"] == "recovering"
        for route, body in [("/api/controller/recover", {}), ("/api/start", {"mode": "train"}),
                            ("/api/gripper", {"operation": "open"})]:
            with pytest.raises(urllib.error.HTTPError) as error: request(live, route, body, headers)
            assert error.value.code == 400
        release.set()
        for _ in range(40):
            with request(live, "/api/status") as response: value = json.load(response)
            if value["controller"]["phase"] == "ready": break
            time.sleep(.025)
        assert value["controller"]["phase"] == "ready"
    finally:
        release.set(); web.shutdown(); web.server_close(); worker.join(timeout=2)


def test_controller_http_rejects_active_actor_and_arbitrary_target(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("hilserl.processes.discover", lambda *_: [{"role": "train", "pid": 1}])
    monkeypatch.setattr("hilserl.controller.ControllerService.recover", lambda *_: calls.append("ssh"))
    config = replace(Config(), root=tmp_path, min_free_gib=0)
    # Avoid reading a fake Actor's real control directory while acquiring the UI token.
    monkeypatch.setattr("hilserl.processes.Manager.status", lambda *_: {"config": config.snapshot(), "launch": {}})
    web = make_server(config, port=0)
    worker = threading.Thread(target=web.serve_forever, daemon=True); worker.start()
    url = f"http://127.0.0.1:{web.server_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/api/status") as response: state = json.load(response)
        live = (url, opener, state, config.output_root)
        headers = {"X-HILSERL-Token": state["token"]}
        for body in ({}, {"command": "anything"}, {"host": "another-controller"}):
            with pytest.raises(urllib.error.HTTPError) as error:
                request(live, "/api/controller/recover", body, headers)
            assert error.value.code == 400
        assert calls == []
    finally:
        web.shutdown(); web.server_close(); worker.join(timeout=2)
