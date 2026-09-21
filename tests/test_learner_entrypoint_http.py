"""Exercise the new HTTP entry points with Manager actions replaced by receipts."""
from contextlib import contextmanager
from dataclasses import replace
import json
import threading
import urllib.error
import urllib.request

import pytest

from hilserl.config import Config
from hilserl.webserver import make_server


@contextmanager
def endpoint(tmp_path, monkeypatch, *, offline=False):
    calls = []
    dispatched = threading.Event()

    def launch(self, *args, **kwargs):
        calls.append(("launch", kwargs))
        dispatched.set()

    def activity(self, command, *, expected):
        calls.append((command, expected))
        return {"phase": "requested", "command": command}

    monkeypatch.setattr("hilserl.processes.Manager.launch", launch)
    monkeypatch.setattr("hilserl.processes.Manager.learner_activity", activity)
    config = replace(Config(), root=tmp_path, min_free_gib=0)
    web = make_server(config, port=0, offline=offline)
    worker = threading.Thread(target=web.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{web.server_port}"
    try:
        with opener.open(url + "/api/status") as response:
            token = json.load(response)["token"]

        def post(path, body, *, authenticated=True):
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["X-HILSERL-Token"] = token
            req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers=headers)
            with opener.open(req) as response:
                return json.load(response)

        yield post, calls, dispatched
    finally:
        web.shutdown(); web.server_close(); worker.join(timeout=2)


def test_http_start_preserves_explicit_resume_and_learner_only(tmp_path, monkeypatch):
    body = {"mode": "train", "learner_only": True, "resume_checkpoint": "/managed/checkpoint_25974"}
    with endpoint(tmp_path, monkeypatch) as (post, calls, dispatched):
        assert post("/api/start", body)["phase"] == "starting"
        assert dispatched.wait(1)
        assert calls == [("launch", body)]


@pytest.mark.parametrize("command", ["pause", "resume"])
def test_http_activity_preserves_exact_expected_identity(tmp_path, monkeypatch, command):
    expected = {"pid": 123, "start_time": "start", "attempt_id": "current"}
    with endpoint(tmp_path, monkeypatch) as (post, calls, _):
        assert post("/api/learner-activity", {"command": command, "expected": expected})["phase"] == "requested"
        assert calls == [(command, expected)]


def test_http_activity_rejects_missing_fence_or_foreign_post(tmp_path, monkeypatch):
    with endpoint(tmp_path, monkeypatch) as (post, calls, _):
        with pytest.raises(urllib.error.HTTPError) as missing:
            post("/api/learner-activity", {"command": "pause"})
        assert missing.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as foreign:
            post("/api/learner-activity", {"command": "pause", "expected": {}}, authenticated=False)
        assert foreign.value.code == 403
        assert calls == []


def test_offline_new_controls_never_dispatch(tmp_path, monkeypatch):
    with endpoint(tmp_path, monkeypatch, offline=True) as (post, calls, _):
        for route, body in [("/api/learner-activity", {"command": "pause", "expected": {}}),
                            ("/api/start", {"mode": "train", "learner_only": True, "resume_checkpoint": "/managed/checkpoint_0"})]:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(route, body)
            assert error.value.code == 400
        assert calls == []
