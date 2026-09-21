"""Deterministic regressions for learner/actor policy handshakes.

These tests intentionally avoid JAX, FR3, cameras, and a live Agentlace socket.
The Agentlace in-process adapter is extracted from its source so the cache and
revision contract remains testable in the minimal workstation test environment.
"""

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
from typing import Callable, Optional
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from hilserl.config import Config
from hilserl.files import atomic_json
from hilserl.processes import Manager
from hilserl.control import StopRequested


ROOT = Path(__file__).resolve().parents[1]


def extract_functions(path, *names, **globals_):
    """Compile selected top-level functions without importing JAX."""
    tree = ast.parse(Path(path).read_text())
    wanted = [node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(os=os, np=np, time=__import__("time"), threading=threading, **globals_)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def extract_class(path, name, **globals_):
    """Compile one dependency-light class from a source file."""
    tree = ast.parse(Path(path).read_text())
    wanted = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    namespace = dict(Optional=Optional, Callable=Callable, **globals_)
    namespace.update(globals_)
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_inprocess_snapshot_is_available_to_late_subscriber_and_revision_is_monotonic():
    """A late actor can pull the latest policy even when it missed PUB/SUB."""
    interface = extract_class(ROOT / "upstream/agentlace/agentlace/trainer.py", "TrainerSMInterface")()

    assert interface.get_network() is None
    interface.publish_network({"weights": [1]})
    first = interface.get_network()
    interface.publish_network({"weights": [2]})
    second = interface.get_network()

    assert first == {"params": {"weights": [1]}, "revision": 1}
    assert second == {"params": {"weights": [2]}, "revision": 2}
    assert second["revision"] > first["revision"]


def test_actor_pulls_snapshot_without_waiting_for_a_broadcast(monkeypatch):
    """The startup gate must use get_network when the initial broadcast was lost."""
    monkeypatch.setenv("HILSERL_MODE", "train")
    monkeypatch.setenv("ACTOR_LEARNER_PARAMS_TIMEOUT", "1")
    monkeypatch.setenv("ACTOR_LEARNER_PARAMS_SNAPSHOT_PERIOD", "0.01")

    class State:
        def __init__(self, params):
            self.params = params

        def replace(self, *, params):
            return State(params)

    class Agent:
        def __init__(self, state=None):
            self.state = state or State("bootstrap")

        def replace(self, *, state):
            return Agent(state)

    class Operator:
        state = {"phase": "starting"}

        def publish(self, phase, **fields):
            self.state.update(phase=phase, **fields)

        def raise_if_stop(self):
            raise AssertionError("snapshot should arrive before the timeout poll")

    class Client:
        instances = []

        def __init__(self, *args, **kwargs):
            self.callback = None
            self.get_network_calls = 0
            self.stopped = False
            Client.instances.append(self)

        def recv_network_callback(self, callback):
            self.callback = callback

        def get_network(self):
            self.get_network_calls += 1
            return {"params": "late-snapshot", "revision": 41}

        def stop(self):
            self.stopped = True

    marker = object()
    fake_episodes = ModuleType("hilserl.episodes")

    def run_episodes(*args, **kwargs):
        return marker

    fake_episodes.run_episodes = run_episodes
    monkeypatch.setitem(sys.modules, "hilserl.episodes", fake_episodes)
    base = SimpleNamespace(
        recorder=SimpleNamespace(check=lambda: None),
        operator=Operator(),
        raw_state=lambda: {"values": {}},
    )
    env = SimpleNamespace(unwrapped=base, action_space=SimpleNamespace(shape=(7,)))
    flags = SimpleNamespace(ip="127.0.0.1", eval_checkpoint_step=0, checkpoint_path="unused", eval_n_trajs=1)
    config = SimpleNamespace(max_steps=10, buffer_period=0)
    ns = extract_functions(
        ROOT / "_run_actor.py",
        "actor",
        FLAGS=flags,
        TrainerClient=Client,
        _make_runtime_trainer_config=lambda: {},
        config=config,
        StopRequested=StopRequested,
    )

    result = ns["actor"](Agent(), None, None, env, None)

    assert result is marker
    assert len(Client.instances) == 1
    assert Client.instances[0].get_network_calls >= 1
    assert Client.instances[0].stopped is True


def test_actor_rejects_out_of_order_delta_after_newer_snapshot(monkeypatch):
    """A delayed old broadcast must not roll the actor back to an older revision."""
    monkeypatch.setenv("HILSERL_MODE", "train")
    monkeypatch.setenv("ACTOR_LEARNER_PARAMS_TIMEOUT", "1")
    monkeypatch.setenv("ACTOR_LEARNER_PARAMS_SNAPSHOT_PERIOD", "0.01")

    class State:
        def __init__(self, params):
            self.params = params

        def replace(self, *, params):
            return State(params)

    class Agent:
        def __init__(self, state=None):
            self.state = state or State("bootstrap")

        def replace(self, *, state):
            return Agent(state)

        def sample_actions(self, **kwargs):
            return np.zeros(7, dtype=np.float32)

    class FakeRandom:
        @staticmethod
        def split(rng):
            return rng, rng

    class FakeJax:
        random = FakeRandom()

        @staticmethod
        def device_put(value):
            return value

        @staticmethod
        def device_get(value):
            return value

    class Operator:
        state = {"phase": "starting"}

        def publish(self, phase, **fields):
            self.state.update(phase=phase, **fields)

        def raise_if_stop(self):
            raise AssertionError("policy snapshot should arrive before timeout")

    class Client:
        instances = []

        def __init__(self, *args, **kwargs):
            self.callback = None
            self.stopped = False
            Client.instances.append(self)

        def recv_network_callback(self, callback):
            self.callback = callback

        def get_network(self):
            # Simulate a late snapshot followed by out-of-order PUB/SUB data.
            self.callback("snapshot-5", 5)
            self.callback("stale-4", 4)
            self.callback("delta-6", 6)
            return None

        def stop(self):
            self.stopped = True

    fake_episodes = ModuleType("hilserl.episodes")
    observed = {}

    def run_episodes(env, sample, *args, **kwargs):
        _, info = sample({}, 0)
        observed.update(info)

    fake_episodes.run_episodes = run_episodes
    monkeypatch.setitem(sys.modules, "hilserl.episodes", fake_episodes)
    base = SimpleNamespace(
        recorder=SimpleNamespace(check=lambda: None),
        operator=Operator(),
        raw_state=lambda: {"values": {}},
    )
    env = SimpleNamespace(unwrapped=base, action_space=SimpleNamespace(shape=(7,)))
    flags = SimpleNamespace(ip="127.0.0.1", eval_checkpoint_step=0, checkpoint_path="unused", eval_n_trajs=1)
    config = SimpleNamespace(max_steps=10, buffer_period=0)
    ns = extract_functions(
        ROOT / "_run_actor.py",
        "actor",
        FLAGS=flags,
        TrainerClient=Client,
        _make_runtime_trainer_config=lambda: {},
        config=config,
        StopRequested=StopRequested,
        jax=FakeJax,
    )

    ns["actor"](Agent(), None, None, env, None)

    assert observed["revision"] == 6
    assert observed["source"] == "learner"
    assert Client.instances[0].stopped is True


def test_startup_policy_retry_survives_transient_publish_error():
    """A single PUB failure must not kill the bounded retry worker."""
    ns = extract_functions(ROOT / "_run_actor.py", "_start_initial_network_retry")
    attempts = []

    def publish(reason):
        attempts.append(reason)
        if len(attempts) == 1:
            raise RuntimeError("transient socket error")

    stop, thread = ns["_start_initial_network_retry"](publish, period=0.1, grace_seconds=0.35)
    thread.join(timeout=1)
    stop.set()

    assert not thread.is_alive()
    assert len(attempts) >= 2


@pytest.fixture
def process_cfg(tmp_path):
    return replace(Config(), root=tmp_path, data_dir="runs", min_free_gib=0)


def _process_pair(cfg):
    common = {
        "AGENTLACE_PORT": "5588",
        "AGENTLACE_BROADCAST_PORT": "5589",
        "HILSERL_CLASSIFIER_CKPT": str(cfg.path(cfg.classifier_ckpt)),
    }
    checkpoint = str(cfg.root / "checkpoints")
    learner = dict(
        pid=100,
        start_time="same",
        role="learner",
        args=["python", "_run_actor.py", "--learner", "--exp_name=plug_insertion", "--seed=0"],
        env=dict(common),
        checkpoint=checkpoint,
        run_dir=None,
    )
    actor = dict(
        pid=101,
        start_time="same",
        role="train",
        args=["python", "_run_actor.py", "--actor", "--exp_name=plug_insertion", "--seed=0", "--ip=127.0.0.1"],
        env=dict(common),
        checkpoint=checkpoint,
        run_dir=None,
    )
    return learner, actor


def test_source_fingerprint_mismatch_blocks_pair_reuse(process_cfg, monkeypatch):
    cfg = process_cfg
    learner, actor = _process_pair(cfg)
    run = cfg.root / "run"
    run.mkdir()
    learner["run_dir"] = actor["run_dir"] = str(run)
    (run / "learner.json").write_text(json.dumps({"pid": 100, "start_time": "same", "source_sha256": {"_run_actor.py": "old"}}))
    (run / "actor.json").write_text(json.dumps({"pid": 101, "start_time": "same", "source_sha256": {}}))
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})

    with pytest.raises(RuntimeError, match="source fingerprint"):
        Manager(cfg).launch()


def test_config_fingerprint_mismatch_blocks_pair_reuse(process_cfg, monkeypatch):
    cfg = process_cfg
    learner, actor = _process_pair(cfg)
    run = cfg.root / "run"
    run.mkdir()
    learner["run_dir"] = actor["run_dir"] = str(run)
    record = {"pid": 100, "start_time": "same", "source_sha256": {}}
    (run / "learner.json").write_text(json.dumps({**record, "config_sha256": "learner-config"}))
    (run / "actor.json").write_text(json.dumps({**record, "pid": 101, "config_sha256": "actor-config"}))
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [learner, actor])
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})

    with pytest.raises(RuntimeError, match="config fingerprint"):
        Manager(cfg).launch()


def test_status_reconciles_stale_running_launch_after_actor_stopped(process_cfg, monkeypatch):
    cfg = process_cfg
    run = cfg.root / "run"
    control = run / "control"
    control.mkdir(parents=True)
    cfg.control_root.mkdir(parents=True)
    atomic_json(control / "status.json", {
        "pid": 101,
        "start_time": "same",
        "attempt_id": "attempt",
        "phase": "stopped",
        "reason": "operator_stop",
    })
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "running",
        "run_dir": str(run),
        "actor_pid": 101,
        "learner_pid": 100,
    })
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [{
        "pid": 100,
        "start_time": "same",
        "role": "learner",
        "args": [],
        "env": {},
        "checkpoint": str(cfg.root / "checkpoints"),
        "run_dir": str(run),
    }])
    status = Manager(cfg).status()

    # The Learner is still alive, so the pair is degraded rather than fully
    # stopped; the console must not present a live Learner as a healthy task.
    assert status["launch"]["phase"] == "degraded"
    assert "Learner 仍在运行" in status["launch"]["error"]
    assert json.loads((cfg.control_root / "launch.json").read_text())["phase"] == "degraded"


def test_pending_launch_with_dead_owner_is_reported_as_fault(process_cfg, monkeypatch):
    cfg = process_cfg
    cfg.control_root.mkdir(parents=True)
    atomic_json(cfg.control_root / "launch.json", {
        "phase": "waiting_actor",
        "request_id": "request",
        "owner_pid": 999,
        "owner_start": "gone",
    })
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: None)
    monkeypatch.setattr("hilserl.processes.discover", lambda root: [])

    status = Manager(cfg).status()

    assert status["launch"]["phase"] == "fault"
    assert "启动入口已退出" in status["launch"]["error"]
