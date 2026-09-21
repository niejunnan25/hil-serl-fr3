"""Runtime behavior without importing JAX or accessing a device."""
import ast
import copy
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hilserl.control import StopRequested

ROOT = Path(__file__).resolve().parents[1]


def functions(*names, **globals):
    tree = ast.parse((ROOT / "_run_actor.py").read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = dict(os=os, np=np, time=time, threading=threading, copy=copy, **globals)
    exec(compile(module, str(ROOT / "_run_actor.py"), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("cancel", [False, True])
def test_no_policy_never_enters_reset_or_step(monkeypatch, cancel):
    monkeypatch.setenv("HILSERL_MODE", "train")
    monkeypatch.setenv("ACTOR_LEARNER_PARAMS_TIMEOUT", "0")
    events = []
    stops = []
    class Operator:
        state = {"phase":"starting"}
        def publish(self, phase, **fields):
            self.state.update(phase=phase, **fields)
        def raise_if_stop(self):
            if cancel:
                raise StopRequested()
    class Client:
        def __init__(self, *args, **kwargs):
            assert kwargs["wait_for_server"] is False
        def recv_network_callback(self, callback):
            pass
        def stop(self):
            stops.append("stop")
    base = SimpleNamespace(recorder=SimpleNamespace(check=lambda:events.append("check")), operator=Operator())
    env = SimpleNamespace(unwrapped=base, reset=lambda:events.append("reset"), step=lambda a:events.append("step"))
    flags = SimpleNamespace(ip="127.0.0.1",eval_checkpoint_step=0)
    actor = functions("actor", FLAGS=flags, TrainerClient=Client, _make_runtime_trainer_config=lambda:{})["actor"]
    with pytest.raises(StopRequested if cancel else RuntimeError):
        actor(object(),None,None,env,None)
    assert events == ["check"]
    assert stops == ["stop"]


def test_initial_network_retry_republishes_for_late_actor():
    ns = functions("_start_initial_network_retry")
    published = []
    stop, thread = ns["_start_initial_network_retry"](
        lambda reason: published.append(reason), period=0.01, grace_seconds=0.25
    )
    thread.join(timeout=1)
    stop.set()
    assert not thread.is_alive()
    assert published and set(published) == {"startup-retry"}


@pytest.mark.parametrize("state", [
    {"dq": np.full(7,.36), "tcp_force": np.zeros(3)},
    {"dq": np.zeros(7), "tcp_force": np.array([46,0,0])},
])
def test_raw_state_limit_prevents_next_action(state):
    ns = functions("_as_array","_extract_state","_extract_relz","_check_runtime_safety",
                   ACTOR_SAFETY_DQ_MAX=.35, ACTOR_SAFETY_FORCE_MAX=45,
                   ACTOR_RELZ_ABS_MAX=.35, ACTOR_SAFETY_LOG_PERIOD=1e9,
                   SERL_FLAT_RELZ_STATE_INDEX=6, _LAST_SAFETY_LOG=time.time())
    with pytest.raises(RuntimeError, match="actor_safety_fatal"):
        ns["_check_runtime_safety"]({"state":np.zeros((1,19))},0,state)


def test_success_credit_keeps_raw_parent_and_does_not_mutate_raw():
    source = {"observations":{"state":np.arange(19)}, "actions":np.ones(7),
              "rewards":0.0,"dones":False,"masks":1.0,
              "infos":{"raw_transition_id":"run/cap/ep/0","source_action":"human"}}
    transform = functions("_make_success_credit_transition")["_make_success_credit_transition"]
    credited = transform(source)
    assert source["rewards"] == 0
    assert "derived" not in source["infos"]
    assert credited["infos"]["derived"]
    assert credited["infos"]["parent_raw_transition_id"] == source["infos"]["raw_transition_id"]
