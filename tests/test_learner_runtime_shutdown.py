"""Execute the actual Learner shutdown path without GPU, sockets or OS signals.

Only runtime function definitions are compiled from _run_actor.py. The real
LearnerControl and checkpoint completion checks run against temporary files;
training, progress bars, background workers and the checkpoint backend are fakes.
"""
import ast
from contextlib import contextmanager, nullcontext
import gc
import itertools
import json
import os
from pathlib import Path
import platform
import queue
import signal
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import hilserl.learner_control as control_module
from hilserl.learner_control import LearnerControl, request_stop
from hilserl.training_gate import TrainingGate


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def collect_with_callback(callback):
    """Run an actual CPython GC callback while preserving global GC settings."""
    enabled = gc.isenabled()
    gc.disable()

    def on_collection(phase, _info):
        if phase == "start":
            callback()

    gc.callbacks.append(on_collection)
    try:
        yield gc.collect
    finally:
        gc.callbacks.remove(on_collection)
        if enabled:
            gc.enable()


class FakeSignals:
    SIGINT = signal.SIGINT
    SIGTERM = signal.SIGTERM

    def __init__(self):
        self.original = {self.SIGINT: object(), self.SIGTERM: object()}
        self.handlers = dict(self.original)

    def getsignal(self, signum):
        return self.handlers[signum]

    def signal(self, signum, handler):
        previous = self.handlers[signum]
        self.handlers[signum] = handler
        return previous

    def call(self, signum=SIGINT):
        # Deliberately call the Python function: never os.kill/raise_signal.
        self.handlers[signum](signum, None)


class Runtime:
    def __init__(self, tmp_path, monkeypatch):
        self.events = []
        self.signals = FakeSignals()
        monkeypatch.setattr(control_module, "signal", self.signals)
        self.control = LearnerControl(tmp_path / "run", pid=321, start_time="runtime-start", attempt_id="runtime-attempt")
        self.checkpoint_dir = tmp_path / "checkpoints"
        self.checkpoint_dir.mkdir()
        self.config = SimpleNamespace(
            training_starts=1, batch_size=4, max_steps=4, cta_ratio=1,
            steps_per_update=10, log_period=10, checkpoint_period=0,
        )
        self.replay_size = 1
        self.update_count = 0
        self.sleep_count = 0
        self.on_update = lambda _count: None
        self.on_sleep = lambda: None
        self.on_save = lambda: None
        self.save_mode = "committed"
        self.stop_error = None
        self.start_step = 0
        self.sleep_limit = 2
        self.actor_phase = "collecting"
        self.actor_alive = True
        self.online_count = 1
        self.actor_status()
        runtime = self

        class Agent:
            def __init__(self, completed=0):
                self.state = SimpleNamespace(params={"completed_updates": completed})

            def update(self, batch, *, networks_to_update):
                count = runtime.update_count
                runtime.events.append(("update_start", count, networks_to_update))
                runtime.on_update(count)
                runtime.update_count += 1
                updated = Agent(runtime.update_count)
                runtime.events.append(("update_end", count, networks_to_update))
                return updated, {"loss": 0.0}

        class Buffer:
            def __init__(self, name):
                self.name = name

            def __len__(self):
                return runtime.replay_size if self.name == "replay" else 1

            def get_iterator(self, **_kwargs):
                runtime.events.append(("iterator", self.name))
                return itertools.repeat({"buffer": self.name})

        class Server:
            def __init__(self, *_args, **_kwargs):
                runtime.events.append(("server_create",))
                runtime.server = self
                self.snapshot = None

            def register_data_store(self, name, _store):
                runtime.events.append(("server_register", name))

            def start(self, **_kwargs):
                runtime.events.append(("server_start",))

            def get_data_activity(self, _name):
                return dict(received_count=runtime.online_count, last_received_monotonic=time.monotonic(),
                            client_id="actor-attempt")

            def publish_network(self, _params):
                self.snapshot = _params
                runtime.events.append(("publish",))

            def broadcast_network(self, _params):
                runtime.events.append(("broadcast",))

            def stop(self):
                runtime.events.append(("server_stop", runtime.status()))
                if runtime.stop_error:
                    raise runtime.stop_error

        class Worker:
            def __init__(self, *, target=None, name="startup-retry", daemon=True):
                self.name = name

            def start(self):
                runtime.events.append(("worker_start", self.name))

            def join(self, *, timeout):
                runtime.events.append(("worker_join", self.name, timeout))

        class Progress:
            def __init__(self, values=(), *, initial=0, **_kwargs):
                self.values = values
                self.n = initial

            def __iter__(self):
                return iter(self.values)

            def update(self, delta):
                self.n += delta

            def close(self):
                runtime.events.append(("progress_close",))

        class Timer:
            def context(self, _name):
                return nullcontext()

            def get_average_times(self):
                return {}

        self.agent_type = Agent
        self.agent = Agent()
        self.replay = Buffer("replay")
        self.demo = Buffer("demo")
        self.training_gate = TrainingGate(
            self.control.run_dir,
            lambda name: dict(received_count=self.online_count, last_received_monotonic=time.monotonic(),
                              client_id="actor-attempt"),
            process_info=lambda pid: ("actor-start", "R") if self.actor_alive else (None, None),
        )
        namespace = dict(
            os=os, queue=queue,
            threading=SimpleNamespace(Event=threading.Event, Lock=threading.Lock, Thread=Worker),
            time=SimpleNamespace(monotonic=time.monotonic, sleep=self.sleep),
            tqdm=SimpleNamespace(tqdm=Progress), config=self.config,
            FLAGS=SimpleNamespace(checkpoint_path=str(self.checkpoint_dir)),
            AGENTLACE_PORT=5588, AGENTLACE_BROADCAST_PORT=5589,
            TrainerServer=Server, _make_runtime_trainer_config=lambda: {},
            _start_initial_network_retry=lambda *_args, **_kwargs: (threading.Event(), Worker()),
            print_green=lambda _text: None, sharding=SimpleNamespace(replicate=lambda: None),
            Timer=Timer, SACAgent=Agent, concat_batches=lambda first, second, **_kwargs: (first, second),
            jax=SimpleNamespace(block_until_ready=self.block_until_ready),
            checkpoints=SimpleNamespace(save_checkpoint=self.save_checkpoint),
        )
        tree = ast.parse((ROOT / "_run_actor.py").read_text())
        names = {"learner", "_save_training_checkpoint", "_save_final_checkpoint"}
        body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in body} == names
        exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / "_run_actor.py"), "exec"), namespace)
        self.namespace = namespace
        self.learner = namespace["learner"]

    def status(self):
        return json.loads(self.control.status_path.read_text())

    def stop_file(self):
        return request_stop(
            self.control.run_dir, pid=321, start_time="runtime-start", attempt_id="runtime-attempt",
        )

    def actor_status(self, phase=None):
        if phase is not None:
            self.actor_phase = phase
        record = dict(pid=654, start_time="actor-start", attempt_id="actor-attempt", role="actor", mode="train",
                      run_dir=str(self.control.run_dir.resolve()))
        (self.control.run_dir / "actor.json").write_text(json.dumps(record))
        status = dict(record, phase=self.actor_phase, monotonic_ns=time.monotonic_ns(),
                      collection_started_monotonic_ns=1)
        (self.control.directory / "status.json").write_text(json.dumps(status))

    def sleep(self, seconds):
        self.sleep_count += 1
        assert self.sleep_count <= self.sleep_limit, "Learner failed to observe stop while paused"
        self.events.append(("buffer_wait", seconds))
        self.on_sleep()

    def block_until_ready(self, value):
        self.events.append(("block_until_ready",))
        return value

    def save_checkpoint(self, directory, state, *, step, keep, overwrite):
        self.events.append(("save_start", step, state.params, self.status()))
        self.on_save()
        if self.save_mode == "raise":
            raise OSError("simulated disk full")
        path = Path(directory) / f"checkpoint_{step}"
        if self.save_mode != "missing":
            path.mkdir()
            metadata = {"commit_timestamp_nsecs": time.time_ns()} if self.save_mode == "committed" else {}
            (path / "_CHECKPOINT_METADATA").write_text(json.dumps(metadata))
        self.events.append(("save_end", step, str(path)))
        return str(path)

    def run(self):
        with self.control.signal_handlers():
            self.learner(None, self.agent, self.replay, self.demo, control=self.control,
                         start_step=self.start_step, training_gate=self.training_gate)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    return Runtime(tmp_path, monkeypatch)


@pytest.mark.parametrize("trigger", ["file", "gc_handler"])
@pytest.mark.parametrize("cta_ratio", [1, 3])
def test_stop_during_update_finishes_current_step_and_never_begins_next(runtime, monkeypatch, trigger, cta_ratio):
    runtime.config.cta_ratio = cta_ratio
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def request_during_first_update(count):
        if count:
            return
        if trigger == "file":
            runtime.stop_file()
        else:
            with collect_with_callback(runtime.signals.call) as collect:
                collect()
        runtime.events.append(("stop_issued_during_update",))

    runtime.on_update = request_during_first_update
    runtime.run()

    assert runtime.update_count == cta_ratio
    assert unraisable == []
    names = [entry[0] for entry in runtime.events]
    assert names.index("update_start") < names.index("stop_issued_during_update") < names.index("update_end")
    assert names.count("save_start") == names.count("save_end") == 1
    assert names.index("save_end") < names.index("worker_join") < names.index("server_stop")
    receipt = runtime.status()
    assert receipt["phase"] == "stopped"
    assert receipt["step"] == 0
    assert receipt["checkpoint_saved"] is True
    assert receipt["checkpoint_error"] is None
    assert receipt["stop_requested"] is True
    assert receipt["stop_source"] == ("file" if trigger == "file" else "signal")
    saved = next(entry for entry in runtime.events if entry[0] == "save_start")
    assert saved[1] == 0
    assert saved[2] == {"completed_updates": cta_ratio}
    assert saved[3]["phase"] == "saving_checkpoint"
    assert saved[3]["checkpoint_saved"] is False
    server_stop_state = next(entry[1] for entry in runtime.events if entry[0] == "server_stop")
    assert server_stop_state["phase"] == "closing"
    assert server_stop_state["checkpoint_saved"] is True
    assert runtime.signals.handlers == runtime.signals.original


def test_one_completed_step_at_zero_is_saved_on_natural_completion(runtime):
    runtime.config.max_steps = 1
    runtime.run()
    assert runtime.update_count == 1
    receipt = runtime.status()
    assert receipt["phase"] == "stopped"
    assert receipt["step"] == 0
    assert receipt["checkpoint_saved"] is True
    assert receipt["stop_requested"] is False
    assert Path(receipt["checkpoint_path"]).name == "checkpoint_0"
    assert json.loads((Path(receipt["checkpoint_path"]) / "_CHECKPOINT_METADATA").read_text())["commit_timestamp_nsecs"] > 0


@pytest.mark.parametrize("save_mode", ["raise", "missing", "uncommitted"])
def test_failed_or_uncommitted_save_reports_fault_and_still_closes_transport(runtime, save_mode):
    runtime.save_mode = save_mode
    runtime.on_update = lambda count: runtime.stop_file() if count == 0 else None
    with pytest.raises(RuntimeError, match="checkpoint save failed"):
        runtime.run()
    assert runtime.update_count == 1
    receipt = runtime.status()
    assert receipt["phase"] == "fault"
    assert receipt["checkpoint_saved"] is False
    assert receipt["checkpoint_error"]
    names = [entry[0] for entry in runtime.events]
    assert names.index("save_start") < names.index("server_stop")
    stop_state = next(entry[1] for entry in runtime.events if entry[0] == "server_stop")
    assert stop_state["phase"] == "fault"
    assert stop_state["checkpoint_saved"] is False
    assert runtime.signals.handlers == runtime.signals.original


def test_uncommitted_existing_checkpoint_cannot_be_overwritten_or_reported_saved(runtime):
    existing = runtime.checkpoint_dir / "checkpoint_0"
    existing.mkdir()
    metadata = existing / "_CHECKPOINT_METADATA"
    metadata.write_text("{}")
    runtime.on_update = lambda count: runtime.stop_file() if count == 0 else None
    with pytest.raises(RuntimeError, match="no committed save receipt"):
        runtime.run()
    assert metadata.read_text() == "{}"
    assert not any(entry[0] == "save_start" for entry in runtime.events)
    assert runtime.status()["phase"] == "fault"
    assert runtime.status()["checkpoint_saved"] is False


@pytest.mark.parametrize("trigger", ["file", "handler"])
def test_stop_while_waiting_for_initial_replay_never_trains_or_claims_a_save(runtime, trigger):
    runtime.replay_size = 0
    runtime.on_sleep = runtime.stop_file if trigger == "file" else runtime.signals.call
    runtime.run()
    assert runtime.sleep_count == 1
    assert runtime.update_count == 0
    names = [entry[0] for entry in runtime.events]
    assert "iterator" not in names and "save_start" not in names
    assert "server_stop" in names
    receipt = runtime.status()
    assert receipt["phase"] == "stopped"
    assert receipt["step"] == -1
    assert receipt["checkpoint_saved"] is False
    assert receipt["checkpoint_path"] is None
    assert receipt["stop_requested"] is True


def test_duplicate_stops_during_save_cannot_interrupt_save_or_regress_its_phase(runtime):
    first = []

    def stop_update(count):
        if count == 0:
            first.append(runtime.stop_file())

    def repeat_during_save():
        for _ in range(3):
            assert runtime.stop_file() == first[0]
            runtime.signals.call(signal.SIGINT)
            runtime.signals.call(signal.SIGTERM)
            assert runtime.control.stop_requested()
            assert runtime.status()["phase"] == "saving_checkpoint"
            assert runtime.status()["checkpoint_saved"] is False

    runtime.on_update = stop_update
    runtime.on_save = repeat_during_save
    runtime.run()
    names = [entry[0] for entry in runtime.events]
    assert runtime.update_count == 1
    assert names.count("save_start") == names.count("save_end") == 1
    receipt = runtime.status()
    assert receipt["phase"] == "stopped"
    assert receipt["checkpoint_saved"] is True
    assert receipt["stop_request_id"] == first[0]["request_id"]


def test_transport_cleanup_failure_preserves_successful_checkpoint_receipt(runtime):
    runtime.on_update = lambda count: runtime.stop_file() if count == 0 else None
    runtime.stop_error = RuntimeError("simulated server cleanup failure")
    runtime.run()
    receipt = runtime.status()
    assert receipt["phase"] == "fault"
    assert receipt["checkpoint_saved"] is True
    assert receipt["checkpoint_error"] is None
    assert "Transport cleanup failed" in receipt["error"]
    assert Path(receipt["checkpoint_path"]).exists()


@pytest.mark.skipif(platform.python_implementation() != "CPython", reason="CPython GC callback exception semantics")
def test_gc_swallows_legacy_keyboard_interrupt_but_cooperative_handler_survives(runtime, monkeypatch):
    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    with collect_with_callback(lambda: signal.default_int_handler(signal.SIGINT, None)) as collect:
        # CPython reports the callback exception as unraisable and returns.
        collect()
    assert len(unraisable) == 1
    assert unraisable[0].exc_type is KeyboardInterrupt
    assert runtime.control.stop_requested() is False
    unraisable.clear()
    before = runtime.status()
    with runtime.control.signal_handlers():
        with collect_with_callback(runtime.signals.call) as collect:
            collect()
        assert unraisable == []
        assert runtime.status() == before  # The callback itself performs no I/O.
        assert runtime.control.stop_requested() is True
        assert runtime.status()["phase"] == "stop_requested"
    assert runtime.signals.handlers == runtime.signals.original


def test_actor_fault_pauses_updates_and_resume_preserves_step_and_buffers(runtime):
    runtime.sleep_limit = 5
    runtime.on_update = lambda count: runtime.actor_status("fault") if count == 0 else None

    def paused():
        receipt = runtime.status()
        assert receipt["phase"] == "paused"
        assert receipt["step"] == 0 and receipt["next_step"] == 1
        assert runtime.update_count == 1
        assert receipt["checkpoint_saved"] is True
        assert receipt["pause_code"] == "actor_not_collecting"
        assert not any(event[0] == "server_stop" for event in runtime.events)
        if runtime.sleep_count == 3:
            runtime.actor_status("collecting")

    runtime.on_sleep = paused
    runtime.run()
    assert runtime.update_count == 4
    saves = [event for event in runtime.events if event[0] == "save_start"]
    assert [event[1] for event in saves] == [0, 3]
    assert saves[-1][2] == {"completed_updates": 4}
    assert [event[1] for event in runtime.events if event[0] == "iterator"] == ["replay", "demo"]


def test_normal_reset_first_batch_wait_does_not_save_but_later_stream_loss_does(runtime):
    clock, received, window_started = [100.0], [100.0], [90.0]
    runtime.sleep_limit = 3
    runtime.training_gate = TrainingGate(
        runtime.control.run_dir,
        lambda _name: dict(received_count=1, last_received_monotonic=received[0], client_id="actor-attempt"),
        clock=lambda: clock[0], process_info=lambda _pid: ("actor-start", "R"),
    )

    def publish(phase):
        runtime.actor_status(phase)
        path = runtime.control.directory / "status.json"
        state = json.loads(path.read_text())
        state.update(monotonic_ns=int(clock[0] * 1e9),
                     collection_started_monotonic_ns=int(window_started[0] * 1e9))
        path.write_text(json.dumps(state))

    def during_update(count):
        if count == 0:
            clock[0] = 108.5
            publish("resetting")
        elif count == 1:
            clock[0] = 111.0
            publish("collecting")  # Fresh Actor heartbeat, real 2.4 second stream loss.

    def during_pause():
        saves = [event for event in runtime.events if event[0] == "save_start"]
        if runtime.sleep_count == 1:
            assert runtime.status()["pause_code"] == "actor_not_collecting"
            assert not saves
            window_started[0] = clock[0]
            publish("collecting")
        elif runtime.sleep_count == 2:
            assert runtime.status()["pause_code"] == "waiting_online_data"
            assert runtime.update_count == 1 and not saves
            clock[0] = received[0] = 108.6
            publish("collecting")
        else:
            assert runtime.status()["pause_code"] == "online_data_stale"
            assert runtime.update_count == 2
            assert [event[1] for event in saves] == [1]
            runtime.stop_file()

    publish("collecting")
    runtime.on_update, runtime.on_sleep = during_update, during_pause
    runtime.run()
    assert runtime.update_count == 2
    assert [event[1] for event in runtime.events if event[0] == "save_start"] == [1]
    assert runtime.status()["checkpoint_saved"] is True


@pytest.mark.parametrize("cause", ["fault", "waiting_reset", "resetting", "dead", "no_data"])
def test_stop_while_automatically_paused_never_updates_or_loses_request(runtime, cause):
    if cause == "dead":
        runtime.actor_alive = False
    elif cause == "no_data":
        runtime.online_count = 0
    else:
        runtime.actor_status(cause)

    def pause_then_stop():
        assert runtime.status()["phase"] == "paused"
        assert runtime.update_count == 0
        runtime.stop_file()

    runtime.on_sleep = pause_then_stop
    runtime.run()
    assert runtime.update_count == 0 and runtime.sleep_count == 1
    assert runtime.status()["stop_requested"] is True
    assert runtime.status()["checkpoint_saved"] is False


def test_pause_save_failure_is_visible_and_closes_without_retrying_write(runtime):
    runtime.save_mode = "raise"
    runtime.on_update = lambda count: runtime.actor_status("fault") if count == 0 else None
    with pytest.raises(RuntimeError, match="Pause checkpoint save failed"):
        runtime.run()
    assert runtime.update_count == 1
    assert runtime.status()["phase"] == "fault"
    assert runtime.status()["checkpoint_error"]
    assert len([event for event in runtime.events if event[0] == "save_start"]) == 1
    assert any(event[0] == "server_stop" for event in runtime.events)


def test_explicit_resume_starts_at_next_step_and_initial_pause_preserves_checkpoint(runtime):
    runtime.start_step = 10
    runtime.config.max_steps = 12
    runtime.agent = runtime.agent_type(10)
    runtime.update_count = 10
    previous = runtime.checkpoint_dir / "checkpoint_9"
    previous.mkdir()
    (previous / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1}')
    runtime.control.publish("initializing", step=9, next_step=10, checkpoint_saved=True, checkpoint_path=previous)
    runtime.actor_status("waiting_reset")

    def resume_collecting():
        assert runtime.status()["step"] == 9 and runtime.status()["next_step"] == 10
        assert runtime.status()["checkpoint_saved"] is True
        assert runtime.update_count == 10
        runtime.actor_status("collecting")

    runtime.on_sleep = resume_collecting
    runtime.run()
    assert runtime.update_count == 12
    assert runtime.status()["step"] == 11
    assert (previous / "_CHECKPOINT_METADATA").read_text() == '{"commit_timestamp_nsecs": 1}'
    assert [event[1] for event in runtime.events if event[0] == "save_start"] == [11]


def test_initial_policy_publisher_remains_available_during_pause(runtime):
    runtime.namespace["threading"].Thread = threading.Thread
    runtime.actor_status("waiting_reset")

    def snapshot_then_stop():
        deadline = time.monotonic() + 1
        while runtime.server.snapshot is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runtime.server.snapshot == {"completed_updates": 0}
        assert runtime.status()["phase"] == "paused"
        assert runtime.update_count == 0
        assert not any(event[0] == "server_stop" for event in runtime.events)
        runtime.stop_file()

    runtime.on_sleep = snapshot_then_stop
    runtime.run()
    assert runtime.status()["phase"] == "stopped"


@pytest.mark.parametrize("phase", ["waiting_reset", "awaiting_label", "resetting"])
def test_normal_episode_wait_does_not_add_a_large_checkpoint(runtime, phase):
    runtime.on_update = lambda count: runtime.actor_status(phase) if count == 0 else None

    def resume():
        assert runtime.status()["phase"] == "paused"
        assert not any(event[0] == "save_start" for event in runtime.events)
        runtime.actor_status("collecting")

    runtime.on_sleep = resume
    runtime.run()
    assert [event[1] for event in runtime.events if event[0] == "save_start"] == [3]


def test_robot_fault_discovered_while_already_paused_saves_only_once(runtime):
    runtime.sleep_limit = 4
    runtime.on_update = lambda count: runtime.actor_status("waiting_reset") if count == 0 else None

    def wait():
        if runtime.sleep_count == 1:
            assert not any(event[0] == "save_start" for event in runtime.events)
            path = runtime.control.directory / "status.json"
            status = json.loads(path.read_text())
            status["robot_state_error"] = "controller stale"
            path.write_text(json.dumps(status))
        else:
            assert runtime.status()["phase"] == "paused"
            assert runtime.status()["checkpoint_saved"] is True
            if runtime.sleep_count == 3:
                runtime.stop_file()

    runtime.on_sleep = wait
    runtime.run()
    assert runtime.update_count == 1
    assert [event[1] for event in runtime.events if event[0] == "save_start"] == [0]


def test_failed_partial_update_saves_only_last_complete_group(runtime):
    runtime.config.cta_ratio = 3

    def fail_second_group(count):
        if count == 4:
            raise RuntimeError("simulated optimizer failure")

    runtime.on_update = fail_second_group
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        runtime.run()
    receipt = runtime.status()
    assert receipt["phase"] == "fault"
    assert receipt["step"] == 0
    assert receipt["checkpoint_saved"] is True
    saved = next(event for event in runtime.events if event[0] == "save_start")
    assert saved[1] == 0 and saved[2] == {"completed_updates": 3}
