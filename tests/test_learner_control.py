"""Learner shutdown protocol tests; no process signals, training or hardware I/O."""
import json
import signal

import pytest

import hilserl.learner_control as module
from hilserl.learner_control import LearnerControl, request_activity, request_stop


@pytest.fixture
def control(tmp_path):
    return LearnerControl(tmp_path, pid=101, start_time="start-1", attempt_id="attempt-1")


def read(path):
    return json.loads(path.read_text())


def request(control, **changes):
    value = dict(command="stop", pid=101, start_time="start-1", attempt_id="attempt-1",
                 request_id="request-1", **{"protocol_version": 1})
    value.update(changes)
    control.request_path.write_text(json.dumps(value))
    return value


def test_initial_status_is_owned_and_does_not_claim_a_checkpoint(control):
    state = read(control.status_path)
    assert state["phase"] == "initializing"
    assert (state["pid"], state["start_time"], state["attempt_id"]) == (101, "start-1", "attempt-1")
    assert state["protocol_version"] == 1
    assert state["revision"] == 1
    assert state["unix_ns"] > 0 and state["monotonic_ns"] > 0
    assert state["checkpoint_saved"] is False
    assert state["checkpoint_path"] is None
    assert state["stop_requested"] is False
    assert control.stop_requested() is False
    assert not control.request_path.exists()


def test_default_identity_uses_current_process_and_attempt_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(module.os, "getpid", lambda: 321)
    monkeypatch.setattr(module, "_process_start", lambda pid: "real-start" if pid == 321 else None)
    monkeypatch.setenv("HILSERL_ATTEMPT_ID", "env-attempt")
    item = LearnerControl(tmp_path)
    assert (item.state["pid"], item.state["start_time"], item.state["attempt_id"]) == (321, "real-start", "env-attempt")


@pytest.mark.parametrize("changes", [
    {"pid": 102}, {"pid": "101"}, {"pid": True}, {"start_time": "old-start"},
    {"start_time": None}, {"attempt_id": "old-attempt"}, {"attempt_id": None},
    {"command": "save"}, {"request_id": None}, {"request_id": ""},
    {"request_id": 123}, {"protocol_version": 2}, {"protocol_version": True},
])
def test_stop_requires_complete_exact_identity_and_valid_request(control, changes):
    request(control, **changes)
    assert control.stop_requested() is False
    assert control.state["phase"] == "initializing"
    assert control.state["revision"] == 1


@pytest.mark.parametrize("text", ["{broken", "[]", "null", "\"stop\"", "x" * 65537])
def test_malformed_request_cannot_stop_or_crash_learner(control, text):
    control.request_path.write_text(text)
    assert control.stop_requested() is False


def test_accepted_stop_is_latched_and_duplicate_polling_does_not_change_status(control):
    control.publish("training", step=41)
    request(control)
    assert control.stop_requested() is True
    state = read(control.status_path)
    assert state["phase"] == "stop_requested"
    assert state["step"] == 41
    assert state["stop_request_id"] == "request-1"
    assert state["stop_requested"] is True
    assert state["stop_source"] == "file"
    assert state["checkpoint_saved"] is False
    assert control.stop_requested() is True
    assert read(control.status_path) == state
    control.request_path.unlink()
    assert control.stop_requested() is True
    assert read(control.status_path) == state


def test_duplicate_stop_never_regresses_saving_or_closing_phase(control):
    request(control)
    assert control.stop_requested()
    for phase in ("saving_checkpoint", "closing", "stopped"):
        control.publish(phase, step=42)
        previous = read(control.status_path)
        assert control.stop_requested()
        assert read(control.status_path) == previous


def test_stopped_and_closing_do_not_imply_checkpoint_saved(control):
    for phase in ("stop_requested", "saving_checkpoint", "closing", "stopped"):
        control.publish(phase, step=42, checkpoint_path="checkpoints/checkpoint_42")
        assert read(control.status_path)["checkpoint_saved"] is False


def test_only_explicit_completed_save_receipt_marks_checkpoint_saved(control, tmp_path):
    path = tmp_path / "checkpoints" / "checkpoint_42"
    control.publish("saving_checkpoint", step=42, checkpoint_path=path)
    assert control.state["checkpoint_saved"] is False
    # The runtime supplies this receipt only after its real save function returns.
    control.publish("closing", checkpoint_saved=True, checkpoint_path=path)
    control.publish("stopped")
    state = read(control.status_path)
    assert state["checkpoint_saved"] is True
    assert state["checkpoint_path"] == str(path)
    assert state["checkpoint_error"] is None
    assert state["step"] == 42


def test_failed_new_save_clears_previous_receipt_and_preserves_error(control):
    control.publish("training", checkpoint_path="checkpoint_40", checkpoint_saved=True)
    control.publish("saving_checkpoint", step=42, checkpoint_path="checkpoint_42")
    assert control.state["checkpoint_saved"] is False
    control.publish("fault", checkpoint_error="disk full")
    control.publish("stopped")
    state = read(control.status_path)
    assert state["checkpoint_saved"] is False
    assert state["checkpoint_error"] == "disk full"


def test_cleanup_fault_does_not_erase_successful_save_receipt(control):
    control.publish("closing", checkpoint_path="checkpoint_42", checkpoint_saved=True)
    control.publish("fault", error="worker join timed out")
    assert control.state["checkpoint_saved"] is True
    assert control.state["checkpoint_error"] is None


@pytest.mark.parametrize("fields", [
    {"pid": 200}, {"attempt_id": "overwrite"}, {"revision": 9},
    {"stop_requested": True}, {"checkpoint_saved": 1},
    {"checkpoint_saved": True},
])
def test_publish_rejects_owned_fields_and_invalid_save_receipts(control, fields):
    with pytest.raises(ValueError):
        control.publish(**fields)
    assert read(control.status_path)["revision"] == 1


def test_publish_rejects_unknown_phase_and_saved_while_saving(control):
    with pytest.raises(ValueError, match="Unknown"):
        control.publish("finished_maybe")
    with pytest.raises(ValueError, match="while saving"):
        control.publish("saving_checkpoint", checkpoint_path="checkpoint_42", checkpoint_saved=True)


def test_request_writer_is_idempotent_and_does_not_touch_learner_owned_status(control):
    before = read(control.status_path)
    first = request_stop(control.run_dir, pid=101, start_time="start-1", attempt_id="attempt-1", request_id="explicit")
    again = request_stop(control.run_dir, pid=101, start_time="start-1", attempt_id="attempt-1", request_id="retry")
    assert first == again == read(control.request_path)
    assert first["request_id"] == "explicit"
    assert read(control.status_path) == before
    assert control.stop_requested()


def test_old_request_is_replaced_for_new_attempt_but_cannot_stop_wrong_learner(control):
    request(control, attempt_id="old")
    assert not control.stop_requested()
    value = request_stop(control.run_dir, pid=101, start_time="start-1", attempt_id="attempt-1")
    assert value["attempt_id"] == "attempt-1"
    assert control.stop_requested()


@pytest.mark.parametrize("changes", [
    {"pid": None}, {"pid": 0}, {"pid": True}, {"start_time": None},
    {"start_time": ""}, {"attempt_id": ""}, {"request_id": ""},
])
def test_request_writer_rejects_incomplete_identity(control, changes):
    arguments = dict(pid=101, start_time="start-1", attempt_id="attempt-1")
    arguments.update(changes)
    with pytest.raises(ValueError):
        request_stop(control.run_dir, **arguments)
    assert not control.request_path.exists()


def fake_signals(monkeypatch):
    handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
    original = dict(handlers)
    calls = []
    monkeypatch.setattr(module.signal, "getsignal", handlers.__getitem__)

    def install(signum, handler):
        calls.append((signum, handler))
        handlers[signum] = handler

    monkeypatch.setattr(module.signal, "signal", install)
    return handlers, original, calls


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_handlers_latch_without_io_or_raise_and_restore(control, monkeypatch, signum):
    handlers, original, calls = fake_signals(monkeypatch)
    control.publish("training", step=12)
    before = read(control.status_path)
    with control.signal_handlers() as active:
        assert active is control
        handlers[signum](signum, None)
        handlers[signum](signum, None)
        assert read(control.status_path) == before
        assert control.stop_requested()
        assert control.state["phase"] == "stop_requested"
        assert control.state["stop_signal"] == int(signum)
        assert control.state["stop_source"] == "signal"
    assert handlers == original
    assert [entry[0] for entry in calls] == [signal.SIGINT, signal.SIGTERM, signal.SIGTERM, signal.SIGINT]


def test_late_file_request_after_signal_adds_receipt_without_regressing_phase(control, monkeypatch):
    handlers, _original, _calls = fake_signals(monkeypatch)
    with control.signal_handlers():
        handlers[signal.SIGINT](signal.SIGINT, None)
        assert control.stop_requested()
        control.publish("saving_checkpoint", step=42)
        request(control)
        assert control.stop_requested()
        assert control.state["phase"] == "saving_checkpoint"
        assert control.state["stop_request_id"] == "request-1"
        assert control.state["stop_signal"] == int(signal.SIGINT)


def test_signal_handlers_restore_when_body_raises(control, monkeypatch):
    handlers, original, _calls = fake_signals(monkeypatch)
    with pytest.raises(RuntimeError, match="training failure"):
        with control.signal_handlers():
            raise RuntimeError("training failure")
    assert handlers == original


def test_partial_signal_installation_rolls_back(control, monkeypatch):
    handlers, original, _calls = fake_signals(monkeypatch)
    install = module.signal.signal

    def fail_second(signum, handler):
        if signum == signal.SIGTERM:
            raise ValueError("main thread required")
        install(signum, handler)

    monkeypatch.setattr(module.signal, "signal", fail_second)
    with pytest.raises(ValueError, match="main thread"):
        with control.signal_handlers():
            pytest.fail("must not enter")
    assert handlers == original


def test_no_run_directory_still_supports_cooperative_signal_shutdown(monkeypatch):
    handlers, original, _calls = fake_signals(monkeypatch)
    item = LearnerControl(pid=101, start_time="start-1", attempt_id="standalone")
    assert item.status_path is None and item.request_path is None
    with item.signal_handlers():
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert item.stop_requested()
        item.publish("stopped")
    assert item.state["checkpoint_saved"] is False
    assert handlers == original


def test_manual_pause_resume_targets_identity_and_acknowledges_each_request(control):
    arguments = dict(pid=101, start_time="start-1", attempt_id="attempt-1")
    pause = request_activity(control.run_dir, command="pause", **arguments)
    assert control.activity_paused()
    assert control.state["activity_request_id"] == pause["request_id"]
    assert request_activity(control.run_dir, command="pause", **arguments) == pause
    before = read(control.status_path)
    assert control.activity_paused() and read(control.status_path) == before
    resume = request_activity(control.run_dir, command="resume", **arguments)
    assert resume["request_id"] != pause["request_id"]
    assert not control.activity_paused()
    assert control.state["activity_request_id"] == resume["request_id"]


def test_old_attempt_activity_command_cannot_pause_new_learner(control):
    request_activity(control.run_dir, command="pause", pid=101, start_time="start-1", attempt_id="old-attempt")
    assert not control.activity_paused()
    assert control.state["activity_request_id"] is None


def test_paused_learner_still_acknowledges_stop(control):
    control.publish("paused", step=99, pause_reason="Actor offline")
    request_stop(control.run_dir, pid=101, start_time="start-1", attempt_id="attempt-1")
    assert control.stop_requested()
    assert control.state["phase"] == "stop_requested"
    assert control.state["step"] == 99


def test_manual_initial_pause_and_owned_protocol_fields(tmp_path):
    control = LearnerControl(tmp_path, pid=101, start_time="s", attempt_id="a", initial_paused=True)
    assert control.activity_paused()
    for fields in ({"manual_paused": False}, {"activity_request_id": "forged"}):
        with pytest.raises(ValueError, match="owned"):
            control.publish(**fields)


@pytest.mark.parametrize("changes", [
    {"pid": True}, {"pid": 0}, {"attempt_id": ""}, {"attempt_id": False},
    {"start_time": ""}, {"start_time": 123},
])
def test_constructor_rejects_invalid_explicit_identity(tmp_path, changes):
    arguments = dict(pid=101, start_time="start-1", attempt_id="attempt-1")
    arguments.update(changes)
    with pytest.raises(ValueError):
        LearnerControl(tmp_path, **arguments)
    assert not (tmp_path / "control" / "learner_status.json").exists()


def test_failed_status_write_preserves_last_complete_state(control, monkeypatch):
    before = read(control.status_path)
    real_replace = module.os.replace

    def fail_replace(source, destination):
        if destination == control.status_path:
            raise OSError("disk unavailable")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        control.publish("saving_checkpoint", step=42)
    assert control.state == before
    assert read(control.status_path) == before
    assert not list(control.directory.glob("*.partial"))
