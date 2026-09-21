"""Actor/process/transport agreement, without GPU, sockets or robot access."""
import json

import pytest

from hilserl.training_gate import TrainingGate


@pytest.fixture
def scenario(tmp_path):
    (tmp_path / "control").mkdir()
    record = dict(pid=123, start_time="1234", attempt_id="actor-one", role="actor", mode="train", run_dir=str(tmp_path.resolve()))
    state = dict(record, phase="collecting", monotonic_ns=100_000_000_000,
                 collection_started_monotonic_ns=90_000_000_000)
    activity = dict(received_count=1, last_received_monotonic=100.0, client_id="actor-one")
    clock = [100.0]
    process = ["1234", "R"]
    gate = TrainingGate(tmp_path, lambda name: activity, clock=lambda: clock[0], process_info=lambda pid: tuple(process))

    def write():
        (tmp_path / "actor.json").write_text(json.dumps(record))
        (tmp_path / "control" / "status.json").write_text(json.dumps(state))

    write()
    return gate, record, state, activity, clock, process, write


def test_only_collecting_live_identity_with_fresh_own_data_is_admitted(scenario):
    gate, _, _, _, _, _, _ = scenario
    result = gate.check()
    assert result["allowed"] and result["pause_reason"] is None
    assert result["online_received_count"] == 1
    assert result["online_data_age_seconds"] == result["actor_heartbeat_age_seconds"] == 0


@pytest.mark.parametrize("phase", ["starting", "waiting_reset", "resetting", "waiting_controller", "paused", "awaiting_label", "fault", "stopped"])
def test_non_collecting_phase_pauses_even_with_recent_data(scenario, phase):
    gate, _, state, _, _, _, write = scenario
    state["phase"] = phase
    write()
    assert gate.check()["pause_code"] == "actor_not_collecting"


@pytest.mark.parametrize("change", [{"pid": 124}, {"start_time": "reused"}, {"attempt_id": "old"}])
def test_status_must_match_run_actor_identity(scenario, change):
    gate, _, state, _, _, _, write = scenario
    state.update(change)
    write()
    assert gate.check()["pause_code"] == "actor_identity_missing"


@pytest.mark.parametrize("process", [("reused", "R"), (None, None), ("1234", "Z"), ("1234", "X")])
def test_dead_zombie_or_reused_pid_pauses(scenario, process):
    gate, _, _, _, _, actual, _ = scenario
    actual[:] = process
    assert gate.check()["pause_code"] == "actor_not_alive"


@pytest.mark.parametrize("age", [2.001, 40, -1])
def test_heartbeat_stale_or_future_pauses(scenario, age):
    gate, _, state, _, _, _, write = scenario
    state["monotonic_ns"] = int((100 - age) * 1e9)
    write()
    assert gate.check()["pause_code"] == "actor_heartbeat_stale"


@pytest.mark.parametrize("change,code", [
    ({"received_count": 0, "last_received_monotonic": None}, "waiting_online_data"),
    ({"last_received_monotonic": 97.999}, "online_data_stale"),
    ({"last_received_monotonic": 101}, "online_data_stale"),
    ({"last_received_monotonic": float("nan")}, "online_data_stale"),
    ({"client_id": "previous-actor"}, "waiting_online_data"),
    ({"received_count": True}, "data_activity_invalid"),
])
def test_replay_prefill_or_old_transport_receipt_cannot_start_training(scenario, change, code):
    gate, _, _, activity, _, _, _ = scenario
    activity.update(change)
    assert gate.check()["pause_code"] == code


def test_actor_restart_requires_new_attempt_transport_receipt(scenario):
    gate, record, state, activity, _, _, write = scenario
    assert gate.check()["allowed"]
    record["attempt_id"] = state["attempt_id"] = "actor-two"
    write()
    assert gate.check()["pause_code"] == "waiting_online_data"
    activity.update(client_id="actor-two", received_count=2)
    assert gate.check()["allowed"]


def test_manual_resume_never_bypasses_automatic_gate(scenario):
    gate, _, state, _, _, _, write = scenario
    assert gate.check(manual_paused=True)["pause_code"] == "manual_pause"
    state["phase"] = "fault"
    write()
    assert gate.check(manual_paused=False)["pause_code"] == "actor_not_collecting"


def test_no_run_directory_and_malformed_state_fail_closed(scenario):
    gate, _, _, _, _, _, _ = scenario
    (gate.run_dir / "control" / "status.json").write_text("{broken")
    assert gate.check()["pause_code"] == "actor_identity_missing"
    assert TrainingGate(None, lambda name: {}).check()["pause_code"] == "run_identity_missing"


def test_normal_episode_wait_does_not_request_checkpoint_even_without_recent_heartbeat(scenario):
    gate, _, state, _, clock, _, write = scenario
    state["phase"] = "waiting_reset"
    clock[0] = 1000
    write()
    result = gate.check()
    assert not result["allowed"]
    assert result["save_checkpoint"] is False
    state["robot_state_error"] = "503 stale controller"
    write()
    assert gate.check()["save_checkpoint"] is True


@pytest.mark.parametrize("old_data_age", [8.5, 0.25])
def test_new_collection_waits_for_its_first_batch_without_a_checkpoint(scenario, old_data_age):
    gate, _, state, activity, clock, _, write = scenario
    state["phase"] = "resetting"
    activity["last_received_monotonic"] = clock[0] - old_data_age
    write()
    assert gate.check()["save_checkpoint"] is False

    state.update(phase="collecting", collection_started_monotonic_ns=int(clock[0] * 1e9))
    write()
    decision = gate.check()
    assert decision["allowed"] is False
    assert decision["pause_code"] == "waiting_online_data"
    assert decision["save_checkpoint"] is False

    clock[0] += 0.1
    activity.update(received_count=2, last_received_monotonic=clock[0])
    state["monotonic_ns"] = int(clock[0] * 1e9)
    write()
    assert gate.check()["allowed"] is True

    clock[0] += 2.001
    state["monotonic_ns"] = int(clock[0] * 1e9)
    write()
    decision = gate.check()
    assert decision["allowed"] is False
    assert decision["pause_code"] == "online_data_stale"
    assert decision["save_checkpoint"] is True


def test_first_batch_already_arrived_before_gate_first_reads_new_window(scenario):
    gate, _, state, activity, _, _, write = scenario
    state["collection_started_monotonic_ns"] = 99_900_000_000
    activity.update(received_count=2, last_received_monotonic=99.95)
    write()
    # No earlier gate check observes the reset or establishes an activity baseline.
    assert gate.check()["allowed"] is True


@pytest.mark.parametrize("start", [None, True, "100", -1, 100_000_000_001])
def test_invalid_collection_window_fails_closed(scenario, start):
    gate, _, state, _, _, _, write = scenario
    state["collection_started_monotonic_ns"] = start
    write()
    decision = gate.check()
    assert decision["allowed"] is False
    assert decision["pause_code"] == "collection_window_invalid"


@pytest.mark.parametrize("phase", ["stopped", "fault"])
def test_fault_stop_and_manual_pause_keep_checkpoint_protection_before_first_batch(scenario, phase):
    gate, _, state, activity, _, _, write = scenario
    state.update(phase=phase, collection_started_monotonic_ns=100_000_000_000)
    activity["last_received_monotonic"] = 91.5
    write()
    assert gate.check()["save_checkpoint"] is True
    assert gate.check(manual_paused=True)["pause_code"] == "manual_pause"
    assert gate.check(manual_paused=True)["save_checkpoint"] is True
