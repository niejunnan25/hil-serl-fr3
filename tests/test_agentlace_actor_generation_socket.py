"""Real local REQ/REP transport coverage; no Actor, JAX, GPU, or device I/O."""
from contextlib import ExitStack
import json
import socket
import time
from types import SimpleNamespace

import pytest

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer


@pytest.fixture
def transport(request):
    pipeline = bool(getattr(request, "param", False))
    # Reserve distinct ephemeral ports together before Agentlace binds.
    with ExitStack() as stack:
        ports = []
        for _ in range(3 if pipeline else 2):
            reserved = stack.enter_context(socket.socket())
            reserved.bind(("127.0.0.1", 0))
            ports.append(reserved.getsockname()[1])
    config = TrainerConfig(port_number=ports[0], broadcast_port=ports[1],
                           experimental_pipeline_port=ports[2] if pipeline else None)
    callbacks = []
    callback_failures = {}

    def callback(name, payload):
        callbacks.append((name, payload))
        if name in callback_failures:
            raise callback_failures[name]

    server = TrainerServer(config, data_callback=callback)
    stores = {name: QueuedDataStore(1024) for name in ("online", "human")}
    for name, store in stores.items():
        server.register_data_store(name, store)
    server.start(threaded=True)
    clients = []

    def client(*, stores=None, client_id=None, timeout_ms=800):
        result = TrainerClient("actor", "127.0.0.1", config,
                               data_stores=stores or {"online": QueuedDataStore(1024)},
                               client_id=client_id, timeout_ms=timeout_ms)
        clients.append(result)
        return result

    yield SimpleNamespace(server=server, stores=stores, client=client, config=config,
                          callbacks=callbacks, callback_failures=callback_failures)
    if pipeline:
        # Consumer.stop() otherwise waits indefinitely on a blocking recv.
        # Wake the existing socket after latching stop; an invalid store makes
        # this teardown message unable to alter any tested datastore.
        from agentlace.zmq_wrapper.pipeline import Producer
        wake = Producer("127.0.0.1", ports[2])
        server.consumer.is_kill = True
        wake.send_msg(dict(type="datastore", store_name="__test_shutdown__",
                           payload={"data": [], "last_id": -1}, client_id="teardown"))
        server.consumer.thread.join(timeout=2)
        assert not server.consumer.thread.is_alive()
        consumer_context = server.consumer.results_receiver.context
        server.consumer.results_receiver.close(linger=0)
        consumer_context.term()
        wake_context = wake.zmq_socket.context
        wake.zmq_socket.close(linger=0)
        wake_context.term()
    # TrainerClient.stop currently stops workers; explicitly close its REQ
    # transport as well so teardown cannot depend on garbage collection.
    for item in clients:
        item.stop()
        item.req_rep_client.socket.close(linger=0)
        item.req_rep_client.context.term()
        if pipeline:
            producer_context = item.producer.zmq_socket.context
            item.producer.zmq_socket.close(linger=0)
            producer_context.term()
    server.stop()
    server.thread.join(timeout=2)
    assert not server.thread.is_alive()


def test_new_actor_first_batch_starts_at_zero_after_previous_cursor_155(transport):
    old = transport.client()
    old_store = old.data_stores_map["online"]
    old_store.batch_insert([{"generation": "old", "sequence": i} for i in range(156)])
    assert old.update() is True
    assert old.get_server_last_update_id("online") == 155
    assert len(transport.stores["online"]) == 156

    new_store = QueuedDataStore(1024)
    first = {"generation": "new", "sequence": 0}
    new_store.insert(first)
    # The constructor's first update exercises the actual Actor restart path.
    new = transport.client(stores={"online": new_store})
    assert new.client_id != old.client_id
    assert new.get_server_last_update_id("online") == 0
    assert old.get_server_last_update_id("online") == 155
    assert transport.stores["online"].get_latest_data(155) == [first]
    assert transport.server.get_data_activity("online")["received_count"] == 157


def test_retry_of_committed_last_id_is_idempotent_and_does_not_refresh_activity(transport):
    client = transport.client(client_id="stable-actor-attempt")
    local = client.data_stores_map["online"]
    local.batch_insert([{"step": 0}, {"step": 1}])
    assert client.update() is True
    before = transport.server.get_data_activity("online")
    callback_count = len(transport.callbacks)

    # Pretend the response was lost: explicitly retransmit the same batch and
    # last_id through the real socket, rather than letting update skip it.
    assert client.update_datastore("online", from_id=-1) is True
    assert transport.stores["online"].get_latest_data(-1) == [{"step": 0}, {"step": 1}]
    assert transport.server.get_data_activity("online") == before
    assert len(transport.callbacks) == callback_count

    reconnect = transport.client(stores={"online": local}, client_id=client.client_id)
    assert reconnect.get_server_last_update_id("online") == 1
    assert transport.server.get_data_activity("online") == before
    assert len(transport.stores["online"]) == 2
    local.insert({"step": 2})
    assert reconnect.update() is True
    after = transport.server.get_data_activity("online")
    assert after["received_count"] == 3
    assert after["last_received_monotonic"] > before["last_received_monotonic"]


def test_cursors_are_isolated_by_both_client_and_store(transport):
    first_stores = {name: QueuedDataStore(1024) for name in ("online", "human")}
    second_stores = {name: QueuedDataStore(1024) for name in ("online", "human")}
    first = transport.client(stores=first_stores, client_id="actor-a")
    second = transport.client(stores=second_stores, client_id="actor-b")

    first_stores["online"].batch_insert(["a-online-0", "a-online-1"])
    first_stores["human"].insert("a-human-0")
    assert first.update() is True
    assert first.get_server_last_update_id("online") == 1
    assert first.get_server_last_update_id("human") == 0
    assert second.get_server_last_update_id("online") == -1
    assert second.get_server_last_update_id("human") == -1

    human_activity = transport.server.get_data_activity("human")
    second_stores["online"].insert("b-online-0")
    assert second.update() is True
    assert second.get_server_last_update_id("online") == 0
    assert second.get_server_last_update_id("human") == -1
    assert first.get_server_last_update_id("online") == 1
    assert transport.server.get_data_activity("human") == human_activity
    assert transport.stores["online"].get_latest_data(-1) == ["a-online-0", "a-online-1", "b-online-0"]
    second_stores["human"].insert("b-human-0")
    assert second.update() is True
    assert transport.stores["human"].get_latest_data(-1) == ["a-human-0", "b-human-0"]
    assert transport.server.get_data_activity("online")["received_count"] == 3
    assert transport.server.get_data_activity("human")["received_count"] == 2


def test_bootstrap_polling_empty_and_rejected_requests_do_not_count_as_new_data(transport):
    transport.stores["online"].batch_insert(["bootstrap"] * 7)
    client = transport.client()
    empty = dict(received_count=0, last_received_monotonic=None, client_id=None)
    assert transport.server.get_data_activity("online") == empty
    assert client.get_server_last_update_id("online") == -1
    assert client.get_network() is None
    assert client.update() is True
    assert transport.server.get_data_activity("online") == empty

    # Exercise an empty wire payload too, not only the client's empty shortcut.
    callback_count = len(transport.callbacks)
    result = client._update_ds("online", {"data": [], "last_id": 0})
    assert result["success"] is True
    assert transport.server.get_data_activity("online") == empty
    assert client.get_server_last_update_id("online") == -1
    assert len(transport.callbacks) == callback_count
    result = client._update_ds("unregistered", {"data": ["rejected"], "last_id": 0})
    assert result["success"] is False
    assert transport.server.get_data_activity("online") == empty
    assert transport.server.get_data_activity("unregistered") == empty
    assert len(transport.stores["online"]) == 7

    # An empty packet cannot acknowledge this same client's not-yet-sent seq0.
    client.data_stores_map["online"].insert("new-online")
    assert client.update() is True
    activity = transport.server.get_data_activity("online")
    assert activity["received_count"] == 1 and activity["last_received_monotonic"] is not None
    assert activity["client_id"] == client.client_id
    assert transport.stores["online"].get_latest_data(6) == ["new-online"]


def test_multistore_update_reports_rejected_batch_without_refreshing_failed_store(transport):
    local = {name: QueuedDataStore(1024) for name in ("online", "human")}
    client = transport.client(stores=local)
    local["online"].insert("accepted")
    local["human"].insert("rejected")
    original = transport.server.req_rep_server.impl_callback

    def reject_human(request):
        if request.get("type") == "datastore" and request.get("store_name") == "human":
            return {"success": False, "message": "injected insertion rejection"}
        return original(request)

    # Rejection is a real wire ACK; only the server's store boundary is injected.
    transport.server.req_rep_server.impl_callback = reject_human
    assert client.update() is False
    assert transport.stores["online"].get_latest_data(-1) == ["accepted"]
    assert len(transport.stores["human"]) == 0
    assert transport.server.get_data_activity("online")["received_count"] == 1
    assert transport.server.get_data_activity("human")["received_count"] == 0
    assert transport.server.get_data_activity("human")["last_received_monotonic"] is None
    assert client.get_server_last_update_id("human") == -1

    # After rejection is removed, the failed store's same first batch remains
    # eligible. The already accepted first store must not be counted twice.
    before = transport.server.get_data_activity("online")
    transport.server.req_rep_server.impl_callback = original
    assert client.update() is True
    assert transport.server.get_data_activity("online") == before
    assert transport.stores["human"].get_latest_data(-1) == ["rejected"]
    assert transport.server.get_data_activity("human")["received_count"] == 1


class PartialFailureStore(QueuedDataStore):
    def __init__(self):
        super().__init__(1024)
        self.fail_next = True
        self.batch_calls = 0

    def batch_insert(self, batch):
        self.batch_calls += 1
        if self.fail_next:
            self.insert(batch[0])
            raise OSError("injected failure after first insertion")
        return super().batch_insert(batch)


@pytest.mark.parametrize("failure", ["partial_insert", "callback"])
def test_insertion_failure_freezes_store_without_killing_policy_requests(transport, failure):
    store = PartialFailureStore()
    store.fail_next = failure == "partial_insert"
    transport.server.register_data_store("online", store)
    if failure == "callback":
        transport.callback_failures["online"] = RuntimeError("injected callback failure")
    client = transport.client()
    transport.server.publish_network({"weights": [1, 2]})
    payload = {"data": ["attempt-0", "attempt-1"], "last_id": 1}
    response = client._update_ds("online", payload)
    assert response["success"] is False and response["delivery_unknown"] is True
    assert "injected" in response["message"]
    assert client.get_server_last_update_id("online") == -1
    initial_data = store.get_latest_data(-1)
    assert initial_data == (["attempt-0"] if failure == "partial_insert" else ["attempt-0", "attempt-1"])
    activity = transport.server.get_data_activity("online")
    assert activity["received_count"] == 0 and activity["last_received_monotonic"] is None
    assert "injected" in activity["error"]

    # Removing the injected failure cannot silently clear an ambiguous commit.
    # Both the same attempt and another client remain frozen for this store.
    store.fail_next = False
    transport.callback_failures.clear()
    assert client._update_ds("online", payload)["success"] is False
    different = transport.client()
    assert different._update_ds("online", payload)["success"] is False
    assert store.batch_calls == 1 and store.get_latest_data(-1) == initial_data
    assert transport.server.get_data_activity("online") == activity

    assert transport.server.thread.is_alive()
    assert client.get_network() == {"params": {"weights": [1, 2]}, "revision": 1}
    assert client._update_ds("human", {"data": ["healthy-store"], "last_id": 0})["success"] is True
    assert transport.stores["human"].get_latest_data(-1) == ["healthy-store"]


def test_training_gate_pauses_on_latched_partial_commit_even_with_fresh_prior_data(transport, tmp_path):
    from hilserl.training_gate import TrainingGate

    store = PartialFailureStore()
    store.fail_next = False
    transport.server.register_data_store("actor_env", store)
    local = QueuedDataStore(1024)
    client = transport.client(stores={"actor_env": local}, client_id="active-actor")
    record = dict(pid=123, start_time="start-123", attempt_id=client.client_id,
                  role="actor", mode="train", run_dir=str(tmp_path.resolve()))
    (tmp_path / "control").mkdir()
    (tmp_path / "actor.json").write_text(json.dumps(record))
    (tmp_path / "control/status.json").write_text(json.dumps(dict(record, phase="collecting",
                                                                 collection_started_monotonic_ns=time.monotonic_ns(),
                                                                 monotonic_ns=time.monotonic_ns())))
    gate = TrainingGate(tmp_path, transport.server.get_data_activity,
                        process_info=lambda pid: ("start-123", "R"))
    local.insert("valid-online")
    assert client.update() is True
    assert gate.check()["allowed"] is True
    activity = transport.server.get_data_activity("actor_env")

    store.fail_next = True
    local.batch_insert(["partial-first", "not-inserted"])
    assert client.update() is False
    failed = transport.server.get_data_activity("actor_env")
    assert failed["received_count"] == activity["received_count"] == 1
    assert failed["last_received_monotonic"] == activity["last_received_monotonic"]
    result = gate.check(manual_paused=False)
    assert result["allowed"] is False and result["pause_code"] == "data_store_error"
    assert result["save_checkpoint"] is True
    assert "injected failure" in result["pause_reason"]


def test_intervention_store_error_blocks_gate_but_absent_human_arrivals_do_not(transport, tmp_path):
    from hilserl.training_gate import TrainingGate

    transport.server.register_data_store("actor_env", QueuedDataStore(1024))
    intervention = PartialFailureStore()
    transport.server.register_data_store("actor_env_intvn", intervention)
    local = {name: QueuedDataStore(1024) for name in ("actor_env", "actor_env_intvn")}
    client = transport.client(stores=local, client_id="active-actor")
    record = dict(pid=123, start_time="start-123", attempt_id=client.client_id,
                  role="actor", mode="train", run_dir=str(tmp_path.resolve()))
    (tmp_path / "control").mkdir()
    (tmp_path / "actor.json").write_text(json.dumps(record))
    (tmp_path / "control/status.json").write_text(json.dumps(dict(record, phase="collecting",
                                                                 collection_started_monotonic_ns=time.monotonic_ns(),
                                                                 monotonic_ns=time.monotonic_ns())))
    gate = TrainingGate(tmp_path, transport.server.get_data_activity,
                        process_info=lambda pid: ("start-123", "R"))
    local["actor_env"].insert("policy-step")
    assert client.update() is True
    assert transport.server.get_data_activity("actor_env_intvn")["received_count"] == 0
    assert gate.check()["allowed"] is True

    local["actor_env"].insert("human-step")
    local["actor_env_intvn"].insert("human-step")
    assert client.update() is False
    assert transport.server.get_data_activity("actor_env")["received_count"] == 2
    decision = gate.check()
    assert decision["allowed"] is False and decision["pause_code"] == "data_store_error"
    assert decision["save_checkpoint"] is True
    assert "actor_env_intvn" in decision["pause_reason"]


def eventually(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), "Pipeline consumer did not acknowledge expected cursor"


def test_new_collection_requires_a_new_committed_batch_not_a_previous_window_retry(transport, tmp_path):
    from hilserl.training_gate import TrainingGate

    transport.server.register_data_store("actor_env", QueuedDataStore(1024))
    local = QueuedDataStore(1024)
    client = transport.client(stores={"actor_env": local}, client_id="same-actor")
    local.insert("previous-episode-last-step")
    assert client.update() is True
    previous = transport.server.get_data_activity("actor_env")

    record = dict(pid=123, start_time="start-123", attempt_id=client.client_id,
                  role="actor", mode="train", run_dir=str(tmp_path.resolve()))
    (tmp_path / "control").mkdir()
    (tmp_path / "actor.json").write_text(json.dumps(record))
    state = dict(record, phase="collecting", collection_started_monotonic_ns=time.monotonic_ns(),
                 monotonic_ns=time.monotonic_ns())
    (tmp_path / "control/status.json").write_text(json.dumps(state))
    gate = TrainingGate(tmp_path, transport.server.get_data_activity,
                        process_info=lambda _pid: ("start-123", "R"))
    decision = gate.check()
    assert decision["allowed"] is False and decision["save_checkpoint"] is False
    assert decision["pause_code"] == "waiting_online_data"

    assert client.update_datastore("actor_env", from_id=-1) is True
    assert transport.server.get_data_activity("actor_env") == previous
    assert gate.check()["allowed"] is False
    assert gate.check()["save_checkpoint"] is False

    local.insert("this-episode-first-step")
    assert client.update() is True
    assert gate.check()["allowed"] is True


@pytest.mark.parametrize("transport", [True], indirect=True)
def test_pipeline_duplicate_and_empty_batches_share_reqrep_commit_semantics(transport):
    transport.server.register_data_store("barrier", QueuedDataStore(1024))
    client = transport.client(client_id="pipeline-actor")
    payload = {"data": ["first", "second"], "last_id": 1}
    assert client._update_ds("online", payload)["success"] is True
    eventually(lambda: client.get_server_last_update_id("online") == 1)
    activity = transport.server.get_data_activity("online")
    online_callbacks = [call for call in transport.callbacks if call[0] == "online"]
    assert len(online_callbacks) == 1

    # A later marker on this same PUSH socket proves that the earlier duplicate
    # or empty message has been consumed, without relying on arbitrary sleeps.
    client._update_ds("online", payload)
    client._update_ds("barrier", {"data": ["duplicate-processed"], "last_id": 0})
    eventually(lambda: client.get_server_last_update_id("barrier") == 0)
    assert transport.stores["online"].get_latest_data(-1) == ["first", "second"]
    assert transport.server.get_data_activity("online") == activity

    client._update_ds("online", {"data": [], "last_id": 900})
    client._update_ds("barrier", {"data": ["empty-processed"], "last_id": 1})
    eventually(lambda: client.get_server_last_update_id("barrier") == 1)
    assert client.get_server_last_update_id("online") == 1
    assert transport.server.get_data_activity("online") == activity
    assert [call for call in transport.callbacks if call[0] == "online"] == online_callbacks

    client._update_ds("online", {"data": ["third"], "last_id": 2})
    eventually(lambda: client.get_server_last_update_id("online") == 2)
    assert transport.stores["online"].get_latest_data(-1) == ["first", "second", "third"]
    assert transport.server.get_data_activity("online")["received_count"] == 3
