import copy
import json
import threading
import time

import numpy as np
import pytest

from hilserl.episode_commit import EpisodeCommitServer, demo_subset, envelope
from hilserl.episode_reward import EpisodeRewardPipeline
from hilserl.learning_replay import ContractReplayStore, validate_transition
from hilserl.reward_provider import RoboMeterClient, relabel, RewardSpec, digest
from hilserl.reward_seed import prepare_seed_cache, iter_reward_seed
from scripts.serve_robometer import make_server
from reward_fakes import SPEC, Provider, Store, Transport, episode


def wait_until(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for test event")
        time.sleep(.01)


def request(operation, **fields):
    return dict(operation=operation, run_id="test-run", actor_attempt="actor", gap_id="gap",
                contract_sha256=SPEC.sha256, generation="generation-1", **fields)


def server(tmp_path, online=None, demos=None, generation="generation-1"):
    return EpisodeCommitServer(tmp_path, "test-run", generation, SPEC,
        online if online is not None else Store(), demos if demos is not None else Store(),
        authorize=lambda identity: identity == "actor")


def prepared(raw=None):
    raw = raw or episode()
    return envelope("test-run", "source-actor", "000001", raw, relabel(raw, Provider().score(raw), SPEC), SPEC)


def test_actual_http_batches_are_causal_and_batch_size_does_not_change_scores():
    seen = []

    class Backend:
        def predict_progress_samples(self, samples):
            values = [sample["trajectory"]["frames"][:, 0, 0, 0].tolist() for sample in samples]
            seen.extend(values)
            return [[pixel / 255 for pixel in row] for row in values]

    http = make_server(("127.0.0.1", 0), Backend(), SPEC.model_id)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{http.server_port}"
        raw = episode(12)
        all_results = []
        for batch in (1, 4):
            client = RoboMeterClient(SPEC, url, batch_size=batch, timeout=3)
            try:
                result = client.score(raw)
                all_results.append(result)
            finally:
                client.close()
        assert all_results[0]["scores"] == all_results[1]["scores"]
        assert len(all_results[0]["scores"]) == 13
        for i, values in enumerate(seen):
            query = i % 13
            assert values[0] == 1 and values[-1] == query + 1
            assert max(values) <= query + 1 and len(values) <= 8
        wrong = RoboMeterClient(RewardSpec(model_id="b" * 64, image_profile=SPEC.image_profile), url)
        try:
            with pytest.raises(ValueError, match="identity"):
                wrong.health()
        finally:
            wrong.close()
    finally:
        http.shutdown(); http.server_close(); thread.join(timeout=2)


def test_negative_failure_reward_is_not_human_success_and_sparse_store_rejects_it():
    raw = episode(outcome=0)
    labeled = relabel(raw, Provider().score(raw), SPEC)
    assert labeled[-1]["rewards"] < 0
    assert demo_subset(labeled) == []
    for item in labeled:
        validate_transition(item, SPEC.image_profile, reward_spec=SPEC)
    with pytest.raises(ValueError, match="reward"):
        validate_transition(labeled[-1], SPEC.image_profile)
    assert raw[-1]["rewards"] == 0


@pytest.mark.parametrize("field", ["images", "prefixes", "scores"])
def test_misaligned_noncausal_or_nonfinite_result_is_rejected(field):
    raw = episode()
    scored = Provider().score(raw)
    if field == "images":
        raw[0]["observations"]["side_policy"][0, 0, 0, 0] = 99
    elif field == "prefixes":
        scored["prefixes"][0] = [0, 2]
    else:
        scored["scores"][0] = float("nan")
    with pytest.raises(ValueError):
        relabel(raw, scored, SPEC)


def test_pause_ack_waits_for_current_gpu_group_and_release_is_idempotent(tmp_path):
    service = server(tmp_path)
    assert service.begin_update()
    begin = service.handle(request("begin", source_attempt="source-actor", episode_id="000001"))
    assert begin["success"] and not begin["paused"]
    assert not service.begin_update()
    assert service.handle(request("commit", episode=prepared()))["not_ready"]
    checks = []
    service.complete_update(lambda: checks.append(service.snapshot()["paused"]), 8)
    assert checks == [False]
    assert service.handle(request("status"))["paused"]
    result = service.handle(request("commit", episode=prepared()))
    assert result["success"] and len(service.online) == 3
    assert service.handle(request("commit", episode=prepared())) == result
    assert len(service.online) == 3
    wrong = service.handle(request("release", receipt=dict(result["receipt"], generation="old")))
    assert not wrong["success"] and not service.begin_update()
    payload = request("release", receipt=result["receipt"])
    assert service.handle(payload)["released"]
    assert service.handle(payload)["released"]
    assert service.begin_update()


def test_partial_second_store_failure_blocks_updates_and_restart_rebuilds_both(tmp_path):
    raw = episode(outcome=1)
    raw[0]["infos"]["source_action"] = "human"
    value = prepared(raw)
    service = server(tmp_path, demos=Store(fail_at=1))
    service.handle(request("begin", source_attempt="source-actor", episode_id="000001"))
    response = service.handle(request("commit", episode=value))
    assert not response["success"] and service.snapshot()["error"]
    assert len(service.online) == 3 and len(service.demos) == 1
    assert not service.begin_update()
    restored = server(tmp_path, generation="generation-2")
    assert len(restored.online) == 3 and len(restored.demos) == 2
    assert restored.snapshot()["released_to"] is None
    assert all(receipt["generation"] == "generation-2" for receipt in restored.committed.values())


def test_bad_batch_is_validated_before_any_insert(tmp_path):
    service = server(tmp_path)
    service.handle(request("begin", source_attempt="source-actor", episode_id="000001"))
    value = prepared()
    value["transitions"][1]["actions"][0] = float("nan")
    assert not service.handle(request("commit", episode=value))["success"]
    assert len(service.online) == 0


def test_raw_ids_cannot_be_resubmitted_as_a_different_episode_identity(tmp_path):
    service = server(tmp_path)
    service.handle(request("begin", source_attempt="source-actor", episode_id="000001"))
    receipt = service.handle(request("commit", episode=prepared()))["receipt"]
    service.handle(request("release", receipt=receipt))
    service.handle(request("begin", source_attempt="different-source", episode_id="000001"))
    value = prepared()
    value["source_attempt"] = "different-source"
    value["payload_digest"] = digest({k: v for k, v in value.items() if k != "payload_digest"})
    assert not service.handle(request("commit", episode=value))["success"]
    assert len(service.online) == 3 and len(list(tmp_path.glob("*.pkl"))) == 1


def test_reward_seed_cache_roundtrip_and_wrong_contract_rejected(tmp_path):
    raw = episode(outcome=1)
    directory = tmp_path / "seed"
    prepare_seed_cache(iter(raw), directory, "f" * 64, SPEC, Provider())
    loaded = list(iter_reward_seed(iter(raw), directory, "f" * 64, SPEC))
    assert len(loaded) == 3 and loaded[-1]["rewards"] > 0
    assert raw[-1]["rewards"] == 1
    with pytest.raises(ValueError, match="version"):
        list(iter_reward_seed(iter(raw), directory, "e" * 64, SPEC))


def test_worker_waits_for_pause_commits_once_after_lost_acks_and_releases_after_reset(tmp_path):
    service = server(tmp_path / "journal")
    service.begin_update()
    scoring = threading.Event()
    pipe = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "status.json",
        "test-run", "actor", lambda: Provider(callback=scoring.set),
        lambda: Transport(service, drop={"commit", "release"}), timeout=3)
    raw = episode()
    # run_episodes provides only unfinalized steps during collection.
    raw[-1].update(rewards=0.0, dones=False, masks=1.0)
    raw[-1]["infos"].update(succeed=False, manual_success=False, verdict_source=None)
    try:
        pipe.begin("000001")
        for item in raw:
            pipe.append(item, {})
        pipe.end_collection()
        wait_until(lambda: service.snapshot()["gap"] is not None)
        assert not scoring.is_set()
        service.complete_update(lambda: None, 2)
        assert scoring.wait(2)
        assert len(service.online) == 0  # Human verdict has not arrived yet.
        pipe.finalize(0, "manual")
        assert pipe.committed.wait(3), pipe.error
        assert not service.begin_update()
        assert len(service.online) == 3
        from test_session_recording import ScriptedOperator
        class Recorder:
            def check(self): pass
            def event(self, *args, **kwargs): pass
        pipe.barrier(ScriptedOperator(), Recorder())
        assert len(service.online) == 3 and len(service.demos) == 0
        assert service.begin_update()
    finally:
        pipe.close()


def test_actor_reset_overlaps_reward_and_next_episode_waits_for_commit(tmp_path):
    from hilserl.episodes import run_episodes
    from hilserl.storage import SessionRecorder
    from test_session_recording import FakeEnvironment, ScriptedOperator
    from reward_fakes import observation
    scoring, release, reset_done = threading.Event(), threading.Event(), threading.Event()
    service = server(tmp_path / "journal")
    events, errors = [], []

    def score():
        scoring.set()
        assert release.wait(4)

    class Environment(FakeEnvironment):
        def observation(self): return observation(self.steps + 1)
        def reset(self, *, options):
            _, info = super().reset()
            if self.resets == 2:
                events.append("reset-complete")
                reset_done.set()
            options["hilserl_ready_barrier"](check=lambda: None)
            if self.resets == 2:
                assert release.is_set() and len(service.online) == 2
                events.append("refresh")
            return observation(99), info
        def step(self, action):
            if self.resets == 2:
                assert len(service.online) == 2
                events.append("next-action")
            return super().step(action)

    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    env = Environment(terminal_at=2, recorder=recorder)
    pipe = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "status.json",
        "test-run", "actor", lambda: Provider(callback=score), lambda: Transport(service), timeout=3)
    emitted = []

    def sample(obs, step):
        if env.steps == 0:
            assert obs["side_policy"][0, 0, 0, 0] == 99
        return np.zeros(3), {}

    def run():
        try:
            run_episodes(env, sample, recorder, ScriptedOperator(),
                action_contract="fixed-xyz-v1", max_episodes=2, episode_reward=pipe, emit=emitted.append)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert reset_done.wait(3) and scoring.wait(3), errors
        assert events == ["reset-complete"] and len(service.online) == 0
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and not errors, errors
        assert not emitted  # No per-step streaming in episode mode.
        assert len(service.online) == 4  # Includes the final episode drain.
        assert events.index("reset-complete") < events.index("refresh") < events.index("next-action")
        assert env.closed and env.resets == 2
        saved = json.loads((tmp_path / "recording/manifest.json").read_text())
        assert len(saved["episodes"]) == 2
    finally:
        release.set(); pipe.close(); thread.join(timeout=5)
