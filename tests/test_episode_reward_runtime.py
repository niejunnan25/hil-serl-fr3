"""Exercise real loop definitions and real Agentlace transport without hardware."""
import json
from contextlib import ExitStack
import socket
import threading
import time

import numpy as np
import pytest

from hilserl.episode_commit import REQUEST, EpisodeCommitServer, EpisodeOnlyStore, envelope
from hilserl.episode_reward import EpisodeRewardPipeline, RewardTransport
from hilserl.reward_provider import relabel
from hilserl.training_gate import TrainingGate
from reward_fakes import SPEC, Provider, Store, Transport, episode
from test_episode_reward import wait_until
from test_training_gate import scenario
from test_replay_image_continuity import replay_class


def test_batch_receipt_stays_valid_between_commits_but_actor_heartbeat_does_not(scenario):
    gate, record, state, activity, clock, _, write = scenario
    batch = dict(generation="learner", gap=None, error=None, closed=False,
                 committed_episodes=1, online_count=190, released_to=record["attempt_id"])
    gate.episode_activity = lambda: batch
    activity["last_received_monotonic"] = 1.0
    clock[0] = 120
    state["monotonic_ns"] = 120_000_000_000
    write()
    assert gate.check()["allowed"]
    clock[0] += 3
    assert gate.check()["pause_code"] == "actor_heartbeat_stale"
    clock[0] -= 3
    batch["gap"] = {"id": "next-gap"}
    assert gate.check()["pause_code"] == "reward_gap"
    batch["gap"] = None
    batch["released_to"] = "old-actor"
    assert gate.check()["pause_code"] == "waiting_episode_commit"


def test_actual_learner_finishes_group_before_ack_and_keeps_receiving_during_pause(tmp_path, monkeypatch):
    from test_learner_runtime_shutdown import Runtime
    runtime = Runtime(tmp_path, monkeypatch)
    runtime.config.cta_ratio = 3
    runtime.config.max_steps = 2
    run_id = str(runtime.control.run_dir.resolve())
    service = EpisodeCommitServer(tmp_path / "journal", run_id, "learner", SPEC,
                                  Store(), Store(), authorize=lambda identity: identity == "actor-attempt")
    service.released_to = "actor-attempt"
    service.committed["previous"] = {"prior": True}
    service.online_count = 100
    runtime.training_gate.episode_activity = service.snapshot
    runtime.replay.sample = lambda **kwargs: {"sample": "online"}
    runtime.demo.sample = lambda **kwargs: {"sample": "demo"}
    runtime.namespace["jax"].device_put = lambda value, **kwargs: value
    request = dict(run_id=run_id, actor_attempt="actor-attempt", contract_sha256=SPEC.sha256,
                   gap_id="one-gap", generation="learner")
    saw = []

    def during_update(count):
        if count == 0:
            response = service.handle(dict(request, operation="begin", source_attempt="source", episode_id="000001"))
            assert response["success"] and not response["paused"]
            runtime.actor_status("resetting")
        if count < 3:
            assert not service.snapshot()["paused"]

    def during_pause():
        assert runtime.update_count == 3
        assert service.snapshot()["paused"]
        raw = episode()
        value = envelope(run_id, "source", "000001", raw, relabel(raw, Provider().score(raw), SPEC), SPEC)
        response = service.handle(dict(request, operation="commit", episode=value))
        assert response["success"]
        assert service.handle(dict(request, operation="release", receipt=response["receipt"]))["released"]
        saw.append("received-while-paused")
        runtime.actor_status("collecting")

    runtime.on_update, runtime.on_sleep = during_update, during_pause
    runtime.learner(None, runtime.agent, runtime.replay, runtime.demo, control=runtime.control,
                    training_gate=runtime.training_gate, episode_server=service)
    assert saw == ["received-while-paused"] and runtime.update_count == 6
    assert not any(event[0] == "iterator" for event in runtime.events)
    assert len([event for event in runtime.events if event[0] == "block_until_ready"]) >= 2


def test_real_req_rep_transaction_ack_and_streaming_route_rejection(tmp_path):
    from agentlace.trainer import TrainerConfig, TrainerServer
    from agentlace.zmq_wrapper.req_rep import ReqRepClient
    with ExitStack() as stack:
        ports = []
        for _ in range(2):
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0)); ports.append(sock.getsockname()[1])
    service = EpisodeCommitServer(tmp_path, "run", "learner", SPEC, Store(), Store(), authorize=lambda a: a == "actor")
    server = TrainerServer(TrainerConfig(port_number=ports[0], broadcast_port=ports[1], request_types=[REQUEST]),
                           request_callback=lambda kind, payload: service.handle(payload))
    server.register_data_store("actor_env", EpisodeOnlyStore())
    server.start(threaded=True)
    transport = RewardTransport("127.0.0.1", ports[0])
    request = dict(run_id="run", actor_attempt="actor", contract_sha256=SPEC.sha256,
                   gap_id="gap", generation="learner")
    try:
        assert transport.request(dict(request, operation="begin", source_attempt="source", episode_id="000001"))["paused"]
        raw = episode(outcome=1)
        value = envelope("run", "source", "000001", raw, relabel(raw, Provider().score(raw), SPEC), SPEC)
        first = transport.request(dict(request, operation="commit", episode=value))
        again = transport.request(dict(request, operation="commit", episode=value))
        assert first == again and first["receipt"]["demo_count"] == 1
        assert len(service.online) == 3
        assert transport.request(dict(request, operation="release", receipt=first["receipt"]))["released"]
        denied = transport.client.send_msg(dict(type="datastore", store_name="actor_env", client_id="legacy",
                                                payload=dict(last_id=0, data=[value["transitions"][0]])))
        assert not denied["success"] and len(service.online) == 3
    finally:
        transport.close(); server.stop(); server.thread.join(timeout=2)


def test_reward_transaction_samples_correct_images_and_dense_targets_from_native_replay(tmp_path, replay_class):
    import gymnasium as gym
    from hilserl.learning_replay import ContractReplayStore
    from reward_fakes import observation
    obs_space = gym.spaces.Dict({key: gym.spaces.Box(
        0 if value.dtype == np.uint8 else -1000,
        255 if value.dtype == np.uint8 else 1000, value.shape, value.dtype)
        for key, value in observation(0).items()})
    action_space = gym.spaces.Box(-1, 1, (3,), np.float32)

    def make():
        native = replay_class(obs_space, action_space, capacity=32,
                              image_keys=("side_policy", "wrist_1"))
        return ContractReplayStore(native, reward_spec=SPEC)

    online, demos = make(), make()
    service = EpisodeCommitServer(tmp_path, "run", "learner", SPEC, online, demos, authorize=lambda a: True)
    raw = episode()
    raw[1]["infos"]["source_action"] = "human"
    labeled = relabel(raw, Provider().score(raw), SPEC)
    value = envelope("run", "source", "000001", raw, labeled, SPEC)
    common = dict(run_id="run", actor_attempt="actor", contract_sha256=SPEC.sha256,
                  generation="learner", gap_id="gap")
    assert service.handle(dict(common, operation="begin", source_attempt="source", episode_id="000001"))["paused"]
    assert service.handle(dict(common, operation="commit", episode=value))["success"]
    assert online.transition_count == 3 and demos.transition_count == 1
    batch = online.sample(batch_size=128, pack_obs_and_next_obs=False)
    assert batch["actions"].shape == (128, 3)
    seen = set()
    for i, state in enumerate(batch["observations"]["state"]):
        source = round(float(state[0, 0]) * 100) - 1
        seen.add(source)
        item = labeled[source]
        assert batch["rewards"][i] == pytest.approx(item["rewards"])
        assert batch["masks"][i] == item["masks"]
        for key in ("side_policy", "wrist_1"):
            np.testing.assert_array_equal(batch["observations"][key][i], item["observations"][key])
            np.testing.assert_array_equal(batch["next_observations"][key][i], item["next_observations"][key])
    assert seen == {0, 1, 2}


class Recorder:
    def __init__(self): self.events = []
    def check(self): pass
    def event(self, *args, **kwargs): self.events.append((args, kwargs))


def test_reset_refresh_uses_production_transforms_without_actions_or_moving_reset_origin():
    import copy
    import gymnasium as gym
    from scipy.spatial.transform import Rotation
    from test_fixed_xyz_action_contract import RobotSubstitute, wrapper_factory
    from franka_env.utils.transformations import construct_transform_matrix

    base = RobotSubstitute()
    env = gym.wrappers.RecordEpisodeStatistics(wrapper_factory(base)("fixed-xyz-v1").get_environment(fake_env=True))
    initial, _ = env.reset()
    current = env
    path = {}
    while isinstance(current, gym.Wrapper):
        path[type(current).__name__] = current
        current = current.env
    relative, chunk = path["RelativeFrame"], path["ChunkingWrapper"]
    origin = relative.T_r_o_inv.copy()
    raw = base.observation()
    raw["state"]["tcp_pose"][:3] += [.001, .002, .003]
    raw["state"]["tcp_pose"][3:] = Rotation.from_euler("xyz", [np.pi, 0, .5]).as_quat()
    raw["state"]["tcp_vel"] = np.arange(6, dtype=float)
    for frame in raw["images"].values(): frame.fill(77)
    calls = []
    base._update_currpos = lambda: calls.append("state")
    def get_obs():
        calls.append("images")
        return copy.deepcopy(raw)
    base._get_obs = get_obs
    try:
        fresh = env.get_wrapper_attr("refresh_observation")()
        assert calls == ["state", "images"] and base.resets == 1
        assert base.steps == 0 and base.commands == [] and env.episode_lengths == 0
        np.testing.assert_array_equal(relative.T_r_o_inv, origin)
        np.testing.assert_allclose(relative.transform_matrix, construct_transform_matrix(raw["state"]["tcp_pose"]))
        assert fresh["state"].shape == initial["state"].shape == (1, 19)
        # gripper(1), force(3), pose(6), torque(3), velocity(6)
        np.testing.assert_allclose(fresh["state"][0, 4:7], origin[:3, :3] @ [.001, .002, .003], atol=1e-7)
        np.testing.assert_allclose(fresh["state"][0, -6:], np.linalg.inv(relative.transform_matrix) @ np.arange(6))
        for key in ("side_policy", "wrist_1"):
            assert np.all(fresh[key] == 77) and np.all(chunk.current_obs[0][key] == 77)
    finally:
        env.close()


def test_refresh_failure_does_not_read_images_or_issue_commands():
    from hilserl.errors import RobotStateUnavailable
    from test_fixed_xyz_action_contract import RobotSubstitute, wrapper_factory
    base = RobotSubstitute()
    env = wrapper_factory(base)("fixed-xyz-v1").get_environment(fake_env=True)
    env.reset()
    def unavailable(): raise RobotStateUnavailable("synthetic stale state")
    base._update_currpos = unavailable
    base._get_obs = lambda: pytest.fail("Images must not be read after state failure")
    try:
        with pytest.raises(RobotStateUnavailable):
            env.get_wrapper_attr("refresh_observation")()
        assert base.commands == [] and base.steps == 0 and base.resets == 1
    finally:
        env.close()


def test_labeled_pending_episode_recovers_without_resending_actions(tmp_path):
    from test_session_recording import ScriptedOperator
    service = EpisodeCommitServer(tmp_path / "journal", "run", "learner", SPEC, Store(), Store(),
                                  authorize=lambda a: a in {"actor", "new-actor"})
    scoring = threading.Event()

    class InterruptedProvider(Provider):
        def score(self, raw, *, check):
            scoring.set()
            while True:
                check(); time.sleep(.01)

    first = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "status.json", "run", "actor",
        InterruptedProvider, lambda: Transport(service), timeout=2)
    first.begin("000001")
    raw = episode()
    raw[-1].update(dones=False, masks=1.0)
    raw[-1]["infos"].update(manual_success=False, verdict_source=None)
    for item in raw: first.append(item, {})
    first.end_collection(); first.finalize(1, "manual")
    assert scoring.wait(2)
    first.close()
    assert not first.thread.is_alive() and len(service.online) == 0
    service.authorize = lambda a: a == "new-actor"  # Old Actor process has exited.
    second = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "new-status.json", "run", "new-actor",
        Provider, lambda: Transport(service), timeout=2)
    try:
        second.recover(ScriptedOperator(), Recorder())
        assert len(service.online) == 3 and len(service.demos) == 1
        assert service.released_to == "new-actor"
        second.recover(ScriptedOperator(), Recorder())
        assert len(service.online) == 3
    finally:
        second.close()


def test_operator_stop_during_reward_wait_keeps_data_and_closes_environment(tmp_path):
    from hilserl.control import StopRequested
    from hilserl.episodes import run_episodes
    from hilserl.storage import SessionRecorder
    from test_session_recording import FakeEnvironment, ScriptedOperator
    from reward_fakes import observation
    service = EpisodeCommitServer(tmp_path / "journal", "run", "learner", SPEC, Store(), Store(), authorize=lambda a: True)
    class SlowProvider(Provider):
        def score(self, raw, *, check):
            while True:
                check(); time.sleep(.01)
    class Environment(FakeEnvironment):
        def observation(self): return observation(self.steps + 1)
    class Operator(ScriptedOperator):
        def raise_if_stop(self):
            if self.state["phase"] == "draining_reward":
                raise StopRequested("test stop")
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    env = Environment(terminal_at=2, recorder=recorder)
    pipe = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "status.json", "run", "actor",
        SlowProvider, lambda: Transport(service), timeout=2)
    result = run_episodes(env, lambda *args: (np.zeros(3), {}), recorder, Operator(),
                         action_contract="fixed-xyz-v1", max_episodes=1, episode_reward=pipe,
                         refresh_observation=env.observation)
    assert len(result) == 1 and env.closed
    assert list((tmp_path / "pending").glob("*/labeled.pkl"))
    assert len(service.online) == 0 and not pipe.thread.is_alive()
