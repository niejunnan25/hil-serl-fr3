"""Real reset/transaction code, fake hardware: readiness must follow reward ACK."""
import copy
import threading
import time

import numpy as np
import pytest

from hilserl.control import StopRequested
from hilserl.episode_commit import EpisodeCommitServer
from hilserl.episode_reward import EpisodeRewardPipeline
from hilserl.episodes import run_episodes
from hilserl.storage import SessionRecorder
from reward_fakes import SPEC, Provider, Store, Transport, episode
from test_plug_insertion_reset_safety import _Clock, _make_env
from test_session_recording import ScriptedOperator, FakeEnvironment


def test_actual_reset_waits_for_replay_before_single_cue_and_fresh_observation(tmp_path, monkeypatch):
    from hilserl import reset_feedback
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    env.config.RESET_CLEAR_Z = .15
    env.config.RESET_FEEDBACK = True
    service = EpisodeCommitServer(tmp_path / "journal", "run", "learner", SPEC,
                                  Store(), Store(), authorize=lambda _: True)
    scoring, release, waiting = threading.Event(), threading.Event(), threading.Event()
    events, failures, receipts, checks = [], [], [], []
    def score():
        scoring.set()
        assert release.wait(5)
    pipe = EpisodeRewardPipeline(SPEC, tmp_path / "pending", tmp_path / "status.json", "run", "actor",
        lambda: Provider(callback=score), lambda: Transport(service), timeout=3)
    raw = episode()
    pipe.begin("000001")
    for item in raw:
        value = copy.deepcopy(item)
        value.update(dones=False, masks=1.)
        value["infos"].update(succeed=False, manual_success=False, verdict_source=None)
        pipe.append(value, {})
    pipe.end_collection(); pipe.finalize(1, "manual")

    def cue(*, check):
        assert len(service.online) == 3 and service.gap is None
        events.append("cue")
        check(); clock.sleep(.8); check()
        return dict(state="played", duration_ms=800, pulse_count=1)
    monkeypatch.setattr(reset_feedback, "rumble_once", cue)
    def observe():
        assert events[-1] == "cue"
        boundary, _, check = env._reset_image_barrier
        assert boundary >= 800_000_000
        check(); events.append("fresh")
        return {"state": env.currpos.copy()}
    env._get_obs = observe
    operator = ScriptedOperator()
    def barrier(*, check):
        events.append("motion-complete")
        before = len(env.calls)
        waiting.set()
        def watch():
            check(); checks.append(True)
            assert len(env.calls) == before  # No pose/gripper/recovery calls in the wait.
        pipe.barrier(operator, env.recorder, check=watch)
        events.append("committed")
    def reset():
        try: receipts.append(env.reset(options={"hilserl_ready_barrier": barrier}))
        except BaseException as exc: failures.append(exc)
    thread = threading.Thread(target=reset)
    thread.start()
    try:
        assert waiting.wait(3) and scoring.wait(3), failures
        deadline = time.monotonic() + 3
        while len(checks) < 2 and time.monotonic() < deadline: time.sleep(.01)
        assert len(checks) >= 2 and events == ["motion-complete"] and len(service.online) == 0
        release.set(); thread.join(5)
        assert not thread.is_alive() and not failures, failures
        assert events == ["motion-complete", "committed", "cue", "fresh"]
        assert receipts[0][1]["reset"]["success"]
        assert receipts[0][1]["reset"]["feedback"]["pulse_count"] == 1
        assert not any(phase == "collecting" for phase, _ in env.published)
        assert env.curr_path_length == 0
    finally:
        release.set(); pipe.close(); thread.join(5)


@pytest.mark.parametrize("failure", ["drift", "rotation", "velocity", "force", "gripper", "stop", "reward"])
def test_reward_wait_failure_prevents_cue_and_observation(monkeypatch, failure):
    from hilserl import reset_feedback
    from scipy.spatial.transform import Rotation
    env, _ = _make_env(_Clock(), moving=True)
    env.config.RESET_CLEAR_Z = .15
    env.config.RESET_FEEDBACK = True
    monkeypatch.setattr(reset_feedback, "rumble_once", lambda **kw: pytest.fail("No premature cue"))
    env._get_obs = lambda: pytest.fail("No ready observation after failed wait")
    calls = []
    def barrier(*, check):
        calls.append(len(env.calls)); check()
        if failure == "drift": env.currpos[0] += .001
        elif failure == "rotation": env.currpos[3:] = Rotation.from_rotvec([.02, 0, 0]).as_quat()
        elif failure == "velocity": env.dq[:] = .1
        elif failure == "force": env.currforce[:] = [46., 0, 0]
        elif failure == "gripper": env.curr_gripper_pos[:] = 1.
        elif failure == "stop": raise StopRequested("stop during reward wait")
        else: raise RuntimeError("reward RPC failed")
        check()
    with pytest.raises((RuntimeError, StopRequested)):
        env.reset(options={"hilserl_ready_barrier": barrier})
    assert calls == [len(env.calls)]
    assert not env.last_reset_info["success"]
    assert env.last_reset_info["phase"] == "reward_wait"


def test_native_wrapper_stack_forwards_barrier_and_transforms_only_post_wait_inputs():
    import gymnasium as gym
    from test_fixed_xyz_action_contract import RobotSubstitute, wrapper_factory
    base = RobotSubstitute()
    raw_reset = base.reset
    calls = []
    def reset(*, options, **kwargs):
        _, info = raw_reset(**kwargs)
        options["hilserl_ready_barrier"](check=lambda: calls.append("check"))
        return base.observation(), info
    base.reset = reset
    env = gym.wrappers.RecordEpisodeStatistics(wrapper_factory(base)("fixed-xyz-v1").get_environment(fake_env=True))
    def barrier(*, check):
        check()
        original = base.observation()
        original["images"]["side_policy"].fill(77)
        original["images"]["wrist_1"].fill(88)
        original["state"]["tcp_pose"][:3] += .001
        base.observation = lambda: copy.deepcopy(original)
    try:
        obs, info = env.reset(options={"hilserl_ready_barrier": barrier})
        assert info["reset"]["success"] and calls == ["check"]
        assert obs["state"].shape == (1, 19)
        np.testing.assert_allclose(obs["state"][0, 4:7], 0, atol=1e-7)
        assert np.all(obs["side_policy"] == 77) and np.all(obs["wrist_1"] == 88)
        assert base.commands == [] and base.steps == 0 and base.resets == 1
        assert env.episode_lengths == 0
    finally:
        env.close()


def test_environment_cannot_silently_skip_reward_readiness(tmp_path):
    class IgnoringEnvironment(FakeEnvironment):
        def reset(self, **kwargs): return super().reset()
    class Pipeline:
        def recover(self, *args): pass
        def abort(self, *args): pass
        def close(self): pass
        def check(self): pass
        def snapshot(self): return {}
    env = IgnoringEnvironment()
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    with pytest.raises(RuntimeError, match="skipped the reward-ready barrier"):
        run_episodes(env, lambda *a: pytest.fail("No policy action"), recorder, ScriptedOperator(),
                     max_episodes=1, episode_reward=Pipeline())
    assert env.steps == 0 and env.closed
