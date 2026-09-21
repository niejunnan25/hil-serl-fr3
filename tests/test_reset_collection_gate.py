"""A failed or ambiguous reset never emits a policy action or a training step."""
import json
import time

import numpy as np
import pytest

from hilserl.episodes import run_episodes
from hilserl.storage import SessionRecorder
from test_session_recording import FakeEnvironment, ScriptedOperator


@pytest.mark.parametrize("info", [None, {}, {"succeed": False}, {"succeed": True},
                                    {"reset": {"success": False}}, {"reset": {"success": "true"}}])
def test_failed_reset_never_opens_collection(tmp_path, info):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    env = FakeEnvironment(recorder=recorder)
    env.reset = lambda: (env.observation(), info)
    operator = ScriptedOperator()
    sampled, emitted = [], []
    with pytest.raises(RuntimeError, match="复位未通过"):
        run_episodes(env, lambda *args: sampled.append(args), recorder, operator,
                     max_episodes=1, emit=emitted.append)
    assert env.steps == 0 and not sampled and not emitted
    assert json.loads((tmp_path / "recording/manifest.json").read_text())["episodes"] == []
    assert operator.state["phase"] == "fault"
    assert env.closed
    assert "reset_rejected" in (tmp_path / "recording/events.jsonl").read_text()


def test_positive_reset_evidence_is_kept_with_episode(tmp_path):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    env = FakeEnvironment(terminal_at=1, recorder=recorder)
    run_episodes(env, lambda *args: (np.zeros(7), {}), recorder, ScriptedOperator(), max_episodes=1)
    episode = json.loads((tmp_path / "recording/episodes/000001/episode.json").read_text())
    assert episode["reset_info"]["reset"]["success"] is True
    assert "reset_verified" in (tmp_path / "recording/events.jsonl").read_text()


def test_each_verified_reset_publishes_one_stable_collection_start_before_sampling(tmp_path):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    env = FakeEnvironment(terminal_at=3, recorder=recorder)
    operator = ScriptedOperator()
    windows = {}

    def sample(*_args):
        started = operator.state.get("collection_started_monotonic_ns")
        assert type(started) is int and 0 < started <= time.monotonic_ns()
        windows.setdefault(operator.state["episode_id"], set()).add(started)
        assert env.resets == len(windows)
        return np.zeros(7), {}

    run_episodes(env, sample, recorder, operator, max_episodes=2)
    assert set(windows) == {"000001", "000002"}
    assert all(len(starts) == 1 for starts in windows.values())
    assert next(iter(windows["000001"])) < next(iter(windows["000002"]))
