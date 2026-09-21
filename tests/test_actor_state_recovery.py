"""State outages pause the same Actor without retrying any device command."""
import copy
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hilserl.control import OperatorControl, StopRequested, send_command
from hilserl.episodes import run_episodes
from hilserl.errors import RobotStateUnavailable
from hilserl.files import atomic_json
from hilserl.storage import RecordingError, SessionRecorder, read_step
from test_plug_insertion_reset_safety import _Clock, _make_env
from test_session_recording import FakeEnvironment, ScriptedOperator


def eventually(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), "Actor did not reach expected phase"


@pytest.mark.parametrize("stage", ["reset", "before_step", "step"])
def test_outage_keeps_actor_and_requires_new_reset_confirmation(tmp_path, stage):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)
    operator = OperatorControl(tmp_path / "control", terminal=False)
    failure = RobotStateUnavailable("gripper joint state is missing or stale", status_code=503,
                                   health={"gripper_state_stale": True},
                                   delivery_unknown=stage == "step")

    class Environment(FakeEnvironment):
        failed = False
        commands = 0

        def outage(self):
            self.failed = True
            # These input events arrived during the failing episode. None can
            # confirm a different reset gate or label the next episode.
            operator.pending.extend([
                dict(command="label", value=1, episode_id=operator.state["episode_id"]),
                dict(command="continue", episode_id=None, phase="waiting_reset", gate_id="old"),
                dict(command="pause", episode_id=operator.state["episode_id"]),
                dict(command="gripper", value="open", episode_id=operator.state["episode_id"]),
            ])
            raise failure

        def reset(self):
            result = super().reset()
            if stage == "reset" and not self.failed:
                self.outage()
            return result

        def step(self, action):
            self.commands += 1
            if stage == "step" and not self.failed and self.steps == 1:
                self._step_commands = [dict(kind="pose", pose=np.zeros(7), returned=False,
                                            request_sent=True, delivery_unknown=True)]
                self.outage()
            return super().step(action)

    env = Environment(terminal_at=2, recorder=recorder)
    emitted, results, errors = [], [], []
    operator.gripper_handler = lambda operation: pytest.fail("stale gripper command executed")

    def before_step(obs, step):
        if stage == "before_step" and env.steps == 1 and not env.failed:
            env.outage()

    def run():
        try:
            results.extend(run_episodes(env, lambda *args: (np.zeros(7), {}), recorder, operator,
                                        max_episodes=1, before_step=before_step, emit=emitted.append))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        eventually(lambda: operator.state["phase"] == "waiting_reset" and operator.state.get("gate_id"))
        old_gate = operator.state["gate_id"]
        send_command(operator.directory, "continue")
        eventually(lambda: operator.state.get("recoverable") and operator.state["phase"] == "waiting_reset"
                   and operator.state.get("gate_id") not in (None, old_gate))
        paused = copy.deepcopy(operator.state)
        count = env.commands
        assert env.resets == 1
        assert not env.closed and recorder.error is None and not recorder._closed
        assert not recorder.episode and not operator.pending
        assert paused["robot_state_error"]["status_code"] == 503
        assert "stale" in paused["error"]
        assert not any(item["dones"] for item in emitted)
        if stage != "reset":
            episode = json.loads((recorder.directory / "episodes/000001/episode.json").read_text())
            assert episode["complete"] is False and episode["outcome"] is None
            assert episode["incomplete_reason"] == "robot_state_unavailable"
            assert episode["steps"] == (2 if stage == "step" else 1)
            if stage == "step":
                partial = read_step(recorder.directory / "episodes/000001/000001.npz")
                assert partial["complete_transition"] is False
                assert partial["next_observations"] is None and partial["raw_next_state"] is None
                assert partial["controller_commands"][0]["delivery_unknown"] is True
                assert partial["error_details"]["status_code"] == 503

        # Exercise the actual command queue with an old acknowledgement after
        # the recovery gate exists. The Actor must continue waiting.
        atomic_json(operator.directory / "commands/stale.json", dict(
            command="continue", attempt_id=paused["attempt_id"], episode_id=None,
            phase="waiting_reset", gate_id=old_gate))
        eventually(lambda: not (operator.directory / "commands/stale.json").exists())
        assert env.resets == 1 and env.commands == count

        # One new confirmation is sufficient; there is no automatic reset and
        # no second extra confirmation after the verified reset.
        send_command(operator.directory, "continue", expected=paused)
        eventually(lambda: operator.state["phase"] == "awaiting_label")
        assert env.resets == 2 and env.commands == count + 2
        assert operator.state["error"] is None and operator.state["robot_state_error"] is None
        assert operator.state["recoverable"] is False
        send_command(operator.directory, "label", 0)
        worker.join(timeout=3)
        assert not worker.is_alive() and not errors
        assert len(results) == 1 and results[0]["outcome"] == 0
        assert recorder.error is None and env.closed
    finally:
        if worker.is_alive():
            send_command(operator.directory, "stop")
            worker.join(timeout=3)


def test_recorder_failure_during_outage_finalization_is_still_fatal(tmp_path, monkeypatch):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)

    class Environment(FakeEnvironment):
        def step(self, action):
            self.steps += 1
            self._step_commands = [dict(kind="pose", pose=np.zeros(7), returned=True)]
            raise RobotStateUnavailable("feedback stale", status_code=503)

    monkeypatch.setattr("hilserl.storage.write_step", lambda *args: (_ for _ in ()).throw(OSError("disk failed")))
    env = Environment(recorder=recorder)
    operator = ScriptedOperator()
    with pytest.raises(RecordingError, match="disk failed"):
        run_episodes(env, lambda *args: (np.zeros(7), {}), recorder, operator)
    assert env.steps == 1 and env.resets == 1 and env.closed
    assert operator.state["phase"] == "fault" and recorder.error


def test_persistent_reset_outage_only_retries_after_each_new_confirmation(tmp_path):
    recorder = SessionRecorder(tmp_path / "recording", min_free_bytes=0)

    class Environment(FakeEnvironment):
        def reset(self):
            self.resets += 1
            self._step_commands = [dict(kind="pose", pose=np.zeros(7), request_sent=True,
                                       returned=False, delivery_unknown=True)]
            raise RobotStateUnavailable("reset pose timed out", endpoint="pose", delivery_unknown=True)

    env = Environment(recorder=recorder)
    gates = []

    def wait(phase):
        assert phase == "waiting_reset"
        gates.append(env.resets)
        if len(gates) == 3:
            raise StopRequested()

    result = run_episodes(env, lambda *args: pytest.fail("sampled before reset success"),
                          recorder, ScriptedOperator(on_wait=wait))
    assert gates == [0, 1, 2] and env.resets == 2 and env.steps == 0
    assert not result and recorder.error is None
    events = [json.loads(line) for line in (recorder.directory / "events.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["type"] == "robot_state_unavailable"]
    assert len(failures) == 2
    assert all(event["reset_commands"][0]["delivery_unknown"] for event in failures)
    assert json.loads((recorder.directory / "manifest.json").read_text())["episodes"] == []


def test_503_decodes_reason_and_health_without_retry(monkeypatch):
    env, ns = _make_env(_Clock())
    env.url = "http://no-robot.invalid/"
    request = ns["PlugInsertionEnv"]._robot_post.__get__(env)
    calls = []
    health = {"gripper_state_stale": True, "gripper_state_age_seconds": 13.}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=503, json=lambda: dict(error="gripper joint state is missing or stale",
                                                                  health=health))

    monkeypatch.setattr(ns["requests"], "post", post)
    with pytest.raises(RobotStateUnavailable, match="gripper joint state") as result:
        request("getstate")
    assert len(calls) == 1
    assert result.value.status_code == 503 and result.value.health == health
    assert result.value.delivery_unknown is False


@pytest.mark.parametrize("kind", ["pose", "gripper"])
def test_action_timeout_records_unknown_delivery_and_never_retries(monkeypatch, kind):
    env, ns = _make_env(_Clock())
    env.url = "http://no-robot.invalid/"
    env.clearerr_on_pose = False
    env._step_commands = []
    env.last_gripper_act, env.gripper_sleep = 0., .5
    env.curr_gripper_pos = 1.
    ns["time"].time = lambda: 100.
    env._robot_post = ns["PlugInsertionEnv"]._robot_post.__get__(env)
    calls = []

    def post(url, **kwargs):
        calls.append(url)
        raise ns["requests"].Timeout("response lost")

    monkeypatch.setattr(ns["requests"], "post", post)
    with pytest.raises(RobotStateUnavailable, match="送达未知"):
        if kind == "pose":
            ns["PlugInsertionEnv"]._send_pos_command(env, env.currpos)
        else:
            ns["PlugInsertionEnv"]._send_gripper_command(env, -1.)
    assert calls == [env.url + ("pose" if kind == "pose" else "close_gripper")]
    assert len(env._step_commands) == 1
    assert env._step_commands[0]["returned"] is False
    assert env._step_commands[0]["delivery_unknown"] is True


def test_joint_guard_preserves_state_error_but_not_invalid_joint_value():
    env, ns = _make_env(_Clock())
    env._robot_post = lambda *args, **kwargs: (_ for _ in ()).throw(RobotStateUnavailable("503"))
    with pytest.raises(RobotStateUnavailable):
        env._require_joint_limit_margin()
    with pytest.raises(RuntimeError, match="joint_limit_fatal") as result:
        env._require_joint_limit_margin(np.zeros(6))
    assert not isinstance(result.value, RobotStateUnavailable)


def test_step_rejects_unavailable_feedback_before_any_pose_or_gripper():
    env, ns = _make_env(_Clock())
    env._update_currpos = lambda: (_ for _ in ()).throw(RobotStateUnavailable("503 before action"))
    with pytest.raises(RobotStateUnavailable):
        ns["PlugInsertionEnv"].step(env, np.zeros(7))
    assert env._step_commands == [] and env.calls == []
