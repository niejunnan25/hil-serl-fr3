import ast
from pathlib import Path

import numpy as np
import pytest
from hilserl.errors import RobotStateUnavailable


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "experiments" / "plug_insertion" / "env.py"


FR3_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def _load_reset_safety_helpers():
    source = ENV_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_names = {"joint_limit_margin_degrees", "is_plug_held"}
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "FR3_RESET_LOWER_LIMITS": FR3_LOWER,
        "FR3_RESET_UPPER_LIMITS": FR3_UPPER,
        "RESET_JOINT_LIMIT_MARGIN_DEG": 12.0,
        "DROP_GRIPPER_MIN": 0.40,
        "RESET_GRIPPER_OPEN_THRESHOLD": 0.85,
    }
    exec(compile(module, "<plug-reset-safety>", "exec"), namespace)
    return namespace


def test_joint_limit_margin_uses_real_fr3_limits_and_reports_nearest_joint():
    helpers = _load_reset_safety_helpers()
    q = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.55, 0.0])

    margin_deg, joint_index, all_margins = helpers["joint_limit_margin_degrees"](q)

    assert joint_index == 5
    assert margin_deg == pytest.approx(np.degrees(0.55 - 0.5445))
    assert all_margins.shape == (7,)


def test_gripper_drop_guard_treats_hold_grip_value_as_safe():
    helpers = _load_reset_safety_helpers()

    assert helpers["is_plug_held"](0.567) is True
    assert helpers["is_plug_held"](0.399) is False


def test_reset_calls_fail_closed_joint_and_drop_guards():
    source = ENV_PATH.read_text(encoding="utf-8")

    assert "self._require_joint_limit_margin()" in source
    assert "self._require_plug_held()" in source
    assert "[joint_limit_fatal]" in source
    assert "[drop_plug_fatal]" in source


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return round(self.now * 1_000_000_000)

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


def _load_reset_env(clock):
    """Load the actual reset methods without cameras, JAX, or a live server."""
    import copy
    import os
    import requests
    from scipy.spatial.transform import Rotation

    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    keep = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            or isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper()
                                                   for t in node.targets)]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = dict(np=np, copy=copy, os=os, time=clock, requests=requests,
              Rotation=Rotation, FrankaEnv=object,
              RobotStateUnavailable=RobotStateUnavailable,
              euler_2_quat=lambda e: Rotation.from_euler("xyz", e).as_quat())
    exec(compile(module, str(ENV_PATH), "exec"), ns)
    ns.update(RESET_STRICT=False, MANUAL_RESET=True)
    return ns


def _make_env(clock, *, moving=False):
    import copy
    from types import SimpleNamespace

    ns = _load_reset_env(clock)
    env = object.__new__(ns["PlugInsertionEnv"])
    env.hz = 10
    env.currpos = np.array([0., 0., 0.0859, 0., 0., 0., 1.])
    env.curr_gripper_pos = np.array([0.567])
    env.q = np.array([0., 0., 0., -1., 0., 2., 0.])
    env.currforce = np.zeros(3)
    env.dq = np.zeros(7)
    env.safety_force_max = 45.
    env.safety_dq_max = 0.35
    env.last_reset_info = None
    env.config = SimpleNamespace(TARGET_POSE=np.zeros(6), RESET_CLEAR_Z=0.2095,
                                RESET_POSE=np.array([0.03, 0.0, 0.15, 0., 0., 0.]),
                                RANDOM_RESET=False, PRECISION_PARAM={})
    env.calls = []
    env.published = []
    env.events = []
    env.operator = SimpleNamespace(raise_if_stop=lambda: None,
        publish=lambda phase, **fields: env.published.append((phase, copy.deepcopy(fields))))
    env.recorder = SimpleNamespace(check=lambda: None,
        event=lambda kind, **fields: env.events.append((kind, copy.deepcopy(fields))))
    env._update_currpos = lambda: None
    env._recover = lambda: env.calls.append("recover")

    def robot_post(endpoint, **kwargs):
        env.calls.append(endpoint)
        return SimpleNamespace(json=lambda: {"q": env.q.copy()})

    def send(pose):
        env.calls.append(np.asarray(pose).copy())
        if moving:
            env.currpos = np.asarray(pose).copy()

    env._robot_post = robot_post
    env._send_pos_command = send
    env._get_obs = lambda: {"state": env.currpos.copy()}
    return env, ns


@pytest.mark.parametrize("value,held", [(0., False), (.399, False), (.4, True),
    (.567, True), (.849, True), (.85, False), (1., False), (np.nan, False),
    (np.inf, False), ([], False)])
def test_gripper_precheck_rejects_fully_open_empty_or_invalid(value, held):
    ns = _load_reset_env(_Clock())
    assert ns["is_plug_held"](value) is held


@pytest.mark.parametrize("value", [np.zeros(6), np.zeros(8), np.full(7, np.nan)])
def test_joint_margin_rejects_invalid_measurements(value):
    ns = _load_reset_env(_Clock())
    with pytest.raises(ValueError, match="seven finite"):
        ns["joint_limit_margin_degrees"](value)


@pytest.mark.parametrize("value", [1., 0., np.nan])
def test_open_gripper_reset_stops_before_any_motion_or_recovery(value):
    clock = _Clock()
    env, _ = _make_env(clock)
    env.curr_gripper_pos = np.array([value])
    with pytest.raises(RuntimeError, match="drop_plug_fatal"):
        env.reset()
    assert env.calls == []
    assert env.last_reset_info["success"] is False
    assert env.last_reset_info["phase"] == "precheck"
    assert env.last_reset_info["state"] == "failed"


def test_joint_guard_cannot_be_disabled_by_legacy_reset_strict():
    env, _ = _make_env(_Clock())
    env.q[5] = .55
    with pytest.raises(RuntimeError, match="joint_limit_fatal"):
        env.reset()
    assert env.calls == ["getstate"]
    assert env.last_reset_info["success"] is False


def test_frozen_controller_aborts_clear_in_eight_seconds_without_next_motion():
    clock = _Clock()
    env, _ = _make_env(clock)
    with pytest.raises(RuntimeError, match="clear stalled"):
        env.reset()
    assert 8. <= clock.now < 8.3
    assert env.last_reset_info["reason"] == "stalled"
    assert env.last_reset_info["success"] is False
    phases = env.last_reset_info["phases"]
    assert phases[-1]["name"] == "clear"
    assert phases[-1]["position_error_m"] == pytest.approx(.1236)
    assert all(phase["name"] != "reset_pose" for phase in phases)
    assert all(item[0] == "resetting" for item in env.published)
    assert len(env.events) >= 8  # bounded progress reports, not silent long waits


def test_slow_but_progressing_controller_hits_monotonic_deadline():
    clock = _Clock()
    env, ns = _make_env(clock)
    ns["RESET_MOTION_MAX_TIMEOUT"] = 2.
    before = env._send_pos_command

    def send(pose):
        before(pose)
        env.currpos[2] += .0001

    env._send_pos_command = send
    with pytest.raises(RuntimeError, match="clear timeout"):
        env.reset()
    assert 2. <= clock.now < 2.2
    assert env.last_reset_info["reason"] == "timeout"
    assert env.last_reset_info["phase"] == "clear"


def test_operator_stop_during_reset_cancels_before_the_next_command():
    clock = _Clock()
    env, _ = _make_env(clock)

    class StopRequested(Exception):
        pass

    def check():
        if clock.now >= .6:
            raise StopRequested("test stop")

    env.operator.raise_if_stop = check
    with pytest.raises(StopRequested):
        env.reset()
    assert clock.now <= .7
    assert env.last_reset_info["state"] == "cancelled"
    assert env.last_reset_info["success"] is False
    assert not any(p["name"] == "reset_pose" for p in env.last_reset_info["phases"])


@pytest.mark.parametrize("error", [(0.1236, 0.), (0., .13), (np.nan, 0.)])
def test_nonconvergence_cannot_be_overridden_by_legacy_flags_or_manual_input(error):
    env, _ = _make_env(_Clock())
    env._operator_input = lambda _: pytest.fail("manual confirmation must not bypass reset failure")
    with pytest.raises(RuntimeError, match="not_converged"):
        env._require_reset_settled("clear", error)
    assert env.calls == []
    assert env.last_reset_info["success"] is False


def test_successful_reset_has_measured_evidence_for_both_phases():
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    obs, info = env.reset()
    evidence = info["reset"]
    assert evidence["success"] is True
    assert evidence["state"] == "succeeded"
    assert evidence["phase"] == "ready"
    assert evidence["elapsed_seconds"] == pytest.approx(clock.now)
    motions = [p for p in evidence["phases"] if p["name"] in ("clear", "reset_pose")]
    assert [p["name"] for p in motions] == ["clear", "reset_pose"]
    assert all(p["state"] == "converged" and p["position_error_m"] <= .012
               and p["rotation_error_rad"] <= .12 for p in motions)
    assert evidence["phases"][0]["gripper_check"] == "plausible_width_only"
    assert obs["state"][:3] == pytest.approx(env.config.RESET_POSE[:3])
    assert info["succeed"] is False  # task label remains distinct from reset success


def test_second_motion_failure_does_not_return_ready_or_observation():
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    send = env._send_pos_command

    def stop_after_clear(pose):
        if env.last_reset_info["phase"] != "reset_pose":
            send(pose)

    env._send_pos_command = stop_after_clear
    env._get_obs = lambda: pytest.fail("no observation/episode after failed second reset phase")
    with pytest.raises(RuntimeError, match="reset_pose stalled"):
        env.reset()
    assert env.last_reset_info["success"] is False
    assert env.last_reset_info["phase"] == "reset_pose"
    assert not any(p["name"] == "ready" for p in env.last_reset_info["phases"])


@pytest.mark.parametrize("measurement,value", [("currforce", np.array([46., 0., 0.])),
                                               ("dq", np.full(7, .36))])
def test_reset_uses_existing_force_and_joint_speed_limits(measurement, value):
    env, _ = _make_env(_Clock())
    setattr(env, measurement, value)
    with pytest.raises(RuntimeError, match="safety_limit"):
        env.interpolate_move(np.array([0., 0., .21, 0., 0., 0., 1.]), timeout=8., name="clear")
    assert env.calls == []


def test_robot_request_has_finite_timeout_and_rejects_http_errors(monkeypatch):
    from types import SimpleNamespace

    clock = _Clock()
    env, ns = _make_env(clock)
    env.url = "http://no-robot.invalid/"
    request = ns["PlugInsertionEnv"]._robot_post.__get__(env)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("HTTP 500")))

    monkeypatch.setattr(ns["requests"], "post", post)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        request("getstate")
    assert calls == [("http://no-robot.invalid/getstate", {"timeout": (2., 5.)})]


def test_sensor_jitter_does_not_extend_stall_deadline():
    clock = _Clock()
    env, _ = _make_env(clock)
    initial_z = env.currpos[2]
    reads = 0

    def update():
        nonlocal reads
        reads += 1
        env.currpos[2] = initial_z + (.0001 if reads % 2 else -.0001)

    env._update_currpos = update
    with pytest.raises(RuntimeError, match="stalled"):
        env.reset()
    assert clock.now < 8.3


def test_gripper_reopens_mid_motion_and_stops_without_the_next_command():
    clock = _Clock()
    env, _ = _make_env(clock, moving=True)
    command_count = []

    def update():
        if clock.now >= .4:
            env.curr_gripper_pos = np.array([1.])
            command_count.append(len(env.calls))

    env._update_currpos = update
    with pytest.raises(RuntimeError, match="复位已中止"):
        env.reset()
    assert len(env.calls) == command_count[0]
    assert env.last_reset_info["success"] is False


def test_actual_state_reader_and_pose_sender_preserve_wire_contract(monkeypatch):
    from types import SimpleNamespace

    env, ns = _make_env(_Clock())
    env.url = "http://no-robot.invalid/"
    env.clearerr_on_pose = False
    env._step_commands = []
    state = dict(pose=env.currpos.tolist(), vel=[0.] * 6, force=[1., 2., 3.],
                 torque=[4., 5., 6.], q=env.q.tolist(), dq=[0.] * 7,
                 jacobian=[0.] * 42, gripper_pos=.567, controller_running=True, robot_mode=2,
                 state_age_seconds=.01, state_stale=False,
                 jacobian_age_seconds=.01, jacobian_stale=False,
                 gripper_state_age_seconds=.01, gripper_state_stale=False)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(json=lambda: state, raise_for_status=lambda: None)

    monkeypatch.setattr(ns["requests"], "post", post)
    env._robot_post = ns["PlugInsertionEnv"]._robot_post.__get__(env)
    ns["PlugInsertionEnv"]._update_currpos(env)
    assert env.currforce.tolist() == [1., 2., 3.]
    assert env.currtorque.tolist() == [4., 5., 6.]
    assert env.currjacobian.shape == (6, 7)
    assert env.curr_gripper_pos == pytest.approx(.567)
    assert env._state_capture["server_health"]["state_stale"] is False
    ns["PlugInsertionEnv"]._send_pos_command(env, env.currpos)
    assert [url for url, _ in calls] == [env.url + "getstate", env.url + "pose"]
    assert all(kwargs["timeout"] == (2., 5.) for _, kwargs in calls)
    assert calls[-1][1]["json"]["arr"] == pytest.approx(state["pose"])
    assert env._step_commands[-1]["returned"] is True


def test_network_timeout_fails_reset_before_any_motion(monkeypatch):
    env, ns = _make_env(_Clock())
    env.url = "http://no-robot.invalid/"
    env._robot_post = ns["PlugInsertionEnv"]._robot_post.__get__(env)
    env._update_currpos = ns["PlugInsertionEnv"]._update_currpos.__get__(env)

    def timeout(*args, **kwargs):
        raise ns["requests"].Timeout("no controller response")

    monkeypatch.setattr(ns["requests"], "post", timeout)
    with pytest.raises(RobotStateUnavailable):
        env.reset()
    assert env.calls == []
    assert env.last_reset_info["success"] is False
    assert env.last_reset_info["phases"][-1]["error"] == "RobotStateUnavailable"


@pytest.mark.parametrize("health", [{}, {"controller_running": False},
    {"controller_running": True, "state_stale": True, "state_age_seconds": 7000},
    {"controller_running": True, "state_stale": False, "state_age_seconds": .01,
     "jacobian_stale": False, "jacobian_age_seconds": .01,
     "gripper_state_stale": True, "gripper_state_age_seconds": 7000}])
def test_cached_or_unproven_state_rejected_before_reset_recovery(health):
    from types import SimpleNamespace
    env, ns = _make_env(_Clock())
    env._update_currpos = ns["PlugInsertionEnv"]._update_currpos.__get__(env)
    env._robot_post = lambda *args, **kwargs: SimpleNamespace(json=lambda: health)
    with pytest.raises(RuntimeError, match="FR3"):
        env.reset()
    assert env.calls == []
    assert env.last_reset_info["success"] is False
