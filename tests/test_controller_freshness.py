"""Exercise the staged controller source without importing ROS or running main.

Set HILSERL_CONTROLLER_PATCH_ROOT when validating a copy of the staged patch.
Flask's in-process client executes the actual route functions; all ROS I/O is fake.
"""
import ast
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

from flask import Flask, jsonify, request
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


PATCH_ROOT = Path(os.environ.get(
    "HILSERL_CONTROLLER_PATCH_ROOT",
    "/Users/tacyvan/Code/hilserl-reset-gripper-20260912/controller-patch/robot_servers",
))


class Clock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Publisher:
    def __init__(self, *_args, **_kwargs):
        self.connections = 1
        self.messages = []
        self.fail_connections = False
        self.fail_publish = False

    def get_num_connections(self):
        if self.fail_connections:
            raise OSError("ROS connection unavailable")
        return self.connections

    def publish(self, message):
        if self.fail_publish:
            raise OSError("ROS publisher is closed")
        self.messages.append(message)


class Process:
    def __init__(self):
        self.exit_code = None
        self.poll_error = False

    def poll(self):
        if self.poll_error:
            raise OSError("process cannot be inspected")
        return self.exit_code


class BaseGripper:
    def __init__(self):
        self.gripper_pos = 0

    def activate_gripper(self):
        pass

    def reset_gripper(self):
        pass


def action_goal():
    return SimpleNamespace(goal=SimpleNamespace(epsilon=SimpleNamespace()))


def pose_message():
    return SimpleNamespace(header=SimpleNamespace(), pose=SimpleNamespace())


def state_message(**changes):
    values = dict(robot_mode=2, O_T_EE=np.eye(4).T.reshape(-1).tolist(), q=np.zeros(7),
                  dq=np.ones(7), K_F_ext_hat_K=np.zeros(6))
    values.update(changes)
    return SimpleNamespace(**values)


def _exec_source(path, namespace):
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != "main"]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return tree


@pytest.fixture
def rig():
    clock = Clock()
    rospy = SimpleNamespace(
        Publisher=Publisher,
        Subscriber=lambda *_args, **_kwargs: object(),
        Time=SimpleNamespace(now=lambda: clock.now),
    )
    namespace = dict(np=np, time=clock, threading=threading, rospy=rospy,
                     R=Rotation, STATE_MAX_AGE_SECONDS=1.0,
                     subprocess=SimpleNamespace(
                         Popen=lambda *_args, **_kwargs: pytest.fail("must not start ROS"),
                         TimeoutExpired=subprocess.TimeoutExpired),
                     geom_msg=SimpleNamespace(PoseStamped=pose_message,
                                              Point=lambda *v: v, Quaternion=lambda *v: v),
                     ErrorRecoveryActionGoal=lambda: object(), FrankaState=object,
                     ZeroJacobian=object)
    tree = _exec_source(PATCH_ROOT / "franka_server.py", namespace)
    robot = namespace["FrankaServer"]("unused", "Franka", "unused", np.zeros(7))
    robot.imp = Process()
    gripper_ns = dict(np=np, time=clock, threading=threading, rospy=rospy,
                      GripperServer=BaseGripper, GRIPPER_STATE_MAX_AGE_SECONDS=1.0,
                      MoveActionGoal=action_goal, GraspActionGoal=action_goal, JointState=object)
    _exec_source(PATCH_ROOT / "franka_gripper_server.py", gripper_ns)
    gripper = gripper_ns["FrankaGripperServer"]()
    webapp = Flask(__name__)
    namespace.update(webapp=webapp, jsonify=jsonify, request=request,
                     robot_server=robot, gripper_server=gripper)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    helper_names = {"gripper_snapshot", "unavailable", "read_robot_state", "gripper_command"}
    routes = [node for node in main.body
              if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name in helper_names)]
    exec(compile(ast.Module(body=routes, type_ignores=[]), str(PATCH_ROOT / "franka_server.py"), "exec"),
         namespace)
    return SimpleNamespace(clock=clock, robot=robot, gripper=gripper, client=webapp.test_client(),
                           unavailable=namespace["ControllerUnavailableError"])


def update_all(rig):
    rig.robot._set_jacobian(SimpleNamespace(zero_jacobian=np.arange(42, dtype=float)))
    rig.robot._set_currpos(state_message())
    rig.gripper._update_gripper(SimpleNamespace(position=[0.025, 0.025]))


def test_health_get_is_read_only_and_tracks_new_feedback(rig):
    assert rig.client.get("/health").status_code == 503
    update_all(rig)
    result = rig.client.get("/health")
    assert result.status_code == 200
    assert result.json["ready"] is True
    assert result.json["robot_mode"] == 2
    assert rig.robot.eepub.messages == [] and rig.robot.resetpub.messages == []
    rig.clock.now += 2
    assert rig.client.get("/health").status_code == 503


@pytest.mark.parametrize("mode", [0, 3, 4, 5, 6])
def test_fresh_feedback_in_unsafe_mode_does_not_enable_commands(rig, mode):
    update_all(rig)
    rig.robot._set_currpos(state_message(robot_mode=mode))
    assert rig.client.get("/health").status_code == 503
    assert rig.client.post("/pose", json={"arr": [0, 0, 0, 0, 0, 0, 1]}).status_code == 503
    assert rig.client.post("/close_gripper").status_code == 503
    assert rig.robot.eepub.messages == [] and rig.gripper.grippergrasppub.messages == []


def test_missing_feedback_is_stale_without_fabricated_current_arrays(rig):
    response = rig.client.post("/getstate")
    assert response.status_code == 503
    value = response.get_json()
    assert value["state_age_seconds"] is None
    assert value["state_stale"] is True
    assert value["gripper_state_stale"] is True
    assert value["sequence"] == value["state_sequence"] == 0
    assert "pose" not in value
    assert value["last_state"]["pose"] is None
    assert rig.robot.resetpub.messages == []


def test_healthy_read_preserves_measurements_and_reports_actual_age(rig):
    update_all(rig)
    rig.clock.now += .25
    response = rig.client.post("/getstate")
    assert response.status_code == 200
    value = response.get_json()
    assert value["pose"] == [0., 0., 0., 0., 0., 0., 1.]
    assert value["q"] == [0.] * 7
    assert value["gripper_pos"] == pytest.approx(.625)
    for name in ("state_age_seconds", "jacobian_age_seconds", "gripper_state_age_seconds"):
        assert value[name] == pytest.approx(.25)
    assert value["state_stale"] is value["jacobian_stale"] is value["gripper_state_stale"] is False
    assert value["sequence"] == value["state_sequence"] == 1
    assert rig.robot.resetpub.messages == []


def test_new_jacobian_cannot_make_an_old_franka_state_fresh(rig):
    update_all(rig)
    rig.clock.now += 2.
    rig.robot._set_jacobian(SimpleNamespace(zero_jacobian=np.zeros(42)))
    health = rig.robot.health()
    assert health["state_stale"] is True
    assert health["state_age_seconds"] == 2.
    assert health["jacobian_stale"] is False
    assert health["jacobian_sequence"] == 2
    assert health["state_sequence"] == 1
    assert rig.client.post("/getstate").status_code == 503


def test_new_franka_state_does_not_make_an_old_jacobian_fresh(rig):
    update_all(rig)
    rig.clock.now += 2.
    rig.robot._set_currpos(state_message())
    rig.gripper._update_gripper(SimpleNamespace(position=[.025, .025]))
    assert rig.client.post("/getpos").status_code == 200
    assert rig.client.post("/getvel").status_code == 503
    assert rig.client.post("/getjacobian").status_code == 503
    assert rig.client.post("/getstate").status_code == 503


def test_new_complete_feedback_restores_reads_after_staleness(rig):
    update_all(rig)
    rig.clock.now += 2.
    assert rig.client.post("/getstate").status_code == 503
    update_all(rig)
    response = rig.client.post("/getstate")
    assert response.status_code == 200
    assert response.get_json()["state_age_seconds"] == 0.
    assert response.get_json()["state_sequence"] == 2
    assert response.get_json()["gripper_state_sequence"] == 2


def test_malformed_state_never_commits_a_new_sequence_or_partial_measurement(rig):
    update_all(rig)
    rig.clock.now += .3
    with pytest.raises(ValueError):
        rig.robot._set_currpos(state_message(q=np.ones(7), dq=np.ones(6)))
    with pytest.raises(ValueError):
        rig.robot._set_currpos(state_message(q=np.full(7, np.nan)))
    value = rig.robot.state_snapshot()
    assert value["state_sequence"] == 1
    assert value["state_age_seconds"] == pytest.approx(.3)
    assert value["q"] == [0.] * 7


def test_invalid_jacobian_callback_does_not_refresh_its_timestamp(rig):
    update_all(rig)
    rig.clock.now += .4
    for jacobian in (np.zeros(41), np.full(42, np.nan)):
        with pytest.raises(ValueError):
            rig.robot._set_jacobian(SimpleNamespace(zero_jacobian=jacobian))
    assert rig.robot.health()["jacobian_sequence"] == 1
    assert rig.robot.health()["jacobian_age_seconds"] == pytest.approx(.4)


@pytest.mark.parametrize("endpoint", ["/getpos", "/getpos_euler", "/getvel", "/getforce",
    "/gettorque", "/getq", "/getdq", "/getjacobian", "/get_gripper", "/getstate"])
def test_every_read_endpoint_rejects_old_feedback_without_recovery(rig, endpoint):
    update_all(rig)
    rig.clock.now += 1.01
    response = rig.client.post(endpoint)
    assert response.status_code == 503
    value = response.get_json()
    assert value["success"] is False
    assert value["last_state"]
    assert rig.robot.eepub.messages == rig.robot.resetpub.messages == []


@pytest.mark.parametrize("failure", ["missing_state", "stale", "exited", "missing_process",
    "process_unqueryable", "no_subscriber", "subscriber_unqueryable", "publish_failed"])
def test_pose_rejects_unavailable_controller_before_false_success(rig, failure):
    if failure != "missing_state":
        update_all(rig)
    if failure == "stale":
        rig.clock.now += 2.
    elif failure == "exited":
        rig.robot.imp.exit_code = -6
    elif failure == "missing_process":
        rig.robot.imp = None
    elif failure == "process_unqueryable":
        rig.robot.imp.poll_error = True
    elif failure == "no_subscriber":
        rig.robot.eepub.connections = 0
    elif failure == "subscriber_unqueryable":
        rig.robot.eepub.fail_connections = True
    elif failure == "publish_failed":
        rig.robot.eepub.fail_publish = True
    response = rig.client.post("/pose", json={"arr": [0, 0, 0, 0, 0, 0, 1]})
    assert response.status_code == 503
    value = response.get_json()
    assert value["command_acknowledged"] is False
    assert rig.robot.eepub.messages == rig.robot.resetpub.messages == []


def test_fresh_pose_acknowledges_publication_only(rig):
    update_all(rig)
    response = rig.client.post("/pose", json={"arr": [.1, .2, .3, 0, 0, 0, 1]})
    assert response.status_code == 200
    assert response.get_json() == {"success": True, "command_acknowledged": True, "operation": "pose"}
    assert len(rig.robot.eepub.messages) == 1
    assert rig.robot.state_snapshot()["pose"][:3] == [0., 0., 0.]


@pytest.mark.parametrize("endpoint,field,speed", [("/open_gripper", "grippermovepub", .3),
    ("/close_gripper", "grippergrasppub", .3), ("/close_gripper_slow", "grippergrasppub", .1)])
def test_gripper_ack_is_publication_and_original_speed_force_are_preserved(rig, endpoint, field, speed):
    update_all(rig)
    before = rig.gripper.gripper_pos
    response = rig.client.post(endpoint)
    operation = "open" if endpoint == "/open_gripper" else "close"
    assert response.status_code == 200
    assert response.get_json() == {"success": True, "command_acknowledged": True, "operation": operation}
    messages = getattr(rig.gripper, field).messages
    assert len(messages) == 1
    assert messages[0].goal.speed == speed
    assert messages[0].goal.width == (.09 if operation == "open" else .01)
    if operation == "close":
        assert messages[0].goal.force == 130
        assert messages[0].goal.epsilon.inner == messages[0].goal.epsilon.outer == 1
    assert rig.gripper.gripper_pos == before


def test_external_open_does_not_suppress_the_next_close(rig):
    update_all(rig)
    assert rig.client.post("/close_gripper").status_code == 200
    rig.gripper._update_gripper(SimpleNamespace(position=[.04, .04]))
    before = len(rig.gripper.grippergrasppub.messages)
    assert rig.client.post("/close_gripper").status_code == 200
    assert len(rig.gripper.grippergrasppub.messages) == before + 1


def test_move_gripper_keeps_the_existing_position_conversion(rig):
    update_all(rig)
    response = rig.client.post("/move_gripper", json={"gripper_pos": 255})
    assert response.status_code == 200
    assert response.get_json()["command_acknowledged"] is True
    assert len(rig.gripper.grippermovepub.messages) == 1
    assert rig.gripper.grippermovepub.messages[0].goal.width == .1
    assert rig.gripper.grippermovepub.messages[0].goal.speed == .3


@pytest.mark.parametrize("failure", ["missing_state", "stale", "exited", "no_arm_subscriber",
    "no_gripper_subscriber", "subscriber_unqueryable", "publish_failed"])
def test_gripper_commands_reject_missing_feedback_process_or_subscriber(rig, failure):
    update_all(rig)
    if failure == "missing_state":
        rig.gripper._state_received_at = None
    elif failure == "stale":
        rig.clock.now += 2.
        rig.robot._set_currpos(state_message())
    elif failure == "exited":
        rig.robot.imp.exit_code = -6
    elif failure == "no_arm_subscriber":
        rig.robot.eepub.connections = 0
    elif failure == "no_gripper_subscriber":
        rig.gripper.grippergrasppub.connections = 0
    elif failure == "subscriber_unqueryable":
        rig.gripper.grippergrasppub.fail_connections = True
    elif failure == "publish_failed":
        rig.gripper.grippergrasppub.fail_publish = True
    response = rig.client.post("/close_gripper")
    assert response.status_code == 503
    assert response.get_json()["command_acknowledged"] is False
    assert rig.gripper.grippergrasppub.messages == rig.robot.resetpub.messages == []


def test_invalid_gripper_callback_does_not_refresh_old_state(rig):
    update_all(rig)
    rig.clock.now += .4
    for positions in ([], [0.], [float("nan"), 0.]):
        with pytest.raises(ValueError):
            rig.gripper._update_gripper(SimpleNamespace(position=positions))
    value = rig.gripper.state_snapshot()
    assert value["gripper_state_sequence"] == 1
    assert value["gripper_state_age_seconds"] == pytest.approx(.4)
    assert value["gripper_pos"] == .625


def test_a_defunct_controller_never_returns_current_state_even_before_age_timeout(rig):
    update_all(rig)
    rig.robot.imp.exit_code = -6
    response = rig.client.post("/getstate")
    assert response.status_code == 503
    value = response.get_json()
    assert value["controller_running"] is False
    assert value["controller_exit_code"] == -6
    assert value["state_stale"] is False
    assert "pose" not in value
    assert "pose" in value["last_state"]


def test_unsupported_franka_activation_does_not_pretend_to_succeed(rig):
    update_all(rig)
    assert rig.client.post("/activate_gripper").status_code == 503
    assert rig.client.post("/reset_gripper").status_code == 503
    assert rig.gripper.grippermovepub.messages == rig.gripper.grippergrasppub.messages == []


def test_dead_jointreset_is_rejected_before_any_recovery_or_subprocess(rig):
    update_all(rig)
    rig.robot.imp.exit_code = -6
    assert rig.client.post("/jointreset").status_code == 503
    assert rig.robot.resetpub.messages == []
