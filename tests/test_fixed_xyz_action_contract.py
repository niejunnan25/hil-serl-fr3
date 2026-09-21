"""Execute the production wrapper stack against an in-memory robot substitute.

Heavy hardware imports are excluded by loading the original class/function ASTs.
The actual RelativeFrame transforms, observation wrappers and TrainConfig factory
bodies run unchanged. No camera, controller, robot server or GPU is contacted.
"""
import ast
import copy
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from experiments.config import DefaultTrainingConfig
from experiments.plug_insertion.wrapper import (
    FixedAxesDeviceWrapper, FixedInterventionActionWrapper, FixedXYZActionWrapper,
    GripperPenaltyWrapper, RotationLockWrapper,
)
from hilserl.action_contract import (
    LEGACY, FIXED_XYZ, action_contract_from_env, resolve_action_contract,
)
from hilserl.control import StopRequested
from hilserl.episodes import _training_transition, run_episodes
from hilserl.errors import RobotStateUnavailable
from hilserl.storage import SessionRecorder, read_step, stamp
from scripts.xbox_intervention import HoldGripperWrapper, XboxIntervention
import scripts.xbox_intervention as xbox_module
from test_session_recording import ScriptedOperator


ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "upstream/hil-serl/serl_robot_infra/franka_env"
LAUNCHER = ROOT / "upstream/hil-serl/serl_launcher/serl_launcher"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in names]
    assert {node.name for node in body} == set(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def wrapper_factory(base, *, human_request=None):
    # JAX only supplies a dictionary tree_map for obs_horizon=1 in this suite.
    # Use the installed implementation on the native learner host.
    try:
        import jax
    except ImportError:
        def tree_map(fn, tree, **kwargs):
            return {key: fn(value) for key, value in tree.items()}
        jax = SimpleNamespace(tree_util=SimpleNamespace(tree_map=tree_map))
    namespace = dict(np=np, gym=gym, Env=gym.Env, spaces=gym.spaces, R=R, copy=copy,
                     deque=deque, Optional=Optional, jax=jax, os=os,
                     flatten_space=gym.spaces.flatten_space, flatten=gym.spaces.flatten,
                     DefaultTrainingConfig=DefaultTrainingConfig, DefaultEnvConfig=object,
                     PlugInsertionEnv=lambda **kwargs: base,
                     HoldGripperWrapper=HoldGripperWrapper, RotationLockWrapper=RotationLockWrapper,
                     GripperPenaltyWrapper=GripperPenaltyWrapper,
                     FixedAxesDeviceWrapper=FixedAxesDeviceWrapper,
                     FixedInterventionActionWrapper=FixedInterventionActionWrapper,
                     FixedXYZActionWrapper=FixedXYZActionWrapper,
                     action_contract_from_env=action_contract_from_env,
                     resolve_action_contract=resolve_action_contract)
    definitions(INFRA / "utils/transformations.py",
                {"construct_transform_matrix", "construct_homogeneous_matrix"}, namespace)
    definitions(INFRA / "envs/relative_env.py", {"RelativeFrame"}, namespace)
    definitions(INFRA / "envs/wrappers.py", {"Quat2EulerWrapper"}, namespace)
    definitions(LAUNCHER / "wrappers/serl_obs_wrappers.py", {"SERLObsWrapper"}, namespace)
    definitions(LAUNCHER / "wrappers/chunking.py", {"stack_obs", "space_stack", "ChunkingWrapper"}, namespace)
    definitions(ROOT / "experiments/plug_insertion/config.py", {"_env_bool", "EnvConfig", "TrainConfig"}, namespace)

    def xbox(env):
        wrapper = XboxIntervention(env)
        wrapper._armed = True
        def poll():
            return SimpleNamespace(rb=human_request is not None and base.steps == 1,
                                   left_x=0., left_y=0., right_x=0., right_y=0.,
                                   dpad_x=0., dpad_y=0., a=False, b=False, x=False)
        wrapper._hub = SimpleNamespace(poll=poll)
        # Exercise actual Xbox selection and info reporting; substitute joystick
        # values to include forbidden requests that the device must neutralize.
        wrapper._xbox_action_components = lambda state: (human_request.copy(), np.zeros(3), np.zeros(3))
        return wrapper
    namespace["XboxIntervention"] = xbox
    return namespace["TrainConfig"]


class RobotSubstitute(gym.Env):
    def __init__(self, *, terminal_at=2, fail_step=None):
        self.action_space = gym.spaces.Box(-1., 1., shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "gripper_pose": gym.spaces.Box(0., 1., shape=(1,), dtype=np.float32),
            }),
            "images": gym.spaces.Dict({name: gym.spaces.Box(0, 255, shape=(8, 8, 3), dtype=np.uint8)
                                       for name in ("side_policy", "wrist_1")}),
        })
        self.terminal_at, self.fail_step = terminal_at, fail_step
        self.frame_references = {}
        self.steps = self.resets = 0
        self.commands = []
        self.closed = False

    def pose(self):
        # Orientation changes make use of pre-step versus post-step transforms
        # observable, without any physical device.
        return np.concatenate(([.65, -.01, .15], R.from_euler("xyz", [np.pi, 0., .4 + self.steps * .3]).as_quat()))

    def observation(self):
        return {"state": dict(tcp_pose=self.pose(), tcp_vel=np.zeros(6), tcp_force=np.zeros(3),
                              tcp_torque=np.zeros(3), gripper_pose=np.array([.5])),
                "images": {name: np.full((8, 8, 3), self.steps, dtype=np.uint8)
                           for name in ("side_policy", "wrist_1")}}

    def raw_state(self):
        return {"values": {"tcp_pose": self.pose()}, "capture": stamp()}

    def reset(self, **kwargs):
        self.resets += 1
        self.steps = 0
        return self.observation(), {"reset": {"success": True, "synthetic": True}}

    def step(self, action):
        self.commands.append(np.asarray(action).copy())
        self.steps += 1
        self._step_commands = [dict(kind="pose", returned=True, request_sent=True)]
        if self.steps == self.fail_step:
            raise RobotStateUnavailable("synthetic stale feedback", status_code=503)
        done = self.steps >= self.terminal_at
        return self.observation(), -.02, done, False, dict(
            controller_input_action=np.asarray(action).copy(), controller_commands=self._step_commands,
            termination_reason="classifier" if done else None,
            grasp_penalty=-.02,  # New replay must ignore even stale lower-layer metadata.
        )

    def close(self):
        self.closed = True


def wrappers(env):
    result = []
    while isinstance(env, gym.Wrapper):
        result.append(type(env).__name__)
        env = env.env
    return result


def test_contract_defaults_preserve_legacy_and_explicit_xyz_is_three_dimensional(monkeypatch):
    monkeypatch.delenv("HILSERL_ACTION_CONTRACT", raising=False)
    assert resolve_action_contract().name == LEGACY
    assert action_contract_from_env().learning_dim == 7
    monkeypatch.setenv("HILSERL_ACTION_CONTRACT", FIXED_XYZ)
    contract = action_contract_from_env()
    assert contract.learning_dim == 3 and contract.setup_mode == "single-arm-fixed-gripper"
    assert resolve_action_contract().name == LEGACY
    action = np.array([.2, -.3, .4], dtype=np.float32)
    expanded = contract.to_device_action(action)
    np.testing.assert_array_equal(expanded, np.array([.2, -.3, .4, 0, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(contract.to_learning_action(expanded), action)
    action[:] = 1
    assert expanded[0] != 1


@pytest.mark.parametrize("action", [np.zeros(7), [0, np.nan, 0], [[0, 0, 0]], [0, np.inf, 0], [0, 1.01, 0]])
def test_invalid_model_actions_fail_before_device_step(action):
    base = RobotSubstitute()
    env = wrapper_factory(base)(FIXED_XYZ).get_environment(fake_env=True)
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert base.commands == []
    env.close()


def test_unknown_or_unlocked_contract_is_rejected():
    with pytest.raises(ValueError):
        resolve_action_contract("typo")
    contract = resolve_action_contract(FIXED_XYZ)
    with pytest.raises(ValueError, match="unlocked"):
        contract.to_learning_action([0, 0, 0, .1, 0, 0, 1])


def test_actual_xbox_translation_remains_inside_body_action_cube_at_every_orientation():
    """Prove the live mapping's norm bound; rotated cube corners are not assumed."""
    wrapper = XboxIntervention(RobotSubstitute())
    scale = xbox_module.ACTION_SCALE[:3]
    np.testing.assert_array_equal(scale, np.full(3, .015))
    assert wrapper._dt == .1
    normal_bound = np.sqrt(2) * xbox_module.XBOX_MAX_STEP / scale[0]
    spiral_delta_bound = (xbox_module.XBOX_SPIRAL_RATE * wrapper._dt
                          + 2 * xbox_module.XBOX_SPIRAL_R
                          * np.sin(xbox_module.XBOX_SPIRAL_W * wrapper._dt / 2))
    insertion_bound = np.hypot(xbox_module.XBOX_MAX_STEP + spiral_delta_bound,
                               xbox_module.XBOX_INSERT_REACH) / scale[0]
    assert max(normal_bound, insertion_bound) < .943
    orientations = R.random(64, random_state=np.random.default_rng(7)).as_matrix()
    observed_max = 0.
    for insertion in (False, True):
        for lx in (-1., 0., 1.):
            for ly in (-1., 0., 1.):
                for ry in (-1., 0., 1.):
                    state = SimpleNamespace(left_x=lx, left_y=ly, right_y=ry, right_x=0.,
                                            dpad_x=0., dpad_y=0., a=insertion, b=False, x=False)
                    wrapper._insert_ticks = 0
                    wrapper._spiral_px = wrapper._spiral_py = 0.
                    for _ in range(100):
                        action, _, _ = wrapper._xbox_action_components(state)
                        norm = np.linalg.norm(action[:3])
                        observed_max = max(observed_max, norm)
                        assert norm <= (insertion_bound if insertion else normal_bound) + 1e-6
                        # Each body component is bounded by the invariant norm.
                        body = np.einsum("nji,j->ni", orientations, action[:3])
                        assert np.max(np.abs(body)) < .943
    assert observed_max > .94


def test_rotated_policy_cube_is_valid_before_existing_device_clamps():
    base = RobotSubstitute(terminal_at=5)
    env = wrapper_factory(base)(FIXED_XYZ).get_environment(fake_env=True)
    env.reset()
    action = np.ones(3, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    assert np.max(np.abs(base.commands[-1][:3])) > 1.
    np.testing.assert_array_equal(info["learning_action"], action)
    env.close()


@pytest.mark.parametrize("name,shape,mode", [
    (LEGACY, (7,), "single-arm-learned-gripper"),
    (FIXED_XYZ, (3,), "single-arm-fixed-gripper"),
])
def test_production_factory_preserves_observation_layout_and_selects_agent_contract(name, shape, mode):
    base = RobotSubstitute()
    config = wrapper_factory(base)(name)
    env = config.get_environment(fake_env=True)
    obs, _ = env.reset()
    assert config.setup_mode == mode and env.action_space.shape == shape
    assert obs["state"].shape == (1, 19)
    np.testing.assert_allclose(obs["state"][0, 4:10], 0., atol=1e-7)
    assert ("GripperPenaltyWrapper" in wrappers(env)) == (name == LEGACY)
    assert ("FixedAxesDeviceWrapper" in wrappers(env)) == (name == FIXED_XYZ)
    env.close()


@pytest.mark.parametrize("outcome", [0, 1])
def test_policy_and_takeover_use_xyz_while_raw_requests_and_human_verdict_are_retained(tmp_path, outcome):
    base = RobotSubstitute()
    human = np.array([.2, .4, -.3, .5, -.4, .6, 1.], dtype=np.float32)
    env = wrapper_factory(base, human_request=human)(FIXED_XYZ).get_environment(fake_env=False)
    recorder = SessionRecorder(tmp_path / "capture", min_free_bytes=0)
    emitted = []
    model_action = np.array([.1, -.2, .3], dtype=np.float32)
    result = run_episodes(env, lambda *args: (model_action, {"revision": 4}), recorder,
                          ScriptedOperator(outcome=outcome), action_contract=FIXED_XYZ,
                          max_episodes=1, emit=emitted.append)
    assert len(emitted) == 2 and result[0]["human_steps"] == 1
    assert base.resets == 1 and base.closed and len(base.commands) == 2
    for command in base.commands:
        np.testing.assert_array_equal(command[3:], np.zeros(4))
    expected = R.from_euler("xyz", [np.pi, 0., .7]).as_matrix().T @ human[:3]
    np.testing.assert_allclose(emitted[1]["actions"], expected, atol=1e-7)
    np.testing.assert_array_equal(emitted[0]["actions"], model_action)
    assert [row["rewards"] for row in emitted] == [0., float(outcome)]
    assert [row["dones"] for row in emitted] == [False, True]
    assert [row["masks"] for row in emitted] == [1., 0.]
    assert all("grasp_penalty" not in row for row in emitted)
    raw = read_step(recorder.directory / "episodes/000001/000001.npz")
    for key in ("proposed_action", "policy_action", "actions"):
        assert raw[key].shape == (7,)
        np.testing.assert_array_equal(raw[key][3:], np.zeros(4))
    np.testing.assert_array_equal(raw["requested_action"], human)
    np.testing.assert_array_equal(raw["requested_controller_action"], human)
    assert raw["requested_action_frame"] == "base"
    assert raw["learning_action"].shape == (3,) and raw["action_contract"] == FIXED_XYZ
    assert raw["observed_reward"] == -.02
    metadata = json.loads((recorder.directory / "episodes/000001/episode.json").read_text())
    assert metadata["outcome"] == outcome and metadata["action_contract"] == FIXED_XYZ


def test_outage_retains_requested_axes_without_fabricating_learning_transition(tmp_path):
    base = RobotSubstitute(terminal_at=10, fail_step=2)
    human = np.array([.2, .4, -.3, .5, -.4, .6, 1.], dtype=np.float32)
    env = wrapper_factory(base, human_request=human)(FIXED_XYZ).get_environment(fake_env=False)
    recorder = SessionRecorder(tmp_path / "capture", min_free_bytes=0)
    def on_wait(phase):
        if phase == "waiting_reset" and base.resets:
            raise StopRequested()
    emitted = []
    result = run_episodes(env, lambda *args: (np.zeros(3), {}), recorder,
                          ScriptedOperator(on_wait=on_wait), action_contract=FIXED_XYZ,
                          emit=emitted.append)
    assert not result and len(emitted) == 1 and not emitted[0]["dones"]
    assert emitted[0]["rewards"] == 0 and len(base.commands) == 2 and base.resets == 1
    raw = read_step(recorder.directory / "episodes/000001/000001.npz")
    assert raw["complete_transition"] is False and raw["learning_action"] is None
    assert raw["actions"] is None and raw["next_observations"] is None
    assert raw["proposed_action"].shape == (7,) and raw["policy_action"].shape == (7,)
    attempt = raw["action_attempt"]
    np.testing.assert_array_equal(attempt["requested_controller_action"], human)
    np.testing.assert_array_equal(attempt["controller_input_action"][3:], np.zeros(4))
    metadata = json.loads((recorder.directory / "episodes/000001/episode.json").read_text())
    assert metadata["complete"] is False and metadata["outcome"] is None


def test_legacy_episode_keeps_seven_axis_replay_and_existing_reward_contract(tmp_path):
    base = RobotSubstitute()
    env = wrapper_factory(base)(LEGACY).get_environment(fake_env=True)
    recorder = SessionRecorder(tmp_path / "capture", min_free_bytes=0)
    emitted = []
    action = np.array([.1, .2, .3, .4, .5, .6, 1.], dtype=np.float32)
    run_episodes(env, lambda *args: (action, {}), recorder, ScriptedOperator(outcome=1),
                 max_episodes=1, emit=emitted.append)
    assert emitted[0]["actions"].shape == (7,)
    assert emitted[0]["actions"][6] == 1.
    assert emitted[0]["rewards"] == -.04 and emitted[0]["grasp_penalty"] == -.02
    raw = read_step(recorder.directory / "episodes/000001/000000.npz")
    np.testing.assert_array_equal(raw["proposed_action"], action)
    assert emitted[-1]["rewards"] == 1. and emitted[-1]["masks"] == 0.


def test_replay_rejects_contract_or_recorded_action_disagreement():
    raw = dict(info={}, observations={}, next_observations={}, actions=np.zeros(7),
               learning_action=np.ones(3), action_contract=FIXED_XYZ,
               observed_reward=-.02, source_action="policy", global_step=1,
               id="run/capture/ep/1", episode_id="ep")
    with pytest.raises(ValueError, match="contracts differ"):
        _training_transition(raw)
    with pytest.raises(ValueError, match="differs"):
        _training_transition(raw, action_contract=FIXED_XYZ)
