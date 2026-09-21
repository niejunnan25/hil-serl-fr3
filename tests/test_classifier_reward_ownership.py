import numpy as np
import pytest

from experiments.plug_insertion.success_gate import classify_success
from test_fixed_xyz_action_contract import RobotSubstitute, wrapper_factory, FIXED_XYZ


class LegacyPoseMatch(RobotSubstitute):
    suppress_pose_reward = False

    def pose(self):
        pose = super().pose()
        if self.steps:
            pose[2] -= .1
        return pose

    def step(self, action):
        obs, _, done, truncated, info = super().step(action)
        if not self.suppress_pose_reward:
            return obs, 1, True, truncated, dict(info, termination_reason="environment_success")
        return obs, 0, done, truncated, info


def environment(logit):
    base = LegacyPoseMatch(terminal_at=100)
    factory = wrapper_factory(base)
    ns = factory.get_environment.__globals__
    def construct(**kwargs):
        base.suppress_pose_reward = kwargs["manual_reward"]
        return base
    ns.update(PlugInsertionEnv=construct, _load_classifier_adaptive=lambda **kw: lambda obs: np.array([logit]),
              jnp=np, DEFAULT_CLASSIFIER_THRESHOLD=.78, DEFAULT_DEPTH_THRESHOLD=.08,
              DEFAULT_STREAK_REQUIRED=3, REWARD_RELZ_STATE_INDEX=6, classify_success=classify_success)
    env = factory(FIXED_XYZ).get_environment(fake_env=True, classifier=True, mode="eval")
    env.reset()
    return env


def test_rejected_classifier_cannot_terminate_from_legacy_pose_match():
    env = environment(-10.)
    for _ in range(4):
        _, reward, done, truncated, info = env.step(np.zeros(3))
        assert reward == 0 and not done and not truncated
        assert info.get("termination_reason") != "environment_success"


def test_classifier_streak_remains_the_success_trigger():
    env = environment(10.)
    for index in range(3):
        _, reward, done, _, info = env.step(np.zeros(3))
        assert bool(reward) == (index == 2)
        assert done == (index == 2)
    assert info["termination_reason"] == "classifier"
