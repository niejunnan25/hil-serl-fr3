from pathlib import Path
import sys

import gymnasium as gym
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakePlugEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Box(-np.inf, np.inf, shape=(1, 19), dtype=np.float32)}
    )

    def __init__(self):
        self._step_state = np.zeros((1, 19), dtype=np.float32)

    def reset(self, **kwargs):
        state = np.zeros((1, 19), dtype=np.float32)
        state[0, 0] = 1.0
        state[0, -1] = 0.0
        return {"state": state}, {}

    def step(self, action):
        return {"state": self._step_state.copy()}, 1.0, False, False, {}


def test_gripper_penalty_reads_live_flat_gripper_index_zero():
    from experiments.plug_insertion.wrapper import GripperPenaltyWrapper

    env = GripperPenaltyWrapper(_FakePlugEnv(), penalty=-0.02)
    env.reset()
    _obs, reward, _terminated, _truncated, info = env.step(
        np.array([-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    )

    assert info["grasp_penalty"] == -0.02
    assert reward == 0.98


if __name__ == "__main__":
    test_gripper_penalty_reads_live_flat_gripper_index_zero()
