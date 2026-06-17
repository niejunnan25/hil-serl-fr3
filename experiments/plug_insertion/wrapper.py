"""
plug_insertion 任务专用 wrapper。
"""
import gymnasium as gym
import numpy as np


class GripperPenaltyWrapper(gym.Wrapper):
    """
    夹爪动作惩罚包装器。

    减少无效夹爪切换，引导策略在正确时机开合夹爪。
    """

    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # After SERLObsWrapper, state = [tcp_pose(6), tcp_vel(6), tcp_force(3),
        # tcp_torque(3), gripper_pose(1)] = 25D. Gripper is at the LAST index.
        self.last_gripper_pos = obs["state"][0, -1]
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            penalty = self.penalty
        else:
            penalty = 0.0

        info["grasp_penalty"] = penalty
        # 直接修改 reward，确保惩罚生效 (不依赖下游读取 info)
        reward = reward + penalty

        self.last_gripper_pos = observation["state"][0, -1]
        return observation, reward, terminated, truncated, info
