"""
plug_insertion 任务专用 wrapper。
"""
import gymnasium as gym
import numpy as np

from experiments.plug_insertion.state_layout import gripper_position
from hilserl.action_contract import FIXED_XYZ, resolve_action_contract


class FixedAxesDeviceWrapper(gym.Wrapper):
    """Enforce task locks below human takeover, at the seven-axis device boundary.

    The attempt is saved before forwarding, so a feedback outage after a command
    still retains the original request without claiming a complete transition.
    """

    def __init__(self, env):
        super().__init__(env)
        if env.action_space.shape != (7,):
            raise ValueError(f"expected 7D device action, got {env.action_space.shape}")
        self.contract = resolve_action_contract(FIXED_XYZ)

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32).copy()
        selected = self.contract.project_device_action(requested)
        self.unwrapped._action_contract_attempt = dict(
            action_contract=FIXED_XYZ, requested_controller_action=requested.copy(),
            controller_input_action=selected.copy(),
        )
        obs, reward, terminated, truncated, info = self.env.step(selected)
        info = dict(info)
        info["requested_controller_action"] = requested
        info["controller_input_action"] = selected
        return obs, reward, terminated, truncated, info


class FixedInterventionActionWrapper(gym.Wrapper):
    """Canonicalize base-frame takeover actions before RelativeFrame sees them.

    Xbox reports its original request after the device wrapper has applied the
    locks. Preserve that request separately, then let RelativeFrame rotate the
    selected translation using its pre-step orientation. Its transform is block
    diagonal, so locked rotation and gripper axes remain exactly zero.
    """

    def __init__(self, env):
        super().__init__(env)
        self.contract = resolve_action_contract(FIXED_XYZ)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        selected = np.asarray(info["controller_input_action"], dtype=np.float32)
        # This is still a base-frame request. A valid body-frame policy cube
        # can rotate outside the base cube; existing device clamps own that
        # execution mapping. Check locks here, and model bounds after R^-1.
        if not np.array_equal(selected, self.contract.project_device_action(selected)):
            raise ValueError("fixed-xyz-v1 device action contains unlocked axes")
        if "intervene_action" in info:
            info["requested_intervene_action"] = np.asarray(info["intervene_action"], dtype=np.float32).copy()
            info["intervene_action"] = selected.copy()
        return obs, reward, terminated, truncated, info


class FixedXYZActionWrapper(gym.Wrapper):
    """Expose exactly xyz to SAC, outside the body-frame observation wrappers."""

    action_contract = FIXED_XYZ

    def __init__(self, env):
        super().__init__(env)
        if env.action_space.shape != (7,):
            raise ValueError(f"expected 7D inner action, got {env.action_space.shape}")
        self.contract = resolve_action_contract(FIXED_XYZ)
        self.action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[:3], dtype=np.float32),
            high=np.asarray(env.action_space.high[:3], dtype=np.float32), dtype=np.float32,
        )

    def step(self, action):
        device_action = self.contract.to_device_action(action)
        obs, reward, terminated, truncated, info = self.env.step(device_action)
        info = dict(info)
        selected = np.asarray(info.get("intervene_action", device_action), dtype=np.float32)
        learning_action = self.contract.to_learning_action(selected)
        if "intervene_action" in info:
            info["intervene_action"] = learning_action.copy()
        info.update(action_contract=FIXED_XYZ, learning_action=learning_action,
                    selected_action=self.contract.to_device_action(learning_action))
        return obs, reward, terminated, truncated, info


class EpisodeRewardWrapper(gym.Wrapper):
    """Classifier suggestions belong to one episode and do not supply final labels."""

    def __init__(self, env, reward_func, reward_state):
        super().__init__(env)
        self.reward_func = reward_func
        self.reward_state = reward_state

    def reset(self, **kwargs):
        self.reward_state.clear()
        self.reward_state.update(t=0, streak=0)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_func(obs)
        info["classifier"] = dict(self.reward_state)
        info["succeed"] = bool(reward)
        if reward:
            terminated = True
            info["termination_reason"] = "classifier"
        return obs, reward, terminated, truncated, info


class RotationLockWrapper(gym.ActionWrapper):
    """Lock rotation channels (indices 3:6) to zero.

    Plug insertion only needs xy+z translation + gripper.  The policy
    never needs roll/pitch/yaw so we force them to 0 at the env boundary.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.action_space.shape == (7,), f"expected 7D action, got {env.action_space.shape}"

    def action(self, action):
        a = np.array(action, dtype=np.float32).copy()
        a[3:6] = 0.0
        return a


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
        # SERLObsWrapper flattens gymnasium Dict keys alphabetically; gripper is index 0.
        self.last_gripper_pos = gripper_position(obs["state"])
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

        self.last_gripper_pos = gripper_position(observation["state"])
        return observation, reward, terminated, truncated, info
