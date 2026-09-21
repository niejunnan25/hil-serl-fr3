"""Read fresh reset-side inputs through the factory's explicit observation path."""


def refresh_reset_observation(base, relative, observation_wrappers, chunk):
    """No device action, reset, reward classification, or environment step.

    Preserve the verified reset origin while updating the current body frame and
    reseeding observation history. The factory supplies every observation transform;
    wrappers outside this path affect only actions, rewards, or episode statistics.
    """
    from franka_env.utils.transformations import construct_transform_matrix
    from serl_launcher.wrappers.chunking import stack_obs

    base._update_currpos()
    obs = base._get_obs()
    relative.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
    obs = relative.transform_observation(obs)
    for wrapper in observation_wrappers:
        obs = wrapper.observation(obs)
    chunk.current_obs.clear()
    chunk.current_obs.extend([obs] * chunk.obs_horizon)
    return stack_obs(chunk.current_obs)
