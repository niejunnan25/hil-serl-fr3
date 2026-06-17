import numpy as np

def normalize_action(cartesian_delta: np.ndarray, action_scale: list[float], gripper: float) -> np.ndarray:
    """
    Normalize Cartesian delta actions to [-1, 1] range.

    Args:
        cartesian_delta: 6D array [x, y, z, roll, pitch, yaw]
        action_scale: List of 3 floats [pos_scale, rpy_scale, gripper_scale]
            - action_scale[0]: scale for x, y, z translation
            - action_scale[1]: scale for roll, pitch, yaw rotation
            - action_scale[2]: (unused in this formula, gripper passed directly)
        gripper: Gripper value (already in desired range or raw from GELLO)

    Returns:
        7D normalized action array [x, y, z, roll, pitch, yaw, gripper]
    """
    cartesian_delta = np.asarray(cartesian_delta, dtype=np.float32)

    xyz_normalized = cartesian_delta[:3] / action_scale[0]
    rpy_normalized = cartesian_delta[3:6] / action_scale[1]

    action = np.concatenate([xyz_normalized, rpy_normalized, [gripper]])
    action = np.clip(action, -1.0, 1.0)

    return action
