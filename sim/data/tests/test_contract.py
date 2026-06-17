"""A2/A7: sim/data/contract.py 单源真相断言.

A2: 验证 STATE_KEYS_ORDERED / STATE_DIMS / STATE_DTYPE / ACTION_SCALE 等契约常量与 mainline 一致.
A7: 验证 insertion detection thresholds (8mm/2mm/5°) 已在 contract 中持有.
"""
import pytest


# ---------------------------------------------------------------------------
# A2: state schema
# ---------------------------------------------------------------------------
def test_state_keys_ordered_matches_mainline():
    from sim.data.contract import STATE_KEYS_ORDERED
    assert STATE_KEYS_ORDERED == (
        "tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose",
    )


def test_state_dims_is_25_pending_verify():
    """STATE_DIMS 临时 hardcode = 25 与 wrapper.py 注释一致; VERIFY.md 待 hard-freeze 证据."""
    from sim.data.contract import STATE_DIMS
    assert STATE_DIMS == 25


def test_state_dtype_is_float32():
    from sim.data.contract import STATE_DTYPE
    assert STATE_DTYPE == "float32"


# ---------------------------------------------------------------------------
# A2: action scale
# ---------------------------------------------------------------------------
def test_action_scale_7d_default():
    from sim.data.contract import ACTION_SCALE
    assert len(ACTION_SCALE) == 7
    assert ACTION_SCALE[:3] == (0.015, 0.015, 0.015)  # pos
    assert ACTION_SCALE[3:6] == (0.1, 0.1, 0.1)        # rpy
    assert ACTION_SCALE[6] == 1.0                       # gripper


# ---------------------------------------------------------------------------
# A2: image spec
# ---------------------------------------------------------------------------
def test_image_shape_3_128_128():
    from sim.data.contract import IMAGE_SHAPE, IMAGE_DTYPE
    assert IMAGE_SHAPE == (3, 128, 128)
    assert IMAGE_DTYPE == "uint8"


# ---------------------------------------------------------------------------
# A2: image keys
# ---------------------------------------------------------------------------
def test_policy_image_keys_two():
    from sim.data.contract import POLICY_IMAGE_KEYS
    assert set(POLICY_IMAGE_KEYS) == {"side_policy", "wrist_1"}


def test_classifier_image_keys_and_alias():
    from sim.data.contract import CLASSIFIER_IMAGE_KEYS, CLASSIFIER_ALIAS_OF, IMAGE_KEY_ALIAS_MAP
    assert CLASSIFIER_IMAGE_KEYS == ("side_classifier",)
    assert ("side_classifier", "side_policy") in CLASSIFIER_ALIAS_OF
    assert IMAGE_KEY_ALIAS_MAP["side_classifier"] == "side_policy"


# ---------------------------------------------------------------------------
# A7: insertion detection thresholds
# ---------------------------------------------------------------------------
def test_insertion_depth_threshold_in_contract():
    from sim.data.contract import INSERTION_DEPTH_THRESHOLD
    assert INSERTION_DEPTH_THRESHOLD == pytest.approx(0.008, abs=1e-9)

def test_xy_tolerance_in_contract():
    from sim.data.contract import XY_TOLERANCE
    assert XY_TOLERANCE == pytest.approx(0.002, abs=1e-9)

def test_angle_tolerance_deg_in_contract():
    from sim.data.contract import ANGLE_TOLERANCE_DEG
    assert ANGLE_TOLERANCE_DEG == pytest.approx(5.0, abs=1e-9)


# ---------------------------------------------------------------------------
# A2: episode
# ---------------------------------------------------------------------------
def test_max_episode_length_is_300():
    from sim.data.contract import MAX_EPISODE_LENGTH
    assert MAX_EPISODE_LENGTH == 300
