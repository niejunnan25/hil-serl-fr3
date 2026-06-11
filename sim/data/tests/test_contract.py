"""Contract 必须定义 ACTION_SCALE / STATE_KEYS_ORDERED / STATE_DIMS / IMAGE_SHAPE。

A2 hard-freeze: 25D state 的 ordered keys 与 dim 必须出现在 contract 中。
"""
import pytest


def test_action_scale_present_and_correct():
    from sim.data.contract import ACTION_SCALE
    assert ACTION_SCALE == (0.015, 0.015, 0.015, 0.1, 0.1, 0.1, 1.0)


def test_state_keys_ordered_matches_mainline_proprio():
    from sim.data.contract import STATE_KEYS_ORDERED
    # Mainline: experiments/plug_insertion/config.py:237
    expected = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")
    assert STATE_KEYS_ORDERED == expected


def test_state_dims_is_25():
    from sim.data.contract import STATE_DIMS
    assert STATE_DIMS == 25, f"STATE_DIMS must be 25 (hard-freeze), got {STATE_DIMS}"


def test_state_dtype_float32():
    from sim.data.contract import STATE_DTYPE
    assert STATE_DTYPE == "float32"


def test_image_shape():
    from sim.data.contract import IMAGE_SHAPE
    assert IMAGE_SHAPE == (3, 128, 128)


def test_policy_and_classifier_image_keys():
    from sim.data.contract import (
        POLICY_IMAGE_KEYS, CLASSIFIER_IMAGE_KEYS, CLASSIFIER_ALIAS_OF,
    )
    assert POLICY_IMAGE_KEYS == ("side_policy", "wrist_1")
    assert CLASSIFIER_IMAGE_KEYS == ("side_classifier",)
    # sim 单一 TiledCamera: side_classifier 复用 side_policy
    assert CLASSIFIER_ALIAS_OF == (("side_classifier", "side_policy"),)


def test_max_episode_length():
    from sim.data.contract import MAX_EPISODE_LENGTH
    assert MAX_EPISODE_LENGTH == 150


def test_transition_keys_complete():
    from sim.data.contract import TRANSITION_KEYS
    assert TRANSITION_KEYS == (
        "observations", "next_observations", "actions",
        "rewards", "masks", "dones",
    )


def test_contract_version_is_v2_1():
    from sim.data.contract import CONTRACT_VERSION
    assert CONTRACT_VERSION == "v2.1-real-2026-06-11"
