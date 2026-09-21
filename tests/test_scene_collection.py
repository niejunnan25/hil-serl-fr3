"""The changed mount uses its own profile; human collection owns labels."""
import numpy as np
import pytest

from hilserl.config import Config, validate_classifier_input_contract
from hilserl.image_profile import get_image_profile, preprocess_image
from test_fixed_xyz_action_contract import RobotSubstitute, wrapper_factory, FIXED_XYZ
from test_plug_insertion_reset_safety import _Clock, _make_env


NEW = "insert-front-roi160-v2"
OLD = "insert-front-roi160-v1"


def test_new_mount_has_distinct_identity_and_retains_the_reviewed_crop_pixels():
    Config(action_contract=FIXED_XYZ, image_profile=NEW).validate()
    old, new = get_image_profile(OLD), get_image_profile(NEW)
    assert old["name"] != new["name"]
    assert old["cameras"] == new["cameras"]
    frame = np.random.default_rng(21).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    for camera in new["cameras"]:
        np.testing.assert_array_equal(preprocess_image(frame, camera, NEW),
                                      preprocess_image(frame, camera, OLD))


def test_old_classifier_cannot_silently_run_on_new_mount():
    old_contract = dict(image_key="wrist_1", image_shape=[160, 160, 3],
                        image_dtype="uint8", image_profile=OLD)
    with pytest.raises(ValueError, match="image_profile"):
        validate_classifier_input_contract(old_contract, "wrist_1", NEW)


def test_collect_factory_disables_classifier_even_when_caller_requests_it():
    base = RobotSubstitute()
    factory = wrapper_factory(base)
    namespace = factory.get_environment.__globals__
    seen = {}
    def construct(**kwargs):
        seen.update(kwargs)
        return base
    namespace["PlugInsertionEnv"] = construct
    namespace["_load_classifier_adaptive"] = lambda **kw: pytest.fail("Old classifier loaded in collection")
    env = factory(FIXED_XYZ).get_environment(fake_env=True, mode="collect", classifier=True)
    assert seen["manual_reward"] is True
    assert env.action_space.shape == (3,)


def test_manual_collection_never_terminates_from_pose_reward():
    env, _ = _make_env(_Clock())
    env.manual_reward = True
    assert env.compute_reward({"state": {"tcp_pose": np.zeros(7)}}) == 0
