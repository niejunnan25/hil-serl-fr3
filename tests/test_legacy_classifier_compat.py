"""Legacy processes retain their recorded classifier camera after defaults change."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from hilserl.config import Config
from hilserl.processes import Manager


@pytest.fixture
def scene_manager(tmp_path, monkeypatch):
    config = replace(Config(), root=tmp_path, data_dir="runs", min_free_gib=0,
                     classifier_image_key="wrist_1",
                     classifier_ckpt="classifier_ckpt_wrist_scene_20260920")
    monkeypatch.setattr("hilserl.processes.process_start", lambda pid: "same")
    monkeypatch.setattr("hilserl.processes.listening_ports", lambda pid=None: {5588, 5589})
    manager = Manager(config)
    manager._spawn = Mock(side_effect=AssertionError("no process launch"))
    return manager


def legacy_pair(manager, learner_key=None, actor_key=None):
    common = {"AGENTLACE_PORT": "5588", "AGENTLACE_BROADCAST_PORT": "5589",
              "HILSERL_CLASSIFIER_CKPT": str(manager.config.root / "historical-classifier")}
    checkpoint = str(manager.config.root / "historical-checkpoints")
    learner = dict(pid=100, start_time="same", role="learner", run_dir=None,
                   args=["python", "_run_actor.py", "--learner", "--exp_name=plug_insertion", "--seed=0"],
                   env=dict(common), checkpoint=checkpoint)
    actor = dict(pid=101, start_time="same", role="train", run_dir=None,
                 args=["python", "_run_actor.py", "--actor", "--exp_name=plug_insertion", "--seed=0", "--ip=127.0.0.1"],
                 env=dict(common), checkpoint=checkpoint)
    if learner_key is not None:
        learner["env"]["HILSERL_CLASSIFIER_IMAGE_KEY"] = learner_key
    if actor_key is not None:
        actor["env"]["HILSERL_CLASSIFIER_IMAGE_KEY"] = actor_key
    return learner, actor


@pytest.mark.parametrize("explicit_key,expected", [(None, "side_classifier"),
    ("side_classifier", "side_classifier"), ("wrist_1", "wrist_1")])
def test_snapshotless_learner_uses_own_camera_or_historical_default(scene_manager, explicit_key, expected):
    learner, _ = legacy_pair(scene_manager, learner_key=explicit_key)
    config = scene_manager._learner_config(learner)
    assert config.classifier_image_key == expected
    assert config.classifier_ckpt == learner["env"]["HILSERL_CLASSIFIER_CKPT"]
    assert config.environment("actor", run_dir=scene_manager.config.root / "run")["HILSERL_CLASSIFIER_IMAGE_KEY"] == expected
    scene_manager._spawn.assert_not_called()


@pytest.mark.parametrize("learner_key,actor_key", [(None, "wrist_1"), ("wrist_1", None)])
def test_snapshotless_pair_rejects_different_classifier_cameras(scene_manager, learner_key, actor_key):
    learner, actor = legacy_pair(scene_manager, learner_key, actor_key)
    with pytest.raises(RuntimeError, match="classifier"):
        scene_manager._validate_pair(actor, learner)
    scene_manager._spawn.assert_not_called()


@pytest.mark.parametrize("key", [None, "side_classifier", "wrist_1"])
def test_matching_snapshotless_pair_preserves_classifier_camera(scene_manager, key):
    learner, actor = legacy_pair(scene_manager, key, key)
    config = scene_manager._validate_pair(actor, learner)
    assert config.classifier_image_key == (key or "side_classifier")
    scene_manager._spawn.assert_not_called()
