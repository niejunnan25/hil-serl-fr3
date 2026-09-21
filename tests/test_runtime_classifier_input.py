"""Classifier camera contracts are checked without devices or model execution."""
import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest

from hilserl.action_contract import action_contract_from_env, resolve_action_contract
from hilserl.config import Config, validate_classifier_input_contract
from hilserl.image_profile import observation_image_schema


TASK_CONFIG = Path(__file__).resolve().parents[1] / "experiments/plug_insertion/config.py"
PROFILE = "insert-front-roi160-v1"


def contract(key="wrist_1", profile=PROFILE):
    schema = observation_image_schema(profile)[key]
    return dict(image_key=key, image_shape=schema["shape"][1:],
                image_dtype=schema["dtype"], image_profile=profile)


def runtime():
    tree = ast.parse(TASK_CONFIG.read_text())
    keep = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"_load_classifier_adaptive", "TrainConfig"}]
    module = ast.Module(body=keep, type_ignores=[])
    namespace = dict(os=os, np=np, DefaultTrainingConfig=object,
                     action_contract_from_env=action_contract_from_env,
                     resolve_action_contract=resolve_action_contract,
                     _load_pytorch_classifier=Mock(side_effect=AssertionError("no fallback")))
    exec(compile(module, str(TASK_CONFIG), "exec"), namespace)
    return namespace


def fake_native_loader(monkeypatch):
    module = ModuleType("serl_launcher.networks.reward_classifier")
    module.load_classifier_func = Mock(return_value="loaded")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.load_classifier_func


def test_historical_snapshot_keeps_side_key_and_explicit_env_owns_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HILSERL_CLASSIFIER_IMAGE_KEY", "wrist_1")
    cfg = Config(root=tmp_path)
    assert cfg.classifier_image_key == "side_classifier"
    env = cfg.environment("actor", run_dir=tmp_path / "run")
    assert env["HILSERL_CLASSIFIER_IMAGE_KEY"] == "side_classifier"
    updated = replace(cfg, classifier_image_key="wrist_1")
    assert updated.snapshot()["classifier_image_key"] == "wrist_1"
    assert updated.environment("actor", run_dir=tmp_path / "run")["HILSERL_CLASSIFIER_IMAGE_KEY"] == "wrist_1"
    assert cfg.digest() != updated.digest()


@pytest.mark.parametrize("key", ["side_policy", "pixels", "", None])
def test_config_rejects_unapproved_classifier_camera(key):
    with pytest.raises(ValueError, match="classifier_image_key"):
        replace(Config(), classifier_image_key=key).validate()


def test_train_config_selects_camera_on_instance_without_changing_historical_default(monkeypatch):
    cls = runtime()["TrainConfig"]
    monkeypatch.delenv("HILSERL_CLASSIFIER_IMAGE_KEY", raising=False)
    original = cls(action_contract="fixed-xyz-v1")
    monkeypatch.setenv("HILSERL_CLASSIFIER_IMAGE_KEY", "wrist_1")
    monkeypatch.setenv("HILSERL_IMAGE_PROFILE", PROFILE)
    updated = cls(action_contract="fixed-xyz-v1")
    assert original.classifier_keys == cls.classifier_keys == ["side_classifier"]
    assert updated.classifier_keys == ["wrist_1"]
    assert updated.image_profile == PROFILE
    monkeypatch.setenv("HILSERL_CLASSIFIER_IMAGE_KEY", "side_policy")
    with pytest.raises(ValueError, match="classifier_image_key"):
        cls(action_contract="fixed-xyz-v1")


def test_contract_matches_versioned_hwc_pixels_and_legacy_side_compatibility():
    assert validate_classifier_input_contract(contract(), "wrist_1", PROFILE) == contract()
    assert validate_classifier_input_contract(None, "side_classifier", PROFILE) == contract("side_classifier")
    with pytest.raises(ValueError, match="historical side_classifier"):
        validate_classifier_input_contract(None, "wrist_1", PROFILE)


@pytest.mark.parametrize("field,value", [("image_key", "side_classifier"),
    ("image_shape", [128, 128, 3]), ("image_shape", [1, 160, 160, 3]),
    ("image_dtype", "float32"), ("image_profile", "insert-roi160-v1")])
def test_contract_rejects_camera_geometry_dtype_or_profile_mismatch(field, value):
    with pytest.raises(ValueError, match=field):
        validate_classifier_input_contract(dict(contract(), **{field: value}), "wrist_1", PROFILE)


@pytest.mark.parametrize("key,profile", [("side_classifier", "full-frame128-v1"),
    ("side_classifier", PROFILE), ("wrist_1", PROFILE), ("wrist_1", "insert-roi224-v1")])
def test_loader_builds_native_encoder_from_selected_camera_shape(tmp_path, monkeypatch, key, profile):
    expected = contract(key, profile)
    (tmp_path / "metrics.json").write_text(json.dumps({"input_contract": expected}))
    load = fake_native_loader(monkeypatch)
    ns = runtime()
    assert ns["_load_classifier_adaptive"](str(tmp_path), [key], "rng", profile) == "loaded"
    actual = load.call_args.kwargs
    assert actual["image_keys"] == [key]
    assert list(actual["sample"]) == [key]
    assert actual["sample"][key].shape == (1, *expected["image_shape"])
    assert actual["sample"][key].dtype == np.uint8
    ns["_load_pytorch_classifier"].assert_not_called()


def test_legacy_side_weights_cannot_silently_restore_into_wrist_encoder(tmp_path, monkeypatch):
    load = fake_native_loader(monkeypatch)
    ns = runtime()
    (tmp_path / "reward_classifier.pt").write_bytes(b"legacy fallback")
    with pytest.raises(ValueError, match="historical side_classifier"):
        ns["_load_classifier_adaptive"](str(tmp_path), ["wrist_1"], "rng", PROFILE)
    load.assert_not_called()
    ns["_load_pytorch_classifier"].assert_not_called()
    assert ns["_load_classifier_adaptive"](str(tmp_path), ["side_classifier"], "rng", PROFILE) == "loaded"


def test_metadata_mismatch_is_rejected_before_either_loader(tmp_path, monkeypatch):
    (tmp_path / "metrics.json").write_text(json.dumps({"input_contract": contract("side_classifier")}))
    (tmp_path / "reward_classifier.pt").write_bytes(b"legacy fallback")
    load = fake_native_loader(monkeypatch)
    ns = runtime()
    with pytest.raises(ValueError, match="image_key"):
        ns["_load_classifier_adaptive"](str(tmp_path), ["wrist_1"], "rng", PROFILE)
    load.assert_not_called()
    ns["_load_pytorch_classifier"].assert_not_called()
