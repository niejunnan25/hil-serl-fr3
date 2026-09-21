import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_fake_runtime_modules(monkeypatch):
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Env = object

    jax = types.ModuleType("jax")
    jax.random = types.SimpleNamespace(PRNGKey=lambda seed: seed)
    jax.devices = lambda: ["fake-cpu"]

    jnp = types.ModuleType("jax.numpy")
    jnp.exp = np.exp
    jnp.array = np.array

    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jnp)


def _load_script_module(monkeypatch, relative_path):
    _install_fake_runtime_modules(monkeypatch)
    path = REPO_ROOT / relative_path
    module_name = "_test_success_gate_" + relative_path.replace("/", "_").replace(".py", "")
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_success_gate_matches_train_config_depth_direction():
    from experiments.plug_insertion.success_gate import (
        DEFAULT_CLASSIFIER_THRESHOLD,
        DEFAULT_DEPTH_THRESHOLD,
        DEFAULT_STREAK_REQUIRED,
        classify_success,
    )

    assert DEFAULT_CLASSIFIER_THRESHOLD == 0.78
    assert DEFAULT_DEPTH_THRESHOLD == 0.08
    assert DEFAULT_STREAK_REQUIRED == 3
    assert classify_success(0.79, 0.081) is True
    assert classify_success(0.79, 0.079) is False
    assert classify_success(0.77, 0.081) is False


def test_success_streak_requires_consecutive_hits_and_resets_on_miss():
    from experiments.plug_insertion.success_gate import SuccessStreak

    streak = SuccessStreak(required=3)

    assert streak.update(classifier_prob=0.79, depth=0.081) is False
    assert streak.update(classifier_prob=0.79, depth=0.081) is False
    assert streak.update(classifier_prob=0.79, depth=0.079) is False
    assert streak.current == 0
    assert streak.update(classifier_prob=0.79, depth=0.081) is False
    assert streak.update(classifier_prob=0.79, depth=0.081) is False
    assert streak.update(classifier_prob=0.79, depth=0.081) is True


def test_eval_and_inference_scripts_import_shared_success_gate(monkeypatch):
    for relative_path in ("scripts/inference_service.py", "scripts/eval_rollout.py"):
        module = _load_script_module(monkeypatch, relative_path)

        assert module.CLASSIFIER_THRESHOLD == 0.78
        assert module.Z_HEIGHT_THRESHOLD == 0.08
        assert module.SUCCESS_STREAK_REQUIRED == 3
        assert module.classify_success(0.79, 0.081) is True
