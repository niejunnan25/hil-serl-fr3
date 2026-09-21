import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeEnv:
    def __init__(self, state):
        self._obs = {"state": np.asarray(state, dtype=np.float32).reshape(1, -1)}

    def reset(self):
        return self._obs, {}

    def step(self, _action):
        return self._obs, 0.0, False, True, {}


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
    module_name = "_test_" + relative_path.replace("/", "_").replace(".py", "")
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("relative_path", "call_kwargs"),
    (
        ("scripts/inference_service.py", {"max_steps": 10}),
        ("scripts/eval_rollout.py", {}),
    ),
)
def test_eval_scripts_read_tcp_pose_z_from_flat_state_index_6(
    monkeypatch,
    relative_path,
    call_kwargs,
):
    module = _load_script_module(monkeypatch, relative_path)
    state = np.arange(19, dtype=np.float32)
    state[2] = 99.0
    state[6] = 0.123
    env = _FakeEnv(state)

    episode = module.run_episode(
        env,
        policy_fn=lambda _obs: np.zeros(7, dtype=np.float32),
        classifier_fn=lambda _obs: -10.0,
        episode_id=0,
        **call_kwargs,
    )

    assert episode["final_z_height"] == pytest.approx(0.123)
