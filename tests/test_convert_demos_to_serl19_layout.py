import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "convert_demos_to_serl19.py"


def _load_converter():
    module_name = "_test_convert_demos_to_serl19"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _obs_with_state(state):
    image = np.zeros((3, 128, 128), dtype=np.uint8)
    return {
        "state": np.asarray(state, dtype=np.float32),
        "side_policy": image,
        "wrist_1": image,
        "side_classifier": image,
    }


def test_convert_obs_uses_live_serl_alphabetical_flat_state_order():
    converter = _load_converter()
    state25 = np.zeros(25, dtype=np.float32)
    state25[3:7] = [1.0, 0.0, 0.0, 0.0]
    state25[7:13] = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    state25[13:16] = [2.1, 2.2, 2.3]
    state25[16:19] = [3.1, 3.2, 3.3]
    state25[19:25] = [0.567] * 6

    t_reset = np.eye(4)
    obs = converter.convert_obs(_obs_with_state(state25), t_reset)
    state = obs["state"][0]

    assert state.shape == (19,)
    assert state[0] == pytest.approx(0.567)
    np.testing.assert_allclose(state[1:4], [2.1, 2.2, 2.3])
    np.testing.assert_allclose(state[4:10], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state[10:13], [3.1, 3.2, 3.3])
    np.testing.assert_allclose(state[13:19], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
