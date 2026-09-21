"""Explicit policy resolution must reach the CNN and preserve legacy loading."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "upstream/hil-serl/serl_launcher/serl_launcher"
IMAGE_KEYS = ("side_policy", "wrist_1")


def load_function(path, name, namespace=None):
    """Exercise the production boundary without importing robot/GPU packages."""
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def sample(size):
    return {"state": np.zeros((1, 19), np.float32),
            "side_classifier": np.zeros((1, 128, 128, 3), np.uint8),
            **{key: np.zeros((1, size, size, 3), np.uint8) for key in IMAGE_KEYS}}


@pytest.fixture
def resolve_size():
    return load_function(LAUNCHER / "agents/continuous/sac.py", "_resolve_encoder_image_size")


def test_explicit_resolution_checks_only_policy_images(resolve_size):
    assert resolve_size((224, 224), sample(224), IMAGE_KEYS) == (224, 224)
    assert resolve_size([224, 224], sample(224), IMAGE_KEYS) == (224, 224)
    assert resolve_size((128, 128), sample(128), IMAGE_KEYS) == (128, 128)
    # Legacy callers intentionally retain the encoder's historical resize.
    assert resolve_size(None, sample(224), IMAGE_KEYS) == (128, 128)


@pytest.mark.parametrize("size", [224, (), (224,), (224, 224, 3), (0, 224), (-1, 224),
                                  (224.0, 224), (True, 224), "224x224"])
def test_invalid_resolution_rejected_before_model_creation(resolve_size, size):
    with pytest.raises(ValueError, match="two positive integers"):
        resolve_size(size, sample(224), IMAGE_KEYS)


@pytest.mark.parametrize("defect", ["wrong_size", "missing", "no_spatial_axes"])
def test_mismatched_sample_rejected_before_model_creation(resolve_size, defect):
    obs = sample(224)
    if defect == "wrong_size":
        obs["wrist_1"] = np.zeros((1, 128, 128, 3), np.uint8)
    elif defect == "missing":
        del obs["wrist_1"]
    else:
        obs["wrist_1"] = np.zeros((224, 224), np.uint8)
    with pytest.raises(ValueError, match="wrist_1.*spatial shape"):
        resolve_size((224, 224), obs, IMAGE_KEYS)


@pytest.mark.parametrize("size", [None, (224, 224)])
def test_launcher_forwards_resolution_to_native_sac(size):
    create = Mock(return_value=object())
    factory = load_function(LAUNCHER / "utils/launcher.py", "make_sac_pixel_agent", {
        "SACAgent": SimpleNamespace(create_pixels=create),
        "jax": SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: seed)),
        "nn": SimpleNamespace(tanh=np.tanh),
        "make_batch_augmentation_func": lambda keys: None,
    })
    kwargs = {} if size is None else {"encoder_image_size": size}
    result = factory(0, sample(224 if size else 128), np.zeros(3, np.float32),
                     image_keys=IMAGE_KEYS, **kwargs)
    assert result is create.return_value
    assert create.call_args.kwargs["encoder_image_size"] == size


@pytest.mark.parametrize("size,pool_size", [(None, 4), ((224, 224), 7)])
def test_native_pretrained_backbone_pooling_and_augmentation(size, pool_size, monkeypatch):
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    # Native checks use the already-provisioned artifact and never download it.
    if not (Path.home() / ".serl/resnet10_params.pkl").is_file():
        pytest.skip("native ResNet-10 check requires the provisioned pretrained artifact")
    monkeypatch.syspath_prepend(str(LAUNCHER.parent))
    from flax.core import freeze
    from flax.traverse_util import flatten_dict
    from serl_launcher.utils.launcher import make_sac_pixel_agent

    dimension = 224 if size else 128
    obs = sample(dimension)
    agent = make_sac_pixel_agent(0, obs, np.zeros(3, np.float32), image_keys=IMAGE_KEYS,
                                 encoder_image_size=size)
    assert agent.config["encoder_image_size"] == (dimension, dimension)
    pools = [value for path, value in flatten_dict(agent.state.params).items()
             if path[-2:] == ("SpatialLearnedEmbeddings_0", "kernel")]
    assert len(pools) == len(IMAGE_KEYS)
    assert all(value.shape == (pool_size, pool_size, 512, 8) for value in pools)
    action = np.asarray(jax.device_get(agent.sample_actions(obs, argmax=True)))
    assert action.shape == (3,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1

    batch_obs = {key: np.stack([value, value]) for key, value in obs.items()}
    batch = freeze({"observations": batch_obs, "next_observations": batch_obs})
    augmented = agent.config["augmentation_function"](batch, jax.random.PRNGKey(7))
    for field in ("observations", "next_observations"):
        for key in IMAGE_KEYS:
            assert augmented[field][key].shape == (2, 1, dimension, dimension, 3)
            assert augmented[field][key].dtype == np.uint8
        np.testing.assert_array_equal(augmented[field]["side_classifier"], batch_obs["side_classifier"])
