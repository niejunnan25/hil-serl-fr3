"""Offline input-contract and optimizer regressions; no robot/model initialization."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from scripts.train_reward_classifier_jax import checkpoint_rank, confusion, input_contract, load_images, roc_auc


ROOT = Path(__file__).resolve().parents[1]


def _dataset(tmp_path, *, key="side_classifier", size=128, mode="RGB"):
    manifest = {
        "classifier_key": key,
        "image_profile": ["insert-front-roi160-v1"],
        "files": {"positive": [], "negative": []},
    }
    arrays = []
    for label, offset in (("positive", 10), ("negative", 80)):
        directory = tmp_path / label
        directory.mkdir()
        shape = (size, size, 3) if mode == "RGB" else (size, size)
        array = (np.arange(np.prod(shape)).reshape(shape) + offset).astype(np.uint8)
        Image.fromarray(array).save(directory / "frame.png")
        manifest["files"][label].append(
            {"file": "frame.png", "episode": f"{label}-episode", "step": "0001"})
        arrays.append(array)
    return manifest, arrays


def test_default_input_preserves_existing_side_classifier_pixels(tmp_path):
    manifest, arrays = _dataset(tmp_path)
    images, labels, episodes = load_images(tmp_path, manifest)
    np.testing.assert_array_equal(images, np.stack(arrays))
    np.testing.assert_array_equal(labels, [1.0, 0.0])
    assert episodes.tolist() == ["positive-episode", "negative-episode"]
    assert input_contract(manifest, "side_classifier", 128) == {
        "image_key": "side_classifier", "image_shape": [128, 128, 3],
        "image_dtype": "uint8", "image_profile": "insert-front-roi160-v1",
    }


def test_wrist_160_contract_preserves_pixels_without_resizing(tmp_path):
    manifest, arrays = _dataset(tmp_path, key="wrist_1", size=160)
    manifest["image_shape"] = [160, 160, 3]
    manifest["image_profile"] = "insert-front-roi160-v1"
    images, _, _ = load_images(tmp_path, manifest, image_key="wrist_1", image_size=160)
    np.testing.assert_array_equal(images, np.stack(arrays))
    assert input_contract(manifest, "wrist_1", 160)["image_shape"] == [160, 160, 3]


@pytest.mark.parametrize("manifest_key", ["side_classifier", None])
def test_requested_camera_must_match_manifest(tmp_path, manifest_key):
    manifest, _ = _dataset(tmp_path, key=manifest_key, size=160)
    with pytest.raises(SystemExit, match="classifier_key"):
        load_images(tmp_path, manifest, image_key="wrist_1", image_size=160)


def test_image_shape_mismatch_is_rejected_instead_of_resized(tmp_path):
    manifest, _ = _dataset(tmp_path, key="wrist_1", size=128)
    with pytest.raises(SystemExit, match="unexpected shape/dtype"):
        load_images(tmp_path, manifest, image_key="wrist_1", image_size=160)


def test_declared_shape_must_match_training_input(tmp_path):
    manifest, _ = _dataset(tmp_path, key="wrist_1", size=160)
    manifest["image_shape"] = [128, 128, 3]
    with pytest.raises(SystemExit, match="manifest image_shape"):
        load_images(tmp_path, manifest, image_key="wrist_1", image_size=160)


def test_non_rgb_images_are_rejected_instead_of_changed(tmp_path):
    manifest, _ = _dataset(tmp_path, mode="L")
    with pytest.raises(SystemExit, match="unexpected shape/dtype"):
        load_images(tmp_path, manifest)


def test_frozen_encoder_stays_exact_under_adamw_weight_decay_on_cpu():
    # A subprocess keeps JAX backend selection local to this test and avoids
    # interference from legacy tests that replace jax in sys.modules.
    code = """
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scripts.train_reward_classifier_jax import encoder_mask, make_optimizer

assert jax.default_backend() == "cpu"
params = {
    "encoder_def": {
        "encoder_wrist_1": {
            "pretrained_encoder": {"kernel": jnp.array([2.0, -3.0])},
            "spatial": {"kernel": jnp.array([1.0])},
        }
    },
    "head": {"kernel": jnp.array([4.0])},
    "not_pretrained_encoder": {"kernel": jnp.array([5.0])},
}
mask = encoder_mask(params)
assert mask["encoder_def"]["encoder_wrist_1"]["pretrained_encoder"]["kernel"]
assert not mask["encoder_def"]["encoder_wrist_1"]["spatial"]["kernel"]
assert not mask["head"]["kernel"]
assert not mask["not_pretrained_encoder"]["kernel"]
grads = jax.tree_util.tree_map(jnp.zeros_like, params)
optimizer = make_optimizer(0.1, 0.2, frozen_mask=mask)
state = optimizer.init(params)
updated = params
for _ in range(4):
    updates, state = optimizer.update(grads, state, updated)
    updated = optax.apply_updates(updated, updates)
np.testing.assert_array_equal(
    updated["encoder_def"]["encoder_wrist_1"]["pretrained_encoder"]["kernel"],
    params["encoder_def"]["encoder_wrist_1"]["pretrained_encoder"]["kernel"])
for expected, actual in (
    (params["head"]["kernel"], updated["head"]["kernel"]),
    (params["encoder_def"]["encoder_wrist_1"]["spatial"]["kernel"],
     updated["encoder_def"]["encoder_wrist_1"]["spatial"]["kernel"]),
    (params["not_pretrained_encoder"]["kernel"],
     updated["not_pretrained_encoder"]["kernel"]),
):
    assert np.all(np.asarray(actual) < np.asarray(expected))

unmasked = make_optimizer(0.1, 0.2)
updates, _ = unmasked.update(grads, unmasked.init(params), params)
changed = optax.apply_updates(params, updates)
assert not np.array_equal(
    changed["encoder_def"]["encoder_wrist_1"]["pretrained_encoder"]["kernel"],
    params["encoder_def"]["encoder_wrist_1"]["pretrained_encoder"]["kernel"])
print("CPU freeze regression passed")
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               JAX_PLATFORM_NAME="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU freeze regression passed" in result.stdout


def test_checkpoint_prefers_later_fixed_threshold_f1_even_when_auc_is_already_one():
    labels = np.asarray([0, 0, 1, 1])
    early = np.asarray([0.1, 0.2, 0.8, 0.7])
    later = np.asarray([0.1, 0.2, 0.9, 0.85])
    early_auc, later_auc = roc_auc(labels, early), roc_auc(labels, later)
    assert early_auc == later_auc == 1.0
    early_confusion = confusion(labels, early, 0.78)
    later_confusion = confusion(labels, later, 0.78)
    assert later_confusion["f1"] > early_confusion["f1"]
    assert checkpoint_rank(later_confusion, later_auc) > checkpoint_rank(early_confusion, early_auc)


def test_checkpoint_prefers_lower_fpr_before_auc_when_fixed_threshold_f1_ties():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    more_fp = np.asarray([0.9, 0.85, 0.2, 0.1, 0.99, 0.98, 0.97, 0.7])
    fewer_fp = np.asarray([0.7, 0.6, 0.5, 0.4, 0.99, 0.98, 0.1, 0.1])
    more_confusion = confusion(labels, more_fp, 0.78)
    fewer_confusion = confusion(labels, fewer_fp, 0.78)
    more_auc, fewer_auc = roc_auc(labels, more_fp), roc_auc(labels, fewer_fp)
    assert more_confusion["f1"] == fewer_confusion["f1"]
    assert fewer_confusion["false_positive_rate"] < more_confusion["false_positive_rate"]
    assert fewer_auc < more_auc
    assert checkpoint_rank(fewer_confusion, fewer_auc) > checkpoint_rank(more_confusion, more_auc)
