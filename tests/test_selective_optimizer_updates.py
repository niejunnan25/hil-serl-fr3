"""Real Adam momentum must remain frozen on unselected update steps."""
from functools import partial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import optax
from flax import serialization
from serl_launcher.common.common import JaxRLTrainState


def state():
    params = {name: jnp.array([1., 2.]) for name in ("actor", "critic", "temperature", "shared")}
    return JaxRLTrainState.create(apply_fn=lambda *_: None, params=params, target_params=params,
                                 txs={name: optax.adam(.01) for name in ("actor", "critic", "temperature")})


@partial(jax.jit, static_argnames=("selected", "strict"))
def update(current, selected, strict=True):
    losses = {}
    for name in current.txs:
        if name not in selected:
            losses[name] = lambda params, rng: (0., {})
        else:
            def loss(params, rng, name=name):
                value = jnp.sum(params[name] ** 2)
                if name != "temperature": value += jnp.sum(params["shared"] ** 2)
                return value, {"loss": value}
            losses[name] = loss
    new, _ = current.apply_loss_fns(losses, has_aux=True, optimizer_names=selected if strict else None)
    return new


def assert_tree_equal(first, second):
    assert jax.tree_util.tree_structure(first) == jax.tree_util.tree_structure(second)
    for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
        np.testing.assert_array_equal(a, b)


def test_critic_only_freezes_actor_and_temperature_momentum_and_counts():
    all_names = frozenset({"actor", "critic", "temperature"})
    initial = state()
    trained = update(initial, all_names)
    current = update(trained, frozenset({"critic"}))
    for name in ("actor", "temperature"):
        assert_tree_equal(current.params[name], trained.params[name])
        assert_tree_equal(current.opt_states[name], trained.opt_states[name])
    for name in ("critic", "shared"):
        assert not np.array_equal(current.params[name], trained.params[name])
    assert int(current.opt_states["critic"][0].count) == 2
    resumed = update(current, all_names)
    assert int(resumed.opt_states["critic"][0].count) == 3
    assert int(resumed.opt_states["actor"][0].count) == 2
    assert int(resumed.opt_states["temperature"][0].count) == 2
    assert not np.array_equal(resumed.params["actor"], current.params["actor"])
    restored = serialization.from_bytes(initial, serialization.to_bytes(resumed))
    assert_tree_equal(resumed, restored)


def test_legacy_default_preserves_zero_gradient_adam_behavior():
    trained = update(state(), frozenset({"actor", "critic", "temperature"}), strict=False)
    current = update(trained, frozenset({"critic"}), strict=False)
    assert int(current.opt_states["actor"][0].count) == 2
    assert not np.array_equal(current.params["actor"], trained.params["actor"])
    assert not np.array_equal(current.params["temperature"], trained.params["temperature"])


@pytest.mark.parametrize("names", [frozenset(), frozenset({"missing"})])
def test_invalid_optimizer_selection_is_rejected(names):
    with pytest.raises(ValueError, match="Selected optimizers"):
        update(state(), names)
