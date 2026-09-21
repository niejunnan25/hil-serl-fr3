from collections import namedtuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from hilserl.model_health import require_finite_update


AdamState = namedtuple("AdamState", ("count", "mu", "nu"))


def state():
    return {
        "params": {"weights": jnp.arange(6, dtype=jnp.float32).reshape(2, 3), "unused": None},
        "target_params": {"weights": jnp.ones((2, 3), dtype=jnp.float32), "unused": None},
        "opt_states": {"actor": AdamState(jnp.asarray(4, jnp.int32),
                                         jnp.zeros((2, 3)), jnp.ones((2, 3)))},
        "step": jnp.asarray(4, jnp.int32),
        "rng": jax.random.PRNGKey(0),
        "epsilon": 0.0,
    }


def info():
    return {"actor": {"actor_loss": jnp.asarray(0.2), "temperature": jnp.asarray(0.01)},
            "critic": {"critic_loss": jnp.asarray(0.4), "target_qs": jnp.asarray(0.6)}}


def test_finite_state_nested_optimizer_and_scalar_diagnostics_pass_without_mutation():
    current = state()
    before = jax.tree_util.tree_map(np.asarray, current)
    assert require_finite_update(current, info()) is None
    for actual, expected in zip(jax.tree_util.tree_leaves(current), jax.tree_util.tree_leaves(before)):
        np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("field", ["params", "target_params", "opt_states", "loss", "epsilon"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_every_float_state_component_and_loss_is_checked(field, bad):
    current, diagnostics = state(), info()
    if field in ("params", "target_params"):
        current[field]["weights"] = current[field]["weights"].at[1, 2].set(bad)
    elif field == "opt_states":
        actor = current["opt_states"]["actor"]
        current["opt_states"]["actor"] = actor._replace(nu=actor.nu.at[0, 0].set(bad))
    elif field == "epsilon":
        current["epsilon"] = bad
    else:
        diagnostics["actor"]["actor_loss"] = jnp.asarray(bad)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        require_finite_update(current, diagnostics)


def test_complex_finiteness_checks_real_and_imaginary_components():
    current = state()
    current["params"]["complex"] = jnp.asarray([1 + 2j], dtype=jnp.complex64)
    require_finite_update(current, info())
    current["params"]["complex"] = jnp.asarray([complex(1, float("inf"))], dtype=jnp.complex64)
    with pytest.raises(FloatingPointError):
        require_finite_update(current, info())


def test_integer_boolean_and_typed_rng_key_state_are_allowed():
    current = state()
    current["rng"] = jax.random.key(12)
    current["enabled"] = jnp.asarray(True)
    current["int_limit"] = jnp.asarray(np.iinfo(np.uint32).max, dtype=jnp.uint32)
    require_finite_update(current, {"loss": jnp.asarray(1.0), "sample_count": 3})


def test_only_one_boolean_is_transferred_to_host(monkeypatch):
    current, diagnostics = state(), info()
    require_finite_update(current, diagnostics)  # Compile before observing transfers.
    device_get, observed = jax.device_get, []

    def spy(value):
        observed.append((value.shape, value.dtype))
        return device_get(value)

    monkeypatch.setattr(jax, "device_get", spy)
    require_finite_update(current, diagnostics)
    assert observed == [((), jnp.dtype("bool"))]


def test_registered_train_state_excludes_static_functions():
    from flax import struct

    @struct.dataclass
    class TrainState:
        params: object
        target_params: object
        opt_states: object
        rng: object
        step: object
        apply_fn: object = struct.field(pytree_node=False)

    current = state()
    train_state = TrainState(**{k: current[k] for k in ("params", "target_params", "opt_states", "rng", "step")},
                             apply_fn=lambda: None)
    require_finite_update(train_state, info())
    bad = train_state.replace(target_params={"weights": jnp.asarray(float("nan"))})
    with pytest.raises(FloatingPointError):
        require_finite_update(bad, info())


@pytest.mark.parametrize("diagnostics", [{}, {"actor": {}}, {"loss": None},
                                        {"loss": "0.5"}, {"loss": [1.0, 2.0]}])
def test_malformed_diagnostics_are_not_treated_as_healthy(diagnostics):
    with pytest.raises(ValueError):
        require_finite_update(state(), diagnostics)


@pytest.mark.parametrize("field", ["params", "target_params", "opt_states"])
def test_missing_train_state_component_is_rejected(field):
    current = state()
    current.pop(field)
    with pytest.raises(ValueError, match="missing"):
        require_finite_update(current, info())


def test_array_diagnostic_rejected_even_when_finite():
    with pytest.raises(ValueError, match="scalar"):
        require_finite_update(state(), {"loss": jnp.ones((2,))})
