"""A per-update numerical gate, separate from sampled training diagnostics.

Import this only in the Learner path. All state floating/complex leaves and
scalar update diagnostics are reduced on the device by one cached JIT function;
the caller transfers one boolean to the host before committing an update group.
The function does not mutate state and does not inspect or convert each tensor
on the host. This checks numerical finiteness, not policy quality or robot safety.
"""
from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp


def _state_field(state, name):
    if isinstance(state, Mapping):
        if name not in state:
            raise ValueError(f"TrainState is missing {name}")
        return state[name]
    if not hasattr(state, name):
        raise ValueError(f"TrainState is missing {name}")
    return getattr(state, name)


def _checks(leaves, *, scalar_only=False):
    checks = []
    for leaf in leaves:
        if leaf is None or not hasattr(leaf, "dtype"):
            raise ValueError("Update diagnostics and dynamic state leaves must be numeric")
        if scalar_only and leaf.shape != ():
            raise ValueError("Update diagnostics must be scalar")
        dtype = leaf.dtype
        if jnp.issubdtype(dtype, jnp.inexact):
            checks.append(jnp.all(jnp.isfinite(leaf)))
        elif jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.bool_):
            # Integer counters and old uint32 RNG keys cannot contain NaN/Inf.
            continue
        elif not scalar_only and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            # New typed RNG keys are opaque, not a floating-point array.
            continue
        else:
            raise ValueError(f"Unsupported numeric health-check dtype: {dtype}")
    return checks


@jax.jit
def _update_is_finite(state, update_info):
    # These checks are structural and run on tracing, without reading array data.
    for name in ("params", "target_params", "opt_states"):
        tree = _state_field(state, name)
        if tree is None or (name != "opt_states" and not jax.tree_util.tree_leaves(tree)):
            raise ValueError(f"TrainState {name} must not be missing or empty")
    if not isinstance(update_info, Mapping) or not update_info:
        raise ValueError("update_info must be a nonempty mapping of scalar diagnostics")
    info_leaves = []
    def collect_info(node):
        if isinstance(node, Mapping):
            for key, value in node.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("Diagnostic keys must be nonempty strings")
                collect_info(value)
        else:
            # Keep sequence containers and None as invalid leaves rather than
            # allowing tree_flatten to disguise a batch as independent scalars.
            info_leaves.append(node)
    collect_info(update_info)
    if not info_leaves:
        raise ValueError("update_info contains no scalar diagnostics")
    # Flax excludes static callables/optimizer definitions from the state pytree.
    # None placeholders in the model state are permitted; diagnostics are not.
    checks = _checks(jax.tree_util.tree_leaves(state))
    checks.extend(_checks(info_leaves, scalar_only=True))
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def require_finite_update(state, update_info):
    """Raise before committing a nonfinite complete update group.

    ``state`` is the updated registered TrainState pytree, or a mapping with
    params, target_params and opt_states. State counters, integer/typed RNG keys,
    and None placeholders are allowed. ``update_info`` is a nonempty nested
    mapping of real or complex numeric scalars. Malformed structures raise
    ValueError; nonfinite numbers raise FloatingPointError. Compile/device errors
    propagate instead of allowing an unchecked state to be committed.
    """
    try:
        finite = _update_is_finite(state, update_info)
    except TypeError as exc:
        raise ValueError("TrainState/update_info must be supported numeric JAX pytrees") from exc
    # The only explicit host transfer, independent of the number of model leaves.
    if not bool(jax.device_get(finite)):
        raise FloatingPointError(
            "模型参数、目标网络、优化器状态或训练指标出现 NaN/Inf；本组更新未提交。"
        )
