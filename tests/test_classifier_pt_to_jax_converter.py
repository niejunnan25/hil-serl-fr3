import ast
import sys
import types
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER = REPO_ROOT / "scripts" / "convert_classifier_pt_to_jax.py"


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def numpy(self):
        return self._array.copy()


def _load_mapping_helpers():
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jnp.array = np.array
    sys.modules["jax.numpy"] = fake_jnp
    sys.modules.setdefault("jax", types.ModuleType("jax"))

    source = CONVERTER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_names = {"_map_pytorch_to_jax", "_flatten_params", "_unflatten_params"}
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "<classifier-pt-to-jax>", "exec"), namespace)
    return namespace["_map_pytorch_to_jax"]


def test_mapping_rebuilds_tree_with_mapped_leaf_values():
    map_params = _load_mapping_helpers()
    jax_params = {
        "encoder": {
            "Conv_0": {
                "kernel": np.zeros((3, 3, 1, 2), dtype=np.float32),
                "bias": np.zeros((2,), dtype=np.float32),
            },
        },
        "head": {
            "Dense_0": {
                "kernel": np.zeros((4, 2), dtype=np.float32),
            },
        },
    }
    conv_pt = np.arange(18, dtype=np.float32).reshape(2, 1, 3, 3)
    dense_pt = np.arange(8, dtype=np.float32).reshape(2, 4)
    pt_state = {
        "encoder.Conv_0.weight": _FakeTensor(conv_pt),
        "encoder.Conv_0.bias": _FakeTensor(np.array([0.25, -0.5], dtype=np.float32)),
        "head.Dense_0.weight": _FakeTensor(dense_pt),
    }

    mapped = map_params(pt_state, jax_params, verbose=False)

    np.testing.assert_array_equal(
        mapped["encoder"]["Conv_0"]["kernel"],
        conv_pt.transpose(2, 3, 1, 0),
    )
    np.testing.assert_array_equal(
        mapped["encoder"]["Conv_0"]["bias"],
        np.array([0.25, -0.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mapped["head"]["Dense_0"]["kernel"],
        dense_pt.T,
    )
