"""CPU-only persistence checks, isolated from robot/training imports."""
import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
from flax.training import checkpoints


SOURCE = Path(__file__).resolve().parents[1] / "_run_actor.py"


def load_helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {"_save_training_checkpoint", "_save_final_checkpoint"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {"os": os, "jax": jax, "checkpoints": checkpoints}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class CheckpointRetentionTests(unittest.TestCase):
    def test_enabled_retention_keeps_five_real_committed_models_and_restores_latest(self):
        helper=load_helpers()
        with tempfile.TemporaryDirectory(prefix="hilserl-five-checkpoints-") as folder, patch.dict(os.environ,{"HILSERL_CHECKPOINT_KEEP":"5"}):
            root=Path(folder).resolve()
            (root/"buffer").mkdir();(root/"buffer/transitions_0.pkl.lz4").write_bytes(b"retained replay")
            for step in range(9):
                helper["_save_training_checkpoint"](str(root),{"weight":np.asarray([step],np.float32)},step)
            self.assertEqual(sorted(int(p.name[11:]) for p in root.glob("checkpoint_*") if p.name[11:].isdigit()),[4,5,6,7,8])
            restored=checkpoints.restore_checkpoint(str(root),target=None,step=8)
            np.testing.assert_array_equal(restored["weight"],np.asarray([8],np.float32))
            self.assertEqual((root/"buffer/transitions_0.pkl.lz4").read_bytes(),b"retained replay")

    def test_periodic_and_final_saves_keep_more_than_100_checkpoints(self):
        helper = load_helpers()
        with tempfile.TemporaryDirectory(prefix="hilserl-retention-test-") as folder:
            root = Path(folder)
            state = {"weight": np.asarray([1, 2], dtype=np.float32)}
            helper["_save_training_checkpoint"](folder, state, 1)
            original = hashes(root / "checkpoint_1")
            for step in range(2, 103):
                helper["_save_training_checkpoint"](folder, state, step)
            agent = SimpleNamespace(state=state)
            self.assertEqual(helper["_save_final_checkpoint"](agent, folder, 103), str(root / "checkpoint_103"))
            self.assertEqual(len(list(root.glob("checkpoint_*"))), 103)
            self.assertEqual(hashes(root / "checkpoint_1"), original)
            restored = checkpoints.restore_checkpoint(folder, target=None, step=1)
            np.testing.assert_array_equal(restored["weight"], state["weight"])

    def test_existing_checkpoint_is_not_overwritten_or_removed_on_restart(self):
        helper = load_helpers()
        with tempfile.TemporaryDirectory(prefix="hilserl-no-overwrite-test-") as folder:
            root = Path(folder)
            old_state = {"weight": np.asarray([1], dtype=np.float32)}
            new_state = {"weight": np.asarray([9], dtype=np.float32)}
            helper["_save_training_checkpoint"](folder, old_state, 8)
            helper["_save_training_checkpoint"](folder, old_state, 9)
            before = hashes(root)
            with self.assertRaises((ValueError, checkpoints.errors.InvalidCheckpointError)):
                helper["_save_training_checkpoint"](folder, new_state, 8)
            self.assertEqual(hashes(root), before)
            # Exit-save returns an existing path without replacing its content.
            helper["_save_final_checkpoint"](SimpleNamespace(state=new_state), folder, 9)
            self.assertEqual(hashes(root), before)

    def test_all_training_save_sites_use_preserving_helper(self):
        tree = ast.parse(SOURCE.read_text())
        direct_saves = [n for n in ast.walk(tree)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == "checkpoints"
                        and n.func.attr == "save_checkpoint"]
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_save_training_checkpoint")
        self.assertEqual(len(direct_saves), 1)
        self.assertIn(direct_saves[0], list(ast.walk(helper)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
