import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ACTOR_PATH = REPO_ROOT / "experiments" / "plug_insertion" / "run_actor.py"


def _load_checkpoint_helper():
    source = RUN_ACTOR_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_eval_checkpoint_path"
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"os": os, "PROJECT_ROOT": str(REPO_ROOT)}
    exec(compile(module, "<run-actor-eval-checkpoint>", "exec"), namespace)
    return namespace["_resolve_eval_checkpoint_path"]


def test_eval_checkpoint_path_flag_is_declared_for_deploy_policy():
    source = RUN_ACTOR_PATH.read_text(encoding="utf-8")

    assert "--eval_checkpoint_path" in source


def test_eval_checkpoint_path_takes_explicit_absolute_path():
    helper = _load_checkpoint_helper()
    path = "/tmp/checkpoints/actor_50000"

    assert helper(SimpleNamespace(eval_checkpoint_path=path, eval_checkpoint_step=None)) == path


def test_eval_checkpoint_path_resolves_relative_to_project_root():
    helper = _load_checkpoint_helper()

    assert helper(
        SimpleNamespace(eval_checkpoint_path="checkpoints/actor_50000", eval_checkpoint_step=None)
    ) == str(REPO_ROOT / "checkpoints" / "actor_50000")


def test_eval_checkpoint_step_preserves_existing_actor_directory_convention():
    helper = _load_checkpoint_helper()

    assert helper(
        SimpleNamespace(eval_checkpoint_path=None, eval_checkpoint_step=50000)
    ) == str(REPO_ROOT / "checkpoints" / "actor_50000")


def test_eval_checkpoint_requires_path_or_step():
    helper = _load_checkpoint_helper()

    with pytest.raises(ValueError, match="eval checkpoint"):
        helper(SimpleNamespace(eval_checkpoint_path=None, eval_checkpoint_step=None))
