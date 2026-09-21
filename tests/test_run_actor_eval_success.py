import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_info_success():
    source = (
        REPO_ROOT / "experiments" / "plug_insertion" / "run_actor.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_info_success"
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "<run-actor-info-success>", "exec"), namespace)
    return namespace["_info_success"]


def test_eval_success_reads_succeed_key_used_by_env_wrappers():
    info_success = _load_info_success()

    assert info_success({"succeed": True}) is True
    assert info_success({"succeed": False}) is False


def test_eval_success_remains_backward_compatible_with_success_key():
    info_success = _load_info_success()

    assert info_success({"success": True}) is True
    assert info_success({"success": False}) is False
