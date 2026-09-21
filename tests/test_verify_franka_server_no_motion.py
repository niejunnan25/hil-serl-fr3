import ast
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_franka_server.py"


def _load_main_source():
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _load_should_test_commands():
    source = _load_main_source()
    tree = ast.parse(source)
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "should_test_command_endpoints"
    ]
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"argparse": argparse}
    exec(compile(module, "<verify-franka-server-no-motion>", "exec"), namespace)
    return namespace["should_test_command_endpoints"]


def test_command_endpoints_are_skipped_by_default():
    should_test = _load_should_test_commands()

    assert should_test(SimpleNamespace(live_motion=False, skip_commands=False)) is False


def test_live_motion_flag_is_required_for_command_endpoints():
    should_test = _load_should_test_commands()

    assert should_test(SimpleNamespace(live_motion=True, skip_commands=False)) is True
    assert should_test(SimpleNamespace(live_motion=True, skip_commands=True)) is False


def test_cli_exposes_live_motion_warning_and_no_default_command_phase():
    source = _load_main_source()

    assert "--live-motion" in source
    assert "Command endpoints skipped by default" in source
    assert "if should_test_command_endpoints(args):" in source
