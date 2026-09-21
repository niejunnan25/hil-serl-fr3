"""collect_insert_demo_serl_aligned helper tests."""

from __future__ import annotations

import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)


def test_default_controller_is_xbox_only():
    import collect_insert_demo_serl_aligned as collect

    parser = collect.build_arg_parser()
    args = parser.parse_args([])

    assert args.controller == "xbox"


def test_select_recorder_loads_xbox_by_default(monkeypatch):
    import collect_insert_demo_serl_aligned as collect

    monkeypatch.setattr(collect, "_load_xbox_recorder_run", lambda: "xbox-run")
    monkeypatch.setattr(collect, "_load_gello_recorder_run", lambda: "gello-run")

    assert collect._select_recorder_run("xbox") == "xbox-run"
    assert collect._select_recorder_run("gello") == "gello-run"


def test_project_root_precedes_upstream_examples_when_run_as_script(monkeypatch):
    upstream_robot = os.path.join(ROOT, "upstream", "hil-serl", "serl_robot_infra")
    upstream_examples = os.path.join(ROOT, "upstream", "hil-serl", "examples")
    original = [
        p
        for p in sys.path
        if p not in {ROOT, SCRIPTS, upstream_robot, upstream_examples}
    ]
    monkeypatch.setattr(
        sys,
        "path",
        [SCRIPTS, upstream_robot, upstream_examples, ROOT, *original],
    )

    module_path = os.path.join(SCRIPTS, "collect_insert_demo_serl_aligned.py")
    spec = importlib.util.spec_from_file_location("collect_path_order_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert sys.path.index(ROOT) < sys.path.index(upstream_examples)
