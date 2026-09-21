"""Worktree imports and run paths cannot fall back to active checkout defaults."""
import json
import os
from pathlib import Path
import subprocess
import sys

from hilserl.config import Config


def test_child_prefers_worktree_sources_over_inherited_live_pythonpath(tmp_path, monkeypatch):
    live = tmp_path / "live"
    worktree = live / "artifacts/worktrees/reward"
    packages = {"hilserl": Path("."), "agentlace": Path("upstream/agentlace"),
                "serl_launcher": Path("upstream/hil-serl/serl_launcher"),
                "franka_env": Path("upstream/hil-serl/serl_robot_infra")}
    inherited = []
    for root in (live, worktree):
        for package, parent in packages.items():
            directory = root / parent / package
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text("")
            if root == live:
                inherited.append(str(root / parent))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(inherited))
    config = Config(root=worktree, python=sys.executable)
    env = config.environment("learner", run_dir=config.output_root / "test-run")
    code = ("import importlib, json; "
            f"print(json.dumps({{name: importlib.import_module(name).__file__ for name in {list(packages)!r}}}))")
    # Run outside both trees so cwd cannot hide an incorrect PYTHONPATH order.
    paths = json.loads(subprocess.check_output([sys.executable, "-c", code],
                                              cwd=tmp_path, env=env, text=True))
    assert all(Path(path).is_relative_to(worktree) for path in paths.values()), paths
    assert config.output_root.is_relative_to(worktree)
    assert config.control_root.is_relative_to(worktree)
    assert Path(env["HILSERL_RUN_DIR"]).is_relative_to(worktree)
    assert Path(env["HILSERL_CONTROL_DIR"]).is_relative_to(worktree)
