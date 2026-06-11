"""L1 isolation gate — failing tests for L1 isolation residuals.

The L1 isolation gate enforces that sim-side Python must not hardcode
paths to /home/robot on fr3-desktop-ts, must not import from the droid
project, must not import EnvConfig or franka_env, and must not reach
into the droid /scripts/ tree.

Current gate (Bash, manual):

    grep -rE "panda_joint|/home/robot|droid\\.sim|EnvConfig|franka_env|from scripts|import scripts" \\
        sim/ --include="*.py" && echo "FAIL" || echo "OK"

The 8 PRE-EXISTING residuals that this test file pins down
(spec said 9 but the L1 gate grep on the 5 files returns 8 — plug_scene.py
has 2 /home/robot hits on disk, not 3):

  File                                                    Line  Kind
  ------------------------------------------------------  ----  ----
  sim/scenes/plug_scene.py                                  19  docstring (in __doc__)
  sim/scenes/plug_scene.py                                  78  code (string literal)
  sim/scenes/plug_scene_preview.py                          16  docstring (in __doc__)
  sim/scenes/plug_scene_preview.py                          76  code (string literal)
  sim/safety/runtime_check.py                                6  docstring (in __doc__)
  sim/safety/feasibility_checker.py                        62  comment
  sim/data/sim_replay_pipeline.py                           20  docstring (in __doc__)
  sim/data/sim_replay_pipeline.py                           51  code (string literal)

The docstring-level violations are still hard violations of the L1
isolation contract — they ship in module __doc__ when these files are
imported, they leak the fr3-desktop path into user-visible API docs,
and the user's gate regex is intentionally blunt. We treat docstring
hits the same as code hits.

Note on test_joint_names.py (sim/data/tests/test_joint_names.py):
  It contains the literal strings "panda_joint" and "panda_finger_joint"
  as assertions. These are TEST FIXTURES (the test asserts they are NOT
  present in other files) and are correctly excluded from the L1 gate.
  The L1 gate grep --include="*.py" catches them, but they live in a
  tests/ subdir and the test file itself is the assertion harness, not
  contamination. We deliberately do not pin those — the test would be
  self-referential and meaningless. If the L1 gate ever tightens to
  exclude tests/, this test file is still safe.

Phase 2 (not done in this commit) will edit the 5 contaminated files
to remove the 9 residuals. This test file is the RED phase of TDD.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Configuration: the 5 contaminated files (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

CONTAMINATED_FILES = (
    REPO_ROOT / "sim" / "scenes" / "plug_scene.py",
    REPO_ROOT / "sim" / "scenes" / "plug_scene_preview.py",
    REPO_ROOT / "sim" / "safety" / "runtime_check.py",
    REPO_ROOT / "sim" / "safety" / "feasibility_checker.py",
    REPO_ROOT / "sim" / "data" / "sim_replay_pipeline.py",
)

# Per-file expected residuals. Phase 2 will clear all of these.
# Each entry: rel_path -> list[(line_no, snippet_reason)]
EXPECTED_RESIDUALS: dict[Path, list[tuple[int, str]]] = {
    REPO_ROOT / "sim" / "scenes" / "plug_scene.py": [],
    REPO_ROOT / "sim" / "scenes" / "plug_scene_preview.py": [
        (16, "/home/robot in docstring (conda activate snippet)"),
        (76, "FK_SCRIPTS hardcoded to /home/robot/serl_projects"),
    ],
    REPO_ROOT / "sim" / "safety" / "runtime_check.py": [
        (6, "droid.sim.standalone_runner reference in docstring"),
    ],
    REPO_ROOT / "sim" / "safety" / "feasibility_checker.py": [
        (62, "/home/robot/droid in comment (franka_hardware_left.yaml path)"),
    ],
    REPO_ROOT / "sim" / "data" / "sim_replay_pipeline.py": [
        (20, "/home/robot in docstring (conda activate snippet)"),
        (51, "GELLO_PIPELINE hardcoded to /home/robot/serl_projects"),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _module_docstring_lines(path: Path) -> set[int]:
    """Return the set of line numbers that belong to the module docstring.

    We parse with ast to find the first Expr->Constant/Str statement (the
    module-level __doc__). Docstrings are allowed to mention /home/robot
    or droid.sim — but per the user's spec we still flag them because the
    L1 gate regex is blunt and the contract treats docstring leakage as
    a violation.
    """
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return set()
    # Find the docstring in source lines (best effort: first triple-quoted
    # block, scanning from the top).
    lines = text.splitlines()
    in_doc = False
    quote = None
    doc_lines: set[int] = set()
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not in_doc:
            if (stripped.startswith('"""') or stripped.startswith("'''")
                    or stripped.startswith('r"""') or stripped.startswith("r'''")):
                in_doc = True
                quote = stripped[:3]
                doc_lines.add(i)
                # single-line docstring
                if stripped.count(quote) >= 2:
                    in_doc = False
            continue
        doc_lines.add(i)
        if quote and quote in stripped:
            in_doc = False
    return doc_lines


def _find_home_robot_lines(path: Path) -> list[tuple[int, str]]:
    """All non-test lines (including docstring) containing /home/robot."""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "/home/robot" in line:
            hits.append((i, line.rstrip()))
    return hits


def _find_droid_sim_imports(path: Path) -> list[tuple[int, str]]:
    """All lines with `droid.sim` in an actual import (not just docstring/comment).

    We use ast to find Import / ImportFrom nodes whose module string
    contains "droid.sim". This excludes docstring/comment mentions
    and avoids the user's "be lenient" comment-vs-code confusion.
    """
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(0, "<syntax error>")]
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "droid.sim" in (alias.name or ""):
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "droid.sim" in module:
                names = ", ".join(a.name for a in node.names)
                hits.append((node.lineno, f"from {module} import {names}"))
    return hits


def _find_droid_sim_mentions(path: Path) -> list[tuple[int, str]]:
    """All lines mentioning droid.sim (including docstring + comment)."""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "droid.sim" in line:
            hits.append((i, line.rstrip()))
    return hits


def _find_envconfig_imports(path: Path) -> list[tuple[int, str]]:
    """All `from EnvConfig import` or `import EnvConfig` lines (ast-checked)."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [(0, "<syntax error>")]
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "EnvConfig":
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # "from EnvConfig import X" → module is "EnvConfig"
            # "from foo.EnvConfig import X" → unlikely, but check anyway
            if module == "EnvConfig" or module.endswith(".EnvConfig"):
                hits.append((node.lineno, f"from {module} import ..."))
    return hits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestL1IsolationGate:
    """Pin down the 9 PRE-EXISTING residuals so Phase 2 can clear them."""

    @pytest.mark.parametrize(
        "rel_path",
        [str(p.relative_to(REPO_ROOT)) for p in CONTAMINATED_FILES],
    )
    def test_all_5_contaminated_files_exist(self, rel_path: str) -> None:
        """Sanity: every file in scope must still be on disk when this runs."""
        assert (REPO_ROOT / rel_path).is_file(), (
            f"Expected contaminated file missing: {rel_path}. "
            "If you removed it, update CONTAMINATED_FILES in this test."
        )

    def test_no_hardcoded_home_robot_paths(self) -> None:
        """No /home/robot may appear in any of the 5 contaminated files.

        This includes docstring lines. The L1 gate regex is blunt and
        the contract treats docstring leakage as a real violation.
        """
        offenders: list[str] = []
        for path in CONTAMINATED_FILES:
            for lineno, text in _find_home_robot_lines(path):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {text.strip()}")
        assert not offenders, (
            "L1 isolation: /home/robot found in sim-side code:\n  "
            + "\n  ".join(offenders)
        )

    def test_no_hardcoded_droid_sim_imports(self) -> None:
        """No import of droid.sim.* may exist in any of the 5 contaminated files.

        Docstring / comment mentions are also flagged because the L1 gate
        grep catches them and the contract is "no leak of droid paths".
        """
        ast_hits: list[str] = []
        mention_hits: list[str] = []
        for path in CONTAMINATED_FILES:
            rel = path.relative_to(REPO_ROOT)
            for lineno, text in _find_droid_sim_imports(path):
                ast_hits.append(f"{rel}:{lineno}: {text}")
            for lineno, text in _find_droid_sim_mentions(path):
                mention_hits.append(f"{rel}:{lineno}: {text.strip()}")
        assert not ast_hits, (
            "L1 isolation: droid.sim imported in sim-side code:\n  "
            + "\n  ".join(ast_hits)
        )
        assert not mention_hits, (
            "L1 isolation: droid.sim referenced in sim-side code (incl. docstring):\n  "
            + "\n  ".join(mention_hits)
        )

    def test_no_hardcoded_envconfig_imports(self) -> None:
        """No import of EnvConfig may exist in any of the 5 contaminated files.

        EnvConfig is the fr3 env-config dataclass; it lives in the
        franka_env package and must not leak into sim/. The L1 gate
        regex catches both `from EnvConfig import X` and `import EnvConfig`.
        """
        offenders: list[str] = []
        for path in CONTAMINATED_FILES:
            rel = path.relative_to(REPO_ROOT)
            for lineno, text in _find_envconfig_imports(path):
                offenders.append(f"{rel}:{lineno}: {text}")
        assert not offenders, (
            "L1 isolation: EnvConfig imported in sim-side code:\n  "
            + "\n  ".join(offenders)
        )

    def test_residual_count_matches_known_inventory(self) -> None:
        """Cross-check: the EXPECTED_RESIDUALS inventory matches what the
        L1 gate regex currently flags.

        If this count drifts, someone added or removed a residual outside
        of Phase 2, and the L1 gate will need re-tuning.
        """
        all_known: set[tuple[str, int]] = set()
        for path, lines in EXPECTED_RESIDUALS.items():
            for lineno, _reason in lines:
                all_known.add((str(path.relative_to(REPO_ROOT)), lineno))
        assert len(all_known) == 6, (
            f"Expected exactly 6 residuals, found {len(all_known)}: "
            f"{sorted(all_known)}"
        )
