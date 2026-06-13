"""zed_env_dispatch — pure-stdlib pyzed conda env dispatcher.

This module answers one question: which conda env on fr3-desktop-ts
has a usable `pyzed` install?

Contract
--------
- `detect_zed_env()` probes the candidate envs in priority order
  `hilserl-fr3`, `robot1`, `robot` and returns the first env whose
  site-packages actually has a `pyzed` import spec.
- `env_python_path(name)` returns
  `/home/robot/miniconda3/envs/<name>/bin/python` if it exists,
  otherwise raises FileNotFoundError.
- Importing this module does NOT import pyzed. The `importlib.util`
  spec probe is deferred to `detect_zed_env()`.
- This dispatcher never falls back to system python3. Callers must
  handle a `None` return from `detect_zed_env()` and surface a
  meaningful error (the chain script uses exit code 11).

This module is a sibling, not a wrapper around `scripts.zed_capture`.
It exists to keep the no-motion probe scripts honest about which
interpreter is going to load pyzed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


# Hard-coded candidate envs and their priority. Order is load-bearing:
# tests in tests/test_zed_env_dispatch.py assert this exact order.
CONDA_ROOT: Path = Path(os.environ.get("ZED_CONDA_ROOT", "/home/robot/miniconda3"))
ENV_PRIORITY: Sequence[str] = ("hilserl-fr3", "robot1", "robot")


def env_python_path(name: str) -> Path:
    """Return the python interpreter for a given conda env name.

    Parameters
    ----------
    name : str
        Conda env name (e.g. "hilserl-fr3").

    Returns
    -------
    pathlib.Path
        Absolute path to the env's `python` binary.

    Raises
    ------
    FileNotFoundError
        If the conda env directory or its `bin/python` does not exist.
    """
    if not name:
        raise FileNotFoundError("env name must be non-empty")
    candidate = CONDA_ROOT / "envs" / name / "bin" / "python"
    if not candidate.exists():
        raise FileNotFoundError(f"no python at {candidate}")
    return candidate


def _env_site_packages(python_path: Path) -> Path:
    """Return the most likely site-packages dir for a conda env python.

    We do not invoke the interpreter; we read the conventional
    `<env>/lib/pythonX.Y/site-packages` path. We pick the highest
    pythonX.Y directory that actually exists so the probe is robust
    across minor version bumps.
    """
    lib_dir = python_path.parent.parent / "lib"
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"no lib dir at {lib_dir}")
    candidates = sorted(
        (p for p in lib_dir.iterdir() if p.name.startswith("python")),
        key=lambda p: p.name,
        reverse=True,
    )
    for c in candidates:
        sp = c / "site-packages"
        if sp.is_dir():
            return sp
    raise FileNotFoundError(f"no site-packages under {lib_dir}")


def _pyzed_spec_exists(site_packages: Path) -> bool:
    """Return True if `pyzed` would be importable from `site_packages`.

    We mimic importlib's finder behavior: if a top-level `pyzed`
    directory or a `pyzed.py` / `pyzed-<ver>.dist-info` exists, then
    `importlib.util.find_spec("pyzed")` will succeed. We deliberately
    avoid importing pyzed here — the chain scripts must remain
    safe to run in envs that don't have pyzed.
    """
    if (site_packages / "pyzed").is_dir():
        return True
    if (site_packages / "pyzed.py").is_file():
        return True
    # pyzed 5.x ships as a dist-info package; presence of any matching
    # dist-info is a strong signal that importlib will resolve it.
    for entry in site_packages.iterdir():
        if entry.name.startswith("pyzed-") and entry.name.endswith(".dist-info"):
            return True
    return False


def detect_zed_env() -> Optional[str]:
    """Return the first conda env name (in priority order) with a usable pyzed.

    Returns
    -------
    str | None
        Env name from `ENV_PRIORITY` whose site-packages has a
        resolvable pyzed spec, or `None` if none qualify.

    Notes
    -----
    Never falls back to system python3. The caller is expected to
    surface a clear error when this returns None.
    """
    for name in ENV_PRIORITY:
        try:
            py = env_python_path(name)
        except FileNotFoundError:
            continue
        try:
            sp = _env_site_packages(py)
        except FileNotFoundError:
            continue
        if _pyzed_spec_exists(sp):
            return name
    return None


def has_pyzed_in_current_interpreter() -> bool:
    """Return True if pyzed is importable in the *current* python.

    Useful for tests that want to skip probe runs when pyzed is not
    available locally. Does not call into pyzed — just `find_spec`.
    """
    return importlib.util.find_spec("pyzed") is not None


__all__ = [
    "CONDA_ROOT",
    "ENV_PRIORITY",
    "detect_zed_env",
    "env_python_path",
    "has_pyzed_in_current_interpreter",
]


if __name__ == "__main__":  # pragma: no cover - manual entry point
    name = detect_zed_env()
    if name is None:
        print("no pyzed env found", file=sys.stderr)
        sys.exit(11)
    print(name)
    sys.exit(0)
