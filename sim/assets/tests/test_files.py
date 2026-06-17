"""Smoke test: sim/assets/ USD 文件存在。"""
import os
import pathlib

def test_fr3_usd_exists():
    repo = pathlib.Path(os.environ.get("HILSERL_FR3_ROOT", "/Users/tacyvan/Documents/Code/hilserl-fr3"))
    assert (repo / "sim/assets/fr3.usd").exists()

def test_fr3_gripper_collision_usd_exists():
    repo = pathlib.Path(os.environ.get("HILSERL_FR3_ROOT", "/Users/tacyvan/Documents/Code/hilserl-fr3"))
    assert (repo / "sim/assets/fr3_gripper_collision.usd").exists()


def test_canonical_paths_module_resolves_to_existing_assets():
    """Item 18: sim/assets/paths.py is the single source of truth and every
    USD constant it exposes must point at a file that exists, with no
    /home/robot leak (pure-stdlib import, no isaaclab side-effect)."""
    from sim.assets import paths
    for const in (
        paths.FR3_USD_PATH,
        paths.FR3_GRIPPER_COLLISION_USD_PATH,
        paths.STRIP_USD_PATH,
        paths.PLUG_USD_PATH,
    ):
        assert "/home/robot" not in const, const
        assert os.path.exists(const), const
