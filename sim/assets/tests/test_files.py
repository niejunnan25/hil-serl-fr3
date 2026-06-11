"""Smoke test: sim/assets/ USD 文件存在。"""
import os
import pathlib

def test_fr3_usd_exists():
    repo = pathlib.Path(os.environ.get("HILSERL_FR3_ROOT", "/Users/tacyvan/Documents/Code/hilserl-fr3"))
    assert (repo / "sim/assets/fr3.usd").exists()

def test_fr3_gripper_collision_usd_exists():
    repo = pathlib.Path(os.environ.get("HILSERL_FR3_ROOT", "/Users/tacyvan/Documents/Code/hilserl-fr3"))
    assert (repo / "sim/assets/fr3_gripper_collision.usd").exists()
