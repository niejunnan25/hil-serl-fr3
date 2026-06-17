"""Item 19: the native-FR3 stable_sim PD gains live in ONE place.

Guards against re-divergence of plug_scene._build_fr3_cfg and
official_fr3_loader._build_native_fr3_cfg. Source-string checks only
(isaaclab is GPU-gated and not importable in CI), plus a value pin on the
canonical module which IS importable (no isaaclab dependency).
"""
import os
import pathlib

REPO = pathlib.Path(os.environ.get(
    "HILSERL_FR3_ROOT", "/Users/tacyvan/Documents/Code/hilserl-fr3"))


def test_canonical_module_values():
    from sim.assets import fr3_pd_gains as g
    assert g.ARM_STIFFNESS == 400.0
    assert g.ARM_DAMPING == 80.0
    assert g.GRIPPER_STIFFNESS == 2e3
    assert g.GRIPPER_DAMPING == 1e2
    assert g.ARM_EFFORT_LIMIT_SIM["fr3_joint1"] == 87.0
    assert g.ARM_EFFORT_LIMIT_SIM["fr3_joint7"] == 12.0


def test_plug_scene_uses_canonical_gains():
    text = (REPO / "sim/scenes/plug_scene.py").read_text()
    assert "from sim.assets.fr3_pd_gains import" in text
    assert "stiffness=ARM_STIFFNESS" in text
    assert "damping=ARM_DAMPING" in text
    assert "stiffness=GRIPPER_STIFFNESS" in text
    # the old hard-coded gripper outlier must be gone
    assert "stiffness=200.0" not in text


def test_loader_uses_canonical_gains():
    text = (REPO / "sim/assets/official_fr3_loader.py").read_text()
    assert "from sim.assets.fr3_pd_gains import" in text
    assert "stiffness=ARM_STIFFNESS" in text
    assert "stiffness=GRIPPER_STIFFNESS" in text
