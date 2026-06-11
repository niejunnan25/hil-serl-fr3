"""Smoke test: sim/transforms/ 可 import（8D → 7D 重写在 A2）。"""
def test_delta_actions_importable():
    from sim.transforms import delta_actions
    assert hasattr(delta_actions, "apply_delta")
