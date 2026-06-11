"""Smoke test: sim/kinematics/ 模块可 import。"""


def test_fr3_fk_importable():
    from sim.kinematics import fr3_fk

    # fr3_fk.py 在 A1 阶段只搬运，函数名为 fk_ee_pose（A2 可能改名）。
    assert hasattr(fr3_fk, "fk_ee_pose")
