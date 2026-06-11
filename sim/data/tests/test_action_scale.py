"""A4: gello_replay 的 DEFAULT_POS_SCALE/RPY_SCALE/GRIPPER_SCALE 必须 = contract.ACTION_SCALE。

具体:
  DEFAULT_POS_SCALE     = 0.015
  DEFAULT_RPY_SCALE     = 0.1
  DEFAULT_GRIPPER_SCALE = 1.0
"""
import pytest


def test_default_pos_scale_is_0_015():
    from sim.data import gello_replay
    assert gello_replay.DEFAULT_POS_SCALE == 0.015, (
        f"DEFAULT_POS_SCALE must be 0.015; got {gello_replay.DEFAULT_POS_SCALE}"
    )


def test_default_rpy_scale_is_0_1():
    from sim.data import gello_replay
    assert gello_replay.DEFAULT_RPY_SCALE == 0.1, (
        f"DEFAULT_RPY_SCALE must be 0.1; got {gello_replay.DEFAULT_RPY_SCALE}"
    )


def test_default_gripper_scale_is_1_0():
    from sim.data import gello_replay
    assert hasattr(gello_replay, "DEFAULT_GRIPPER_SCALE"), (
        "gello_replay must expose DEFAULT_GRIPPER_SCALE = 1.0 (A4 new constant)"
    )
    assert gello_replay.DEFAULT_GRIPPER_SCALE == 1.0


def test_action_scale_aligned_with_contract():
    """gello_replay 的 3 个 scale 必须 = sim/data/contract.ACTION_SCALE 的 0/3/6 索引。"""
    from sim.data import gello_replay
    from sim.data.contract import ACTION_SCALE
    assert gello_replay.DEFAULT_POS_SCALE == ACTION_SCALE[0]
    assert gello_replay.DEFAULT_RPY_SCALE == ACTION_SCALE[3]
    assert gello_replay.DEFAULT_GRIPPER_SCALE == ACTION_SCALE[6]


def test_no_hardcoded_0_1_or_0_2_in_gello_replay():
    """grep-style: gello_replay.py 顶部的 DEFAULT_*_SCALE 不能 = 0.1 或 0.2 硬编码 (A4 目标)。

    注: 此测试用 AST 解析赋值，容许出现 "0.1" 作为其他变量 (e.g. friction=0.1) 没问题；
    只检查 'DEFAULT_POS_SCALE' / 'DEFAULT_RPY_SCALE' 的赋值 RHS 是否 = 0.1 / 0.2。
    """
    import ast
    import pathlib
    from sim.data import gello_replay
    src = pathlib.Path(gello_replay.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id in (
                "DEFAULT_POS_SCALE", "DEFAULT_RPY_SCALE", "DEFAULT_GRIPPER_SCALE",
            )
            for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.Constant) and node.value.value in (0.1, 0.2):
            target_name = next(
                t.id for t in node.targets
                if isinstance(t, ast.Name)
            )
            pytest.fail(
                f"{target_name} = {node.value.value} is the OLD hardcoded value; "
                f"A4 requires import from sim.data.contract.ACTION_SCALE"
            )
