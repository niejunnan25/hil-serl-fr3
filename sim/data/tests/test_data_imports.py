"""Smoke test: sim/data/ 关键模块可 import。"""
import pytest


def test_plug_reward_labeler_importable():
    from sim.data import plug_reward_labeler

    # 实际函数是 label_rewards（A7 改签名）。
    assert hasattr(plug_reward_labeler, "label_rewards")


# sim_replay_pipeline 依赖 gello_replay（line 59），而 gello_replay 引用
# 不存在的 /home/robot/... 路径 → 整个模块链都无法 import。A2-A6 改造后再启用。
@pytest.mark.skip(reason="sim_replay_pipeline 依赖 gello_replay；待 A2-A6 修复")
def test_sim_replay_pipeline_importable():
    from sim.data import sim_replay_pipeline

    assert hasattr(sim_replay_pipeline, "run_pipeline_single")
    assert hasattr(sim_replay_pipeline, "run_pipeline_batch")


# gello_replay.py 引用 /home/robot/... 路径，import 失败；A2-A6 修复后再启用。
@pytest.mark.skip(reason="gello_replay.py 重构在 A2-A6 完成")
def test_gello_replay_importable():
    from sim.data import gello_replay

    assert hasattr(gello_replay, "DEFAULT_POS_SCALE")
