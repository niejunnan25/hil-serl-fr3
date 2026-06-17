"""Smoke test: sim/data/ 关键模块可 import。"""


def test_plug_reward_labeler_importable():
    from sim.data import plug_reward_labeler

    # 实际函数是 label_rewards（A7 改签名）。
    assert hasattr(plug_reward_labeler, "label_rewards")


def test_sim_replay_pipeline_importable():
    from sim.data import sim_replay_pipeline

    assert hasattr(sim_replay_pipeline, "run_pipeline_single")
    assert hasattr(sim_replay_pipeline, "run_pipeline_batch")


def test_gello_replay_importable():
    from sim.data import gello_replay

    assert hasattr(gello_replay, "DEFAULT_POS_SCALE")
