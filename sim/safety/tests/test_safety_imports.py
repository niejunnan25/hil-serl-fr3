"""Smoke test: sim/safety/ 可 import。"""


def test_runtime_check_importable():
    from sim.safety import runtime_check

    # runtime_check.py 实际暴露 ensure_real_stack_idle 函数。
    assert hasattr(runtime_check, "ensure_real_stack_idle")
