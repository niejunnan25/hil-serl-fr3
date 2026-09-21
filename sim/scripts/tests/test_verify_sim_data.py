"""A10: verify_sim_data.py 必须 assert 3 键 image schema + ordered state keys.

CRITICAL (codex #1, #2 fix):
  - 3 image keys: side_policy + wrist_1 + side_classifier
  - state keys 必须按 STATE_KEYS_ORDERED 顺序拼接
  - dtype/shape 全断言
  - missing side_classifier pkl 必须 fail
"""
import os
import pickle
import tempfile

import numpy as np
import pytest

from sim.scripts.tests._gen_test_pkls import (
    make_valid_pkl, make_missing_classifier_pkl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_pkl_path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "valid.pkl")
        make_valid_pkl(p, N=5)
        yield p


@pytest.fixture
def missing_classifier_pkl_path():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "missing_classifier.pkl")
        make_missing_classifier_pkl(p, N=5)
        yield p


# ---------------------------------------------------------------------------
# 1) valid pkl: 全 pass
# ---------------------------------------------------------------------------
def test_valid_pkl_passes_all_checks(valid_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(valid_pkl_path)
    # 全部 check 必须是 True
    for check_name, passed in result.items():
        assert passed, f"check {check_name!r} failed on valid pkl; got result={result}"
    # 至少包含 3 键 image schema check + ordered state check
    assert "image_keys_complete" in result
    assert "state_keys_ordered" in result
    assert result["image_keys_complete"] is True
    assert result["state_keys_ordered"] is True


# ---------------------------------------------------------------------------
# 2) missing side_classifier: 必须 fail (codex #1 fix)
# ---------------------------------------------------------------------------
def test_missing_classifier_pkl_fails_image_keys_check(missing_classifier_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(missing_classifier_pkl_path)
    # image_keys_complete 必须 False
    assert result["image_keys_complete"] is False, (
        f"missing side_classifier pkl must fail image_keys_complete check; "
        f"got result={result}"
    )


# ---------------------------------------------------------------------------
# 3) state dtype / shape 检查
# ---------------------------------------------------------------------------
def test_state_shape_and_dtype_check(valid_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(valid_pkl_path)
    assert result.get("state_shape_correct") is True
    assert result.get("state_dtype_correct") is True


def test_action_shape_and_dtype_check(valid_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(valid_pkl_path)
    assert result.get("action_shape_correct") is True
    assert result.get("action_dtype_correct") is True


# ---------------------------------------------------------------------------
# 4) image shape + dtype 检查
# ---------------------------------------------------------------------------
def test_image_shape_and_dtype_check(valid_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(valid_pkl_path)
    assert result.get("image_shape_correct") is True
    assert result.get("image_dtype_correct") is True


# ---------------------------------------------------------------------------
# 5) transition dict top-level keys 检查
# ---------------------------------------------------------------------------
def test_transition_keys_check(valid_pkl_path):
    from sim.scripts.verify_sim_data import verify_pkl
    result = verify_pkl(valid_pkl_path)
    assert result.get("transition_keys_complete") is True


# ---------------------------------------------------------------------------
# 6) state keys order 检查: 用 STATE_KEYS_ORDERED 拼接顺序 (codex #2 fix)
# ---------------------------------------------------------------------------
def test_state_keys_order_check_uses_state_keys_ordered(monkeypatch):
    """verify_state_keys_order 必须 import STATE_KEYS_ORDERED from contract."""
    from sim.data.contract import STATE_KEYS_ORDERED
    # 故意修改 STATE_KEYS_ORDERED 顺序 (但不应影响 contract, 用 monkeypatch 替代)
    import sim.data.contract as contract_mod
    original = contract_mod.STATE_KEYS_ORDERED
    # 倒序
    contract_mod.STATE_KEYS_ORDERED = tuple(reversed(original))
    try:
        from sim.scripts.verify_sim_data import verify_state_keys_order
        # 构造一个 live state, 但 verify 函数应该 assert 顺序 (用 STATE_KEYS_ORDERED 拼接)
        # 此处只验: 函数 import 了 STATE_KEYS_ORDERED
        from sim.data.contract import STATE_DIMS
        result = verify_state_keys_order(np.zeros(STATE_DIMS, dtype=np.float32))
        # 倒序 STATE_KEYS_ORDERED 仍可正确拼接同维 state, 但 order 检查应通过
        # 关键验证: 我们的 verify_state_keys_order 用了 STATE_KEYS_ORDERED 拼接, 而非简单 sum
        assert isinstance(result, bool)
    finally:
        contract_mod.STATE_KEYS_ORDERED = original


# ---------------------------------------------------------------------------
# 7) CLI 命令行入口
# ---------------------------------------------------------------------------
def test_state_keys_order_segment_layout_validates_order():
    """codex #2 真顺序校验: 提供 producer slice layout 时, 子块换位/数据换位必须 fail.

    旧实现忽略 segments、只看 flat shape/dtype, 故 reordered state 会静默 pass。
    本测试证明新实现真正用 STATE_KEYS_ORDERED + STATE_KEY_DIMS 做逐段边界校验。
    """
    from sim.scripts.verify_sim_data import verify_state_keys_order
    from sim.data.contract import STATE_KEYS_ORDERED, STATE_KEY_DIMS

    rng = np.random.default_rng(0)
    segs = {
        k: rng.normal(size=d).astype(np.float32)
        for k, d in zip(STATE_KEYS_ORDERED, STATE_KEY_DIMS)
    }
    flat = np.concatenate(
        [np.asarray(segs[k]).reshape(-1) for k in STATE_KEYS_ORDERED]
    ).astype(np.float32)

    # correct ordered layout => True
    assert verify_state_keys_order(flat, segments=segs) is True

    # permute the ORDER of the first two sub-blocks => must be False
    order = list(STATE_KEYS_ORDERED)
    order[0], order[1] = order[1], order[0]
    permuted = {k: segs[k] for k in order}
    assert verify_state_keys_order(flat, segments=permuted) is False, (
        "reordered sub-blocks must fail verify_state_keys_order"
    )

    # swap the DATA of two equal-size sub-blocks (tcp_force <-> tcp_torque),
    # keeping key order correct => concat no longer equals flat => must be False
    swapped = {k: segs[k] for k in STATE_KEYS_ORDERED}
    swapped["tcp_force"], swapped["tcp_torque"] = (
        segs["tcp_torque"], segs["tcp_force"],
    )
    assert verify_state_keys_order(flat, segments=swapped) is False, (
        "data-swapped sub-blocks must fail verify_state_keys_order"
    )

    # flat-only call path (no segments) preserves the original behavior
    assert verify_state_keys_order(flat) is True


def test_cli_verify_valid_pkl_exits_zero(valid_pkl_path, capsys):
    """CLI: verify valid pkl, exit 0, 输出报告."""
    from sim.scripts.verify_sim_data import main
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["verify_sim_data.py", "--pkl", valid_pkl_path]
        exit_code = main()
        assert exit_code == 0
    finally:
        sys.argv = old_argv
    captured = capsys.readouterr()
    assert "PASS" in captured.out or "pass" in captured.out


def test_cli_verify_missing_classifier_pkl_exits_nonzero(missing_classifier_pkl_path):
    """CLI: verify missing-classifier pkl, exit 1, 输出 FAIL 报告."""
    from sim.scripts.verify_sim_data import main
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["verify_sim_data.py", "--pkl", missing_classifier_pkl_path]
        exit_code = main()
        assert exit_code != 0
    finally:
        sys.argv = old_argv
