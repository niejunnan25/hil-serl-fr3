import json
from pathlib import Path

import numpy as np
import pytest

from hilserl import training_metrics
from hilserl.training_metrics import LearnerMetrics, read_metrics_tail


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_actual_sac_aux_keys_and_attempt_local_utd(tmp_path):
    # SACTrainState.apply_loss_fns preserves the named optimizer/loss tree.
    info = {
        "critic": {"critic_loss": np.float32(0.2), "predicted_qs": np.array(0.4),
                   "target_qs": 0.6, "rewards": 0.1},
        "actor": {"actor_loss": -0.1, "entropy": 0.2, "temperature": 0.01},
        "temperature": {"temperature_loss": -0.2},
        "grasp_critic": {"grasp_critic_loss": 0.5},
        "actor_lr": np.array(3e-4),
    }
    with LearnerMetrics(tmp_path, "resumed", cta_ratio=2, start_step=4317) as writer:
        sample = writer.write(step=4366, update_groups=50, unique_online_count=100,
                              update_info=info)
        assert sample["metrics"]["critic/target_qs"] == 0.6
        assert sample["metrics"]["actor/temperature"] == 0.01
        assert sample["metrics"]["temperature/temperature_loss"] == -0.2
        assert sample["invalid_fields"] == []
        assert sample["critic_updates"] == 100
        assert sample["critic_utd"] == 1.0
        assert sample["critic_utd_since_sample"] == 1.0
        assert sample["start_step"] == 4317
        assert sample["counter_scope"] == "learner_attempt"
        assert sample["restored_replay_included"] is False
    start, _, end = records(writer.path)
    assert start["step"] == 4316
    assert start["update_groups"] == 0
    assert start["critic_utd"] is None
    assert end["event"] == "attempt_end"
    assert end["critic_updates"] == 100
    assert end["metrics"] == {}


def test_step_or_time_sampling_and_no_unsampled_device_transfers(tmp_path):
    clock = Clock()

    class DeviceScalar:
        shape = ()
        transfers = 0

        def __float__(self):
            self.transfers += 1
            return 1.25

    value = DeviceScalar()
    with LearnerMetrics(tmp_path, "live", cta_ratio=2, clock=clock) as writer:
        assert writer.write(step=0, update_groups=1, unique_online_count=10,
                            update_info={"loss": value}) is None
        assert value.transfers == 0
        assert writer.should_write(update_groups=49) is False
        assert writer.should_write(update_groups=50) is True
        first = writer.write(step=49, update_groups=50, unique_online_count=20,
                             update_info={"loss": value})
        assert first["metrics"]["loss"] == 1.25
        assert value.transfers == 1
        clock.value += 10
        assert writer.should_write(update_groups=51) is True
        second = writer.write(step=50, update_groups=51, unique_online_count=24,
                              update_info={"loss": value})
        assert second["elapsed_seconds"] == 10
        assert second["critic_utd"] == 102 / 24
        assert second["critic_utd_since_sample"] == 2 / 4
        assert second["new_online_since_sample"] == 4
    assert value.transfers == 2


def test_zero_new_online_is_null_not_infinite_or_historical_average(tmp_path):
    with LearnerMetrics(tmp_path, "empty-window", cta_ratio=3) as writer:
        first = writer.write(step=0, update_groups=1, unique_online_count=0, force=True)
        assert first["critic_utd"] is None
        assert first["critic_utd_since_sample"] is None
        second = writer.write(step=1, update_groups=2, unique_online_count=2, force=True)
        assert second["critic_utd"] == 3
        assert second["critic_utd_since_sample"] == 1.5
        third = writer.write(step=2, update_groups=3, unique_online_count=2, force=True)
        assert third["critic_utd"] == 4.5
        assert third["critic_utd_since_sample"] is None


def test_nonfinite_values_are_visible_strict_json_nulls(tmp_path):
    with LearnerMetrics(tmp_path, "invalid", cta_ratio=2) as writer:
        sample = writer.write(step=0, update_groups=1, unique_online_count=10, force=True,
                              update_info={"critic": {"loss": np.array(float("nan"))},
                                           "positive": float("inf"), "negative": -float("inf"),
                                           "missing": None, "valid": 2.0})
        assert sample["invalid_fields"] == ["critic/loss", "missing", "negative", "positive"]
        assert sample["invalid_reasons"]["critic/loss"] == "nan"
        assert sample["invalid_reasons"]["positive"] == "positive_infinity"
        assert sample["metrics"]["critic/loss"] is None
        assert sample["metrics"]["valid"] == 2
        assert writer.last_invalid_fields == tuple(sample["invalid_fields"])
    assert "NaN" not in writer.path.read_text()
    assert "Infinity" not in writer.path.read_text()


def test_extra_scalar_timing_and_buffer_counts_keep_a_separate_namespace(tmp_path):
    with LearnerMetrics(tmp_path, "extra", cta_ratio=2) as writer:
        sample = writer.write(step=0, update_groups=1, unique_online_count=3, force=True,
                              update_info={"actor": {"actor_loss": 0.2}},
                              extra={"timer": {"train": 0.07}, "demo_transitions": 204,
                                     "policy_age_seconds": float("nan")})
        assert sample["metrics"]["actor/actor_loss"] == 0.2
        assert sample["metrics"]["runtime/timer/train"] == 0.07
        assert sample["metrics"]["runtime/demo_transitions"] == 204
        assert sample["invalid_fields"] == ["runtime/policy_age_seconds"]


@pytest.mark.parametrize("value", [np.ones((1,)), np.zeros((128, 128, 3)), [1.0], "loss",
                                  np.array(1 + 2j), complex(1, 2)])
def test_rejects_tensors_strings_and_complex_instead_of_silent_reduction(tmp_path, value):
    with LearnerMetrics(tmp_path, "scalar-only", cta_ratio=2) as writer:
        with pytest.raises((TypeError, ValueError), match="scalar"):
            writer.write(step=0, update_groups=1, unique_online_count=1,
                         update_info={"bad": value}, force=True)
    assert len(records(writer.path)) == 2


def test_tensor_shape_rejected_before_materialization(tmp_path):
    class DeviceImage:
        shape = (128, 128, 3)

        def __float__(self):
            raise AssertionError("Must not transfer an image")

        def __array__(self):
            raise AssertionError("Must not transfer an image")

    with LearnerMetrics(tmp_path, "no-image", cta_ratio=2) as writer:
        with pytest.raises(ValueError, match="scalar"):
            writer.write(step=0, update_groups=1, unique_online_count=1,
                         update_info={"image": DeviceImage()}, force=True)


def test_close_keeps_latest_unsampled_counters_and_is_idempotent(tmp_path):
    writer = LearnerMetrics(tmp_path, "short", cta_ratio=2)
    assert writer.write(step=2, update_groups=3, unique_online_count=17) is None
    writer.close()
    writer.close()
    assert records(writer.path)[-1]["critic_updates"] == 6
    assert records(writer.path)[-1]["unique_online_count"] == 17
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(step=2, update_groups=3, unique_online_count=17)


def test_existing_attempt_is_not_overwritten_and_resume_has_own_window(tmp_path):
    with LearnerMetrics(tmp_path, "one", cta_ratio=2) as writer:
        writer.write(step=49, update_groups=50, unique_online_count=20)
    original = writer.path.read_bytes()
    with pytest.raises(FileExistsError):
        LearnerMetrics(tmp_path, "one", cta_ratio=2)
    assert writer.path.read_bytes() == original
    with LearnerMetrics(tmp_path, "two", cta_ratio=2, start_step=50) as resumed:
        sample = resumed.write(step=50, update_groups=1, unique_online_count=2, force=True)
        assert sample["critic_updates"] == 2
        assert sample["critic_utd"] == 1


@pytest.mark.parametrize("values", [
    dict(step=0, update_groups=2, unique_online_count=2),
    dict(step=0, update_groups=1, unique_online_count=-1),
    dict(step=0, update_groups=True, unique_online_count=1),
])
def test_invalid_counter_contract_fails_explicitly(tmp_path, values):
    with LearnerMetrics(tmp_path, "counter-check", cta_ratio=2) as writer:
        with pytest.raises(ValueError):
            writer.write(**values)


def test_counter_regression_is_not_mistaken_for_a_new_resume_window(tmp_path):
    with LearnerMetrics(tmp_path, "no-reset", cta_ratio=2) as writer:
        writer.write(step=9, update_groups=10, unique_online_count=12)
        with pytest.raises(ValueError, match="cannot decrease"):
            writer.write(step=10, update_groups=11, unique_online_count=11)


@pytest.mark.parametrize("attempt", ["", "../outside", "/absolute", "..", "a/b", "x" * 129])
def test_unsafe_attempt_identifier_rejected_before_file_creation(tmp_path, attempt):
    with pytest.raises(ValueError, match="attempt_id"):
        LearnerMetrics(tmp_path, attempt, cta_ratio=2)
    assert not (tmp_path / "metrics").exists()


def test_records_and_close_are_fsynced_and_failures_propagate(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr(training_metrics.os, "fsync", lambda fd: syncs.append(fd))
    writer = LearnerMetrics(tmp_path, "durable", cta_ratio=2)
    assert len(syncs) == 3  # Start record, file directory entry, metrics directory entry.
    writer.write(step=0, update_groups=1, unique_online_count=1, force=True)
    assert len(syncs) == 4

    def no_space(fd):
        raise OSError("No space left on device")

    monkeypatch.setattr(training_metrics.os, "fsync", no_space)
    with pytest.raises(OSError, match="No space"):
        writer.write(step=1, update_groups=2, unique_online_count=2, force=True)
    with pytest.raises(RuntimeError, match="previous I/O failure"):
        writer.write(step=2, update_groups=3, unique_online_count=3, force=True)
    writer.close()


def test_bounded_tail_excludes_partial_record_and_reports_truncation(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(b"".join((json.dumps({"step": step}) + "\n").encode() for step in range(20))
                     + b'{"step":20')
    result = read_metrics_tail(path, limit=3, max_bytes=90)
    assert result["records"] == [{"step": 17}, {"step": 18}, {"step": 19}]
    assert result["bytes_read"] == 90
    assert result["truncated_head"] is True
    assert result["incomplete_last_line"] is True
    assert read_metrics_tail(path, max_bytes=4)["incomplete_last_line"] is True


@pytest.mark.parametrize("payload", [b'{"loss":NaN}\n', b'{"loss":1e999}\n', b'{broken}\n', b'[]\n'])
def test_tail_rejects_corrupt_complete_records(tmp_path, payload):
    path = tmp_path / "corrupt.jsonl"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_metrics_tail(path)


def test_mapping_size_depth_and_collisions_are_bounded(tmp_path):
    with LearnerMetrics(tmp_path, "bounded", cta_ratio=2) as writer:
        def check(info, extra=None):
            return writer.write(step=0, update_groups=1, unique_online_count=0,
                                update_info=info, extra=extra, force=True)

        with pytest.raises(ValueError, match="Too many"):
            check({f"key-{i}": i for i in range(257)})
        with pytest.raises(ValueError, match="colliding"):
            check({"a/b": 1, "a": {"b": 2}})
        with pytest.raises(ValueError, match="collide"):
            check({"runtime/count": 1}, extra={"count": 2})
        nested = 1
        for _ in range(9):
            nested = {"deep": nested}
        with pytest.raises(ValueError, match="depth"):
            check(nested)


def test_context_manager_persists_end_counters_on_training_exception(tmp_path):
    with pytest.raises(RuntimeError, match="training failed"):
        with LearnerMetrics(tmp_path, "fault", cta_ratio=2) as writer:
            writer.write(step=1, update_groups=2, unique_online_count=10)
            raise RuntimeError("training failed")
    end = records(writer.path)[-1]
    assert end["event"] == "attempt_end"
    assert end["update_groups"] == 2
    assert end["metrics_scope"] is None
