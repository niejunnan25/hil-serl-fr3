"""CPU-only scoring checks with synthetic checkpoints and evaluation evidence."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import stat

import pytest

from hilserl import checkpoint_scores as scores


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    ticks = itertools.count(1_800_000_000_000_000_000)
    monkeypatch.setattr(scores.time, "time_ns", lambda: next(ticks))
    config = dict(action_contract="fixed-xyz-v1", image_profile="insert-roi160-v1",
                  classifier_threshold=0.78, safety_force_max=45.0, random_reset=True,
                  random_xy_range=0.006, max_episode_steps=190, control_hz=10.0,
                  root=str(root), python="python", data_dir="runs", demo_dir="demos",
                  batch_size=128, learner_steps=100000, min_free_gib=8.0)
    return root, checkpoints, config


def model(directory, step, *, committed=True, legacy=False):
    path = directory / f"checkpoint_{step}"
    if legacy:
        path.write_bytes(b"synthetic legacy msgpack")
    else:
        path.mkdir()
        (path / "weights").write_bytes(b"original model data")
        if committed:
            (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": step + 1}))
    return path


def episodes(successes=7, count=10):
    return [dict(episode_id=f"{index:06d}", mode="eval", outcome=int(index < successes),
                 steps=15, human_steps=0, collection_seconds=1.5, termination_reason="operator_label")
            for index in range(count)]


def evaluate(setup, step, *, successes=7, count=10, name=None, config=None, seed=0):
    root, directory, original_config = setup
    if not (directory / f"checkpoint_{step}").exists():
        model(directory, step)
    run = root / (name or f"eval_{step}")
    run.mkdir(exist_ok=True)
    return scores.record_evaluation(directory, step, evaluation_run=run,
                                    summaries=episodes(successes, count), expected_episodes=count,
                                    seed=seed, config_snapshot=config or original_config)


def read_ledger(directory):
    return json.loads((directory / "retention_scores.json").read_text())


def test_valid_evaluations_rank_by_success_then_newer_step_and_do_not_touch_models(setup):
    root, directory, config = setup
    for step in (1, 2, 3, 4):
        model(directory, step, legacy=step == 4)
    sentinel = directory / "buffer"
    sentinel.mkdir()
    (sentinel / "transitions.pkl").write_bytes(b"untouched replay")
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in directory.rglob("*") if p.is_file()}
    original = copy.deepcopy(config)
    evaluate(setup, 1, successes=9)
    evaluate(setup, 2, successes=9)
    result = evaluate(setup, 3, successes=6)
    evaluate(setup, 4, successes=0)
    assert scores.ranked_steps(directory, [1, 2, 3, 4]) == [2, 1, 3, 4]
    assert result["score"] == 0.6 and result["status"] == "ranked"
    assert read_ledger(directory)["schema_version"] == 1
    assert json.loads((root / "eval_3/evaluation.json").read_text()) == result
    assert original == config
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert not (directory / "checkpoint_scores.json").exists()


@pytest.mark.parametrize("change,expected,reason", [
    (lambda items: items.pop(), 10, "incomplete_evaluation"),
    (lambda items: None, 9, "at_least_10_episodes_required"),
    (lambda items: items[0].update(human_steps=1), 10, "intervention_present"),
    (lambda items: items[0].update(mode="train"), 10, "non_eval_episode"),
    (lambda items: items[0].update(outcome=True), 10, "invalid_outcome"),
    (lambda items: items[0].update(outcome=2), 10, "invalid_outcome"),
    (lambda items: items[0].update(outcome=float("nan")), 10, "invalid_outcome"),
    (lambda items: items[0].update(steps=0), 10, "invalid_steps"),
    (lambda items: items[0].update(steps=True), 10, "invalid_steps"),
    (lambda items: items[0].update(human_steps=False), 10, "invalid_human_steps"),
    (lambda items: items[0].pop("human_steps"), 10, "invalid_human_steps"),
    (lambda items: items[0].update(complete=False), 10, "incomplete_episode"),
    (lambda items: items[0].update(episode_id=""), 10, "invalid_episode_id"),
    (lambda items: items[0].update(episode_id=items[1]["episode_id"]), 10, "duplicate_episode_id"),
])
def test_unranked_evidence_is_written_without_inventing_a_zero_score(setup, change, expected, reason):
    root, directory, config = setup
    model(directory, 1)
    run = root / "eval"
    run.mkdir()
    summaries = episodes(count=expected)
    change(summaries)
    result = scores.record_evaluation(directory, 1, evaluation_run=run, summaries=summaries,
                                     expected_episodes=expected, seed=0, config_snapshot=config)
    assert result["status"] == "unranked" and result["reason"] == reason
    assert "score" not in result and "successes" not in result
    assert not (directory / "retention_scores.json").exists()
    assert json.loads((run / "evaluation.json").read_text()) == result
    assert scores.ranked_steps(directory, [1]) == []


def test_idempotent_retry_does_not_replace_newer_evaluation_for_same_step(setup):
    _, directory, _ = setup
    first = evaluate(setup, 1, successes=9, name="first")
    first_bytes = (directory / "retention_scores.json").read_bytes()
    assert evaluate(setup, 1, successes=9, name="first") == first
    assert (directory / "retention_scores.json").read_bytes() == first_bytes
    second = evaluate(setup, 1, successes=4, name="second")
    assert second["unix_ns"] > first["unix_ns"]
    assert read_ledger(directory)["entries"]["1"] == second
    assert evaluate(setup, 1, successes=9, name="first") == first
    assert read_ledger(directory)["entries"]["1"] == second
    with pytest.raises(ValueError, match="different result"):
        evaluate(setup, 1, successes=8, name="first")


def test_unranked_later_evaluation_does_not_erase_existing_score(setup):
    _, directory, _ = setup
    first = evaluate(setup, 1, successes=8, name="first")
    result = evaluate(setup, 1, count=3, name="incomplete")
    assert result["status"] == "unranked"
    assert read_ledger(directory)["entries"]["1"] == first


def test_condition_hash_ignores_training_storage_but_keeps_unknown_environment_fields(setup):
    _, _, config = setup
    first = evaluate(setup, 1)
    changed = dict(config, root="elsewhere", python="other-python", data_dir="other-runs",
                   demo_dir="other-demos", seed_dataset_sha256="0" * 64, seed=876,
                   batch_size=256, cta_ratio=4, actor_steps=200000, learner_steps=200000,
                   replay_buffer_capacity=4, training_starts=1000, steps_per_update=100,
                   port=1234, broadcast_port=1235, console_port=1236,
                   checkpoint_keep_latest=5, checkpoint_retention_policy="score", min_free_gib=1)
    second = evaluate(setup, 2, config=changed)
    assert first["conditions_sha256"] == second["conditions_sha256"]
    third = evaluate(setup, 3, config=dict(changed, new_environment_parameter="new"))
    assert first["conditions_sha256"] != third["conditions_sha256"]


@pytest.mark.parametrize("change", [
    {"image_profile": "insert-roi192-v1"}, {"safety_force_max": 40},
    {"classifier_threshold": 0.8}, {"random_reset": False}, {"max_episode_steps": 180},
    {"random_xy_range": 0.01}, {"control_hz": 20}, {"video_fps": 15},
])
def test_environment_input_safety_and_success_parameters_separate_conditions(setup, change):
    _, directory, config = setup
    first = evaluate(setup, 1, successes=10)
    second = evaluate(setup, 2, successes=3, config=dict(config, **change))
    assert first["conditions_sha256"] != second["conditions_sha256"]
    assert scores.ranked_steps(directory, [1, 2]) == [2]
    assert set(read_ledger(directory)["entries"]) == {"1", "2"}


def test_latest_evaluation_conditions_win_not_largest_checkpoint_step_or_best_score(setup):
    _, directory, config = setup
    evaluate(setup, 100, successes=10)
    evaluate(setup, 2, successes=3, config=dict(config, random_reset=False))
    evaluate(setup, 3, successes=6, config=dict(config, random_reset=False))
    assert scores.ranked_steps(directory, [2, 3, 100]) == [3, 2]
    evaluate(setup, 4, successes=7, seed=1)
    assert scores.ranked_steps(directory, [2, 3, 4, 100]) == [4]
    evaluate(setup, 5, successes=10, count=20, seed=1)
    assert scores.ranked_steps(directory, [2, 3, 4, 5, 100]) == [5]
    # Reserving the latest checkpoint first must not switch to older conditions.
    assert scores.ranked_steps(directory, [2, 3, 4, 100]) == []


@pytest.mark.parametrize("available", [[1, True], [True, 1], [1, "2"], [-1], [1.0]])
def test_candidate_steps_are_strict_nonnegative_integers(setup, available):
    _, directory, _ = setup
    with pytest.raises(ValueError, match="available_steps"):
        scores.ranked_steps(directory, available)


@pytest.mark.parametrize("tamper", [
    lambda entry: entry.update(step=True),
    lambda entry: entry.update(step=2),
    lambda entry: entry.update(checkpoint="/outside/checkpoint_1"),
    lambda entry: entry.update(evaluation_run="../eval_1"),
    lambda entry: entry.update(unix_ns=True),
    lambda entry: entry.update(successes=-1),
    lambda entry: entry.update(successes=11),
    lambda entry: entry.update(episodes=0),
    lambda entry: entry.update(score=float("nan")),
    lambda entry: entry.update(score=float("inf")),
    lambda entry: entry.update(score=0.9),
    lambda entry: entry.update(conditions_sha256="0" * 64),
    lambda entry: entry["conditions"].update(eval_protocol="training-success"),
    lambda entry: entry["conditions"].update(seed=True),
    lambda entry: entry["summaries"][0].update(human_steps=1),
])
def test_invalid_entries_never_influence_ranking_even_if_receipt_matches(setup, tamper):
    root, directory, _ = setup
    evaluate(setup, 1, successes=7)
    ledger = read_ledger(directory)
    entry = ledger["entries"]["1"]
    tamper(entry)
    (directory / "retention_scores.json").write_text(json.dumps(ledger))
    (root / "eval_1/evaluation.json").write_text(json.dumps(entry))
    assert scores.ranked_steps(directory, [1]) == []


def test_noncanonical_step_key_and_missing_evidence_are_unscored(setup):
    root, directory, _ = setup
    evaluate(setup, 1)
    ledger = read_ledger(directory)
    ledger["entries"]["01"] = ledger["entries"].pop("1")
    (directory / "retention_scores.json").write_text(json.dumps(ledger))
    assert scores.ranked_steps(directory, [1]) == []
    ledger["entries"]["1"] = ledger["entries"].pop("01")
    (directory / "retention_scores.json").write_text(json.dumps(ledger))
    (root / "eval_1/evaluation.json").unlink()
    assert scores.ranked_steps(directory, [1]) == []


def test_removed_or_uncommitted_models_are_never_ranked(setup):
    _, directory, _ = setup
    evaluate(setup, 1)
    (directory / "checkpoint_1/_CHECKPOINT_METADATA").unlink()
    assert scores.ranked_steps(directory, [1]) == []
    with pytest.raises(ValueError, match="committed"):
        evaluate(setup, 1, name="later")


@pytest.mark.parametrize("unsafe", ["source", "receipt", "ledger", "evaluation_inside_model", "mpa"])
def test_symlinks_model_evidence_paths_and_mpa_are_rejected(setup, unsafe):
    root, directory, config = setup
    checkpoint = model(directory, 1, legacy=unsafe == "mpa")
    run = root / "eval"
    run.mkdir()
    outside = root / "outside"
    outside.write_bytes(b"outside unchanged")
    if unsafe == "source":
        (directory / "checkpoint_2").symlink_to(checkpoint, target_is_directory=True)
        step = 2
    else:
        step = 1
    if unsafe == "receipt":
        (run / "evaluation.json").symlink_to(outside)
    elif unsafe == "ledger":
        (directory / "retention_scores.json").symlink_to(outside)
    elif unsafe == "evaluation_inside_model":
        run = checkpoint
    elif unsafe == "mpa":
        (directory / "checkpoint_1_gda").mkdir()
    with pytest.raises(ValueError):
        scores.record_evaluation(directory, step, evaluation_run=run, summaries=episodes(),
                                 expected_episodes=10, seed=0, config_snapshot=config)
    assert outside.read_bytes() == b"outside unchanged"
    assert not (checkpoint / "evaluation.json").exists()


def test_corrupt_ledger_does_not_promote_scores_or_get_silently_overwritten(setup):
    _, directory, _ = setup
    path = directory / "retention_scores.json"
    path.write_text("incomplete JSON")
    assert scores.ranked_steps(directory, [1]) == []
    with pytest.raises(ValueError):
        evaluate(setup, 1)
    assert path.read_text() == "incomplete JSON"


def test_special_file_ledger_is_rejected_without_opening_it(setup):
    _, directory, _ = setup
    scores.os.mkfifo(directory / "retention_scores.json")
    assert scores.ranked_steps(directory, [1]) == []
    with pytest.raises(ValueError, match="regular file"):
        evaluate(setup, 1)


def test_atomic_write_failure_is_retryable_without_changing_old_score(setup, monkeypatch):
    root, directory, _ = setup
    evaluate(setup, 1, successes=9)
    before = (directory / "retention_scores.json").read_bytes()
    original = scores.os.replace
    def fail_ledger(source, destination):
        if Path(destination).name == "retention_scores.json":
            raise OSError("simulated ledger write failure")
        return original(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(scores.os, "replace", fail_ledger)
        with pytest.raises(OSError, match="ledger write"):
            evaluate(setup, 2, successes=10)
    assert (directory / "retention_scores.json").read_bytes() == before
    saved = json.loads((root / "eval_2/evaluation.json").read_text())
    assert evaluate(setup, 2, successes=10) == saved
    assert scores.ranked_steps(directory, [1, 2]) == [2, 1]
    assert not list(directory.glob("*.partial"))


def test_evidence_and_containing_directories_are_fsynced(setup, monkeypatch):
    modes = []
    original = scores.os.fsync
    def observe(fd):
        modes.append(scores.os.fstat(fd).st_mode)
        return original(fd)
    monkeypatch.setattr(scores.os, "fsync", observe)
    evaluate(setup, 1)
    assert sum(stat.S_ISREG(mode) for mode in modes) == 2
    assert sum(stat.S_ISDIR(mode) for mode in modes) == 2
