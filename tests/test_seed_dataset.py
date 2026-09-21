"""Real NPZ extraction/admission checks, without robot, JAX or network access."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hilserl import seed_dataset as seed
from hilserl.storage import write_step, read_step


def make_episode(tmp_path, episode="000001", *, outcome=1, steps=3, human_steps=None):
    run = tmp_path / "run_A"
    capture = run / "recordings" / "capture_A"
    directory = capture / "episodes" / episode
    directory.mkdir(parents=True)
    config = run / "attempts" / "config.json"
    config.parent.mkdir(exist_ok=True)
    config.write_text(json.dumps({"action_max_step": 0.008}))
    (capture / "manifest.json").write_text(json.dumps({"metadata": {"config_file": str(config)}}))
    meta = {"id": episode, "complete": True, "status": "complete", "steps": steps, "committed_steps": steps,
            "human_steps": steps if human_steps is None else human_steps, "outcome": outcome,
            "verdict_source": "human", "reset_info": {"reset": {"success": True}}}
    (directory / "episode.json").write_text(json.dumps(meta))
    def raw_state(i):
        return {"values": {"tcp_pose": np.array([0.6, 0.1, 0.15-i*.001, 1., 0., 0., 0.]),
                           "tcp_vel": np.array([0., 0., -.01, 0., 0., 0.]),
                           "tcp_force": np.array([1., 2., 3.]), "tcp_torque": np.array([4., 5., 6.]),
                           "gripper_pose": np.array(.56)}}
    def obs(i):
        state = np.array([[.56, 1, 2, 3, 0, 0, i*.001, 0, 0, 0, 4, 5, 6, 0, 0, .01, 0, 0, 0]], np.float32)
        return {"state": state, **{k: np.full((1, 128, 128, 3), i, np.uint8) for k in seed.CAMERAS}}
    for i in range(steps):
        raw = {"id": f"run_A/capture_A/{episode}/{i:06d}", "episode_id": episode, "episode_step": i,
               "global_step": i, "complete_transition": True, "source_action": "human", "observations": obs(i),
               "next_observations": obs(i+1), "actions": np.array([0, 0, .25, 0, 0, 0, 0], np.float32),
               "controller_input_action": np.array([0, 0, -.25, 0, 0, 0, 0], np.float32),
               "controller_commands": [{"kind": "pose", "request_sent": True, "returned": True}],
               "raw_state": raw_state(i), "raw_next_state": raw_state(i+1), "observed_reward": float(i==steps-1),
               "terminated": i==steps-1, "truncated": False}
        write_step(directory / f"{i:06d}.npz", raw)
    return run, directory


def rewrite_raw(path, change):
    raw = read_step(path)
    change(raw)
    write_step(path, raw)


def build(tmp_path):
    run, source = make_episode(tmp_path)
    output = tmp_path / "derived"
    manifest = seed.build_seed_dataset([run], output)
    return source, output, manifest


def update_npz(output, mutate):
    p = output / "episode_0000.npz"
    with np.load(p, allow_pickle=False) as original:
        arrays = {k: original[k].copy() for k in original.files}
    mutate(arrays)
    np.savez_compressed(p, **arrays)
    m = json.loads((output / "manifest.json").read_text())
    m["episodes"][0].update(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(m))


def test_real_extraction_preserves_arrays_and_true_terminal_without_duplicates(tmp_path):
    source, output, manifest = build(tmp_path)
    transitions = list(seed.iter_seed_transitions(output))
    assert manifest["counts"] == {"episodes": 1, "transitions": 3, "human_transitions": 3, "positive_rewards": 1, "terminal_transitions": 1}
    for i, transition in enumerate(transitions):
        original = read_step(source / f"{i:06d}.npz")
        assert np.array_equal(transition["actions"], original["actions"][:3])
        for field in ("observations", "next_observations"):
            for key in seed.OBSERVATION_SCHEMA:
                assert transition[field][key].tobytes() == original[field][key].tobytes()
        assert transition["infos"]["raw_transition_id"] == original["id"]
        assert transition["infos"]["action_contract"] == seed.ACTION_CONTRACT
        assert transition["infos"]["verdict_source"] == ("human" if i==2 else None)
    assert [t["rewards"] for t in transitions] == [0, 0, 1]
    assert [t["dones"] for t in transitions] == [False, False, True]
    assert [t["masks"] for t in transitions] == [1, 1, 0]
    assert max(manifest["episodes"][0]["frame_max_errors"].values()) < 1e-5


def test_frozen_manifest_hash_rejects_changed_dataset_for_validate_and_loader(tmp_path):
    _, output, _ = build(tmp_path)
    expected = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    assert seed.validate_seed_dataset(output, expected_manifest_sha256=expected)["counts"]["transitions"] == 3
    with (output / "manifest.json").open("a") as stream:
        stream.write("\n")
    for call in (lambda: seed.validate_seed_dataset(output, expected_manifest_sha256=expected),
                 lambda: next(seed.iter_seed_transitions(output, expected_manifest_sha256=expected))):
        with pytest.raises(seed.SeedDatasetError, match="frozen run"):
            call()


def test_legacy_seed_directory_and_other_contract_cannot_load(tmp_path):
    (tmp_path / "demo_success.pkl").write_bytes(b"legacy, even when named success")
    with pytest.raises(seed.SeedDatasetError, match="legacy pickle"):
        seed.validate_seed_dataset(tmp_path)
    _, output, _ = build(tmp_path)
    with pytest.raises(seed.SeedDatasetError, match="action contract"):
        seed.validate_seed_dataset(output, expected_action_contract="legacy-hybrid-v0")


def test_mixed_and_failed_episodes_are_not_used_as_expert_seed(tmp_path):
    run, _ = make_episode(tmp_path, "000001")
    make_episode(tmp_path, "000002", human_steps=2)
    make_episode(tmp_path, "000003", outcome=0)
    manifest = seed.build_seed_dataset([run], tmp_path / "derived")
    assert manifest["counts"]["episodes"] == 1
    assert manifest["omitted_counts"] == {"mixed_or_policy_episode_not_expert_demo": 1,
                                          "human_labeled_failure_retained_only_in_source": 1}


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r["actions"].__setitem__(6, 1), "not locked"),
    (lambda r: r["controller_input_action"].__setitem__(3, .1), "not locked"),
    (lambda r: r["actions"].__setitem__(2, -.25), "disagree"),
    (lambda r: r["observations"]["state"].__setitem__((0,4), .6), "reset-relative origin"),
    (lambda r: r["next_observations"]["state"].__setitem__((0,6), .4), "frame mismatch"),
    (lambda r: r["next_observations"]["state"].__setitem__((0,0), np.nan), "nonfinite"),
    (lambda r: r.__setitem__("source_action", "policy"), "raw step does not"),
    (lambda r: r["controller_commands"][0].__setitem__("returned", False), "incomplete controller"),
    (lambda r: r.__setitem__("complete_transition", False), "incomplete raw"),
    (lambda r: r.__setitem__("terminated", True), "before final"),
])
def test_raw_contract_errors_omit_whole_episode_never_repair_labels(tmp_path, mutation, reason):
    run, directory = make_episode(tmp_path)
    rewrite_raw(directory / "000000.npz", mutation)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    with pytest.raises(seed.SeedDatasetError, match="no episodes satisfy"):
        seed.build_seed_dataset([run], tmp_path / "derived")
    failure = json.loads((tmp_path / "derived" / "build-failure.json").read_text())
    assert reason in failure["omitted"][0]["reason"]
    assert not (tmp_path / "derived" / "manifest.json").exists()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_nonterminal_classifier_score_is_not_human_reward(tmp_path):
    run, directory = make_episode(tmp_path)
    rewrite_raw(directory / "000000.npz", lambda r: r.__setitem__("observed_reward", 1.))
    manifest = seed.build_seed_dataset([run], tmp_path / "derived")
    assert manifest["episodes"][0]["nonterminal_classifier_positive_ignored"] == 1
    assert [x["rewards"] for x in seed.iter_seed_transitions(tmp_path / "derived")] == [0, 0, 1]


@pytest.mark.parametrize("mutate,reason", [
    (lambda a: a["rewards"].__setitem__(0, 1), "final human label"),
    (lambda a: a["masks"].__setitem__(-1, 1), "must not bootstrap"),
    (lambda a: a["dones"].__setitem__(0, True), "only the last"),
    (lambda a: a["actions"].__setitem__((0, 0), np.nan), "nonfinite"),
    (lambda a: a.__setitem__("actions", np.zeros((3, 7), np.float32)), "expected shape"),
    (lambda a: a["raw_transition_ids"].__setitem__(1, a["raw_transition_ids"][0]), "duplicate transition"),
    (lambda a: a["next_obs__wrist_1"].__setitem__((0, 0, 0, 0, 0), 55), "continuity"),
])
def test_loader_checks_schema_even_with_recomputed_file_checksum(tmp_path, mutate, reason):
    _, output, _ = build(tmp_path)
    update_npz(output, mutate)
    with pytest.raises(seed.SeedDatasetError, match=reason):
        next(seed.iter_seed_transitions(output))


def test_corruption_rejects_whole_dataset_before_first_transition(tmp_path):
    run, _ = make_episode(tmp_path, "000001")
    make_episode(tmp_path, "000002")
    output = tmp_path / "derived"
    seed.build_seed_dataset([run], output)
    p = output / "episode_0001.npz"
    p.write_bytes(p.read_bytes()[:-9])
    with pytest.raises(seed.SeedDatasetError, match="checksum mismatch"):
        next(seed.iter_seed_transitions(output))


def test_builder_never_overwrites_existing_data(tmp_path):
    run, _ = make_episode(tmp_path)
    output = tmp_path / "derived"
    output.mkdir()
    keep = output / "existing.txt"
    keep.write_text("preserve")
    with pytest.raises(FileExistsError):
        seed.build_seed_dataset([run], output)
    assert keep.read_text() == "preserve"
