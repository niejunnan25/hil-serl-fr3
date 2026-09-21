"""Versioned exports retain original raw actions and true terminal labels."""
import json
import pickle
import zipfile

import numpy as np
import pytest

from hilserl.action_contract import FIXED_XYZ, LEGACY
from hilserl.archive import Archive
from hilserl.storage import SessionRecorder, read_step, write_step


def recording(tmp_path, *, contract=FIXED_XYZ, outcome=1):
    root = tmp_path / "runs"
    path = root / "run/recordings/capture"
    metadata = {"action_contract": contract} if contract is not None else {}
    recorder = SessionRecorder(path, metadata=metadata, min_free_bytes=0)
    recorder.start_episode(**metadata)
    recorder.episode.update(verdict_source="human", human_steps=2)
    for i, source in enumerate(("human", "human", "policy")):
        raw = dict(id=f"run/capture/000001/{i:06d}", episode_id="000001", episode_step=i, global_step=i,
                   complete_transition=True, observations={"state": np.full((1,19), i, np.float32)},
                   next_observations={"state": np.full((1,19), i+1, np.float32)},
                   actions=np.array([.1*i, -.2, .3, 0, 0, 0, 0], np.float32), source_action=source,
                   observed_reward=.5, info={}, termination_reason="manual" if i==2 else None, **metadata)
        if contract == FIXED_XYZ:
            raw["learning_action"] = raw["actions"][:3].copy()
        recorder.append_step(raw)
    recorder.end_collection("manual")
    recorder.finish_episode(outcome)
    recorder.close()
    return Archive(root, min_free_bytes=0), path


def exported(archive, **kwargs):
    result = archive.export("run/capture", **kwargs)
    with zipfile.ZipFile(result["path"]) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        transitions = pickle.loads(bundle.read("episodes/000001/transitions.pkl")) if kwargs.get("format") == "serl" else None
    return manifest, transitions


@pytest.mark.parametrize("outcome", [0, 1])
def test_fixed_serl_export_uses_three_dimensions_and_actual_human_terminal(tmp_path, outcome):
    archive, _ = recording(tmp_path, outcome=outcome)
    manifest, transitions = exported(archive, format="serl")
    assert manifest["action_contract"] == FIXED_XYZ and manifest["action_contracts"] == [FIXED_XYZ]
    assert manifest["episodes"][0]["exported_action_dim"] == 3
    assert all(t["actions"].shape == (3,) and t["infos"]["action_contract"] == FIXED_XYZ for t in transitions)
    assert [t["rewards"] for t in transitions] == [0, 0, outcome]
    assert [t["dones"] for t in transitions] == [False, False, True]
    assert [t["masks"] for t in transitions] == [1, 1, 0]
    assert [t["infos"]["verdict_source"] for t in transitions] == [None, None, "human"]


def test_human_filter_does_not_move_policy_terminal_to_last_selected_human_step(tmp_path):
    archive, _ = recording(tmp_path)
    manifest, transitions = exported(archive, format="serl", source="human")
    assert manifest["episodes"][0]["selected_step_indices"] == [0, 1]
    assert manifest["episodes"][0]["outcome"] == 1
    assert not manifest["episodes"][0]["is_complete_episode_selection"]
    assert [t["rewards"] for t in transitions] == [0, 0]
    assert [t["dones"] for t in transitions] == [False, False]
    assert [t["masks"] for t in transitions] == [1, 1]
    np.testing.assert_array_equal(transitions[-1]["next_observations"]["state"], np.full((1,19), 2))


def test_fixed_raw_npz_export_is_byte_exact_seven_dimensions(tmp_path):
    archive, path = recording(tmp_path)
    result = archive.export("run/capture", format="npz")
    with zipfile.ZipFile(result["path"]) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["action_representation"] == "raw_device_7d"
        assert manifest["episodes"][0]["exported_action_dim"] == 7
        for i in range(3):
            name = f"episodes/000001/{i:06d}.npz"
            assert bundle.read(name) == (path / name).read_bytes()
            assert read_step(path / name)["actions"].shape == (7,)


@pytest.mark.parametrize("contract", [None, LEGACY])
def test_old_unversioned_and_explicit_legacy_exports_remain_seven_dimensions(tmp_path, contract):
    archive, _ = recording(tmp_path, contract=contract)
    manifest, transitions = exported(archive, format="serl")
    assert manifest["action_contract"] == LEGACY
    assert all(t["actions"].shape == (7,) for t in transitions)
    assert [t["rewards"] for t in transitions] == [.5, .5, 1]


@pytest.mark.parametrize("format", ["npz", "serl"])
def test_mixed_raw_contract_is_rejected_even_if_filtered_step_would_be_excluded(tmp_path, format):
    archive, path = recording(tmp_path)
    raw_path = path / "episodes/000001/000002.npz"
    raw = read_step(raw_path)
    raw["action_contract"] = LEGACY
    write_step(raw_path, raw)
    with pytest.raises(ValueError, match="contracts differ"):
        archive.export("run/capture", format=format, source="human")
    assert not list((archive.root / "exports").iterdir())


def test_fixed_recording_rejects_fabricated_human_provenance(tmp_path):
    archive, path = recording(tmp_path)
    p = path / "episodes/000001/episode.json"
    metadata = json.loads(p.read_text())
    metadata["verdict_source"] = "classifier"
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="no final human verdict"):
        archive.export("run/capture", format="serl")


def test_explicit_raw_version_is_used_when_old_container_has_no_version(tmp_path):
    archive, path = recording(tmp_path)
    for p, parent in ((path / "manifest.json", "metadata"), (path / "episodes/000001/episode.json", None)):
        value = json.loads(p.read_text())
        (value[parent] if parent else value).pop("action_contract")
        p.write_text(json.dumps(value))
    manifest, transitions = exported(archive, format="serl", source="policy")
    assert manifest["action_contract"] == FIXED_XYZ
    assert len(transitions) == 1 and transitions[0]["dones"]
