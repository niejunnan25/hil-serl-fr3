"""Exact frame provenance, H.264 decoding and non-image preservation checks."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from hilserl import recrop_dataset as recrop
from hilserl import seed_dataset as seed
from hilserl.image_profile import INSERT_ROI, get_image_profile, preprocess_image
from hilserl.storage import read_step, write_step
from test_seed_dataset import make_episode


CAPTURE_IDS = [100, 107, 119, 142]


def reference(camera, index):
    return dict(camera=camera, capture_id=CAPTURE_IDS[index], monotonic_ns=1000 + index * 10,
                unix_ns=10000 + index * 10)


def write_index(path, camera, mutate=None):
    rows = [dict(reference(camera, i), frame=i, fps=30, segment=i // 2) for i in range(4)]
    if mutate:
        mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def encode_video(path, frames):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for H.264 integration checks")
    size = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", size, "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-threads", "2", "-bf", "0", str(path)],
                   input=b"".join(frame.tobytes() for frame in frames), check=True, capture_output=True)


@pytest.fixture
def recorded_seed(tmp_path):
    run, raw_dir = make_episode(tmp_path)
    capture = raw_dir.parent.parent
    for i in range(3):
        path = raw_dir / f"{i:06d}.npz"
        raw = read_step(path)
        raw["frame_references"] = {camera: reference(camera, i) for camera in recrop.POLICY_CAMERAS}
        raw["next_frame_references"] = {camera: reference(camera, i + 1) for camera in recrop.POLICY_CAMERAS}
        write_step(path, raw)
    source = tmp_path / "source_seed"
    manifest = seed.build_seed_dataset([run], source)
    for camera in recrop.POLICY_CAMERAS:
        video_dir = capture / "video" / camera
        video_dir.mkdir(parents=True)
        write_index(video_dir / "frames.jsonl", camera)
        frames = []
        for i in range(4):
            frame = np.empty((720, 1280, 3), np.uint8)
            frame[..., 0] = np.arange(1280, dtype=np.uint16)[None, :] % 256
            frame[..., 1] = np.arange(720, dtype=np.uint16)[:, None] % 256
            frame[..., 2] = 40 + i * 40
            frames.append(frame)
        for segment in range(2):
            encode_video(video_dir / f"{segment:06d}.mp4", frames[segment * 2:segment * 2 + 2])
    return source, capture, manifest


def build(recorded_seed, tmp_path):
    source, _, _ = recorded_seed
    output = tmp_path / "roi224_seed"
    manifest = recrop.build_recropped_dataset(source, output, image_profile=get_image_profile(INSERT_ROI),
                                               expected_source_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest())
    return output, manifest


def test_index_uses_segment_row_ordinal_with_sparse_capture_ids(tmp_path):
    path = tmp_path / "frames.jsonl"
    write_index(path, "wrist_1")
    index = recrop.load_frame_index(path, "wrist_1", {107, 119})
    assert index["segment_counts"] == {0: 2, 1: 2}
    assert set(index["frames"]) == {107, 119}
    assert index["frames"][107]["segment_frame"] == 1
    assert index["frames"][119]["segment_frame"] == 0
    assert index["frames"][119]["frame"] == 2
    assert Path(index["frames"][119]["video_path"]).name == "000001.mp4"


@pytest.mark.parametrize("mutate, reason", [
    (lambda rows: rows[1].update(capture_id=100), "duplicate capture_id"),
    (lambda rows: rows[1].update(frame=4), "noncontiguous indexed frame"),
    (lambda rows: rows[1].update(camera="side_policy"), "camera mismatch"),
    (lambda rows: rows[1].update(monotonic_ns=1), "nonmonotonic"),
    (lambda rows: rows[2].update(segment=3), "noncontiguous indexed segment"),
    (lambda rows: rows[2].update(fps=20), "fps changed"),
])
def test_index_rejects_ambiguous_or_inconsistent_rows(tmp_path, mutate, reason):
    path = tmp_path / "frames.jsonl"
    write_index(path, "wrist_1", mutate)
    with pytest.raises(recrop.RecropDatasetError, match=reason):
        recrop.load_frame_index(path, "wrist_1", {119})


def test_missing_capture_has_no_nearest_timestamp_fallback(tmp_path):
    path = tmp_path / "frames.jsonl"
    write_index(path, "wrist_1")
    with pytest.raises(recrop.RecropDatasetError, match="missing capture_id"):
        recrop.load_frame_index(path, "wrist_1", {118})


@pytest.mark.parametrize("size", [128,160,192])
def test_reconstruction_uses_versioned_resolution_without_changing_labels(recorded_seed,tmp_path,size):
    source,_,_=recorded_seed
    profile=get_image_profile(f"insert-roi{size}-v1")
    output=tmp_path/f"roi{size}"
    result=recrop.build_recropped_dataset(source,output,image_profile=profile,
        expected_source_manifest_sha256=hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest())
    assert result["image_profile"]==profile
    transitions=list(seed.iter_seed_transitions(output,expected_image_profile=profile["name"]))
    assert len(transitions)==3 and [t["rewards"] for t in transitions]==[0,0,1]
    assert transitions[0]["observations"]["wrist_1"].shape==(1,size,size,3)
    assert transitions[0]["observations"]["side_classifier"].shape==(1,128,128,3)


def test_rebuild_decodes_original_frames_and_preserves_all_other_bytes(recorded_seed, tmp_path, monkeypatch):
    source, capture, source_manifest = recorded_seed
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for base in (source, capture) for p in base.rglob("*") if p.is_file()}
    opened = []
    real_decode = recrop.iter_video_frames
    def trace(path, targets, **kwargs):
        opened.append(str(path))
        yield from real_decode(path, targets, **kwargs)
    monkeypatch.setattr(recrop, "iter_video_frames", trace)
    output, manifest = build(recorded_seed, tmp_path)
    assert len(opened) == len(set(opened)) == 4
    assert manifest["counts"] == source_manifest["counts"]
    assert manifest["image_rebuild"]["source_encoding"] == "h264_lossy_video"
    assert manifest["image_rebuild"]["frame_mapping"]["rows"] == 12
    assert manifest["image_rebuild"]["unique_images"] == 8
    assert not (output / ".policy_frames.npy").exists()
    with np.load(source / "episode_0000.npz", allow_pickle=False) as old, np.load(output / "episode_0000.npz", allow_pickle=False) as new:
        for key in old.files:
            if key not in recrop.REPLACED_ARRAYS:
                assert old[key].dtype == new[key].dtype and old[key].shape == new[key].shape
                assert old[key].tobytes() == new[key].tobytes()
        assert new["obs__side_classifier"].shape == (3, 1, 128, 128, 3)
        for camera in recrop.POLICY_CAMERAS:
            pixels = dict(real_decode(capture / "video" / camera / "000001.mp4", {0, 1}, expected_count=2))
            assert np.array_equal(new[f"obs__{camera}"][2, 0], preprocess_image(pixels[0], camera, INSERT_ROI))
            assert np.array_equal(new[f"next_obs__{camera}"][2, 0], preprocess_image(pixels[1], camera, INSERT_ROI))
            assert np.array_equal(new[f"next_obs__{camera}"][:-1], new[f"obs__{camera}"][1:])
            assert new[f"obs__{camera}"].shape == (3, 1, 224, 224, 3)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in protected.items())
    recrop.validate_recropped_dataset(output, verify_pixels=True)
    transitions = list(seed.iter_seed_transitions(output, expected_image_profile=INSERT_ROI))
    assert len(transitions) == 3 and sum(bool(t["dones"]) for t in transitions) == 1


def test_raw_timestamp_must_equal_index_even_when_capture_id_matches(recorded_seed, tmp_path):
    source, capture, _ = recorded_seed
    path = capture / "video/wrist_1/frames.jsonl"
    write_index(path, "wrist_1", lambda rows: rows[1].update(unix_ns=99999))
    with pytest.raises(recrop.RecropDatasetError, match="timestamp/reference mismatch"):
        build(recorded_seed, tmp_path)
    assert not (tmp_path / "roi224_seed").exists()


def test_decode_count_mismatch_aborts_candidate(recorded_seed, tmp_path):
    _, capture, _ = recorded_seed
    path = capture / "video/wrist_1/frames.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(dict(camera="wrist_1", capture_id=150, monotonic_ns=1040, unix_ns=10040,
                                     frame=4, fps=30, segment=1)) + "\n")
    with pytest.raises(recrop.RecropDatasetError, match="decoded frame count"):
        build(recorded_seed, tmp_path)
    output = tmp_path / "roi224_seed"
    assert (output / "build-failure.json").exists()
    assert not (output / "manifest.json").exists()


def test_original_seed_manifest_is_frozen(recorded_seed, tmp_path):
    source, _, _ = recorded_seed
    with pytest.raises(seed.SeedDatasetError, match="frozen run"):
        recrop.build_recropped_dataset(source, tmp_path / "candidate", image_profile=get_image_profile(INSERT_ROI),
                                        expected_source_manifest_sha256="0" * 64)


def test_rebuild_never_overwrites_existing_directory(recorded_seed, tmp_path):
    (tmp_path / "roi224_seed").mkdir()
    keep = tmp_path / "roi224_seed/existing.txt"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        build(recorded_seed, tmp_path)
    assert keep.read_text() == "keep"


def test_non_image_mutation_rejected_even_if_npz_and_manifest_rehashed(recorded_seed, tmp_path):
    output, manifest = build(recorded_seed, tmp_path)
    path = output / "episode_0000.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["actions"][0, 0] = .125
    np.savez_compressed(path, **arrays)
    manifest["episodes"][0].update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(recrop.RecropDatasetError, match="preserved seed array changed: actions"):
        recrop.validate_recropped_dataset(output)


def test_128_video_is_rejected_instead_of_upscaled(tmp_path):
    path = tmp_path / "small.mp4"
    encode_video(path, [np.zeros((128, 128, 3), np.uint8)])
    with pytest.raises(recrop.RecropDatasetError, match="source resolution"):
        recrop.probe_video(path)
