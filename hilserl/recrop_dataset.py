"""Rebuild policy images from indexed H.264 recordings; never mutate a seed.

The source is decoded 1280 x 720 video, not uncompressed camera pixels. Every
output image is bound to an exact raw transition, capture reference, index row,
video segment and decoded frame. The classifier and all other arrays are copied
byte for byte. No frame interpolation, approximate seek or missing-frame fallback
is allowed.
"""
from __future__ import annotations

from collections import defaultdict
import copy
from fractions import Fraction
from functools import lru_cache
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from hilserl import seed_dataset as seed

POLICY_CAMERAS = ("wrist_1", "side_policy")
REPLACED_ARRAYS = {f"{prefix}__{camera}" for prefix in ("obs", "next_obs") for camera in POLICY_CAMERAS}
REFERENCE_FIELDS = ("camera", "capture_id", "monotonic_ns", "unix_ns")
LOGGER = logging.getLogger(__name__)


class RecropDatasetError(seed.SeedDatasetError):
    """Source evidence or reconstructed data failed the image contract."""


def _require(condition, reason):
    if not condition:
        raise RecropDatasetError(reason)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _file_info(path, expected=None):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"missing or symlink source: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"source changed while hashing: {path}")
    value = {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest.hexdigest()}
    if expected is not None:
        _require(value["bytes"] == expected.get("bytes") and value["sha256"] == expected.get("sha256"),
                 f"source checksum mismatch: {path}")
    return value


def _read_bound_bytes(source):
    path = Path(source["path"])
    _require(path.is_file() and not path.is_symlink(), f"missing or symlink source: {path}")
    data = path.read_bytes()
    _require(len(data) == source["bytes"] and _sha(data) == source["sha256"], f"source checksum mismatch: {path}")
    return data


def load_frame_index(path, camera, capture_ids=None):
    """Validate a whole index and retain only requested capture IDs, if supplied.

    ``segment_frame`` is the row's ordinal within a contiguous indexed segment.
    It is never inferred from acquisition IDs, timestamps or nominal frame rate.
    """
    path = Path(path)
    source = _file_info(path)
    data = _read_bound_bytes(source)
    requested = None if capture_ids is None else set(capture_ids)
    selected, segment_counts, seen = {}, defaultdict(int), set()
    previous_capture = previous_monotonic = -1
    previous_segment, fps = 0, None
    lines = data.splitlines()
    _require(bool(lines), f"empty frame index: {path}")
    for ordinal, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RecropDatasetError(f"invalid index JSON at {path}:{ordinal + 1}") from exc
        _require(isinstance(row, dict), f"invalid index row: {path}:{ordinal + 1}")
        for key in ("capture_id", "frame", "segment", "monotonic_ns", "unix_ns"):
            _require(type(row.get(key)) is int and row[key] >= 0, f"invalid {key} at {path}:{ordinal + 1}")
        _require(row.get("camera") == camera, f"camera mismatch in index: {path}:{ordinal + 1}")
        _require(row["capture_id"] not in seen, f"duplicate capture_id in index: {path}:{ordinal + 1}")
        _require(row["capture_id"] > previous_capture and row["monotonic_ns"] > previous_monotonic,
                 f"nonmonotonic capture sequence: {path}:{ordinal + 1}")
        _require(row["frame"] == ordinal, f"noncontiguous indexed frame: {path}:{ordinal + 1}")
        _require(row["segment"] in (previous_segment, previous_segment + 1) and
                 (ordinal > 0 or row["segment"] == 0), f"noncontiguous indexed segment: {path}:{ordinal + 1}")
        _require(type(row.get("fps")) in (int, float) and np.isfinite(row["fps"]) and row["fps"] > 0,
                 f"invalid indexed fps: {path}:{ordinal + 1}")
        if fps is None:
            fps = row["fps"]
        _require(row["fps"] == fps, f"index fps changed: {path}:{ordinal + 1}")
        segment = row["segment"]
        if requested is None or row["capture_id"] in requested:
            selected[row["capture_id"]] = {
                **{key: row[key] for key in REFERENCE_FIELDS}, "index_path": source["path"],
                "index_line": ordinal + 1, "frame": row["frame"], "segment": segment,
                "segment_frame": segment_counts[segment],
                "video_path": str((path.parent / f"{segment:06d}.mp4").resolve()),
            }
        segment_counts[segment] += 1
        seen.add(row["capture_id"])
        previous_capture, previous_monotonic, previous_segment = row["capture_id"], row["monotonic_ns"], segment
    if requested is not None:
        _require(set(selected) == requested, f"missing capture_id in index {path}: {sorted(requested - set(selected))[:8]}")
    return {"source": source, "camera": camera, "fps": fps, "rows": len(lines),
            "segment_counts": dict(segment_counts), "frames": selected}


def probe_video(path, *, raw_size=(1280, 720), fps=None):
    """Require one unrotated H.264 stream at the recorded source resolution."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    _require(result.returncode == 0 and not result.stderr.strip(), f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        streams = json.loads(result.stdout)["streams"]
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    except (KeyError, ValueError, TypeError) as exc:
        raise RecropDatasetError(f"invalid ffprobe response: {path}") from exc
    _require(len(videos) == 1, f"expected exactly one video stream: {path}")
    video = videos[0]
    _require(video.get("codec_name") == "h264", f"expected H.264 source video: {path}")
    _require([video.get("width"), video.get("height")] == list(raw_size),
             f"source resolution differs from raw_size {list(raw_size)}: {path}")
    rotations = [video.get("tags", {}).get("rotate", 0)]
    rotations += [item.get("rotation", 0) for item in video.get("side_data_list", [])]
    _require(all(float(value) == 0 for value in rotations), f"rotated source video is unsupported: {path}")
    try:
        rate = Fraction(video["avg_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise RecropDatasetError(f"invalid video frame rate: {path}") from exc
    if fps is not None:
        _require(rate == Fraction(str(fps)), f"video frame rate differs from index: {path}")
    return {"codec": "h264", "pixel_format": video.get("pix_fmt"), "width": raw_size[0],
            "height": raw_size[1], "frame_rate": str(rate)}


@lru_cache(maxsize=1)
def _decoder_capabilities():
    help_result = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"], capture_output=True, text=True, check=True)
    if "-fps_mode" in help_result.stdout:
        sync_args = ["-fps_mode", "passthrough"]
    else:
        _require("-vsync" in help_result.stdout, "ffmpeg has no supported frame-passthrough option")
        sync_args = ["-vsync", "0"]
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    return {"frame_sync_args": sync_args, "version": version}


def iter_video_frames(path, target_indices, *, expected_count=None, raw_size=(1280, 720)):
    """Yield selected ``(zero_based_frame, BGR uint8)`` values in one decode pass.

    Rebuilds supply ``expected_count`` and decode the entire segment, rejecting
    decoder errors and frame-count discrepancies. Read-only visual sampling may
    omit it, in which case decoding stops after the final requested frame.
    """
    targets = set(target_indices)
    _require(bool(targets) and all(type(i) is int and i >= 0 for i in targets), "invalid target frame indices")
    _require(expected_count is None or (type(expected_count) is int and expected_count > max(targets)),
             "target frame lies outside indexed segment")
    width, height = raw_size
    frame_bytes = width * height * 3
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
               "-err_detect", "explode", "-threads", "2", "-noautorotate", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", *_decoder_capabilities()["frame_sync_args"], "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-threads", "2", "pipe:1"]
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_log)
        count, found, sampling_complete = 0, set(), False
        try:
            while True:
                data = process.stdout.read(frame_bytes)
                if not data:
                    break
                _require(len(data) == frame_bytes, f"partial decoded frame in {path} at {count}")
                if count in targets:
                    found.add(count)
                    yield count, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
                count += 1
                if expected_count is None and found == targets:
                    sampling_complete = True
                    break
            if sampling_complete:
                process.terminate()
            else:
                status = process.wait(timeout=30)
                error_log.seek(0)
                errors = error_log.read().decode(errors="replace").strip()
                _require(status == 0 and not errors, f"video decode failed for {path}: {errors}")
                if expected_count is not None:
                    _require(count == expected_count,
                             f"decoded frame count {count} differs from index {expected_count}: {path}")
            _require(found == targets, f"missing decoded target frames in {path}: {sorted(targets - found)}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _profile(value):
    from hilserl.image_profile import get_image_profile

    _require(isinstance(value, dict) and isinstance(value.get("name"), str), "image_profile must be a named configuration")
    profile = copy.deepcopy(value)
    _require(profile == get_image_profile(profile["name"]), "image_profile differs from the registered frozen configuration")
    _require(profile.get("raw_size") == [1280, 720] and profile.get("color") == "RGB", "unsupported source image contract")
    for camera in POLICY_CAMERAS:
        _require(profile["cameras"][camera]["size"] in ([128, 128], [160, 160], [192, 192], [224, 224]), "unsupported policy image size")
    _require(profile["cameras"]["side_classifier"]["size"] == [128, 128], "classifier image contract must remain 128 x 128")
    _require(len({tuple(profile["cameras"][camera]["size"]) for camera in POLICY_CAMERAS}) == 1, "policy camera sizes must agree")
    return profile


def _preprocess(frame, camera, profile):
    from hilserl.image_profile import preprocess_image

    result = preprocess_image(frame, camera, profile)
    _require(result.dtype == np.uint8 and result.shape == (*profile["cameras"][camera]["size"][::-1], 3), "preprocessor returned an invalid policy image")
    return result


def plan_seed_recrop(directory, *, expected_manifest_sha256):
    """Resolve all source references without decoding or writing dataset files."""
    directory = Path(directory).resolve()
    manifest = seed.validate_seed_dataset(directory, expected_manifest_sha256=expected_manifest_sha256)
    source = _file_info(directory / "manifest.json")
    _require(source["sha256"] == expected_manifest_sha256, "source manifest changed during validation")
    _require("image_rebuild" not in manifest, "rebuild source must be the original audited seed")
    for camera in POLICY_CAMERAS:
        _require(manifest["observation_schema"][camera]["shape"] == [1, 128, 128, 3], "rebuild source is not the original 128 seed")
    records, requested, meta_sources = [], defaultdict(set), {}
    for entry in manifest["episodes"]:
        for key in ("source_capture", "source_config", "source_episode"):
            info = entry[key]
            if info["path"] not in meta_sources:
                meta_sources[info["path"]] = _file_info(info["path"], info)
        capture = Path(entry["source_capture"]["path"]).parent.resolve()
        for i, raw_source in enumerate(entry["source_steps"]):
            raw_path = Path(raw_source["path"])
            _require(raw_path.parent.parent.parent.resolve() == capture, f"raw step belongs to a different capture: {raw_path}")
            data = _read_bound_bytes(raw_source)
            try:
                with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                    raw = json.loads(str(archive["__tree__"]))
            except (KeyError, ValueError, OSError) as exc:
                raise RecropDatasetError(f"invalid raw NPZ reference tree: {raw_path}") from exc
            _require(raw.get("id") == raw_source["raw_transition_id"] and raw.get("episode_id") == entry["episode_id"]
                     and raw.get("episode_step") == i and raw.get("complete_transition") is True,
                     f"raw transition identity mismatch: {raw_path}")
            for prefix, ref_field in (("obs", "frame_references"), ("next_obs", "next_frame_references")):
                for camera in POLICY_CAMERAS:
                    ref = raw.get(ref_field, {}).get(camera)
                    _require(isinstance(ref, dict) and set(ref) == set(REFERENCE_FIELDS), f"missing or unsupported {ref_field}/{camera}: {raw_path}")
                    _require(ref.get("camera") == camera, f"camera reference mismatch: {raw_path}")
                    for key in REFERENCE_FIELDS[1:]:
                        _require(type(ref.get(key)) is int and ref[key] >= 0, f"invalid reference {key}: {raw_path}")
                    index_path = str(capture / "video" / camera / "frames.jsonl")
                    requested[(index_path, camera)].add(ref["capture_id"])
                    records.append({"episode_file": entry["file"], "episode_step": i,
                                    "raw_transition_id": raw["id"], "raw_source_sha256": raw_source["sha256"],
                                    "observation": prefix, **ref, "index_path": index_path})
    indexes = {key: load_frame_index(key[0], key[1], ids) for key, ids in requested.items()}
    videos, slots = {}, {}
    for record in records:
        index = indexes[(record["index_path"], record["camera"])]
        frame = index["frames"][record["capture_id"]]
        _require(all(record[key] == frame[key] for key in REFERENCE_FIELDS),
                 f"capture timestamp/reference mismatch: {record['raw_transition_id']} {record['observation']} {record['camera']}")
        record.update(frame)
        identity = (frame["video_path"], frame["segment_frame"])
        if identity not in slots:
            slots[identity] = len(slots)
        record["image_slot"] = slots[identity]
        videos[frame["video_path"]] = {"path": frame["video_path"], "camera": record["camera"],
                                       "indexed_frames": index["segment_counts"][frame["segment"]], "fps": index["fps"]}
    _require(len(records) == manifest["counts"]["transitions"] * 4, "image mapping count mismatch")
    return {"manifest": manifest, "source_seed": source, "records": records,
            "source_indexes": [index["source"] for index in indexes.values()],
            "source_metadata": list(meta_sources.values()), "videos": videos, "unique_images": len(slots)}


def _decode_plan(plan, profile, consume):
    selected = defaultdict(dict)
    for row in plan["records"]:
        selected[row["video_path"]][row["segment_frame"]] = row
    sources = []
    for ordinal, path in enumerate(sorted(selected), 1):
        meta = plan["videos"][path]
        source = _file_info(path)
        source.update(probe_video(path, raw_size=profile["raw_size"], fps=meta["fps"]))
        for index, bgr in iter_video_frames(path, selected[path], expected_count=meta["indexed_frames"], raw_size=profile["raw_size"]):
            row = selected[path][index]
            consume(row["image_slot"], _preprocess(bgr, row["camera"], profile))
        _file_info(path, source)
        source.update(camera=meta["camera"], indexed_frames=meta["indexed_frames"], decoded_frames=meta["indexed_frames"])
        sources.append(source)
        LOGGER.info("Decoded segment %d/%d: %s; %d indexed frames, %d selected images", ordinal, len(selected),
                    path, meta["indexed_frames"], len(selected[path]))
    return sources


def _assert_preserved(original, rebuilt, profile):
    _require(set(original) == set(rebuilt), "reconstructed array inventory differs from source seed")
    for key, before in original.items():
        after = rebuilt[key]
        if key in REPLACED_ARRAYS:
            _require(after.dtype == np.uint8 and after.shape == (len(before), 1, *profile["cameras"][key.split("__", 1)[1]]["size"][::-1], 3), f"invalid rebuilt policy image: {key}")
        else:
            _require(before.dtype == after.dtype and before.shape == after.shape and before.tobytes() == after.tobytes(),
                     f"preserved seed array changed: {key}")
    for camera in POLICY_CAMERAS:
        _require(np.array_equal(rebuilt[f"next_obs__{camera}"][:-1], rebuilt[f"obs__{camera}"][1:]),
                 f"reconstructed observation continuity mismatch: {camera}")


def build_recropped_dataset(directory, output, *, image_profile, expected_source_manifest_sha256):
    """Build a separate immutable candidate, preserving every original seed row."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    _require(output != directory and directory not in output.parents, "output must be outside the original seed directory")
    profile = _profile(image_profile)
    plan = plan_seed_recrop(directory, expected_manifest_sha256=expected_source_manifest_sha256)
    LOGGER.info("Resolved %d image mappings, %d unique images, %d video segments", len(plan["records"]),
                plan["unique_images"], len(plan["videos"]))
    output.mkdir(parents=True, exist_ok=False)
    written = []
    cache_path = output / ".policy_frames.npy"
    try:
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8,
                                         shape=(plan["unique_images"], *profile["cameras"][POLICY_CAMERAS[0]]["size"][::-1], 3))
        source_videos = _decode_plan(plan, profile, lambda slot, pixels: cache.__setitem__(slot, pixels))
        cache.flush()
        hashes = {slot: _sha(cache[slot].tobytes()) for slot in range(plan["unique_images"])}
        by_episode = defaultdict(list)
        for row in plan["records"]:
            by_episode[row["episode_file"]].append(row)
        manifest = copy.deepcopy(plan["manifest"])
        for entry in manifest["episodes"]:
            original = seed._episode_arrays(directory, entry)
            arrays = dict(original)
            for key in REPLACED_ARRAYS:
                arrays[key] = np.empty((entry["steps"], 1, *profile["cameras"][key.split("__", 1)[1]]["size"][::-1], 3), np.uint8)
            for row in by_episode[entry["file"]]:
                arrays[f"{row['observation']}__{row['camera']}"][row["episode_step"], 0] = cache[row["image_slot"]]
            _assert_preserved(original, arrays, profile)
            entry["source_seed_episode"] = _file_info(directory / entry["file"], entry)
            entry["image_profile"] = profile["name"]
            entry["preserved_array_sha256"] = {key: _sha(value.tobytes()) for key, value in original.items() if key not in REPLACED_ARRAYS}
            with (output / entry["file"]).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            info = _file_info(output / entry["file"])
            entry.update(bytes=info["bytes"], sha256=info["sha256"])
            written.append(entry["file"])
            LOGGER.info("Wrote episode %d/%d: %s; %d transitions", len(written), len(manifest["episodes"]),
                        entry["file"], entry["steps"])
        del cache
        cache_path.unlink()
        mapping_path = output / "frame_mapping.jsonl"
        with mapping_path.open("x") as stream:
            for row in plan["records"]:
                stream.write(json.dumps({**row, "image_sha256": hashes[row["image_slot"]]}, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        mapping = _file_info(mapping_path)
        mapping.pop("path")
        mapping.update(file=mapping_path.name, rows=len(plan["records"]))
        manifest["image_profile"] = profile
        for camera in POLICY_CAMERAS:
            manifest["observation_schema"][camera]["shape"] = [1, *profile["cameras"][camera]["size"][::-1], 3]
        manifest["image_rebuild"] = {
            "schema_version": 1, "source_seed": plan["source_seed"], "source_encoding": "h264_lossy_video",
            "source_description": "Decoded 1280x720 H.264 recordings; not uncompressed raw camera pixels and not enlarged 128px seed images.",
            "decoder": "ffmpeg sequential BGR24, xerror, err_detect=explode; full indexed segment frame count checked",
            "decoder_runtime": _decoder_capabilities(),
            "mapping_rule": "exact camera/capture_id/monotonic_ns/unix_ns; index row ordinal within segment; zero based",
            "frame_mapping": mapping, "unique_images": plan["unique_images"],
            "source_indexes": plan["source_indexes"], "source_videos": source_videos,
            "source_metadata": plan["source_metadata"], "replaced_arrays": sorted(REPLACED_ARRAYS),
            "builder_source_sha256": _sha(Path(__file__).read_bytes()),
        }
        with (output / "manifest.json").open("xb") as stream:
            stream.write(_json_bytes(manifest)); stream.flush(); os.fsync(stream.fileno())
        validate_recropped_dataset(output)
        return manifest
    except Exception as exc:
        (output / "build-failure.json").write_bytes(_json_bytes({"status": "incomplete", "error": str(exc), "episodes_written": written}))
        raise


def validate_recropped_dataset(directory, *, expected_manifest_sha256=None, verify_pixels=False):
    """Audit source hashes, exact mappings, copied arrays and optionally re-decode.

    Source recordings must remain available for this offline audit. Training uses
    the ordinary seed loader with its frozen manifest and image-profile checks.
    """
    directory = Path(directory).resolve()
    _require(not (directory / "build-failure.json").exists(), "dataset build did not finish successfully")
    data = (directory / "manifest.json").read_bytes()
    _require(expected_manifest_sha256 is None or _sha(data) == expected_manifest_sha256, "reconstructed manifest checksum mismatch")
    manifest = json.loads(data)
    _require(manifest.get("status") == "complete" and manifest.get("schema_version") == 1, "incomplete reconstructed dataset")
    profile = _profile(manifest.get("image_profile"))
    rebuild = manifest["image_rebuild"]
    _require(rebuild.get("schema_version") == 1 and rebuild.get("source_encoding") == "h264_lossy_video"
             and rebuild.get("replaced_arrays") == sorted(REPLACED_ARRAYS), "unsupported image reconstruction contract")
    source = rebuild["source_seed"]
    _file_info(source["path"], source)
    source_dir = Path(source["path"]).parent
    plan = plan_seed_recrop(source_dir, expected_manifest_sha256=source["sha256"])
    expected_manifest = copy.deepcopy(plan["manifest"])
    expected_manifest["image_profile"] = profile
    for camera in POLICY_CAMERAS:
        expected_manifest["observation_schema"][camera]["shape"] = [1, *profile["cameras"][camera]["size"][::-1], 3]
    for key, value in expected_manifest.items():
        if key != "episodes":
            _require(manifest.get(key) == value, f"source seed manifest field changed: {key}")
    _require(len(manifest["episodes"]) == len(plan["manifest"]["episodes"]), "source episode count changed")
    _require({p.name for p in directory.glob("*.npz")} == {e["file"] for e in manifest["episodes"]}, "reconstructed file inventory mismatch")
    _require(rebuild["source_indexes"] == plan["source_indexes"] and rebuild["source_metadata"] == plan["source_metadata"], "source provenance changed")
    _require(rebuild["unique_images"] == plan["unique_images"], "unique image count mismatch")
    video_sources = {item["path"]: item for item in rebuild["source_videos"]}
    _require(len(video_sources) == len(rebuild["source_videos"]) and set(video_sources) == set(plan["videos"]), "source video inventory mismatch")
    for path, value in video_sources.items():
        _file_info(path, value)
        expected = plan["videos"][path]
        _require(value.get("codec") == "h264" and [value.get("width"), value.get("height")] == profile["raw_size"]
                 and value.get("indexed_frames") == value.get("decoded_frames") == expected["indexed_frames"]
                 and value.get("camera") == expected["camera"] and value.get("frame_rate") == str(Fraction(str(expected["fps"]))),
                 "source video metadata mismatch")
    mapping = rebuild["frame_mapping"]
    _require(mapping.get("file") == "frame_mapping.jsonl", "unsafe mapping filename")
    mapping_data = _read_bound_bytes({**mapping, "path": str(directory / mapping["file"])})
    rows = [json.loads(line) for line in mapping_data.splitlines()]
    _require(len(rows) == mapping["rows"] == len(plan["records"]), "frame mapping row count mismatch")
    by_episode, image_hashes = defaultdict(list), {}
    for actual, expected in zip(rows, plan["records"]):
        _require(set(actual) == set(expected) | {"image_sha256"} and all(actual[key] == value for key, value in expected.items()), "exact frame mapping mismatch")
        slot = actual["image_slot"]
        _require(slot not in image_hashes or image_hashes[slot] == actual["image_sha256"], "one source frame has conflicting image hashes")
        image_hashes[slot] = actual["image_sha256"]
        by_episode[actual["episode_file"]].append(actual)
    for entry, source_entry in zip(manifest["episodes"], plan["manifest"]["episodes"]):
        for key, value in source_entry.items():
            if key not in ("bytes", "sha256", "image_profile"):
                _require(entry.get(key) == value, f"source episode metadata changed: {key}")
        _require(entry.get("image_profile") == profile["name"], "episode image profile mismatch")
        original = seed._episode_arrays(source_dir, source_entry)
        _require(entry.get("source_seed_episode") == _file_info(source_dir / source_entry["file"], source_entry), "source seed episode provenance mismatch")
        target_data = _read_bound_bytes({**entry, "path": str(directory / entry["file"])})
        rebuilt = seed._load_npz_bytes(target_data, entry["file"])
        _assert_preserved(original, rebuilt, profile)
        _require(entry.get("preserved_array_sha256") == {key: _sha(value.tobytes()) for key, value in original.items() if key not in REPLACED_ARRAYS}, "preserved array hashes mismatch")
        for row in by_episode[entry["file"]]:
            pixels = rebuilt[f"{row['observation']}__{row['camera']}"][row["episode_step"], 0]
            _require(_sha(pixels.tobytes()) == row["image_sha256"], "rebuilt image checksum mismatch")
    if verify_pixels:
        def compare(slot, pixels):
            _require(_sha(pixels.tobytes()) == image_hashes[slot], f"redecoded source pixels differ at image slot {slot}")
        _decode_plan(plan, profile, compare)
    return manifest
