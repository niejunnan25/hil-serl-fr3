"""Audited fixed-XYZ demonstrations, derived without changing raw recordings.

A dataset contains complete human demonstrations, never success-credit copies.
Only the final executed transition receives the final human label. Arrays are
stored in NPZ (allow_pickle=False); a manifest binds every file to its sources.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from hilserl.image_profile import FULL_FRAME, get_image_profile, observation_image_schema

ACTION_CONTRACT = "fixed-xyz-v1"
OBSERVATION_CONTRACT = "reset-relative-pose_body-velocity_base-wrench_19d-v1"
SELECTION = "complete_human_success_episodes"
CAMERAS = ("side_classifier", "side_policy", "wrist_1")
OBSERVATION_SCHEMA = {"state": {"shape": [1, 19], "dtype": "float32"},
                      **{k: {"shape": [1, 128, 128, 3], "dtype": "uint8"} for k in CAMERAS}}
REWARD_RULE = "nonterminal=0; final=human_episode_outcome; final_done=true; final_mask=0"


class SeedDatasetError(ValueError):
    """A dataset cannot be safely admitted under the requested learning contract."""


def _require(condition, reason):
    if not condition:
        raise SeedDatasetError(reason)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def _array(value, shape, name, dtype=None):
    a = np.asarray(value)
    _require(a.shape == tuple(shape), f"{name}: expected shape {tuple(shape)}, got {a.shape}")
    _require(a.dtype.kind in "biuf" and np.all(np.isfinite(a)), f"{name}: nonnumeric or nonfinite values")
    if dtype:
        _require(str(a.dtype) == dtype, f"{name}: expected dtype {dtype}, got {a.dtype}")
    return a


def _rotation(q):
    q = _array(q, (4,), "raw quaternion").astype(np.float64)
    _require(abs(np.linalg.norm(q) - 1) < 1e-5, "raw quaternion is not normalized")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _euler_rotation(euler):
    x, y, z = euler
    cx, cy, cz = np.cos([x, y, z]); sx, sy, sz = np.sin([x, y, z])
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])


def _schema(profile=FULL_FRAME):
    return {"state": OBSERVATION_SCHEMA["state"], **observation_image_schema(profile)}


def _observation(obs, name, schema=OBSERVATION_SCHEMA):
    _require(isinstance(obs, dict) and set(obs) == set(schema), f"{name}: observation keys mismatch")
    for key, spec in schema.items():
        _array(obs[key], spec["shape"], f"{name}/{key}", spec["dtype"])


def _verify_frame(obs, raw_state, origin, origin_rotation, evidence, schema=OBSERVATION_SCHEMA):
    """Prove an already-relative recording matches its raw state; do not convert."""
    _observation(obs, "raw observation", schema)
    values = raw_state["values"]
    pose = _array(values["tcp_pose"], (7,), "raw tcp_pose")
    rotation = _rotation(pose[3:])
    state = obs["state"][0]
    errors = {
        "position_m": np.max(np.abs(state[4:7] - origin_rotation.T @ (pose[:3] - origin[:3]))),
        "rotation_matrix": np.max(np.abs(_euler_rotation(state[7:10]) - origin_rotation.T @ rotation)),
        "gripper": abs(state[0] - float(_array(values["gripper_pose"], (), "raw gripper"))),
        "force": np.max(np.abs(state[1:4] - _array(values["tcp_force"], (3,), "raw force"))),
        "torque": np.max(np.abs(state[10:13] - _array(values["tcp_torque"], (3,), "raw torque"))),
    }
    velocity = _array(values["tcp_vel"], (6,), "raw velocity")
    errors["velocity"] = max(np.max(np.abs(state[13:16] - rotation.T @ velocity[:3])),
                             np.max(np.abs(state[16:19] - rotation.T @ velocity[3:])))
    for key, value in errors.items():
        _require(value <= 2e-5, f"observation frame mismatch: {key} error {float(value)}")
        evidence[key] = max(evidence.get(key, 0.0), float(value))
    return rotation


def _eligibility(meta):
    if meta.get("complete") is not True or meta.get("status") != "complete":
        return "incomplete_episode"
    if meta.get("verdict_source") != "human" or type(meta.get("outcome")) is not int or meta["outcome"] not in (0, 1):
        return "missing_final_human_label"
    if meta["outcome"] != 1:
        return "human_labeled_failure_retained_only_in_source"
    if type(meta.get("steps")) is not int or meta["steps"] <= 0 or meta["steps"] != meta.get("committed_steps"):
        return "inconsistent_committed_steps"
    if meta.get("human_steps") != meta["steps"]:
        return "mixed_or_policy_episode_not_expert_demo"
    if meta.get("reset_info", {}).get("reset", {}).get("success") is not True:
        return "reset_success_not_proven"
    return None


def _read_stable_json(path):
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SeedDatasetError(f"Invalid source JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"Invalid source object: {path}")
    return value, {"path": str(path.resolve()), "sha256": _sha(data), "bytes": len(data)}


def _extract_episode(meta_path, meta, meta_source):
    from hilserl.storage import read_step

    n = meta["steps"]
    paths = sorted(meta_path.parent.glob("*.npz"))
    _require([p.name for p in paths] == [f"{i:06d}.npz" for i in range(n)], "raw step sequence has gaps or extra files")
    capture_path = meta_path.parents[2] / "manifest.json"
    capture, capture_source = _read_stable_json(capture_path)
    config_path = Path(capture.get("metadata", {}).get("config_file", ""))
    _require(config_path.is_absolute() and config_path.is_file(), "capture has no readable immutable config snapshot")
    config, config_source = _read_stable_json(config_path)
    image_profile = get_image_profile(config.get("image_profile", FULL_FRAME))["name"]
    schema = _schema(image_profile)
    _require(config.get("action_max_step") == 0.008, "source uses a different displacement limit")
    raw_sources, steps, frame_errors = [], [], {}
    origin = origin_rotation = previous = None
    original_nonterminal_positive = 0
    for i, path in enumerate(paths):
        before = path.read_bytes()
        raw = read_step(path)
        _require(_sha(path.read_bytes()) == _sha(before), f"Source changed while reading: {path}")
        _require(raw.get("complete_transition") is True, "incomplete raw transition")
        _require(raw.get("image_profile", FULL_FRAME) == image_profile, "raw/config image profile mismatch")
        _require(raw.get("episode_step") == i and raw.get("episode_id") == meta["id"], "raw episode/step identity mismatch")
        expected_id = f"{meta_path.parents[4].name}/{meta_path.parents[2].name}/{meta['id']}/{i:06d}"
        _require(raw.get("id") == expected_id, "raw transition id does not match its archived source")
        _require(raw.get("source_action") == "human", "episode summary says human but raw step does not")
        action = _array(raw["actions"], (7,), "raw action", "float32")
        controller = _array(raw["controller_input_action"], (7,), "controller action", "float32")
        _require(np.all(action[3:] == 0) and np.all(controller[3:] == 0), "rotation or gripper was not locked")
        _require(np.max(np.abs(action[:3])) <= 1.0, "raw body XYZ lies outside normalized policy action space")
        commands = raw.get("controller_commands")
        _require(isinstance(commands, list) and len(commands) > 0, "missing executed controller command")
        _require(all(c.get("kind") == "pose" and c.get("request_sent") is True and c.get("returned") is True for c in commands),
                 "non-pose, failed or incomplete controller command")
        if origin is None:
            origin = _array(raw["raw_state"]["values"]["tcp_pose"], (7,), "origin raw pose").copy()
            origin_rotation = _rotation(origin[3:])
            _observation(raw["observations"], "initial observation", schema)
            _require(np.max(np.abs(raw["observations"]["state"][0, 4:10])) < 1e-6,
                     "first observation is not the recorded reset-relative origin")
        rotation = _verify_frame(raw["observations"], raw["raw_state"], origin, origin_rotation, frame_errors, schema)
        _verify_frame(raw["next_observations"], raw["raw_next_state"], origin, origin_rotation, frame_errors, schema)
        body_error = float(np.max(np.abs(rotation @ action[:3] - controller[:3])))
        _require(body_error <= 2e-6, f"body action and actual controller input disagree: {body_error}")
        frame_errors["body_to_controller_action"] = max(frame_errors.get("body_to_controller_action", 0.), body_error)
        if previous is not None:
            _require(all(np.array_equal(previous[k], raw["observations"][k]) for k in OBSERVATION_SCHEMA),
                     "adjacent raw observations are discontinuous")
        previous = raw["next_observations"]
        _require(np.isfinite(raw.get("observed_reward", np.nan)), "raw reward is not finite")
        original_nonterminal_positive += int(i < n-1 and raw["observed_reward"] > 0)
        if i < n-1:
            _require(not raw.get("terminated") and not raw.get("truncated"), "environment terminated before final executed step")
        raw_sources.append({"path": str(path.resolve()), "sha256": _sha(before), "bytes": len(before), "raw_transition_id": raw["id"]})
        steps.append(raw)
    for source in (meta_source, capture_source, config_source):
        _require(_sha(Path(source["path"]).read_bytes()) == source["sha256"], f"Source metadata changed: {source['path']}")
    arrays = {f"{prefix}__{key}": np.stack([s[field][key] for s in steps])
              for prefix, field in (("obs", "observations"), ("next_obs", "next_observations")) for key in OBSERVATION_SCHEMA}
    arrays.update(actions=np.stack([s["actions"][:3] for s in steps]),
                  rewards=np.zeros(n, np.float32), dones=np.zeros(n, np.bool_), masks=np.ones(n, np.float32),
                  episode_steps=np.arange(n, dtype=np.int64),
                  global_steps=np.asarray([s["global_step"] for s in steps], dtype=np.int64),
                  raw_transition_ids=np.asarray([s["id"] for s in steps]))
    arrays["rewards"][-1] = float(meta["outcome"]); arrays["dones"][-1] = True; arrays["masks"][-1] = 0.
    entry = {"image_profile": image_profile, "episode_id": meta["id"], "steps": n, "outcome": meta["outcome"], "verdict_source": "human",
             "source_action": "human", "source_episode": meta_source, "source_capture": capture_source,
             "source_config": config_source, "source_steps": raw_sources, "frame_max_errors": frame_errors,
             "nonterminal_classifier_positive_ignored": original_nonterminal_positive,
             "selection_reason": "complete successful trajectory with every executed action supplied by human",
             "reset_origin_raw_pose_xyzw": origin.tolist()}
    return arrays, entry


def _load_npz_bytes(data, name):
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
            return {k: loaded[k].copy() for k in loaded.files}
    except (ValueError, OSError, EOFError) as exc:
        raise SeedDatasetError(f"Cannot load seed episode {name}: {exc}") from exc


def _validate_arrays(arrays, entry, schema=None):
    schema = schema or _schema(entry.get("image_profile", FULL_FRAME))
    n = entry.get("steps")
    _require(type(n) is int and n > 0, "invalid episode transition count")
    fields = {"actions", "rewards", "dones", "masks", "episode_steps", "global_steps", "raw_transition_ids"}
    expected = fields | {f"{prefix}__{key}" for prefix in ("obs", "next_obs") for key in schema}
    _require(set(arrays) == expected, "seed episode array schema mismatch")
    for prefix in ("obs", "next_obs"):
        for key, spec in schema.items():
            _array(arrays[f"{prefix}__{key}"], [n] + spec["shape"], key, spec["dtype"])
    _array(arrays["actions"], (n, 3), "learning actions", "float32")
    _require(np.all(np.abs(arrays["actions"]) <= 1), "learning actions exceed normalized bounds")
    for key, dtype in (("rewards", "float32"), ("dones", "bool"), ("masks", "float32"),
                       ("episode_steps", "int64"), ("global_steps", "int64")):
        _array(arrays[key], (n,), key, dtype)
    _require(np.array_equal(arrays["episode_steps"], np.arange(n)), "episode steps are not contiguous")
    _require(arrays["global_steps"][0] >= 0 and np.all(np.diff(arrays["global_steps"]) == 1), "global steps are not contiguous")
    _require(np.all(arrays["rewards"][:-1] == 0) and arrays["rewards"][-1] == entry["outcome"], "reward differs from final human label")
    _require(not np.any(arrays["dones"][:-1]) and bool(arrays["dones"][-1]), "terminal flags must mark only the last true step")
    _require(np.all(arrays["masks"][:-1] == 1) and arrays["masks"][-1] == 0, "terminal must not bootstrap")
    ids = arrays["raw_transition_ids"]
    _require(ids.shape == (n,) and ids.dtype.kind == "U", "invalid transition ids")
    _require(len(set(ids.tolist())) == n, "duplicate transition id within episode")
    sources = entry.get("source_steps", [])
    _require(len(sources) == n and ids.tolist() == [s.get("raw_transition_id") for s in sources], "source provenance does not match transitions")
    for i, identity in enumerate(ids):
        _require(identity.endswith(f"/{entry['episode_id']}/{i:06d}"), "transition id episode/step mismatch")
    for key in schema:
        _require(np.array_equal(arrays[f"next_obs__{key}"][:-1], arrays[f"obs__{key}"][1:]), "dataset observation continuity mismatch")
    _require(np.max(np.abs(arrays["obs__state"][0, 0, 4:10])) < 1e-6, "dataset initial pose is not reset-relative")


def _manifest(directory, expected_action_contract, expected_manifest_sha256=None, expected_image_profile=FULL_FRAME):
    path = directory / "manifest.json"
    _require(path.is_file(), "fixed-xyz-v1 requires an audited seed manifest; legacy pickle demos are incompatible")
    _require(not (directory / "build-failure.json").exists(), "dataset build did not finish successfully")
    data = path.read_bytes()
    if expected_manifest_sha256 is not None:
        _require(isinstance(expected_manifest_sha256, str) and len(expected_manifest_sha256) == 64
                 and _sha(data) == expected_manifest_sha256, "seed manifest SHA256 differs from frozen run configuration")
    try:
        manifest = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise SeedDatasetError(f"Invalid seed manifest: {exc}") from exc
    _require(isinstance(manifest, dict), "seed manifest must be an object")
    _require(manifest.get("schema_version") == 1 and manifest.get("status") == "complete", "incomplete or unsupported seed dataset")
    _require(expected_action_contract == ACTION_CONTRACT and manifest.get("action_contract") == expected_action_contract, "incompatible seed action contract")
    profile = get_image_profile(expected_image_profile)
    _require(manifest.get("image_profile", get_image_profile(FULL_FRAME)) == profile, "incompatible seed image profile")
    _require(manifest.get("observation_contract") == OBSERVATION_CONTRACT and manifest.get("observation_schema") == _schema(profile),
             "incompatible seed observation contract")
    _require(manifest.get("action_schema") == {"shape": [3], "dtype": "float32", "frame": "current_end_effector_body", "bounds": [-1, 1]},
             "incompatible seed action schema")
    _require(manifest.get("reward_rule") == REWARD_RULE and manifest.get("selection") == SELECTION, "unsupported seed reward/selection rule")
    _require(manifest.get("execution") == {"rotation_locked": True, "gripper_locked": True, "action_max_step_m": 0.008},
             "seed execution contract differs")
    entries = manifest.get("episodes")
    _require(isinstance(entries, list) and len(entries) > 0, "empty seed dataset")
    paths = []
    for entry in entries:
        _require(isinstance(entry, dict), "invalid seed episode manifest")
        rel = entry.get("file", "")
        _require(isinstance(rel, str) and Path(rel).name == rel and rel.endswith(".npz"), "unsafe seed episode filename")
        _require(entry.get("outcome") == 1 and type(entry.get("outcome")) is int and entry.get("verdict_source") == "human"
                 and entry.get("source_action") == "human", "seed episode lacks successful human provenance")
        paths.append(rel)
    _require(len(paths) == len(set(paths)), "duplicate seed episode file")
    _require(set(paths) == {p.name for p in directory.glob("*.npz")}, "seed episode file inventory mismatch")
    return manifest


def _episode_arrays(directory, entry, schema=None):
    path = directory / entry["file"]
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlink seed episode: {entry['file']}")
    data = path.read_bytes()
    _require(len(data) == entry.get("bytes") and _sha(data) == entry.get("sha256"), f"Seed checksum mismatch: {entry['file']}")
    arrays = _load_npz_bytes(data, entry["file"])
    _validate_arrays(arrays, entry, schema)
    return arrays


def validate_seed_dataset(directory, *, expected_action_contract=ACTION_CONTRACT, expected_manifest_sha256=None, expected_image_profile=FULL_FRAME):
    """Validate all content before training admission; no robot/JAX dependency."""
    directory = Path(directory)
    manifest = _manifest(directory, expected_action_contract, expected_manifest_sha256, expected_image_profile)
    seen, steps = set(), 0
    for entry in manifest["episodes"]:
        arrays = _episode_arrays(directory, entry, manifest["observation_schema"])
        ids = set(arrays["raw_transition_ids"].tolist())
        _require(not seen.intersection(ids), "duplicate raw transition across seed episodes")
        seen.update(ids); steps += entry["steps"]
    _require(manifest.get("counts") == {"episodes": len(manifest["episodes"]), "transitions": steps,
             "human_transitions": steps, "positive_rewards": len(manifest["episodes"]), "terminal_transitions": len(manifest["episodes"])},
             "seed summary counts do not match content")
    return manifest


def iter_seed_transitions(directory, *, expected_action_contract=ACTION_CONTRACT, expected_manifest_sha256=None, expected_image_profile=FULL_FRAME):
    """Yield canonical transitions only after the whole dataset passes validation."""
    directory = Path(directory)
    manifest = validate_seed_dataset(directory, expected_action_contract=expected_action_contract,
                                     expected_manifest_sha256=expected_manifest_sha256, expected_image_profile=expected_image_profile)
    for entry in manifest["episodes"]:
        arrays = _episode_arrays(directory, entry, manifest["observation_schema"])
        for i in range(entry["steps"]):
            terminal = bool(arrays["dones"][i])
            yield {"observations": {k: arrays[f"obs__{k}"][i].copy() for k in OBSERVATION_SCHEMA},
                   "next_observations": {k: arrays[f"next_obs__{k}"][i].copy() for k in OBSERVATION_SCHEMA},
                   "actions": arrays["actions"][i].copy(), "rewards": np.float32(arrays["rewards"][i]),
                   "dones": np.bool_(terminal), "masks": np.float32(arrays["masks"][i]),
                   "infos": {"image_profile": get_image_profile(expected_image_profile)["name"], "source_action": "human", "manual_success": terminal, "succeed": terminal,
                             "step": int(arrays["global_steps"][i]), "episode_id": entry["episode_id"],
                             "raw_transition_id": str(arrays["raw_transition_ids"][i]),
                             "verdict_source": "human" if terminal else None, "action_contract": ACTION_CONTRACT}}


def build_seed_dataset(run_dirs: Iterable[Path], output):
    """Create a new immutable candidate. A manifest is written only when complete.

    Source data is read-only. Interrupted builds stay incomplete and cannot load.
    Every complete fully-human success is included if its full raw contract proves
    valid. Mixed/failed episodes are inventoried, never relabeled or truncated.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries, omitted = [], []
    try:
        paths = sorted({p.resolve() for run in run_dirs for p in Path(run).glob("recordings/*/episodes/*/episode.json")})
        _require(len(paths) > 0, "no raw episode metadata found")
        for meta_path in paths:
            meta, source = _read_stable_json(meta_path)
            reason = _eligibility(meta)
            if reason:
                omitted.append({"source_episode": source, "reason": reason, "steps": meta.get("steps"),
                                "human_steps": meta.get("human_steps"), "outcome": meta.get("outcome")})
                continue
            try:
                arrays, entry = _extract_episode(meta_path, meta, source)
                _validate_arrays(arrays, entry)
            except (SeedDatasetError, KeyError, TypeError, ValueError) as exc:
                omitted.append({"source_episode": source, "reason": f"raw_contract_rejected: {exc}",
                                "steps": meta.get("steps"), "human_steps": meta.get("human_steps"), "outcome": meta.get("outcome")})
                continue
            name = f"episode_{len(entries):04d}.npz"
            with (output / name).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush(); os.fsync(stream.fileno())
            data = (output / name).read_bytes()
            entry.update(file=name, bytes=len(data), sha256=_sha(data))
            entries.append(entry)
        _require(len(entries) > 0, "no episodes satisfy the fixed-XYZ demonstration contract")
        profiles = {e.get("image_profile", FULL_FRAME) for e in entries}
        _require(len(profiles) == 1, "cannot mix image profiles in one seed dataset")
        image_profile = profiles.pop()
        n = sum(e["steps"] for e in entries)
        manifest = {"schema_version": 1, "status": "complete", "action_contract": ACTION_CONTRACT,
                    "observation_contract": OBSERVATION_CONTRACT, "observation_schema": _schema(image_profile),
                    "image_profile": get_image_profile(image_profile),
                    "action_schema": {"shape": [3], "dtype": "float32", "frame": "current_end_effector_body", "bounds": [-1, 1]},
                    "execution": {"rotation_locked": True, "gripper_locked": True, "action_max_step_m": 0.008},
                    "state_layout": {"gripper": [0, 1], "base_force": [1, 4], "reset_relative_xyz": [4, 7],
                                     "reset_relative_euler_xyz": [7, 10], "base_torque": [10, 13], "body_velocity": [13, 19]},
                    "reward_rule": REWARD_RULE, "selection": SELECTION,
                    "selection_rationale": "Complete fully-human successes preserve true successful endings and expert action continuity; mixed and failed trajectories remain in original recordings and are not reused as expert seed.",
                    "counts": {"episodes": len(entries), "transitions": n, "human_transitions": n,
                               "positive_rewards": len(entries), "terminal_transitions": len(entries)},
                    "episodes": entries, "omitted": omitted,
                    "omitted_counts": dict(Counter(x["reason"] for x in omitted)),
                    "builder_source_sha256": _sha(Path(__file__).read_bytes())}
        for entry in entries:
            _episode_arrays(output, entry)
        temporary = output / ".manifest.partial"
        with temporary.open("xb") as stream:
            stream.write(_json_bytes(manifest)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, output / "manifest.json")
        temporary.unlink()
        fd = os.open(output, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return validate_seed_dataset(output, expected_image_profile=image_profile)
    except Exception as exc:
        (output / "build-failure.json").write_bytes(_json_bytes({"status": "incomplete", "error": str(exc), "episodes_written": len(entries), "omitted": omitted}))
        raise
