"""Immutable reward cache for the pinned demonstrations, separate from source."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import pickle

from hilserl.episode_commit import envelope, save_json, save_pickle, validate_envelope
from hilserl.reward_provider import digest, relabel


def episodes(transitions):
    seen = set()
    for parent, group in itertools.groupby(transitions,
            key=lambda t: t["infos"]["raw_transition_id"].rpartition("/")[0]):
        if parent in seen:
            raise ValueError("Seed episode order is not contiguous")
        seen.add(parent)
        yield parent, list(group)


def prepare_seed_cache(transitions, directory, source_sha256, spec, provider, *, check=lambda: None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for parent, raw in episodes(transitions):
        check()
        scores = provider.score(raw, check=check)
        labeled = relabel(raw, scores, spec)
        value = envelope("seed:" + source_sha256, parent, raw[0]["infos"]["episode_id"], raw, labeled, spec)
        validate_envelope(value, "seed:" + source_sha256, spec, len(raw))
        filename = f"{len(entries):06d}.pkl"
        save_pickle(directory / filename, value)
        save_json(directory / f"{len(entries):06d}.scores.json", scores)
        entries.append(dict(file=filename, sha256=hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                            raw_digest=digest(raw), steps=len(raw)))
    if not entries:
        raise ValueError("No seed episodes")
    manifest = dict(version=1, source_sha256=source_sha256, contract_sha256=spec.sha256, episodes=entries)
    save_json(directory / "manifest.json", manifest)
    return manifest


def iter_reward_seed(transitions, directory, source_sha256, spec):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (manifest.get("version") != 1 or manifest.get("source_sha256") != source_sha256
            or manifest.get("contract_sha256") != spec.sha256):
        raise ValueError("Demo reward cache does not match seed/model/input/reward version")
    source = episodes(transitions)
    for entry in manifest["episodes"]:
        try:
            _, raw = next(source)
        except StopIteration as exc:
            raise ValueError("Too many cached seed episodes") from exc
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("Invalid seed cache path")
        blob = (directory / filename).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError("Demo reward cache checksum mismatch")
        value = pickle.loads(blob)  # Locally generated, pinned training artifact.
        validate_envelope(value, "seed:" + source_sha256, spec, len(raw))
        if value["raw_digest"] != digest(raw) or len(raw) != entry["steps"]:
            raise ValueError("Reward cache and source demo observations/actions disagree")
        yield from value["transitions"]
    if next(source, None) is not None:
        raise ValueError("Demo reward cache is incomplete")
