"""Browse and export completed raw episodes, without importing the training stack."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import pickle
import re
import shutil
import uuid
import zipfile

from hilserl.action_contract import LEGACY, resolve_action_contract
from hilserl.files import read_json, stamp
from hilserl.image_profile import FULL_FRAME, get_image_profile


def _identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError("Invalid archive identifier")
    return value


def _export_contract(*records):
    """Resolve explicit archive versions; absence preserves historical exports."""
    names = [record["action_contract"] for record in records if "action_contract" in record]
    contracts = [resolve_action_contract(name) for name in names]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Recording, episode and raw action contracts differ; export cancelled")
    return contracts[0] if contracts else resolve_action_contract(LEGACY)


def _export_image_profile(*records):
    names = [get_image_profile(record["image_profile"])["name"] for record in records if "image_profile" in record]
    if len(set(names)) > 1:
        raise ValueError("Recording, episode and raw image profiles differ; export cancelled")
    return names[0] if names else FULL_FRAME


class Archive:
    def __init__(self, output_root, *, min_free_bytes=8 * 2**30):
        self.root = Path(output_root).resolve()
        self.min_free_bytes = min_free_bytes

    def recording(self, recording_id):
        parts = recording_id.split("/")
        if len(parts) != 2:
            raise ValueError("Recording ID must name a run and capture")
        run, capture = map(_identifier, parts)
        path = (self.root / run / "recordings" / capture).resolve()
        if not path.is_relative_to(self.root) or not (path / "manifest.json").is_file():
            raise ValueError("Recording is unavailable")
        return path

    def list(self):
        recordings = []
        for path in sorted(self.root.glob("*/recordings/*/manifest.json"), reverse=True):
            manifest = read_json(path)
            run = path.parents[2]
            episodes = [read_json(p) for p in sorted((path.parent / "episodes").glob("*/episode.json"))]
            good = [episode for episode in episodes if episode.get("complete")]
            recordings.append(dict(id=f"{run.name}/{path.parent.name}", run_id=run.name, capture_id=path.parent.name,
                                   mode=manifest.get("metadata", {}).get("mode", "train"), status=manifest["status"],
                                   started=manifest["started"], ended=manifest.get("ended"),
                                   episodes=len(episodes), complete_episodes=len(good),
                                   successes=sum(x.get("outcome") == 1 for x in good), error=manifest.get("error"),
                                   synthetic=bool(manifest.get("metadata", {}).get("synthetic"))))
        return recordings

    def detail(self, recording_id):
        path = self.recording(recording_id)
        manifest = read_json(path / "manifest.json")
        episodes = [read_json(p) for p in sorted((path / "episodes").glob("*/episode.json"))]
        videos = {}
        for camera in (path / "video").glob("*") if (path / "video").exists() else []:
            if not camera.is_dir():
                continue
            segments = []
            for movie in sorted(camera.glob("*.mp4")):
                segments.append(dict(name=movie.name, bytes=movie.stat().st_size))
            videos[camera.name] = dict(segments=segments, index_exists=(camera / "frames.jsonl").is_file())
        return dict(id=recording_id, manifest=manifest, episodes=episodes, videos=videos)

    def video_file(self, recording_id, camera, filename):
        path = self.recording(recording_id)
        _identifier(camera)
        if not re.fullmatch(r"\d{6}\.mp4", filename):
            raise ValueError("Invalid video segment")
        result = (path / "video" / camera / filename).resolve()
        if not result.is_relative_to(path) or not result.is_file():
            raise ValueError("Video segment is unavailable")
        return result

    def timeline(self, recording_id, camera):
        _identifier(camera)
        path = self.recording(recording_id) / "video" / camera / "frames.jsonl"
        frames = []
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:
                        frame = json.loads(line)
                    except json.JSONDecodeError:
                        # A live writer may currently be appending this final line.
                        continue
                    frames.append({k: frame[k] for k in ("capture_id", "monotonic_ns", "frame", "fps", "segment")})
        return frames

    def export(self, recording_id, *, outcome="all", source="all", format="npz", episode_ids=None):
        from hilserl.episodes import _training_transition
        from hilserl.storage import read_step
        if outcome not in {"all", "success", "failure"} or source not in {"all", "human", "policy"}:
            raise ValueError("Invalid export filter")
        if format not in {"npz", "serl"}:
            raise ValueError("Export format must be npz or serl")
        path = self.recording(recording_id)
        manifest = read_json(path / "manifest.json")
        selected = []
        for meta in sorted((path / "episodes").glob("*/episode.json")):
            episode = read_json(meta)
            if not episode.get("complete") or episode.get("outcome") not in (0, 1):
                continue
            if outcome != "all" and (episode["outcome"] == 1) != (outcome == "success"):
                continue
            if episode_ids is not None and episode["id"] not in episode_ids:
                continue
            selected.append((meta.parent, episode))
        if not selected:
            raise ValueError("筛选结果没有已完成且已标注的 episode。")
        export_dir = self.root / "exports"
        export_dir.mkdir(exist_ok=True)
        name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{format}_{uuid.uuid4().hex[:6]}.zip"
        destination = export_dir / name
        partial = destination.with_suffix(".partial")
        exported = []
        try:
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                for directory, episode in selected:
                    transitions, indices = [], []
                    contract = _export_contract(manifest.get("metadata", {}), episode)
                    image_profile = _export_image_profile(manifest.get("metadata", {}), episode)
                    image_inferred_from_raw = all("image_profile" not in entry for entry in (manifest.get("metadata", {}), episode))
                    inferred_from_raw = "action_contract" not in manifest.get("metadata", {}) and "action_contract" not in episode
                    files = sorted(directory.glob("*.npz"))
                    if len(files) != episode["steps"]:
                        raise ValueError(f"Episode {episode['id']} 的步文件数量不一致，已取消导出。")
                    for file in files:
                        if shutil.disk_usage(export_dir).free < self.min_free_bytes:
                            raise OSError("导出达到磁盘保留容量下限；原始记录保留。")
                        raw = read_step(file)
                        if not raw.get("complete_transition", True) or raw.get("next_observations") is None:
                            raise ValueError(f"Episode {episode['id']} 包含不完整动作记录，已取消导出。")
                        if inferred_from_raw and not indices and file == files[0]:
                            contract = _export_contract(raw)
                        _export_contract({"action_contract": contract.name}, raw)
                        if image_inferred_from_raw and file == files[0]:
                            image_profile = _export_image_profile(raw)
                        _export_image_profile({"image_profile": image_profile},
                                              {"image_profile": raw.get("image_profile", FULL_FRAME)})
                        if contract.fixed_xyz and episode.get("verdict_source") != "human":
                            raise ValueError("fixed-xyz-v1 episode has no final human verdict; export cancelled")
                        if source != "all" and raw["source_action"] != source:
                            continue
                        indices.append(raw["episode_step"])
                        if format == "npz":
                            bundle.write(file, f"episodes/{episode['id']}/{file.name}", compress_type=zipfile.ZIP_STORED)
                        else:
                            terminal = raw["episode_step"] == episode["steps"] - 1
                            raw["termination_reason"] = episode["termination_reason"] if terminal else None
                            transitions.append(_training_transition(raw, terminal=terminal,
                                                outcome=episode["outcome"] if terminal else None,
                                                action_contract=contract))
                    if not indices:
                        continue
                    metadata = dict(episode, selected_step_indices=indices, source_filter=source,
                                    is_complete_episode_selection=len(indices) == episode["steps"],
                                    action_contract=contract.name,
                                    image_profile=image_profile,
                                    exported_action_dim=7 if format == "npz" else contract.learning_dim)
                    exported.append(metadata)
                    bundle.writestr(f"episodes/{episode['id']}/episode.json", json.dumps(metadata, ensure_ascii=False, indent=2))
                    if format == "serl":
                        # One episode at a time bounds export RAM; each transition keeps its original next_obs.
                        with bundle.open(f"episodes/{episode['id']}/transitions.pkl", "w", force_zip64=True) as output:
                            pickle.dump(transitions, output, protocol=pickle.HIGHEST_PROTOCOL)
                if not exported:
                    raise ValueError("所选 episode 中没有匹配动作来源的步骤。")
                index = dict(schema_version=1, format=format, recording_id=recording_id, exported=stamp(),
                             filters=dict(outcome=outcome, source=source, episode_ids=episode_ids), episodes=exported,
                             source_manifest=manifest, videos_included=False,
                             notes="有效零动作保留。来源筛选保留原始步索引与 next_observations，不拼接间隔，不加入训练信用补偿样本。")
                contracts = sorted({episode["action_contract"] for episode in exported})
                index.update(action_contracts=contracts,
                             image_profiles=sorted({episode["image_profile"] for episode in exported}),
                             action_representation="raw_device_7d" if format == "npz" else "learning_contract")
                if len(contracts) == 1:
                    index["action_contract"] = contracts[0]
                bundle.writestr("manifest.json", json.dumps(index, ensure_ascii=False, indent=2))
            partial.rename(destination)
        except BaseException:
            # Only this newly-created incomplete export is removed; archives/checkpoints are untouched.
            partial.unlink(missing_ok=True)
            raise
        return dict(file=name, path=str(destination), episodes=len(exported),
                    steps=sum(len(e["selected_step_indices"]) for e in exported), bytes=destination.stat().st_size)
