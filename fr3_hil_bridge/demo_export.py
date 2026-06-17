from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np

from fr3_hil_bridge.demo_recording import validate_transition


EXPORT_SCHEMA_VERSION = "plug_zed_lerobot_sidecar_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _obs_summary(obs: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in obs.items():
        arr = np.asarray(value)
        summary[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return summary


def export_episode_to_lerobot_sidecar(episode_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    episode = Path(episode_dir)
    transitions_path = episode / "transitions.pkl"
    metadata_path = episode / "metadata.json"
    if not transitions_path.exists():
        raise FileNotFoundError(f"missing transitions: {transitions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata: {metadata_path}")

    with transitions_path.open("rb") as handle:
        transitions = pickle.load(handle)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    out_dir = Path(output_dir) if output_dir is not None else episode
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / "lerobot_sidecar.jsonl"
    dataset_info = out_dir / "lerobot_dataset_info.json"

    rows = []
    with sidecar.open("w", encoding="utf-8") as handle:
        for idx, transition in enumerate(transitions):
            validation = validate_transition(transition)
            if validation.get("ok") is not True:
                raise ValueError(f"transition {idx} invalid: {validation}")
            infos = transition.get("infos") or {}
            row = {
                "episode_id": metadata.get("episode_id"),
                "frame_index": idx,
                "timestamp": idx,
                "observation_summary": _obs_summary(transition["observations"]),
                "action": _jsonable(transition["actions"]),
                "reward": _jsonable(transition["rewards"]),
                "done": bool(transition["dones"]),
                "info": _jsonable({k: v for k, v in infos.items() if k != "teleop"}),
                "complementary_data": {
                    "teleop": _jsonable(infos.get("teleop", {})),
                    "policy_action_norm": _jsonable(infos.get("policy_action_norm")),
                    "human_action_norm": _jsonable(infos.get("human_action_norm")),
                    "executed_action_norm": _jsonable(infos.get("executed_action_norm")),
                },
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    info = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "episode_id": metadata.get("episode_id"),
        "task": metadata.get("task"),
        "source_demo_schema_version": metadata.get("demo_schema_version"),
        "action_schema_version": metadata.get("action_schema_version"),
        "transition_count": len(rows),
        "primary_format": "HIL-SERL transition pickle",
        "export_format": "LeRobot-compatible JSONL sidecar",
        "video_written": False,
        "parquet_written": False,
        "motion_command": False,
        "gripper_command": False,
        "droid_mutation": False,
        "openpi_dependency": False,
    }
    dataset_info.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    return {"sidecar": str(sidecar), "dataset_info": str(dataset_info), **info}


def validate_lerobot_sidecar(path: str | Path, expected_count: int | None = None) -> dict[str, Any]:
    sidecar = Path(path)
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected_count is not None and len(rows) != expected_count:
        return {"ok": False, "reason": "ROW_COUNT_MISMATCH", "row_count": len(rows), "expected_count": expected_count}
    for idx, row in enumerate(rows):
        if len(row.get("action", [])) != 7:
            return {"ok": False, "reason": "ACTION_SHAPE_MISMATCH", "row": idx}
        if "complementary_data" not in row:
            return {"ok": False, "reason": "MISSING_COMPLEMENTARY_DATA", "row": idx}
    return {"ok": True, "reason": "LEROBOT_SIDECAR_ACCEPTED", "row_count": len(rows)}
