from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
import time
from typing import Any
from uuid import uuid4

import numpy as np

from fr3_hil_bridge.plug_bridge import validate_action_7d


DEMO_SCHEMA_VERSION = "hilserl_executed_action_demo_v1"


@dataclass(frozen=True)
class DemoMetadata:
    task: str
    episode_id: str
    action_schema_version: str = "xbox_7d_v1"
    demo_schema_version: str = DEMO_SCHEMA_VERSION
    control_frame: str = "base"
    fixture_axis_base: tuple[float, float, float] = (0.0, 0.0, -1.0)
    source: str = "xbox_no_motion_dry_run"
    created_unix_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fixture_axis_base"] = list(self.fixture_axis_base)
        return data


def _action_array(action: Any) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    validation = validate_action_7d(arr)
    if not validation.valid:
        raise ValueError(validation.reason)
    return arr


def make_transition(
    observations: dict[str, Any],
    action: Any,
    next_observations: dict[str, Any],
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    executed = _action_array(action)
    info = dict(info or {})
    info["executed_action_norm"] = executed.copy()
    return {
        "observations": observations,
        "actions": executed,
        "next_observations": next_observations,
        "rewards": np.float32(reward),
        "masks": np.float32(1.0 - float(done)),
        "dones": bool(done),
        "infos": info,
    }


def make_hil_transition(
    observations: dict[str, Any],
    policy_action: Any,
    human_action: Any | None,
    intervention: bool,
    next_observations: dict[str, Any],
    reward: float,
    done: bool,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = _action_array(policy_action)
    human = _action_array(human_action) if human_action is not None else None
    executed = human if intervention and human is not None else policy
    info = dict(info or {})
    info["policy_action_norm"] = policy.copy()
    info["human_action_norm"] = None if human is None else human.copy()
    if intervention:
        info["intervene_action"] = executed.copy()
    return make_transition(observations, executed, next_observations, reward, done, info)


def validate_transition(transition: dict[str, Any]) -> dict[str, Any]:
    required = {"observations", "actions", "next_observations", "rewards", "masks", "dones", "infos"}
    missing = sorted(required - set(transition))
    if missing:
        return {"ok": False, "reason": "MISSING_TRANSITION_KEYS", "missing": missing}
    action = np.asarray(transition["actions"], dtype=np.float32)
    validation = validate_action_7d(action)
    if not validation.valid:
        return {"ok": False, "reason": validation.reason, "action_shape": list(action.shape)}
    infos = transition.get("infos") or {}
    executed = infos.get("executed_action_norm")
    if executed is None or not np.allclose(np.asarray(executed, dtype=np.float32), action):
        return {"ok": False, "reason": "EXECUTED_ACTION_INFO_MISMATCH"}
    if "intervene_action" in infos and not np.allclose(np.asarray(infos["intervene_action"], dtype=np.float32), action):
        return {"ok": False, "reason": "INTERVENE_ACTION_MISMATCH"}
    return {
        "ok": True,
        "reason": "TRANSITION_ACCEPTED",
        "action_shape": list(action.shape),
        "actions_are_executed_actions": True,
        "has_intervention": "intervene_action" in infos,
    }


class EpisodeWriter:
    def __init__(self, root: str | Path, metadata: DemoMetadata):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.transitions: list[dict[str, Any]] = []
        self.sidecar_path = self.root / "controller_sidecar.jsonl"
        self._sidecar = self.sidecar_path.open("w", encoding="utf-8")

    @classmethod
    def create(cls, root: str | Path, task: str = "plug_zed_insertion") -> "EpisodeWriter":
        episode_id = f"episode_{uuid4().hex}"
        metadata = DemoMetadata(task=task, episode_id=episode_id, created_unix_s=time.time())
        return cls(Path(root) / episode_id, metadata)

    def append(self, transition: dict[str, Any], sidecar: dict[str, Any] | None = None) -> None:
        result = validate_transition(transition)
        if not result["ok"]:
            raise ValueError(result["reason"])
        self.transitions.append(transition)
        if sidecar is not None:
            self._sidecar.write(json.dumps(sidecar, sort_keys=True) + "\n")
            self._sidecar.flush()

    def close(self) -> dict[str, Any]:
        self._sidecar.close()
        with (self.root / "transitions.pkl").open("wb") as handle:
            pickle.dump(self.transitions, handle)
        metadata = self.metadata.to_dict()
        metadata["transition_count"] = len(self.transitions)
        with (self.root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
        return {
            "episode_dir": str(self.root),
            "metadata": metadata,
            "transitions": str(self.root / "transitions.pkl"),
            "sidecar": str(self.sidecar_path),
        }
