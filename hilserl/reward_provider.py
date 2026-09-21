"""Versioned, causal episode rewards; no robot, JAX or model imports here."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
import time
from typing import Callable

MODE = "robometer-episode"
PROTOCOL = "hilserl-robometer-prefix-v1"


def digest(value):
    """Hash values, not pickle memoization or ndarray sharing/layout."""
    import numpy as np
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays are not reward data")
            h.update(b"array" + str(item.dtype).encode() + repr(item.shape).encode())
            h.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            h.update(b"dict[")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError("Reward metadata keys must be strings")
                visit(key); visit(item[key])
            h.update(b"]")
        elif isinstance(item, (list, tuple)):
            h.update(b"list[")
            for child in item:
                visit(child)
            h.update(b"]")
        else:
            h.update(json.dumps(item, allow_nan=False, ensure_ascii=True).encode() + b";")

    visit(value)
    return h.hexdigest()


@dataclass(frozen=True)
class RewardSpec:
    model_id: str
    task: str = "Insert the plug into the socket."
    image_key: str = "side_policy"
    image_profile: str = "insert-front-roi160-v2"
    gamma: float = 0.98
    scale: float = 1.0
    max_frames: int = 8
    protocol: str = PROTOCOL

    def validate(self):
        if (not isinstance(self.model_id, str) or len(self.model_id) != 64
                or any(c not in "0123456789abcdef" for c in self.model_id)
                or not isinstance(self.task, str) or not self.task.strip() or self.image_key not in {"side_policy", "wrist_1"}
                or self.protocol != PROTOCOL or type(self.max_frames) is not int
                or not 2 <= self.max_frames <= 8):
            raise ValueError("Invalid RoboMeter identity, task, view or prefix contract")
        if not (type(self.gamma) in (int, float) and math.isfinite(self.gamma) and 0 < self.gamma <= 1
                and type(self.scale) in (int, float) and math.isfinite(self.scale) and self.scale >= 0):
            raise ValueError("Invalid reward gamma/scale")
        from hilserl.profile_names import PROFILE_NAMES
        if self.image_profile not in PROFILE_NAMES:
            raise ValueError("Unknown reward image profile")
        return self

    @property
    def sha256(self):
        return digest(asdict(self))

    @classmethod
    def from_env(cls):
        text = os.environ.get("HILSERL_REWARD_SPEC")
        return cls(**json.loads(text)).validate() if text else None


def image_from(obs, spec):
    import numpy as np
    image = np.asarray(obs[spec.image_key])
    from hilserl.image_profile import observation_image_schema
    expected = tuple(observation_image_schema(spec.image_profile)[spec.image_key]["shape"])
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(f"Reward image must be uint8 {expected}: {spec.image_key}")
    return np.ascontiguousarray(image[0])


def observation_queries(transitions, spec):
    """Keep actual next_obs; only reuse the matching adjacent obs boundary."""
    import numpy as np
    frames, pairs = [], []
    for item in transitions:
        obs, nxt = (image_from(item[key], spec) for key in ("observations", "next_observations"))
        if not frames or not np.array_equal(frames[-1], obs):
            frames.append(obs)
        before = len(frames) - 1
        frames.append(nxt)
        pairs.append((before, len(frames) - 1))
    return frames, pairs


def prefix_indices(query, maximum):
    import numpy as np
    return np.linspace(0, query, min(maximum, query + 1), dtype=int).tolist()


class RoboMeterClient:
    """Stateless prefix batches: retries cannot inherit another episode's cache."""
    def __init__(self, spec, url, *, batch_size=8, timeout=120.0, session=None):
        self.spec = spec.validate()
        if type(batch_size) is not int or not 1 <= batch_size <= 64 or not 0 < timeout < float("inf"):
            raise ValueError("Invalid reward batch/timeout")
        self.url, self.batch_size, self.timeout = url.rstrip("/"), batch_size, timeout
        if session is None:
            import requests
            session = requests.Session()
            session.trust_env = False
        self.session = session

    def health(self):
        response = self.session.get(self.url + "/health", timeout=min(self.timeout, 10))
        response.raise_for_status()
        value = response.json()
        if (value.get("protocol") != PROTOCOL or value.get("model_id") != self.spec.model_id
                or value.get("ready") is not True):
            raise ValueError("RoboMeter service identity/protocol/readiness mismatch")
        return value

    def score(self, transitions, *, check: Callable = lambda: None):
        import numpy as np
        check(); self.health(); check()
        started = time.monotonic()
        frames, pairs = observation_queries(transitions, self.spec)
        if not frames:
            raise ValueError("Cannot score an empty episode")
        scores, histories = [], []
        for start in range(0, len(frames), self.batch_size):
            check()
            queries = list(range(start, min(start + self.batch_size, len(frames))))
            samples = {}
            for i, query in enumerate(queries):
                selected = prefix_indices(query, self.spec.max_frames)
                histories.append(selected)
                samples[f"sample_{i}"] = np.stack([frames[j] for j in selected])
            metadata = dict(protocol=PROTOCOL, model_id=self.spec.model_id,
                            task=self.spec.task, query_ids=queries, image_key=self.spec.image_key)
            stream = io.BytesIO()
            np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **samples)
            response = self.session.post(self.url + "/score", data=stream.getvalue(),
                                         headers={"Content-Type": "application/x-npz"}, timeout=self.timeout)
            response.raise_for_status()
            value = response.json()
            if (value.get("model_id") != self.spec.model_id or value.get("protocol") != PROTOCOL
                    or value.get("query_ids") != queries):
                raise ValueError("Reward batch response identity/order mismatch")
            batch = np.asarray(value.get("scores"), dtype=float)
            if batch.shape != (len(queries),) or not np.isfinite(batch).all() or np.any((batch < 0) | (batch > 1)):
                raise ValueError("Reward service returned invalid progress scores")
            scores.extend(batch.tolist()); check()
        return dict(scores=scores, pairs=pairs, prefixes=histories,
                    image_sha256=[digest(image) for image in frames],
                    inference_seconds=time.monotonic() - started,
                    contract=asdict(self.spec), contract_sha256=self.spec.sha256)

    def close(self):
        self.session.close()


def relabel(transitions, scored, spec):
    """Human outcome remains authoritative, including negative terminal rewards."""
    import numpy as np
    if scored.get("contract_sha256") != spec.sha256:
        raise ValueError("Reward result contract mismatch")
    frames, pairs = observation_queries(transitions, spec)
    if ([list(p) for p in scored.get("pairs", [])] != [list(p) for p in pairs]
            or scored.get("image_sha256") != [digest(image) for image in frames]
            or scored.get("prefixes") != [prefix_indices(i, spec.max_frames) for i in range(len(frames))]):
        raise ValueError("Reward scores do not match episode images/prefixes")
    scores = np.asarray(scored["scores"], dtype=float)
    if scores.shape != (len(frames),) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid episode progress scores")
    result = copy.deepcopy(transitions)
    for item, (before, after) in zip(result, pairs):
        task_reward = float(item["rewards"])
        potential, next_potential = float(scores[before]), float(scores[after])
        shaping = spec.scale * (spec.gamma * float(item["masks"]) * next_potential - potential)
        item["rewards"] = task_reward + shaping
        item["infos"]["reward"] = dict(contract_sha256=spec.sha256, task_reward=task_reward,
            potential=potential, next_potential=next_potential, shaping=shaping,
            obs_image_sha256=scored["image_sha256"][before],
            next_image_sha256=scored["image_sha256"][after])
    return result


def validate_reward(item, spec):
    info = item["infos"]
    value = info.get("reward", {})
    if value.get("contract_sha256") != spec.sha256:
        raise ValueError("Replay reward version differs from this run")
    task, potential, nxt, shaping = (float(value[k]) for k in
                                   ("task_reward", "potential", "next_potential", "shaping"))
    if not all(math.isfinite(v) for v in (task, potential, nxt, shaping)):
        raise ValueError("Nonfinite reward metadata")
    if task not in (0, 1) or (task and not item["dones"]) or not 0 <= potential <= 1 or not 0 <= nxt <= 1:
        raise ValueError("Invalid task reward/potential")
    if bool(task) != bool(info.get("succeed")):
        raise ValueError("Task reward and human success disagree")
    expected = spec.scale * (spec.gamma * float(item["masks"]) * nxt - potential)
    if not math.isclose(shaping, expected, abs_tol=1e-6) or not math.isclose(float(item["rewards"]), task + expected, abs_tol=1e-6):
        raise ValueError("Reward formula mismatch")
    for field, key in (("observations", "obs_image_sha256"), ("next_observations", "next_image_sha256")):
        if value.get(key) != digest(image_from(item[field], spec)):
            raise ValueError("Replay reward image mismatch")
