from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import requests

from fr3_hil_bridge.image_adapter import ZED_IMAGE_KEYS, validate_zed_observation


DEFAULT_ZED_SERVICE_URL = "http://127.0.0.1:54819"


@dataclass(frozen=True)
class ZedObservationResult:
    ok: bool
    status: str
    base_url: str
    image_keys: list[str]
    validation: dict[str, Any]
    image_payloads_b64: dict[str, str]
    camera: dict[str, Any]
    runtime_claims: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_png(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    encoded = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("REJECT_BAD_PNG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fetch_zed_observation(base_url: str = DEFAULT_ZED_SERVICE_URL, timeout: float = 2.0) -> ZedObservationResult:
    normalized = base_url.rstrip("/")
    health = requests.get(normalized + "/healthz", timeout=timeout)
    health.raise_for_status()
    health_payload = health.json()
    if health_payload.get("ok") is not True:
        raise RuntimeError(f"ZED_SERVICE_HEALTH_NOT_OK {health_payload}")

    response = requests.get(normalized + "/frames", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if payload.get("ok") is not True:
        raise RuntimeError(f"ZED_SERVICE_FRAME_NOT_OK {payload}")

    frames = payload.get("frames")
    if not isinstance(frames, dict):
        raise RuntimeError("ZED_SERVICE_MISSING_FRAMES")
    image_payloads_b64 = {key: frames[key]["png_b64"] for key in ZED_IMAGE_KEYS}
    images = {key: _decode_png(image_payloads_b64[key]) for key in ZED_IMAGE_KEYS}
    validation = validate_zed_observation(images)
    return ZedObservationResult(
        ok=validation.get("valid") is True,
        status="ZED_OBSERVATION_FETCHED" if validation.get("valid") is True else "ZED_OBSERVATION_REJECTED",
        base_url=normalized,
        image_keys=list(ZED_IMAGE_KEYS),
        validation=validation,
        image_payloads_b64=image_payloads_b64,
        camera=health_payload.get("camera", {}),
        runtime_claims=payload.get("runtime_claims", {}),
    )


def synthetic_zed_observation() -> dict[str, np.ndarray]:
    left = np.zeros((128, 128, 3), dtype=np.uint8)
    right = np.zeros((128, 128, 3), dtype=np.uint8)
    left[:, :, 0] = 64
    left[:, :, 1] = np.arange(128, dtype=np.uint8)[None, :]
    right[:, :, 2] = 96
    right[:, :, 1] = np.arange(128, dtype=np.uint8)[:, None]
    return {"zed_left": left, "zed_right": right}
