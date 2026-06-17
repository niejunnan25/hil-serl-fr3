from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import cv2
import numpy as np


ZED_IMAGE_KEYS = ["zed_left", "zed_right"]


@dataclass(frozen=True)
class ImageValidation:
    valid: bool
    reason: str
    key: str | None
    shape: list[int] | None
    dtype: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _shape(image: object) -> list[int] | None:
    try:
        return list(np.asarray(image).shape)
    except Exception:
        return None


def validate_uint8_hwc(image: object, key: str | None = None) -> ImageValidation:
    arr = np.asarray(image)
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    if arr.dtype != np.uint8:
        return ImageValidation(False, "REJECT_IMAGE_DTYPE", key, shape, dtype)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return ImageValidation(False, "REJECT_IMAGE_NOT_HWC3", key, shape, dtype)
    return ImageValidation(True, "ACCEPT_UINT8_HWC3", key, shape, dtype)


def validate_expected_keys(images: Mapping[str, object], expected_keys: list[str]) -> dict[str, object]:
    missing = [key for key in expected_keys if key not in images]
    if missing:
        return {"valid": False, "reason": "REJECT_MISSING_IMAGE_KEY", "missing": missing}
    return {"valid": True, "reason": "ACCEPT_IMAGE_KEYS", "keys": list(expected_keys)}


def validate_hilserl_image(image: object, key: str | None = None) -> ImageValidation:
    base = validate_uint8_hwc(image, key=key)
    if not base.valid:
        return base
    arr = np.asarray(image)
    if arr.shape != (128, 128, 3):
        return ImageValidation(False, "REJECT_HILSERL_IMAGE_SHAPE", key, list(arr.shape), str(arr.dtype))
    return ImageValidation(True, "ACCEPT_HILSERL_128_IMAGE", key, list(arr.shape), str(arr.dtype))


def validate_hilserl_images(images: Mapping[str, object], expected_keys: list[str]) -> dict[str, object]:
    key_result = validate_expected_keys(images, expected_keys)
    if key_result.get("valid") is not True:
        return key_result
    per_key = {key: validate_hilserl_image(images[key], key=key).to_dict() for key in expected_keys}
    rejected = {key: result for key, result in per_key.items() if result["valid"] is not True}
    if rejected:
        return {"valid": False, "reason": "REJECT_IMAGE_VALUE", "rejected": rejected, "per_key": per_key}
    return {"valid": True, "reason": "ACCEPT_HILSERL_IMAGES", "keys": list(expected_keys), "per_key": per_key}


def validate_zed_observation(images: Mapping[str, object]) -> dict[str, object]:
    return validate_hilserl_images(images, ZED_IMAGE_KEYS)


def resize_with_pad_224(image: object) -> np.ndarray:
    base = validate_uint8_hwc(image)
    if not base.valid:
        raise ValueError(base.reason)
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    scale = min(224 / float(height), 224 / float(width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = cv2.resize(arr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    top = (224 - new_height) // 2
    left = (224 - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas
