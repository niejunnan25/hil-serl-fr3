"""Versioned camera geometry shared by live observations and seed reconstruction."""
from __future__ import annotations

import copy
import cv2
import numpy as np

from hilserl.profile_names import (FRONT_ROI160, FRONT_ROI160_V2, FULL_FRAME, INSERT_ROI, POLICY_SIZES,
                                   PROFILE_NAMES)

__all__ = ["FRONT_ROI160", "FRONT_SIDE_CROP", "FULL_FRAME", "INSERT_ROI", "PROFILE_NAMES",
           "crop_image", "get_image_profile", "observation_image_schema", "preprocess_image"]

CAMERAS = ("wrist_1", "side_policy", "side_classifier")

# Camera geometry belongs to a physical mount. The `insert-roi*` family was
# calibrated for the side mount of the external ZED (serial 36276705). Moving
# that camera invalidates its crop, so the relocated mount gets its own profile
# family (`insert-front-roi*`) rather than edited numbers under the existing
# name. The wrist mount (serial 13132609) did not move, so `wrist_1` keeps the
# side-mount crop in both families.
FRONT_SIDE_CROP = [460, 280, 440, 440]

# The reward classifier is retrained for every mount, so its resize must match
# the pixels written by scripts/collect_classifier_images.py (INTER_AREA). The
# historical side-mount profile keeps INTER_LINEAR for its already-trained
# model; changing it would change that classifier's live input.
FRONT_CLASSIFIER_INTERPOLATION = "area"


def get_image_profile(name=FULL_FRAME):
    profiles = {
        FULL_FRAME: dict(name=FULL_FRAME, raw_size=[1280, 720], color="RGB", cameras={
            key: dict(crop_xywh=None, size=[128, 128], interpolation="linear") for key in CAMERAS}),
    }
    for size in POLICY_SIZES:
        profile_name = f"insert-roi{size}-v1"
        profiles[profile_name] = dict(name=profile_name, raw_size=[1280, 720], color="RGB", cameras={
            "wrist_1": dict(crop_xywh=[360, 0, 720, 720], size=[size, size], interpolation="area"),
            "side_policy": dict(crop_xywh=[180, 280, 400, 400], size=[size, size], interpolation="area"),
            # The existing reward classifier retains its own trained view.
            "side_classifier": dict(crop_xywh=None, size=[128, 128], interpolation="linear"),
        })
        front_name = f"insert-front-roi{size}-v1"
        profiles[front_name] = dict(name=front_name, raw_size=[1280, 720], color="RGB", cameras={
            "wrist_1": dict(crop_xywh=[360, 0, 720, 720], size=[size, size], interpolation="area"),
            "side_policy": dict(crop_xywh=list(FRONT_SIDE_CROP), size=[size, size], interpolation="area"),
            "side_classifier": dict(crop_xywh=None, size=[128, 128],
                                    interpolation=FRONT_CLASSIFIER_INTERPOLATION),
        })
    # The 2026-09-21 mount was physically adjusted. Keep the broad verified
    # crops, but bind new recordings/models to the new physical scene.
    profiles[FRONT_ROI160_V2] = copy.deepcopy(profiles[FRONT_ROI160])
    profiles[FRONT_ROI160_V2]["name"] = FRONT_ROI160_V2
    if isinstance(name, dict):
        canonical = get_image_profile(name.get("name"))
        if name != canonical:
            raise ValueError("Image profile differs from its versioned geometry")
        return canonical
    if not isinstance(name, str) or name not in profiles:
        raise ValueError(f"Unknown image profile: {name}")
    return copy.deepcopy(profiles[name])


def observation_image_schema(profile=FULL_FRAME):
    return {key: dict(shape=[1, spec["size"][1], spec["size"][0], 3], dtype="uint8")
            for key, spec in get_image_profile(profile)["cameras"].items()}


def crop_image(frame_bgr, camera, profile=FULL_FRAME):
    profile = get_image_profile(profile)
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Camera frame must be BGR uint8 HxWx3")
    if camera not in profile["cameras"]:
        raise ValueError(f"Unknown camera: {camera}")
    if profile["name"] != FULL_FRAME and list(frame_bgr.shape[:2][::-1]) != profile["raw_size"]:
        raise ValueError("Camera dimensions differ from calibrated crop geometry")
    crop = profile["cameras"][camera]["crop_xywh"]
    if crop is None:
        return frame_bgr
    x, y, width, height = crop
    return frame_bgr[y:y+height, x:x+width]


def preprocess_image(frame_bgr, camera, profile=FULL_FRAME):
    profile = get_image_profile(profile)
    crop = crop_image(frame_bgr, camera, profile)
    spec = profile["cameras"][camera]
    interpolation = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}[spec["interpolation"]]
    return cv2.resize(crop, tuple(spec["size"]), interpolation=interpolation)[..., ::-1].copy()
