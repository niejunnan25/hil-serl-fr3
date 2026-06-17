#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


REQUIRED_SOURCE_KEYS = {
    "wrist": "observation/wrist_image",
    "front": "observation/image",
}


def fail(reason: str, detail: object | None = None) -> None:
    payload = {"success": False, "reason": reason}
    if detail is not None:
        payload["detail"] = detail
    raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))


def finite_list(values: object, length: int) -> bool:
    return isinstance(values, list) and len(values) == length and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--no-camera-capture", action="store_true")
    parser.add_argument("--no-frame-read", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--no-fci", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    if not (args.dry_run and args.manifest_only and args.no_camera_capture and args.no_frame_read and args.no_robot and args.no_fci and args.no_gripper):
        fail("missing_no_capture_guards")
    manifest = json.loads(Path(args.manifest_file).read_text(encoding="utf-8"))
    views = manifest.get("camera_views")
    if not isinstance(views, list):
        fail("missing_camera_views")
    by_role = {view.get("role"): view for view in views if isinstance(view, dict)}
    if set(by_role) != {"wrist", "front"}:
        fail("camera_roles_bad", sorted(by_role))
    for role, source_key in REQUIRED_SOURCE_KEYS.items():
        view = by_role[role]
        if view.get("source_key") != source_key:
            fail("source_key_bad", {role: view.get("source_key")})
        if view.get("output_size") != [128, 128]:
            fail("output_size_bad", {role: view.get("output_size")})
        crop = view.get("crop_xywh_norm")
        if not finite_list(crop, 4) or crop[2] <= 0 or crop[3] <= 0 or crop[0] < 0 or crop[1] < 0 or crop[0] + crop[2] > 1.0 or crop[1] + crop[3] > 1.0:
            fail("crop_bad", {role: crop})
        if view.get("sample_frame_path") != "":
            fail("sample_frame_path_present", {role: view.get("sample_frame_path")})
    fixture = manifest.get("fixture", {})
    if fixture.get("fixture_frame") != "fixture_plug_socket_v1":
        fail("fixture_frame_bad", fixture)
    if not finite_list(fixture.get("fr3_base_to_fixture_xyz_rpy"), 6):
        fail("fixture_transform_bad", fixture)
    mn = fixture.get("safe_workspace_xyz_min")
    mx = fixture.get("safe_workspace_xyz_max")
    if not finite_list(mn, 3) or not finite_list(mx, 3) or any(hi <= lo for lo, hi in zip(mn, mx)):
        fail("workspace_bounds_bad", fixture)
    result = {
        "success": True,
        "process_name": "hilserl_camera_fixture_no_capture",
        "roles": sorted(by_role),
        "source_keys": REQUIRED_SOURCE_KEYS,
        "output_size": [128, 128],
        "fixture_frame": fixture["fixture_frame"],
        "manifest_checked": True,
        "calibration_runtime_executed": True,
        "camera_capture": False,
        "frame_read": False,
        "fr3_connection": False,
        "fci_state_read": False,
        "motion_command": False,
        "gripper_command": False,
        "fixture_contact": False,
        "policy_rollout": False,
        "remote_network_call": False,
        "no_camera_capture": True,
        "no_frame_read": True,
        "no_fr3_connection": True,
        "no_fci_state_read": True,
        "no_motion_command": True,
        "no_gripper_command": True,
        "no_fixture_contact": True,
        "no_policy_rollout": True,
        "no_droid_dependency": True,
        "timestamp": time.time(),
    }
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_file).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
