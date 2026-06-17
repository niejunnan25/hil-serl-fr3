#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


sys.dont_write_bytecode = True

APPROVAL_PHRASE = "APPROVE PLUG-ZED CAMERA FIXTURE CAPTURE NO-MOTION"
READY_STATUS = "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME_SCAFFOLD_READY"
ACCEPT_STATUS = "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME_PROVEN"
BLOCK_STATUS = "BLOCK_PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME"
CAPTURE_ROOT = Path("/home/robot/serl_projects/hil-serl-fr3/experiments/plug_zed_insertion/calibration/zed_fixture_capture")
OBSERVATION_SCHEMA_HASH = "plugzedobs:v1:zed_left_zed_right:fci_gripper_state"
ACTION_SCHEMA_HASH = "hilserl7d:v1:cartesian_delta_rpy_gripper"
REQUIRED_PRECONDITIONS = {
    "zed_camera_service_runtime": "ZED_CAMERA_SERVICE_RUNTIME_PROVEN",
    "plug_zed_remote_source_contract": "PLUG_ZED_REMOTE_SOURCE_CONTRACT_PROVEN",
    "plug_zed_camera_gate_alignment_contract": "PLUG_ZED_CAMERA_GATE_ALIGNMENT_CONTRACT_PROVEN",
    "plug_zed_camera_fixture_capture_result_contract": "PROVEN_FOR_STATIC_ZED_RESULT_CONTRACT",
    "plug_zed_camera_fixture_capture_runtime_scaffold_deployed": "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME_SCAFFOLD_DEPLOYED",
}
RESIDUE_PATTERNS = (
    "zed_camera_service.py",
    "plug_zed_camera_fixture_capture_runtime.py",
    "franka_server",
    "polymetis",
    "RobotEnv",
)


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


def common_payload() -> dict:
    return {
        "approval_phrase_required": APPROVAL_PHRASE,
        "capture_root": str(CAPTURE_ROOT),
        "required_roles": ["zed_left", "zed_right"],
        "required_source_keys": {"zed_left": "zed_left", "zed_right": "zed_right"},
        "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
        "action_schema_hash": ACTION_SCHEMA_HASH,
        "required_precondition_statuses": REQUIRED_PRECONDITIONS,
        "runtime_execution_performed": False,
        "camera_api_imported": False,
        "camera_source_open": False,
        "sample_frame_capture": False,
        "frame_read": False,
        "fr3_connection": False,
        "fci_state_read": False,
        "motion_command": False,
        "gripper_command": False,
        "policy_rollout": False,
        "checkpoint_promotion": False,
        "reward_label_collection": False,
        "classifier_training": False,
        "demo_collection": False,
        "droid_mutation": False,
        "openpi_dependency": False,
        "remote_training_side_effect": False,
        "starts_socket": False,
        "no_ssh": True,
    }


def reject(code: str, detail: object | None = None) -> None:
    payload = common_payload()
    payload.update({
        "ok": False,
        "status": BLOCK_STATUS,
        "reject_code": code,
    })
    if detail is not None:
        payload["detail"] = detail
    emit(payload, 64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_residue(*, current_pid: int | None = None) -> dict:
    command = ["ps", "-Ao", "pid=,ppid=,command="]
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    matches: list[str] = []
    excluded_pids = {pid for pid in (current_pid, os.getppid()) if pid is not None}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=2)
        try:
            pid = int(parts[0])
        except (IndexError, ValueError):
            pid = None
        if pid in excluded_pids:
            continue
        if any(pattern in stripped for pattern in RESIDUE_PATTERNS):
            matches.append(stripped)
    return {
        "command": " ".join(command),
        "patterns": list(RESIDUE_PATTERNS),
        "returncode": result.returncode,
        "stdout": "\n".join(matches),
        "stderr": result.stderr,
        "empty": result.returncode == 0 and not matches and not result.stderr.strip(),
    }


def write_role_frame(*, role: str, source_key: str, serial: str, image, run_dir: Path, timestamp: float) -> dict:
    import cv2

    raw_height, raw_width = image.shape[:2]
    output = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{role}_frame_000001.png"
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"FRAME_WRITE_FAILED: {path}")
    return {
        "role": role,
        "serial": serial,
        "source_key": source_key,
        "captured_frame_path": str(path),
        "sha256": sha256_file(path),
        "raw_size": [raw_width, raw_height],
        "crop_xywh": [0, 0, raw_width, raw_height],
        "output_size": [128, 128],
        "dtype": str(output.dtype),
        "capture_timestamp": timestamp,
    }


def capture_once(args: argparse.Namespace) -> dict:
    preflight_scan = scan_residue(current_pid=os.getpid())
    if not preflight_scan["empty"]:
        reject("PREFLIGHT_RESIDUE_NOT_EMPTY", preflight_scan)

    import cv2
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER

    camera = sl.Camera()
    opened = camera.open(init)
    if opened != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED_OPEN_FAILED: {opened}")

    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    right = sl.Mat()
    run_id = args.run_id or time.strftime("plug_zed_camera_capture_%Y%m%dT%H%M%S%z")
    run_dir = CAPTURE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    camera_closed = False
    try:
        for _ in range(max(0, args.warmup_frames)):
            camera.grab(runtime)
        grabbed = camera.grab(runtime)
        if grabbed != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED_GRAB_FAILED: {grabbed}")
        camera.retrieve_image(left, sl.VIEW.LEFT)
        camera.retrieve_image(right, sl.VIEW.RIGHT)
        left_bgr = cv2.cvtColor(left.get_data(), cv2.COLOR_BGRA2BGR)
        right_bgr = cv2.cvtColor(right.get_data(), cv2.COLOR_BGRA2BGR)
        info = camera.get_camera_information()
        serial = f"ZED_2I_{int(info.serial_number)}"
        timestamp = time.time()
        frames = [
            write_role_frame(role="zed_left", source_key="zed_left", serial=serial, image=left_bgr, run_dir=run_dir, timestamp=timestamp),
            write_role_frame(role="zed_right", source_key="zed_right", serial=serial, image=right_bgr, run_dir=run_dir, timestamp=timestamp),
        ]
    finally:
        camera.close()
        camera_closed = True

    postflight_scan = scan_residue(current_pid=os.getpid())
    if not postflight_scan["empty"]:
        raise RuntimeError(f"POSTFLIGHT_RESIDUE_NOT_EMPTY: {postflight_scan}")
    result = {
        "approval_phrase": APPROVAL_PHRASE,
        "run_id": run_id,
        "capture_root": str(CAPTURE_ROOT),
        "status": ACCEPT_STATUS,
        "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
        "action_schema_hash": ACTION_SCHEMA_HASH,
        "frames": frames,
        "fixture": {
            "fixture_frame": args.fixture_frame,
            "fixture_id": args.fixture_id,
        },
        "precondition_statuses": dict(REQUIRED_PRECONDITIONS),
        "preflight_residue_empty": preflight_scan["empty"],
        "preflight_scan": preflight_scan,
        "postflight_residue_empty": postflight_scan["empty"],
        "postflight_scan": postflight_scan,
        "camera_closed_before_postflight_scan": camera_closed,
        "runtime_claims": {
            "camera_source_open": True,
            "sample_frame_capture": True,
            "frame_read": True,
            "camera_closed_before_postflight_scan": camera_closed,
            "fr3_connection": False,
            "fci_state_read": False,
            "motion_command": False,
            "gripper_command": False,
            "fixture_contact": False,
            "policy_rollout": False,
            "live_checkpoint_promotion": False,
            "reward_label_collection": False,
            "classifier_training": False,
            "demo_collection": False,
            "droid_mutation": False,
            "openpi_dependency": False,
            "remote_training_side_effect": False,
        },
    }
    result_path = run_dir / "plug_zed_camera_fixture_capture_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approval-gated Plug-ZED fixture camera capture runtime scaffold.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-residue-scan", action="store_true")
    parser.add_argument("--approval-phrase", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--fixture-frame", default="fixture_plug_socket_v1")
    parser.add_argument("--fixture-id", default="bench-plug-fixture-001")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.self_check:
        payload = common_payload()
        payload.update({
            "ok": True,
            "status": READY_STATUS,
            "future_accept_status": ACCEPT_STATUS,
            "imports_camera_api_on_self_check": False,
            "exact_approval_required_before_camera_import": True,
            "writes_formal_result_json": True,
            "writes_captured_frame_path": True,
            "computes_sha256": True,
            "records_raw_size": True,
            "records_crop_xywh": True,
            "records_output_size": True,
            "records_dtype": True,
            "runs_residue_scan": True,
            "camera_closed_before_postflight_scan": True,
            "residue_scan_patterns": list(RESIDUE_PATTERNS),
        })
        emit(payload)
    if args.self_check_residue_scan:
        scan = scan_residue(current_pid=os.getpid())
        payload = common_payload()
        payload.update({
            "ok": scan["empty"],
            "status": "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RESIDUE_SCAN_SELF_CHECKED" if scan["empty"] else BLOCK_STATUS,
            "residue_scan": scan,
            "runtime_execution_performed": False,
            "camera_api_imported": False,
            "camera_source_open": False,
            "sample_frame_capture": False,
            "frame_read": False,
        })
        emit(payload, 0 if scan["empty"] else 64)

    if args.approval_phrase != APPROVAL_PHRASE:
        reject("APPROVAL_PHRASE_REQUIRED")
    if args.fixture_frame != "fixture_plug_socket_v1":
        reject("FIXTURE_FRAME_MISMATCH", args.fixture_frame)
    if not args.fixture_id.strip():
        reject("FIXTURE_ID_REQUIRED")

    try:
        result = capture_once(args)
    except Exception as exc:
        reject("PLUG_ZED_CAMERA_CAPTURE_EXCEPTION", str(exc))
    emit(result)


if __name__ == "__main__":
    main()
