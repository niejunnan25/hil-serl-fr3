#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


sys.dont_write_bytecode = True

APPROVAL_PHRASE = "APPROVE PLUG-ZED FIXTURE AXIS CALIBRATION NO-MOTION"
BROAD_APPROVALS = {
    "批准一切权限",
    "批准一切真机权限",
    "全部都批准",
    "我批准一切",
    "approve all",
    "all approved",
}
READY_STATUS = "PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_RUNTIME_READY"
READY_NOT_EXECUTED_STATUS = "PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_READY_RUNTIME_NOT_EXECUTED"
ACCEPT_STATUS = "PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_PROVEN"
BLOCK_STATUS = "BLOCK_PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_RUNTIME"
SYMBOLIC_AXIS = "fixture_z_negative"
FIXTURE_FRAME = "fixture_plug_socket_v1"
ACTIVE_TASK = "plug_zed_insertion"
MAX_ANGULAR_UNCERTAINTY_RAD = 0.10
RESULT_ROOT = Path("/sda/fzt/hilserl-fr3/plug_zed_insertion")
DEFAULT_RUN_ROOT = RESULT_ROOT / "calibration/fixture_axis"
REQUIRED_PRECONDITIONS = {
    "plug_zed_formal_camera_capture_runtime": "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME_PROVEN",
    "fci_gripper_state_only_runtime": "FR3_FCI_GRIPPER_STATE_ONLY_RUNTIME_PROVEN",
    "plug_zed_remote_source_contract": "PLUG_ZED_REMOTE_SOURCE_CONTRACT_PROVEN",
    "contact_micro_motion_live_control_body": "PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY_SOURCE_COMPILED_AND_SELF_CHECKED",
}
ALLOWED_AXIS_SOURCES = {
    "measured_fixture_transform",
    "human_measured_fixture_axis",
    "calibrated_camera_fixture_transform",
}
FORBIDDEN_AXIS_SOURCES = {
    "assumed_default",
    "guessed_from_name",
    "camera_image_only_without_scale",
    "old_front_wrist_only_gate",
    "symbolic_fixture_axis_only",
    "synthetic_validator_axis",
}
RESIDUE_PATTERNS = (
    "plug_zed_fixture_axis_calibration_runtime.py",
    "plug_zed_contact_micro_motion_live_control",
    "plug_zed_contact_micro_motion_runtime.py",
    "plug_zed_camera_fixture_capture_runtime.py",
    "zed_camera_service.py",
    "franka_server",
    "polymetis",
    "RobotEnv",
)
OLD_FRONT_WRIST_MARKERS = (
    "plug_insertion_v1",
    "camera_fixture_no_capture",
    "observation/wrist_image",
    "observation/image",
    "STATIC_WRIST_SERIAL_PLACEHOLDER",
    "STATIC_FRONT_SERIAL_PLACEHOLDER",
)


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


def common_payload() -> dict:
    return {
        "approval_phrase_required": APPROVAL_PHRASE,
        "ready_status": READY_STATUS,
        "ready_not_executed_status": READY_NOT_EXECUTED_STATUS,
        "future_accept_status": ACCEPT_STATUS,
        "block_status": BLOCK_STATUS,
        "active_task": ACTIVE_TASK,
        "fixture_frame": FIXTURE_FRAME,
        "symbolic_axis": SYMBOLIC_AXIS,
        "result_root": str(RESULT_ROOT),
        "allowed_axis_sources": sorted(ALLOWED_AXIS_SOURCES),
        "forbidden_axis_sources": sorted(FORBIDDEN_AXIS_SOURCES),
        "max_angular_uncertainty_rad": MAX_ANGULAR_UNCERTAINTY_RAD,
        "required_preconditions": dict(REQUIRED_PRECONDITIONS),
        "runtime_execution_performed": False,
        "fr3_connection": False,
        "fci_state_read": False,
        "motion_command": False,
        "gripper_command": False,
        "fixture_contact": False,
        "live_micro_motion": False,
        "policy_rollout": False,
        "remote_training": False,
        "droid_mutation": False,
        "starts_socket": False,
        "no_ssh": True,
    }


def reject(code: str, detail: object | None = None) -> None:
    payload = common_payload()
    payload.update({
        "ok": False,
        "status": BLOCK_STATUS,
        "reject_code": code,
        "safe_to_claim_100_percent": False,
        "overall_goal_complete": False,
    })
    if detail is not None:
        payload["detail"] = detail
    emit(payload, 64)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def normalize_axis(axis: list[float]) -> tuple[list[float], float]:
    if len(axis) != 3 or not all(finite(value) for value in axis):
        reject("BAD_AXIS_VECTOR", axis)
    norm = math.sqrt(sum(float(value) ** 2 for value in axis))
    if not math.isfinite(norm) or norm <= 0:
        reject("BAD_AXIS_VECTOR", {"axis": axis, "norm": norm})
    normalized = [float(value) / norm for value in axis]
    return normalized, norm


def check_unit_axis(axis: list[float]) -> tuple[list[float], float]:
    normalized, original_norm = normalize_axis(axis)
    if not (0.999 <= original_norm <= 1.001):
        reject("AXIS_NOT_UNIT_LENGTH", {"axis": axis, "norm": original_norm})
    return normalized, original_norm


def matmul_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Robotics xyz_rpy convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def axis_from_transform(xyz_rpy: list[float]) -> tuple[list[float], float]:
    if len(xyz_rpy) != 6 or not all(finite(value) for value in xyz_rpy):
        reject("BAD_TRANSFORM_XYZ_RPY", xyz_rpy)
    roll, pitch, yaw = map(float, xyz_rpy[3:])
    axis = matmul_vec(rotation_from_rpy(roll, pitch, yaw), [0.0, 0.0, -1.0])
    normalized, norm = normalize_axis(axis)
    return normalized, norm


def scan_residue(*, current_pid: int | None = None) -> dict:
    result = subprocess.run(["ps", "-Ao", "pid=,ppid=,command="], text=True, capture_output=True, check=False, timeout=10)
    excluded = {pid for pid in (current_pid, os.getppid()) if pid is not None}
    matches: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            pid = int(stripped.split(maxsplit=1)[0])
        except ValueError:
            pid = None
        if pid in excluded:
            continue
        if any(pattern in stripped for pattern in RESIDUE_PATTERNS):
            matches.append(stripped)
    return {
        "command": "ps -Ao pid=,ppid=,command=",
        "patterns": list(RESIDUE_PATTERNS),
        "returncode": result.returncode,
        "stdout": "\n".join(matches),
        "stderr": result.stderr,
        "empty": result.returncode == 0 and not matches and not result.stderr.strip(),
    }


def assert_exact_approval(value: str) -> None:
    normalized = value.strip()
    if not normalized:
        reject("MISSING_APPROVAL_PHRASE")
    if normalized in BROAD_APPROVALS:
        reject("BROAD_APPROVAL_REJECTED", {"received": normalized, "required": APPROVAL_PHRASE})
    if normalized != APPROVAL_PHRASE:
        reject("APPROVAL_PHRASE_MISMATCH", {"received": normalized, "required": APPROVAL_PHRASE})


def under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_transform_source(path_text: str) -> tuple[dict | None, str | None]:
    if not path_text:
        return None, None
    path = Path(path_text)
    if not path.is_file():
        reject("TRANSFORM_SOURCE_FILE_MISSING", path_text)
    text = path.read_text(encoding="utf-8")
    if any(marker in text for marker in OLD_FRONT_WRIST_MARKERS):
        reject("OLD_FRONT_WRIST_SOURCE_REJECTED", {"path": path_text})
    try:
        return json.loads(text), sha256_file(path)
    except json.JSONDecodeError as exc:
        reject("TRANSFORM_SOURCE_NOT_JSON", {"path": path_text, "error": str(exc)})


def axis_from_args(args: argparse.Namespace) -> tuple[list[float], str, dict | None, str | None]:
    if args.axis_source in FORBIDDEN_AXIS_SOURCES:
        reject("FORBIDDEN_AXIS_SOURCE", args.axis_source)
    if args.axis_source not in ALLOWED_AXIS_SOURCES:
        reject("AXIS_SOURCE_NOT_ALLOWED", args.axis_source)
    if not args.axis_source_detail.strip():
        reject("AXIS_SOURCE_DETAIL_MISSING")
    if SYMBOLIC_AXIS not in args.axis_source_detail and "[0, 0, -1]" not in args.axis_source_detail:
        reject("AXIS_SOURCE_DETAIL_INSUFFICIENT", args.axis_source_detail)
    if not finite(args.angular_uncertainty_rad) or not (0.0 <= args.angular_uncertainty_rad <= MAX_ANGULAR_UNCERTAINTY_RAD):
        reject("ANGULAR_UNCERTAINTY_TOO_LARGE", args.angular_uncertainty_rad)

    source_payload, source_sha = load_transform_source(args.transform_source_file)
    if args.axis_source == "measured_fixture_transform" or args.axis_source == "calibrated_camera_fixture_transform":
        transform = args.fr3_base_to_fixture_xyz_rpy
        if source_payload:
            candidate = source_payload.get("fr3_base_to_fixture_xyz_rpy")
            if candidate is None:
                candidate = source_payload.get("fixture", {}).get("fr3_base_to_fixture_xyz_rpy")
            if candidate is not None:
                transform = candidate
        axis, _ = axis_from_transform([float(value) for value in transform])
        return axis, "base_R_fixture * [0, 0, -1]", source_payload, source_sha

    axis, _ = check_unit_axis([args.manual_axis_base_x, args.manual_axis_base_y, args.manual_axis_base_z])
    return axis, "human measured base-frame unit axis", source_payload, source_sha


def build_result(args: argparse.Namespace, axis: list[float], derivation: str, source_payload: dict | None, source_sha: str | None, preflight_scan: dict, postflight_scan: dict, *, execute: bool) -> tuple[dict, Path, Path, Path]:
    run_id = args.run_id or time.strftime("plug_zed_fixture_axis_calibration_%Y%m%dT%H%M%S%z")
    result_root = Path(args.result_root)
    run_dir = result_root / run_id
    if not under_root(run_dir, result_root):
        reject("RESULT_DIR_OUTSIDE_ROOT", {"run_dir": str(run_dir), "result_root": str(result_root)})
    result_path = run_dir / "fixture_axis_result.json"
    transform_path = run_dir / "fixture_axis_transform.json"
    notes_path = run_dir / "fixture_axis_notes.md"
    transform_payload = {
        "active_task": ACTIVE_TASK,
        "fixture_frame": FIXTURE_FRAME,
        "symbolic_axis": SYMBOLIC_AXIS,
        "axis_source": args.axis_source,
        "axis_source_detail": args.axis_source_detail,
        "axis_derivation": derivation,
        "fr3_base_to_fixture_xyz_rpy": args.fr3_base_to_fixture_xyz_rpy if args.axis_source != "human_measured_fixture_axis" else None,
        "fr3_base_axis_unit_vector": axis,
        "angular_uncertainty_rad": args.angular_uncertainty_rad,
        "source_payload": source_payload,
        "runtime_execution_performed": False,
    }
    notes = "\n".join([
        "# Plug-ZED Fixture Axis Calibration Notes",
        "",
        f"run_id: {run_id}",
        f"axis_source: {args.axis_source}",
        f"axis_source_detail: {args.axis_source_detail}",
        f"axis_derivation: {derivation}",
        f"fr3_base_axis_unit_vector: {axis}",
        f"execute_no_motion: {execute}",
        "",
    ])
    source_hash = source_sha or hashlib.sha256(json.dumps(transform_payload, sort_keys=True).encode("utf-8")).hexdigest()
    transform_hash = hashlib.sha256(json.dumps(transform_payload, sort_keys=True).encode("utf-8")).hexdigest()
    result_payload = {
        "approval_phrase": APPROVAL_PHRASE,
        "run_id": run_id,
        "status": ACCEPT_STATUS if execute else READY_NOT_EXECUTED_STATUS,
        "active_task": ACTIVE_TASK,
        "fixture_frame": FIXTURE_FRAME,
        "symbolic_axis": SYMBOLIC_AXIS,
        "fr3_base_axis_unit_vector": axis,
        "axis_source": args.axis_source,
        "axis_source_detail": args.axis_source_detail,
        "angular_uncertainty_rad": args.angular_uncertainty_rad,
        "evidence": {
            "source_artifact_path": str(result_path),
            "source_artifact_sha256": source_hash,
            "transform_artifact_path": str(transform_path),
            "transform_artifact_sha256": transform_hash,
            "calibration_notes_path": str(notes_path),
        },
        "preconditions": dict(REQUIRED_PRECONDITIONS),
        "wrapper_ready_not_executed": {
            "status": "PLUG_ZED_CONTACT_MICRO_MOTION_READY_RUNTIME_NOT_EXECUTED",
            "reviewed_live_control_body_confirmed": True,
            "fixture_axis_base_confirmed": True,
            "fixture_axis_base": axis,
            "execute_live": False,
            "runtime_execution_performed": False,
        },
        "cleanup": {
            "preflight_residue_empty": preflight_scan["empty"],
            "postflight_residue_empty": postflight_scan["empty"],
            "preflight_scan": {
                "local_target": preflight_scan["stdout"],
                "remote_target": "",
            },
            "postflight_scan": {
                "local_target": postflight_scan["stdout"],
                "remote_target": "",
            },
        },
        "runtime_claims": {
            "fr3_connection": False,
            "fci_state_read": False,
            "motion_command": False,
            "gripper_command": False,
            "fixture_contact": False,
            "live_micro_motion": False,
            "policy_rollout": False,
            "remote_training": False,
            "checkpoint_promotion": False,
            "reward_label_collection": False,
            "classifier_training": False,
            "demo_collection": False,
            "droid_mutation": False,
            "thread_019e4faf_touched": False,
            "openpi_dependency": False,
            "old_8d_joint_target_action_accepted": False,
        },
    }
    if not execute:
        result_payload["safe_to_claim_100_percent"] = False
        result_payload["overall_goal_complete"] = False
    return result_payload, result_path, transform_path, notes_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-motion Plug-ZED fixture-axis calibration runtime.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-residue-scan", action="store_true")
    parser.add_argument("--approval-phrase", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--result-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--execute-no-motion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--axis-source", default="")
    parser.add_argument("--axis-source-detail", default="")
    parser.add_argument("--angular-uncertainty-rad", type=float, default=float("nan"))
    parser.add_argument("--transform-source-file", default="")
    parser.add_argument("--fr3-base-to-fixture-xyz-rpy", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    parser.add_argument("--manual-axis-base-x", type=float, default=0.0)
    parser.add_argument("--manual-axis-base-y", type=float, default=0.0)
    parser.add_argument("--manual-axis-base-z", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_check:
        payload = common_payload()
        payload.update({
            "ok": True,
            "status": READY_STATUS,
            "imports_robot_api_on_self_check": False,
            "exact_approval_required": True,
            "rejects_broad_approval": True,
            "rejects_old_front_wrist_sources": True,
            "computes_fixture_z_negative_from_transform": True,
            "can_accept_human_measured_axis": True,
            "writes_runtime_proven_result_json_only_with_execute_no_motion": True,
            "safe_to_claim_100_percent": False,
            "overall_goal_complete": False,
        })
        emit(payload)

    if args.self_check_residue_scan:
        scan = scan_residue(current_pid=os.getpid())
        payload = common_payload()
        payload.update({
            "ok": scan["empty"],
            "status": "PLUG_ZED_FIXTURE_AXIS_RESIDUE_SCAN_SELF_CHECKED" if scan["empty"] else BLOCK_STATUS,
            "residue_scan": scan,
            "safe_to_claim_100_percent": False,
            "overall_goal_complete": False,
        })
        emit(payload, 0 if scan["empty"] else 64)

    assert_exact_approval(args.approval_phrase)
    if args.execute_no_motion == args.dry_run:
        reject("SELECT_EXACTLY_ONE_EXECUTION_MODE", {"execute_no_motion": args.execute_no_motion, "dry_run": args.dry_run})
    preflight_scan = scan_residue(current_pid=os.getpid())
    if not preflight_scan["empty"]:
        reject("PREFLIGHT_RESIDUE_NOT_EMPTY", preflight_scan)
    axis, derivation, source_payload, source_sha = axis_from_args(args)
    postflight_scan = scan_residue(current_pid=os.getpid())
    if not postflight_scan["empty"]:
        reject("POSTFLIGHT_RESIDUE_NOT_EMPTY", postflight_scan)
    result_payload, result_path, transform_path, notes_path = build_result(
        args, axis, derivation, source_payload, source_sha, preflight_scan, postflight_scan, execute=args.execute_no_motion
    )
    if args.execute_no_motion:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        transform_payload = {
            "active_task": ACTIVE_TASK,
            "fixture_frame": FIXTURE_FRAME,
            "symbolic_axis": SYMBOLIC_AXIS,
            "axis_source": args.axis_source,
            "axis_source_detail": args.axis_source_detail,
            "axis_derivation": derivation,
            "fr3_base_to_fixture_xyz_rpy": args.fr3_base_to_fixture_xyz_rpy if args.axis_source != "human_measured_fixture_axis" else None,
            "fr3_base_axis_unit_vector": axis,
            "angular_uncertainty_rad": args.angular_uncertainty_rad,
            "runtime_execution_performed": False,
        }
        notes = "\n".join([
            "# Plug-ZED Fixture Axis Calibration Notes",
            "",
            f"run_id: {result_payload['run_id']}",
            f"axis_source: {args.axis_source}",
            f"axis_source_detail: {args.axis_source_detail}",
            f"axis_derivation: {derivation}",
            f"fr3_base_axis_unit_vector: {axis}",
            "execute_no_motion: true",
            "",
        ])
        transform_path.write_text(json.dumps(transform_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        notes_path.write_text(notes, encoding="utf-8")
        # Recompute hashes after writing concrete files.
        result_payload["evidence"]["transform_artifact_sha256"] = sha256_file(transform_path)
        result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result_payload["evidence"]["source_artifact_sha256"] = sha256_file(result_path)
        result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_payload.update({
        "ok": True,
        "accepted_decision": "PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_ACCEPTED" if args.execute_no_motion else "PLUG_ZED_FIXTURE_AXIS_BASE_CALIBRATION_READY_NOT_EXECUTED",
        "runtime_execution_performed": False,
        "motion_command": False,
        "gripper_command": False,
        "fixture_contact": False,
        "live_micro_motion": False,
        "safe_to_claim_100_percent": False,
        "overall_goal_complete": False,
        "result_path": str(result_path),
        "transform_path": str(transform_path),
        "notes_path": str(notes_path),
    })
    emit(result_payload)


if __name__ == "__main__":
    main()
