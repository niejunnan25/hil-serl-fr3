#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


sys.dont_write_bytecode = True

APPROVAL_PHRASE = "APPROVE PLUG-ZED CONTACT ENVELOPE AND MICRO-MOTION LIVE FR3"
BROAD_APPROVALS = {
    "批准一切权限",
    "批准一切真机权限",
    "全部都批准",
    "我批准一切",
    "approve all",
    "all approved",
}
READY_STATUS = "PLUG_ZED_CONTACT_MICRO_MOTION_RUNTIME_SCAFFOLD_READY"
READY_NOT_EXECUTED_STATUS = "PLUG_ZED_CONTACT_MICRO_MOTION_READY_RUNTIME_NOT_EXECUTED"
FUTURE_ACCEPT_STATUS = "PLUG_ZED_CONTACT_MICRO_MOTION_RUNTIME_PROVEN"
FIXTURE_CONTACT_STATUS = "PLUG_FIXTURE_CONTACT_ENVELOPE_PROVEN"
LIVE_MICRO_MOTION_STATUS = "FR3_LIVE_MICRO_MOTION_PROVEN"
BLOCK_STATUS = "BLOCK_PLUG_ZED_CONTACT_MICRO_MOTION_RUNTIME"
REVIEWED_LIVE_CONTROL_BODY_REQUIRED = "REVIEWED_LIVE_CONTROL_BODY_NOT_CONFIRMED"
LIVE_CONTROL_BODY_SOURCE_STATUS = "PLUG_ZED_CONTACT_MICRO_MOTION_LIVE_CONTROL_BODY_SOURCE_COMPILED_AND_SELF_CHECKED"
LIVE_CONTROL_BINARY = Path("/home/robot/serl_projects/hil-serl-fr3/experiments/plug_zed_insertion/calibration/contact_micro_motion/live_control_body/build/plug_zed_contact_micro_motion_live_control")

REMOTE_USER = "fzt"
REMOTE_HOST = "fzt@162.105.195.74"
RESULT_ROOT = Path("/sda/fzt/hilserl-fr3/plug_zed_insertion")
LIVE_RESULT_ROOT = Path(
    os.environ.get(
        "PLUG_ZED_CONTACT_MICRO_MOTION_RESULT_ROOT",
        "/home/robot/serl_projects/hil-serl-fr3/experiments/plug_zed_insertion/calibration/contact_micro_motion/runtime_runs",
    )
)
ACTION_SCHEMA_HASH = "hilserl7d:v1:cartesian_delta_rpy_gripper"
OBSERVATION_SCHEMA_HASH = "plugzedobs:v1:zed_left_zed_right:fci_gripper_state"
REQUIRED_PRECONDITIONS = {
    "plug_zed_formal_camera_capture_runtime": "PLUG_ZED_CAMERA_FIXTURE_CAPTURE_RUNTIME_PROVEN",
    "fci_state_only_runtime": "FR3_FCI_STATE_ONLY_READ_PROVEN",
    "gripper_state_only_runtime": "FR3_GRIPPER_STATE_ONLY_READ_PROVEN",
    "plug_zed_remote_source_contract": "PLUG_ZED_REMOTE_SOURCE_CONTRACT_PROVEN",
}

DEFAULT_ENVELOPE = {
    "fixture_frame": "fixture_plug_socket_v1",
    "tcp_frame": "fr3_tcp",
    "approach_axis": "fixture_z_negative",
    "bounded_micro_motion_envelope_id": "plug_zed_contact_micro_motion_minimal_v1",
    "max_contact_force_n": 5.0,
    "contact_force_limit_n": 8.0,
    "lateral_tolerance_m": 0.003,
    "insertion_depth_limit_m": 0.003,
    "retreat_distance_m": 0.020,
    "max_translation_step_m": 0.001,
    "max_rotation_step_rad": 0.0,
    "max_velocity_m_s": 0.003,
    "max_attempt_duration_s": 1.0,
    "max_retreat_duration_s": 8.0,
}
STRICT_LIMITS = {
    "max_contact_force_n": 5.0,
    "contact_force_limit_n": 8.0,
    "lateral_tolerance_m": 0.003,
    "insertion_depth_limit_m": 0.003,
    "retreat_distance_m": 0.020,
    "max_translation_step_m": 0.001,
    "max_rotation_step_rad": 0.0,
    "max_velocity_m_s": 0.003,
    "max_attempt_duration_s": 1.0,
    "max_retreat_duration_s": 8.0,
}
RESIDUE_PATTERNS = (
    "plug_zed_contact_micro_motion_runtime.py",
    "plug_zed_camera_fixture_capture_runtime.py",
    "zed_camera_service.py",
    "franka_server",
    "polymetis",
    "RobotEnv",
    "generate_cartesian_pose_motion",
    "cartesian_impedance_control",
)


def emit(payload: dict, exit_code: int = 0) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def common_payload() -> dict:
    return {
        "approval_phrase_required": APPROVAL_PHRASE,
        "ready_status": READY_STATUS,
        "ready_not_executed_status": READY_NOT_EXECUTED_STATUS,
        "future_accept_status": FUTURE_ACCEPT_STATUS,
        "future_fixture_contact_envelope_status": FIXTURE_CONTACT_STATUS,
        "future_live_micro_motion_status": LIVE_MICRO_MOTION_STATUS,
        "block_status": BLOCK_STATUS,
        "live_execution_block_code": REVIEWED_LIVE_CONTROL_BODY_REQUIRED,
        "live_control_body_source_status": LIVE_CONTROL_BODY_SOURCE_STATUS,
        "default_live_control_binary": str(LIVE_CONTROL_BINARY),
        "remote_user": REMOTE_USER,
        "remote_host": REMOTE_HOST,
        "result_root": str(RESULT_ROOT),
        "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
        "action_schema_hash": ACTION_SCHEMA_HASH,
        "action_dim": 7,
        "image_keys": ["zed_left", "zed_right"],
        "required_precondition_statuses": dict(REQUIRED_PRECONDITIONS),
        "default_envelope": dict(DEFAULT_ENVELOPE),
        "strict_limits": dict(STRICT_LIMITS),
        "runtime_execution_performed": False,
        "fixture_contact_envelope_executed": False,
        "live_micro_motion_executed": False,
        "camera_capture_executed_by_this_runner": False,
        "camera_capture_precondition_proven": True,
        "fci_state_read": False,
        "gripper_state_read": False,
        "fr3_connection": False,
        "motion_command": False,
        "gripper_command": False,
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


def scan_residue(*, current_pid: int | None = None) -> dict:
    command = ["ps", "-Ao", "pid=,ppid=,command="]
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    excluded_pids = {pid for pid in (current_pid, os.getppid()) if pid is not None}
    matches: list[str] = []
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


def assert_exact_approval(value: str) -> None:
    normalized = value.strip()
    if not normalized:
        reject("MISSING_APPROVAL_PHRASE")
    if normalized in BROAD_APPROVALS:
        reject("BROAD_APPROVAL_REJECTED", {
            "received": normalized,
            "required": APPROVAL_PHRASE,
        })
    if normalized != APPROVAL_PHRASE:
        reject("APPROVAL_PHRASE_MISMATCH", {
            "received": normalized,
            "required": APPROVAL_PHRASE,
        })


def envelope_from_args(args: argparse.Namespace) -> dict:
    return {
        "fixture_frame": args.fixture_frame,
        "tcp_frame": args.tcp_frame,
        "approach_axis": args.approach_axis,
        "bounded_micro_motion_envelope_id": args.bounded_micro_motion_envelope_id,
        "max_contact_force_n": args.max_contact_force_n,
        "contact_force_limit_n": args.contact_force_limit_n,
        "lateral_tolerance_m": args.lateral_tolerance_m,
        "insertion_depth_limit_m": args.insertion_depth_limit_m,
        "retreat_distance_m": args.retreat_distance_m,
        "max_translation_step_m": args.max_translation_step_m,
        "max_rotation_step_rad": args.max_rotation_step_rad,
        "max_velocity_m_s": args.max_velocity_m_s,
        "max_attempt_duration_s": args.max_attempt_duration_s,
        "max_retreat_duration_s": args.max_retreat_duration_s,
    }


def validate_envelope(envelope: dict) -> None:
    for key, expected in (
        ("fixture_frame", DEFAULT_ENVELOPE["fixture_frame"]),
        ("tcp_frame", DEFAULT_ENVELOPE["tcp_frame"]),
        ("approach_axis", DEFAULT_ENVELOPE["approach_axis"]),
        ("bounded_micro_motion_envelope_id", DEFAULT_ENVELOPE["bounded_micro_motion_envelope_id"]),
    ):
        if envelope.get(key) != expected:
            reject(f"{key.upper()}_MISMATCH", {"received": envelope.get(key), "expected": expected})

    for key in (
        "max_contact_force_n",
        "contact_force_limit_n",
        "lateral_tolerance_m",
        "insertion_depth_limit_m",
        "retreat_distance_m",
        "max_translation_step_m",
        "max_rotation_step_rad",
        "max_velocity_m_s",
        "max_attempt_duration_s",
        "max_retreat_duration_s",
    ):
        if not finite_number(envelope.get(key)):
            reject("NONFINITE_ENVELOPE_VALUE", {key: envelope.get(key)})

    max_force = float(envelope["max_contact_force_n"])
    force_limit = float(envelope["contact_force_limit_n"])
    if not (0.0 < max_force <= force_limit <= STRICT_LIMITS["contact_force_limit_n"]):
        reject("CONTACT_FORCE_LIMIT_EXCEEDED", {
            "max_contact_force_n": max_force,
            "contact_force_limit_n": force_limit,
            "strict_limit": STRICT_LIMITS["contact_force_limit_n"],
        })
    if max_force > STRICT_LIMITS["max_contact_force_n"]:
        reject("MAX_CONTACT_FORCE_TOO_LARGE", {
            "max_contact_force_n": max_force,
            "strict_limit": STRICT_LIMITS["max_contact_force_n"],
        })
    if not (0.0 < float(envelope["lateral_tolerance_m"]) <= STRICT_LIMITS["lateral_tolerance_m"]):
        reject("LATERAL_TOLERANCE_TOO_LARGE", envelope)
    if not (0.0 < float(envelope["insertion_depth_limit_m"]) <= STRICT_LIMITS["insertion_depth_limit_m"]):
        reject("INSERTION_DEPTH_TOO_LARGE", envelope)
    if float(envelope["retreat_distance_m"]) < STRICT_LIMITS["retreat_distance_m"]:
        reject("RETREAT_DISTANCE_TOO_SMALL", envelope)
    if not (0.0 < float(envelope["max_translation_step_m"]) <= STRICT_LIMITS["max_translation_step_m"]):
        reject("TRANSLATION_STEP_TOO_LARGE", envelope)
    if float(envelope["max_rotation_step_rad"]) != STRICT_LIMITS["max_rotation_step_rad"]:
        reject("ROTATION_STEP_NOT_ALLOWED", envelope)
    if not (0.0 < float(envelope["max_velocity_m_s"]) <= STRICT_LIMITS["max_velocity_m_s"]):
        reject("VELOCITY_TOO_LARGE", envelope)
    if not (0.0 < float(envelope["max_attempt_duration_s"]) <= STRICT_LIMITS["max_attempt_duration_s"]):
        reject("ATTEMPT_DURATION_TOO_LARGE", envelope)
    minimum_retreat_time = float(envelope["retreat_distance_m"]) / float(envelope["max_velocity_m_s"])
    if not (minimum_retreat_time <= float(envelope["max_retreat_duration_s"]) <= STRICT_LIMITS["max_retreat_duration_s"]):
        reject("RETREAT_DURATION_OUT_OF_RANGE", {
            "minimum_retreat_time_s": minimum_retreat_time,
            "envelope": envelope,
        })


def validate_fixture_axis_base(args: argparse.Namespace) -> dict:
    axis = {
        "x": args.fixture_axis_base_x,
        "y": args.fixture_axis_base_y,
        "z": args.fixture_axis_base_z,
        "confirmed": bool(args.fixture_axis_base_confirmed),
    }
    if not args.fixture_axis_base_confirmed:
        reject("FIXTURE_AXIS_BASE_NOT_CONFIRMED", axis)
    norm = math.sqrt(args.fixture_axis_base_x ** 2 + args.fixture_axis_base_y ** 2 + args.fixture_axis_base_z ** 2)
    if not math.isfinite(norm) or norm < 0.999 or norm > 1.001:
        reject("FIXTURE_AXIS_BASE_NOT_UNIT_LENGTH", {"norm": norm, "axis": axis})
    axis["norm"] = norm
    return axis


def run_live_control_body(args: argparse.Namespace, envelope: dict, gates: dict, preflight_scan: dict, axis: dict) -> None:
    if not args.confirm_reviewed_live_control_body:
        reject(REVIEWED_LIVE_CONTROL_BODY_REQUIRED, {
            "reason": "The compiled C++ live-control body exists, but this wrapper requires an explicit review confirmation before invoking it.",
            "live_control_binary": str(args.live_control_binary),
        })
    binary = Path(args.live_control_binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        reject("LIVE_CONTROL_BINARY_NOT_EXECUTABLE", str(binary))
    command = [
        str(binary),
        "--execute-live",
        "--approval-phrase", APPROVAL_PHRASE,
        "--run-id", args.run_id or time.strftime("plug_zed_contact_micro_motion_live_%Y%m%dT%H%M%S%z"),
        "--result-root", str(LIVE_RESULT_ROOT),
        "--confirm-human-physical-approval-for-this-run",
        "--confirm-estop-supervised",
        "--confirm-workspace-clear",
        "--confirm-robot-idle-initial-pose",
        "--confirm-fixture-staged",
        "--fixture-axis-base-confirmed",
        "--fixture-axis-base-x", str(axis["x"]),
        "--fixture-axis-base-y", str(axis["y"]),
        "--fixture-axis-base-z", str(axis["z"]),
        "--max-contact-force-n", str(envelope["max_contact_force_n"]),
        "--contact-force-limit-n", str(envelope["contact_force_limit_n"]),
        "--lateral-tolerance-m", str(envelope["lateral_tolerance_m"]),
        "--insertion-depth-limit-m", str(envelope["insertion_depth_limit_m"]),
        "--retreat-distance-m", str(envelope["retreat_distance_m"]),
        "--max-translation-step-m", str(envelope["max_translation_step_m"]),
        "--max-rotation-step-rad", str(envelope["max_rotation_step_rad"]),
        "--velocity-m-s", str(envelope["max_velocity_m_s"]),
        "--max-approach-duration-s", str(envelope["max_attempt_duration_s"]),
        "--max-retreat-duration-s", str(envelope["max_retreat_duration_s"]),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        reject("LIVE_CONTROL_BODY_STDOUT_NOT_JSON", {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "preflight_scan": preflight_scan,
            "live_robot_gates": gates,
        })
    payload["wrapper_preflight_scan"] = preflight_scan
    payload["wrapper_live_robot_gates"] = gates
    payload["wrapper_invoked_live_control_body"] = True
    emit(payload, 0 if result.returncode == 0 else 64)


def validate_live_gate_flags(args: argparse.Namespace) -> dict:
    gates = {
        "human_physical_approval_for_this_run": bool(args.confirm_human_physical_approval_for_this_run),
        "estop_supervised": bool(args.confirm_estop_supervised),
        "workspace_clear": bool(args.confirm_workspace_clear),
        "robot_idle_initial_pose": bool(args.confirm_robot_idle_initial_pose),
        "fixture_staged": bool(args.confirm_fixture_staged),
        "final_human_review": False,
    }
    for key in (
        "human_physical_approval_for_this_run",
        "estop_supervised",
        "workspace_clear",
        "robot_idle_initial_pose",
        "fixture_staged",
    ):
        if gates[key] is not True:
            reject("LIVE_ROBOT_GATE_MISSING", {"missing_gate": key, "gates": gates})
    return gates


def ready_payload(args: argparse.Namespace, envelope: dict, gates: dict, preflight_scan: dict) -> dict:
    run_id = args.run_id or time.strftime("plug_zed_contact_micro_motion_ready_%Y%m%dT%H%M%S%z")
    result_dir = RESULT_ROOT / "contact_micro_motion" / run_id
    payload = common_payload()
    payload.update({
        "ok": True,
        "status": READY_NOT_EXECUTED_STATUS,
        "run_id": run_id,
        "result_dir_if_executed": str(result_dir),
        "approval_phrase": APPROVAL_PHRASE,
        "precondition_statuses": dict(REQUIRED_PRECONDITIONS),
        "schema": {
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_schema_hash": ACTION_SCHEMA_HASH,
            "action_dim": 7,
            "image_keys": ["zed_left", "zed_right"],
        },
        "proposed_fixture_contact_envelope": {
            "status": "VALIDATED_RUNTIME_NOT_EXECUTED",
            "fixture_frame": envelope["fixture_frame"],
            "tcp_frame": envelope["tcp_frame"],
            "approach_axis": envelope["approach_axis"],
            "max_contact_force_n": envelope["max_contact_force_n"],
            "contact_force_limit_n": envelope["contact_force_limit_n"],
            "lateral_tolerance_m": envelope["lateral_tolerance_m"],
            "insertion_depth_limit_m": envelope["insertion_depth_limit_m"],
            "retreat_distance_m": envelope["retreat_distance_m"],
        },
        "proposed_live_micro_motion": {
            "status": "VALIDATED_RUNTIME_NOT_EXECUTED",
            "motion_envelope_id": envelope["bounded_micro_motion_envelope_id"],
            "max_translation_step_m": envelope["max_translation_step_m"],
            "max_rotation_step_rad": envelope["max_rotation_step_rad"],
            "max_velocity_m_s": envelope["max_velocity_m_s"],
            "max_attempt_duration_s": envelope["max_attempt_duration_s"],
            "max_retreat_duration_s": envelope["max_retreat_duration_s"],
            "abort_path_tested": False,
            "hold_or_stop_tested": False,
        },
        "live_robot_gates": gates,
        "postflight": {
            "postflight_residue_empty": False,
            "final_human_review_required_after_run": True,
            "droid_mutation": False,
            "thread_019e4faf_touched": False,
        },
        "preflight_residue_empty": preflight_scan["empty"],
        "preflight_scan": preflight_scan,
        "live_control_body_source_status": LIVE_CONTROL_BODY_SOURCE_STATUS,
        "live_control_binary": str(args.live_control_binary),
        "reviewed_live_control_body_confirmed": bool(args.confirm_reviewed_live_control_body),
        "fixture_axis_base": {
            "confirmed": bool(args.fixture_axis_base_confirmed),
            "x": args.fixture_axis_base_x,
            "y": args.fixture_axis_base_y,
            "z": args.fixture_axis_base_z,
        },
        "runtime_claims": {
            "fixture_contact_envelope_executed": False,
            "live_micro_motion_executed": False,
            "fci_state_used": False,
            "gripper_state_used": False,
            "camera_capture_executed_before_run": False,
            "camera_capture_precondition_proven": True,
            "policy_rollout": False,
            "remote_training": False,
            "checkpoint_promotion": False,
            "droid_mutation": False,
            "openpi_dependency": False,
            "old_8d_joint_target_action_accepted": False,
        },
        "abort_path_required_before_accept": "hold_or_stop_on_force_limit_or_state_fault_then_retreat_0.02m",
        "safe_to_claim_100_percent": False,
        "overall_goal_complete": False,
        "next_required_action": "calibrate_fixture_axis_base_and_execute_live_packet_with_postflight_review",
    })
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed Plug-ZED contact/micro-motion runtime scaffold.")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-residue-scan", action="store_true")
    parser.add_argument("--approval-phrase", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--confirm-reviewed-live-control-body", action="store_true")
    parser.add_argument("--confirm-human-physical-approval-for-this-run", action="store_true")
    parser.add_argument("--confirm-estop-supervised", action="store_true")
    parser.add_argument("--confirm-workspace-clear", action="store_true")
    parser.add_argument("--confirm-robot-idle-initial-pose", action="store_true")
    parser.add_argument("--confirm-fixture-staged", action="store_true")
    parser.add_argument("--fixture-axis-base-confirmed", action="store_true")
    parser.add_argument("--fixture-axis-base-x", type=float, default=0.0)
    parser.add_argument("--fixture-axis-base-y", type=float, default=0.0)
    parser.add_argument("--fixture-axis-base-z", type=float, default=0.0)
    parser.add_argument("--live-control-binary", type=Path, default=LIVE_CONTROL_BINARY)
    parser.add_argument("--fixture-frame", default=DEFAULT_ENVELOPE["fixture_frame"])
    parser.add_argument("--tcp-frame", default=DEFAULT_ENVELOPE["tcp_frame"])
    parser.add_argument("--approach-axis", default=DEFAULT_ENVELOPE["approach_axis"])
    parser.add_argument("--bounded-micro-motion-envelope-id", default=DEFAULT_ENVELOPE["bounded_micro_motion_envelope_id"])
    parser.add_argument("--max-contact-force-n", type=float, default=DEFAULT_ENVELOPE["max_contact_force_n"])
    parser.add_argument("--contact-force-limit-n", type=float, default=DEFAULT_ENVELOPE["contact_force_limit_n"])
    parser.add_argument("--lateral-tolerance-m", type=float, default=DEFAULT_ENVELOPE["lateral_tolerance_m"])
    parser.add_argument("--insertion-depth-limit-m", type=float, default=DEFAULT_ENVELOPE["insertion_depth_limit_m"])
    parser.add_argument("--retreat-distance-m", type=float, default=DEFAULT_ENVELOPE["retreat_distance_m"])
    parser.add_argument("--max-translation-step-m", type=float, default=DEFAULT_ENVELOPE["max_translation_step_m"])
    parser.add_argument("--max-rotation-step-rad", type=float, default=DEFAULT_ENVELOPE["max_rotation_step_rad"])
    parser.add_argument("--max-velocity-m-s", type=float, default=DEFAULT_ENVELOPE["max_velocity_m_s"])
    parser.add_argument("--max-attempt-duration-s", type=float, default=DEFAULT_ENVELOPE["max_attempt_duration_s"])
    parser.add_argument("--max-retreat-duration-s", type=float, default=DEFAULT_ENVELOPE["max_retreat_duration_s"])
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.self_check:
        payload = common_payload()
        payload.update({
            "ok": True,
            "status": READY_STATUS,
            "imports_robot_api_on_self_check": False,
            "exact_approval_required_before_any_live_execution": True,
            "rejects_broad_approval": True,
            "validates_live_robot_gate_flags": True,
            "validates_envelope_bounds": True,
            "requires_fixture_axis_base_for_execute_live": True,
            "integrates_compiled_live_control_body": True,
            "live_control_body_source_status": LIVE_CONTROL_BODY_SOURCE_STATUS,
            "live_control_binary": str(args.live_control_binary),
            "runs_residue_scan": True,
            "execute_live_fails_closed": True,
            "writes_runtime_proven_result_json": False,
            "residue_scan_patterns": list(RESIDUE_PATTERNS),
            "safe_to_claim_100_percent": False,
            "overall_goal_complete": False,
        })
        emit(payload)

    if args.self_check_residue_scan:
        scan = scan_residue(current_pid=os.getpid())
        payload = common_payload()
        payload.update({
            "ok": scan["empty"],
            "status": "PLUG_ZED_CONTACT_MICRO_MOTION_RESIDUE_SCAN_SELF_CHECKED" if scan["empty"] else BLOCK_STATUS,
            "residue_scan": scan,
            "safe_to_claim_100_percent": False,
            "overall_goal_complete": False,
        })
        emit(payload, 0 if scan["empty"] else 64)

    assert_exact_approval(args.approval_phrase)
    gates = validate_live_gate_flags(args)
    envelope = envelope_from_args(args)
    validate_envelope(envelope)
    preflight_scan = scan_residue(current_pid=os.getpid())
    if not preflight_scan["empty"]:
        reject("PREFLIGHT_RESIDUE_NOT_EMPTY", preflight_scan)

    if args.execute_live:
        if not args.confirm_reviewed_live_control_body:
            reject(REVIEWED_LIVE_CONTROL_BODY_REQUIRED, {
                "reason": "The compiled C++ live-control body exists, but this wrapper requires an explicit review confirmation before invoking it.",
                "live_control_binary": str(args.live_control_binary),
            })
        axis = validate_fixture_axis_base(args)
        run_live_control_body(args, envelope, gates, preflight_scan, axis)

    emit(ready_payload(args, envelope, gates, preflight_scan))


if __name__ == "__main__":
    main()
