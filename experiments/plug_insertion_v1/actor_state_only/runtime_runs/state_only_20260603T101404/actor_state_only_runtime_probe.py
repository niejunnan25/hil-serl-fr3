#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


ACTION_SPACE = "normalized_cartesian_delta_rpy_gripper"
ACTION_DIM = 7
CHECKPOINT_ID = "mock-plug-policy-ckpt-000"
POLICY_VERSION = "mock-policy-v1"
ACTION_SCHEMA_HASH = "hilserl7d:v1:cartesian_delta_rpy_gripper"


def fail(reason: str, detail: object | None = None) -> None:
    payload = {"success": False, "reason": reason}
    if detail is not None:
        payload["detail"] = detail
    raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def guard_response(response: dict) -> dict:
    if response.get("decision") != "ACCEPT_ACTION":
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "REMOTE_NOT_ACCEPT"}
    if response.get("checkpoint_id") != CHECKPOINT_ID:
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "CHECKPOINT_ID_MISMATCH"}
    if response.get("policy_version") != POLICY_VERSION:
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "POLICY_VERSION_MISMATCH"}
    if response.get("action_schema_hash") != ACTION_SCHEMA_HASH:
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "ACTION_SCHEMA_HASH_MISMATCH"}
    if response.get("action_space") != ACTION_SPACE:
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "ACTION_SPACE_MISMATCH"}
    action = response.get("action")
    if not isinstance(action, list) or len(action) != ACTION_DIM:
        if isinstance(action, list) and len(action) == 8:
            return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "OLD_8D_OPENPI_DROID_ACTION"}
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "WRONG_ACTION_LENGTH"}
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in action):
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "NONFINITE_ACTION"}
    if not all(-1.0 <= float(value) <= 1.0 for value in action):
        return {"decision": "BRIDGE_HOLD_OR_STOP", "reject_code": "OUT_OF_RANGE_ACTION"}
    return {
        "decision": "ACCEPT_ACTION_FOR_STATE_ONLY_TRACE_ONLY",
        "reject_code": "NONE",
        "action_dim": len(action),
        "action_space": ACTION_SPACE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-motion", action="store_true")
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--response-file", required=True)
    parser.add_argument("--negative-response-file", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()
    if not (args.dry_run and args.no_motion and args.state_only and args.no_camera and args.no_gripper):
        fail("missing_no_motion_guards")
    state = load_json(args.state_file)
    response = load_json(args.response_file)
    negative = load_json(args.negative_response_file)
    if state.get("source") != "mock_state_only":
        fail("bad_state_source", state)
    if state.get("fci_state_read") is not False:
        fail("fci_state_read_overclaimed", state)
    if state.get("fr3_connection") is not False:
        fail("fr3_connection_overclaimed", state)
    accepted = guard_response(response)
    rejected = guard_response(negative)
    if accepted.get("decision") != "ACCEPT_ACTION_FOR_STATE_ONLY_TRACE_ONLY":
        fail("accepted_response_not_accepted_for_trace", accepted)
    if rejected.get("decision") != "BRIDGE_HOLD_OR_STOP" or rejected.get("reject_code") != "OLD_8D_OPENPI_DROID_ACTION":
        fail("old_8d_not_rejected", rejected)
    trace = {
        "process_name": "hilserl_desktop_actor_state_only",
        "runtime_mode": "state_only_no_motion",
        "state_source": "mock_state_only",
        "state_age_ms": 0.0,
        "accepted_decision": accepted["decision"],
        "negative_decision": rejected["decision"],
        "negative_reject_code": rejected["reject_code"],
        "accepted_action_dim": accepted["action_dim"],
        "accepted_action_space": accepted["action_space"],
        "timestamp": time.time(),
        "fci_state_read": False,
        "fr3_connection": False,
        "motion_command_sent": False,
        "gripper_command_sent": False,
        "camera_capture": False,
        "frame_read": False,
        "policy_rollout": False,
        "droid_dependency": False,
    }
    Path(args.trace_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace_file).write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    result = {
        "success": True,
        "trace": trace,
        "accepted_decision": accepted["decision"],
        "negative_decision": rejected["decision"],
        "negative_reject_code": rejected["reject_code"],
        "accepted_action_dim": accepted["action_dim"],
        "accepted_action_space": accepted["action_space"],
        "no_fr3_connection": True,
        "no_fci_state_read": True,
        "no_motion_command": True,
        "no_gripper_command": True,
        "no_camera_capture": True,
        "no_frame_read": True,
        "no_droid_dependency": True,
        "no_policy_rollout": True,
    }
    Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_file).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
