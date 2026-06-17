from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from fr3_hil_bridge.demo_recording import make_hil_transition, validate_transition
from fr3_hil_bridge.plug_bridge import validate_action_7d
from fr3_hil_bridge.teleop.xbox_mapping import XboxControllerState, XboxTeleopConfig, map_xbox_to_action, validate_mapping_result


STATUS = "PLUG_ZED_XBOX_TELEOP_DEMO_SCAFFOLD_PROVEN"
BANNED_RUNTIME_PATTERNS = (
    "/home/robot/droid",
    "import droid",
    "from droid",
    "RobotEnv(",
    "/home/robot/openpi",
    "import openpi",
    "from openpi",
)


def fail(reason: str, detail: object | None = None) -> None:
    payload = {"ok": False, "status": "BLOCK_PLUG_ZED_XBOX_TELEOP_DEMO_SCAFFOLD", "reason": reason}
    if detail is not None:
        payload["detail"] = detail
    raise SystemExit(json.dumps(payload, indent=2, sort_keys=True))


def run_json(cmd: list[str], timeout: int = 60) -> dict:
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        fail("subcommand_failed", {"cmd": cmd, "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail("subcommand_bad_json", {"cmd": cmd, "stdout": result.stdout, "stderr": result.stderr, "error": str(exc)})


def controller_state(mode: str, deadman: bool = True, rt: float = 0.45, lt: float = 0.0) -> XboxControllerState:
    return XboxControllerState(
        connected=True,
        axes={"LEFTX": 0.25, "LEFTY": -0.25, "RIGHTX": 0.20, "RIGHTY": 0.0, "LT": lt, "RT": rt},
        buttons={"RB": deadman, "A": False, "B": False},
        mode=mode,
        seq=1,
        t_mono_ns=time.monotonic_ns(),
        backend="mock",
    )


def check_probe_backends() -> dict:
    mock = run_json([sys.executable, "-m", "fr3_experiments.plug_insertion.teleop.xbox_probe", "--backend", "mock", "--duration-sec", "0.01"])
    sdl = run_json([sys.executable, "-m", "fr3_experiments.plug_insertion.teleop.xbox_probe", "--backend", "sdl", "--duration-sec", "0.05"])
    if mock.get("status") != "PLUG_ZED_XBOX_PROBE_MOCK_NO_MOTION":
        fail("mock_probe_bad_status", mock)
    if sdl.get("status") != "PLUG_ZED_XBOX_PROBE_SDL_CONTROLLER_NO_MOTION":
        fail("sdl_probe_bad_status", sdl)
    if sdl.get("pygame_imported") is not True or sdl.get("sdl2_controller_imported") is not True:
        fail("sdl_probe_import_not_proven", sdl)
    for payload in (mock, sdl):
        for flag in ("runtime_execution_performed", "motion_command", "gripper_command", "droid_mutation", "openpi_dependency"):
            if payload.get(flag) is not False:
                fail("probe_runtime_boundary_bad", {"flag": flag, "payload": payload})
    return {
        "mock_status": mock.get("status"),
        "sdl_status": sdl.get("status"),
        "pygame_imported": sdl.get("pygame_imported"),
        "sdl2_controller_imported": sdl.get("sdl2_controller_imported"),
        "physical_xbox_controller_detected": bool(sdl.get("controller_device_opened")),
        "detected_xbox_devices": sdl.get("controller_devices", []),
    }


def check_mapping_contract() -> dict:
    cfg = XboxTeleopConfig()
    modes = {}
    for mode in ("coarse", "fine", "rotation", "insert_axis"):
        result = map_xbox_to_action(controller_state(mode), cfg)
        validation = validate_mapping_result(result)
        if validation.get("ok") is not True:
            fail("mapping_validation_failed", {"mode": mode, "validation": validation})
        modes[mode] = result.to_dict()

    deadman_release = map_xbox_to_action(controller_state("insert_axis", deadman=False), cfg)
    if deadman_release.reject_code != "DEADMAN_RELEASED" or not np.allclose(deadman_release.action, np.zeros(7)):
        fail("deadman_release_not_zero", deadman_release.to_dict())

    stale_state = controller_state("insert_axis")
    stale_state = XboxControllerState(**{**stale_state.to_dict(), "t_mono_ns": time.monotonic_ns() - 10_000_000_000})
    stale = map_xbox_to_action(stale_state, cfg)
    if stale.reject_code != "CONTROLLER_INPUT_STALE":
        fail("stale_input_not_rejected", stale.to_dict())

    insert_action = np.asarray(modes["insert_axis"]["action"], dtype=np.float32)
    if insert_action.shape != (7,) or insert_action[2] >= 0.0:
        fail("insert_axis_does_not_move_down", modes["insert_axis"])

    old_8d = validate_action_7d(np.zeros(8, dtype=np.float32))
    if old_8d.valid or old_8d.reason != "REJECT_OLD_OPENPI_DROID_8D_ACTION":
        fail("old_8d_not_rejected", old_8d.to_dict())

    return {
        "modes_checked": sorted(modes),
        "deadman_release_zero": True,
        "stale_input_rejected": True,
        "old_8d_rejected": True,
        "insert_axis_moves_down": True,
    }


def check_transition_contract() -> dict:
    obs = {
        "zed_left": np.zeros((128, 128, 3), dtype=np.uint8),
        "zed_right": np.zeros((128, 128, 3), dtype=np.uint8),
        "state": np.zeros((1, 7), dtype=np.float32),
        "gripper_state": np.array([0.08], dtype=np.float32),
    }
    human_action = map_xbox_to_action(controller_state("insert_axis"), XboxTeleopConfig()).action
    policy_action = np.zeros(7, dtype=np.float32)
    transition = make_hil_transition(obs, policy_action, human_action, True, obs, 1.0, True, {"teleop_mode": "insert_axis"})
    validation = validate_transition(transition)
    if validation.get("ok") is not True:
        fail("transition_validation_failed", validation)
    if not np.allclose(transition["actions"], transition["infos"]["intervene_action"]):
        fail("intervene_action_not_executed_action")
    return validation


def check_demo_writer(artifact_root: Path) -> dict:
    payload = run_json(
        [
            sys.executable,
            "-m",
            "fr3_experiments.plug_insertion.teleop.xbox_demo_dry_run",
            "--artifact-root",
            str(artifact_root / "demo_dry_run"),
            "--steps",
            "3",
        ]
    )
    if payload.get("ok") is not True or payload.get("status") != "PLUG_ZED_XBOX_DEMO_DRY_RUN_WRITTEN":
        fail("demo_dry_run_bad_payload", payload)
    episode = payload.get("episode", {})
    for key in ("transitions", "sidecar"):
        if not Path(episode.get(key, "")).exists():
            fail("demo_dry_run_missing_output", payload)
    return payload


def check_state_only_actor(artifact_root: Path) -> dict:
    payload = run_json(
        [
            sys.executable,
            "-m",
            "fr3_experiments.plug_insertion.teleop.xbox_state_only_actor",
            "--artifact-root",
            str(artifact_root / "state_only_actor"),
            "--steps",
            "4",
        ]
    )
    if payload.get("ok") is not True or payload.get("status") != "PLUG_ZED_XBOX_STATE_ONLY_ACTOR_DRY_RUN_WRITTEN":
        fail("state_only_actor_bad_payload", payload)
    if payload.get("fr3_connection") is not False or payload.get("fci_state_read") is not False:
        fail("state_only_actor_overclaims_runtime", payload)
    return payload


def check_lerobot_export(demo_payload: dict, artifact_root: Path) -> dict:
    episode_dir = demo_payload.get("episode", {}).get("episode_dir")
    if not episode_dir:
        fail("demo_episode_dir_missing", demo_payload)
    payload = run_json(
        [
            sys.executable,
            "-m",
            "fr3_experiments.plug_insertion.teleop.xbox_export_lerobot_sidecar",
            "--episode-dir",
            episode_dir,
            "--output-dir",
            str(artifact_root / "lerobot_sidecar"),
        ]
    )
    if payload.get("ok") is not True or payload.get("status") != "PLUG_ZED_XBOX_LEROBOT_SIDECAR_EXPORTED":
        fail("lerobot_export_bad_payload", payload)
    validation = payload.get("validation", {})
    if validation.get("ok") is not True or validation.get("row_count") != demo_payload.get("transition_count"):
        fail("lerobot_export_validation_bad", payload)
    return payload


def check_demo_collection_preflight(artifact_root: Path) -> dict:
    payload = run_json(
        [
            sys.executable,
            "-m",
            "fr3_experiments.plug_insertion.teleop.xbox_demo_collection_preflight",
            "--artifact-root",
            str(artifact_root / "demo_collection_preflight"),
            "--probe-duration-sec",
            "0.05",
            "--steps",
            "3",
        ]
    )
    if payload.get("ok") is not True or payload.get("status") != "PLUG_ZED_XBOX_DEMO_COLLECTION_PREFLIGHT_NO_MOTION_PROVEN":
        fail("demo_collection_preflight_bad_payload", payload)
    if payload.get("physical_xbox_controller_detected") is not False:
        fail("demo_collection_preflight_physical_boundary_bad", payload)
    if payload.get("ready_for_live_packet_generation") is not False:
        fail("demo_collection_preflight_overclaims_live_packet_ready", payload)
    if payload.get("lerobot_sidecar_export_status") != "PLUG_ZED_XBOX_LEROBOT_SIDECAR_EXPORTED":
        fail("demo_collection_preflight_missing_lerobot_export", payload)
    validation = payload.get("lerobot_sidecar_validation", {})
    if validation.get("ok") is not True or validation.get("row_count") != payload.get("transition_count"):
        fail("demo_collection_preflight_lerobot_validation_bad", payload)
    for flag in (
        "runtime_execution_performed",
        "live_demo_collection",
        "motion_command",
        "gripper_command",
        "policy_rollout",
        "remote_training",
        "droid_mutation",
        "openpi_dependency",
        "safe_to_execute_live_now",
        "old_packet_reuse_allowed",
    ):
        if payload.get(flag) is not False:
            fail("demo_collection_preflight_runtime_boundary_bad", {flag: payload.get(flag), "payload": payload})
    return payload


def check_sources(root: Path) -> dict:
    files = [
        root / "fr3_hil_bridge/teleop/xbox_mapping.py",
        root / "fr3_hil_bridge/teleop/xbox_backend.py",
        root / "fr3_hil_bridge/demo_recording.py",
        root / "fr3_hil_bridge/demo_export.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_probe.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_action_preview.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_demo_dry_run.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_state_only_actor.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_export_lerobot_sidecar.py",
        root / "fr3_experiments/plug_insertion/teleop/xbox_demo_collection_preflight.py",
    ]
    hits: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = [pattern for pattern in BANNED_RUNTIME_PATTERNS if pattern in text]
        if found:
            hits[str(path)] = found
    if hits:
        fail("banned_runtime_pattern_found", hits)
    return {"files_checked": [str(path) for path in files], "banned_runtime_patterns_found": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Plug-ZED Xbox teleop/demo scaffold without runtime")
    parser.add_argument("--artifact-root", default="artifacts/plug_zed_xbox_teleop_no_motion")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    probes = check_probe_backends()
    mapping = check_mapping_contract()
    transition = check_transition_contract()
    demo = check_demo_writer(artifact_root)
    state_only = check_state_only_actor(artifact_root)
    export = check_lerobot_export(demo, artifact_root)
    demo_collection_preflight = check_demo_collection_preflight(artifact_root)
    sources = check_sources(root)

    payload = {
        "ok": True,
        "status": STATUS,
        "probe_contract": probes,
        "mapping_contract": mapping,
        "transition_contract": transition,
        "demo_dry_run": {
            "status": demo.get("status"),
            "transition_count": demo.get("transition_count"),
            "episode_dir": demo.get("episode", {}).get("episode_dir"),
        },
        "state_only_actor": {
            "status": state_only.get("status"),
            "transition_count": state_only.get("transition_count"),
        },
        "lerobot_sidecar_export": {
            "status": export.get("status"),
            "row_count": export.get("validation", {}).get("row_count"),
            "schema_version": export.get("export", {}).get("schema_version"),
        },
        "demo_collection_preflight": {
            "status": demo_collection_preflight.get("status"),
            "transition_count": demo_collection_preflight.get("transition_count"),
            "physical_xbox_controller_detected": demo_collection_preflight.get("physical_xbox_controller_detected"),
            "ready_for_live_packet_generation": demo_collection_preflight.get("ready_for_live_packet_generation"),
            "lerobot_sidecar_schema_version": demo_collection_preflight.get("lerobot_sidecar_schema_version"),
        },
        "source_contract": sources,
        "xbox_backend_runtime_required": False,
        "pygame_imported": probes.get("pygame_imported"),
        "sdl2_controller_imported": probes.get("sdl2_controller_imported"),
        "controller_device_opened": probes.get("physical_xbox_controller_detected"),
        "physical_xbox_controller_detected": probes.get("physical_xbox_controller_detected"),
        "detected_xbox_devices": probes.get("detected_xbox_devices"),
        "fr3_connection": False,
        "fci_state_read": False,
        "motion_command": False,
        "gripper_command": False,
        "policy_rollout": False,
        "remote_training": False,
        "live_demo_collection": False,
        "demo_collection_preflight_ready": True,
        "droid_mutation": False,
        "openpi_dependency": False,
        "actions_are_executed_actions": True,
        "xbox_raw_fci_allowed": False,
        "xbox_bounded_7d_action_required": True,
        "safe_to_execute_live_now": False,
    }
    summary_path = artifact_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
