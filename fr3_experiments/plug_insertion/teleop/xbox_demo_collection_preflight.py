from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fr3_hil_bridge.demo_export import export_episode_to_lerobot_sidecar, validate_lerobot_sidecar
from fr3_hil_bridge.demo_recording import EpisodeWriter, make_hil_transition, validate_transition
from fr3_hil_bridge.teleop.xbox_backend import pygame_sdl_controller_probe
from fr3_hil_bridge.teleop.xbox_mapping import XboxControllerState, XboxTeleopConfig, map_xbox_to_action


PREFLIGHT_STATUS = "PLUG_ZED_XBOX_DEMO_COLLECTION_PREFLIGHT_NO_MOTION_PROVEN"
PREFLIGHT_WITH_CONTROLLER_STATUS = "PLUG_ZED_XBOX_DEMO_COLLECTION_PREFLIGHT_PHYSICAL_CONTROLLER_PROVEN"
BLOCK_STATUS = "BLOCK_PLUG_ZED_XBOX_DEMO_COLLECTION_PREFLIGHT"
REQUIRED_HUMAN_FACTS = [
    "E-stop supervised current for this packet",
    "workspace clear current for this packet",
    "fixture/contact area clear or staged current for this packet",
    "robot idle initial pose visually unchanged current for this packet",
    "Xbox controller USB connected on fr3-desktop-ts for this packet/probe",
    "final review after run current for this packet",
]


def synthetic_obs(step: int) -> dict:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = (step * 17) % 255
    image[:, :, 1] = (step * 31) % 255
    return {
        "zed_left": image.copy(),
        "zed_right": image.copy(),
        "state": np.zeros((1, 7), dtype=np.float32),
        "gripper_state": np.array([0.08], dtype=np.float32),
        "ee_pose": np.array([0.55, 0.0, 0.53, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }


def mock_insert_controller_state(step: int) -> XboxControllerState:
    return XboxControllerState(
        connected=True,
        axes={"LEFTX": 0.04, "LEFTY": -0.03, "RIGHTX": 0.0, "RIGHTY": 0.0, "LT": 0.0, "RT": 0.35},
        buttons={"RB": True, "A": False, "B": False},
        hats={"DPAD": (-1, 0)},
        mode="insert_axis",
        seq=step,
        t_mono_ns=time.monotonic_ns(),
        backend="preflight_mock",
    )


def write_preflight_episode(root: Path, steps: int) -> dict:
    writer = EpisodeWriter.create(root, task="plug_zed_insertion_xbox_preflight")
    cfg = XboxTeleopConfig()
    validations = []
    for step in range(steps):
        mapped = map_xbox_to_action(mock_insert_controller_state(step), cfg)
        transition = make_hil_transition(
            observations=synthetic_obs(step),
            policy_action=np.zeros(7, dtype=np.float32),
            human_action=mapped.action,
            intervention=True,
            next_observations=synthetic_obs(step + 1),
            reward=0.0,
            done=step == steps - 1,
            info={
                "step_idx": step,
                "teleop": mapped.info,
                "demo_collection_preflight": True,
                "live_packet_required_before_motion": True,
            },
        )
        validations.append(validate_transition(transition))
        writer.append(
            transition,
            sidecar={
                "step_idx": step,
                "teleop": mapped.info,
                "demo_collection_preflight": True,
                "live_packet_required_before_motion": True,
            },
        )
    episode = writer.close()
    export = export_episode_to_lerobot_sidecar(episode["episode_dir"], root / "lerobot_sidecar")
    validation = validate_lerobot_sidecar(export["sidecar"], expected_count=steps)
    return {
        "episode": episode,
        "transition_validations": validations,
        "transition_count": steps,
        "lerobot_export": export,
        "lerobot_validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="No-motion preflight before physical Xbox Plug-ZED demo collection.")
    parser.add_argument("--artifact-root", default="artifacts/plug_zed_xbox_demo_collection_preflight")
    parser.add_argument("--probe-duration-sec", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--require-controller", action="store_true")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    probe = pygame_sdl_controller_probe(args.probe_duration_sec)
    controller_detected = bool(probe.get("controller_device_opened"))
    if args.require_controller and not controller_detected:
        payload = {
            "ok": False,
            "status": BLOCK_STATUS,
            "reject_code": "PHYSICAL_XBOX_CONTROLLER_NOT_DETECTED",
            "controller_probe": probe,
            "required_human_physical_facts": REQUIRED_HUMAN_FACTS,
            "runtime_execution_performed": False,
            "live_demo_collection": False,
            "motion_command": False,
            "gripper_command": False,
            "policy_rollout": False,
            "remote_training": False,
            "droid_mutation": False,
            "openpi_dependency": False,
            "safe_to_execute_live_now": False,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    episode = write_preflight_episode(artifact_root / "episode_preflight", args.steps)
    ok = (
        all(item.get("ok") for item in episode["transition_validations"])
        and episode["lerobot_validation"].get("ok") is True
    )
    status = PREFLIGHT_WITH_CONTROLLER_STATUS if controller_detected else PREFLIGHT_STATUS
    payload = {
        "ok": ok,
        "status": status,
        "artifact_root": str(artifact_root),
        "controller_probe_status": probe.get("status"),
        "physical_xbox_controller_detected": controller_detected,
        "detected_xbox_devices": probe.get("controller_devices", []),
        "demo_output_root_writable": True,
        "episode": episode["episode"],
        "transition_count": episode["transition_count"],
        "lerobot_sidecar_export_status": "PLUG_ZED_XBOX_LEROBOT_SIDECAR_EXPORTED",
        "lerobot_sidecar_schema_version": episode["lerobot_export"].get("schema_version"),
        "lerobot_sidecar_validation": episode["lerobot_validation"],
        "ready_for_live_packet_generation": controller_detected,
        "required_human_physical_facts": REQUIRED_HUMAN_FACTS,
        "runtime_execution_performed": False,
        "live_demo_collection": False,
        "motion_command": False,
        "gripper_command": False,
        "policy_rollout": False,
        "remote_training": False,
        "droid_mutation": False,
        "openpi_dependency": False,
        "safe_to_execute_live_now": False,
        "old_packet_reuse_allowed": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
