from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


def ensure_import_paths(root: Path) -> None:
    paths = [
        root,
        root / "upstream" / "hil-serl" / "serl_robot_infra",
        root / "upstream" / "hil-serl" / "examples",
    ]
    for path in paths:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def dry_run_summary(
    role: str,
    headless: bool,
    camera_profile: str,
    check_camera_service: bool,
    camera_service_url: str,
    check_fci_state: bool,
) -> dict[str, Any]:
    root = Path(os.environ.get("SERL_FR3_ROOT", "/home/robot/serl_projects/hil-serl-fr3"))
    ensure_import_paths(root)
    if headless:
        os.environ.setdefault("PYNPUT_BACKEND", "dummy")

    if camera_profile == "legacy_realsense":
        from fr3_experiments.plug_insertion.config import TrainConfig

        cfg = TrainConfig()
        train_config = "fr3_experiments.plug_insertion.config.TrainConfig"
        observation_contract = {
            "images": list(cfg.image_keys),
            "classifier_images": list(cfg.classifier_keys),
            "proprio": list(cfg.proprio_keys),
        }
        zed_observation_validation = None
        camera_service_validation = None
        fci_state_validation = None
    elif camera_profile == "zed_stereo":
        from fr3_experiments.plug_insertion.zed_config import ZedTrainConfig

        cfg = ZedTrainConfig()
        train_config = "fr3_experiments.plug_insertion.zed_config.ZedTrainConfig"
        observation_contract = cfg.observation_contract().to_dict()
        zed_observation_validation = cfg.validate_synthetic_observation()
        camera_service_validation = None
        fci_state_validation = None
        if check_camera_service:
            from fr3_hil_bridge.zed_camera_client import fetch_zed_observation

            camera_service_validation = fetch_zed_observation(camera_service_url).to_dict()
        if check_fci_state:
            from fr3_hil_bridge.fci_state_client import fetch_fci_gripper_state

            fci_state_validation = fetch_fci_gripper_state().to_dict()
    else:
        raise ValueError(f"UNKNOWN_CAMERA_PROFILE {camera_profile}")

    if check_camera_service and camera_profile != "zed_stereo":
        raise ValueError("CAMERA_SERVICE_CHECK_REQUIRES_ZED_STEREO")

    return {
        "role": role,
        "dry_run": True,
        "headless": headless,
        "camera_profile": camera_profile,
        "check_camera_service": check_camera_service,
        "check_fci_state": check_fci_state,
        "camera_service_url": camera_service_url if camera_profile == "zed_stereo" else None,
        "pynput_backend": os.environ.get("PYNPUT_BACKEND"),
        "train_config": train_config,
        "uses_upstream_config_mapping": False,
        "env_constructed": False,
        "action_shape": [7],
        "observation_contract": observation_contract,
        "image_keys": list(cfg.image_keys),
        "classifier_keys": list(cfg.classifier_keys),
        "proprio_keys": list(cfg.proprio_keys),
        "zed_observation_validation": zed_observation_validation,
        "camera_service_validation": camera_service_validation,
        "fci_state_validation": fci_state_validation,
        "no_reset_step_fci_camera_or_gripper": not check_camera_service and not check_fci_state,
        "no_reset_step_or_motion": True,
        "no_gripper_command": True,
        "camera_source_requested": check_camera_service,
        "fci_state_requested": check_fci_state,
        "gripper_state_requested": check_fci_state,
        "camera_service_check_does_not_start_service": True,
        "live_actor_requires_separate_gate": True,
        "intervention_device_decision_required": True,
    }


def main(default_role: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local FR3 plug runner wrapper")
    parser.add_argument("--role", choices=["learner", "actor"], default=default_role)
    parser.add_argument("--camera-profile", choices=["legacy_realsense", "zed_stereo"], default="legacy_realsense")
    parser.add_argument("--check-camera-service", action="store_true")
    parser.add_argument("--camera-service-url", default="http://127.0.0.1:54819")
    parser.add_argument("--check-fci-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.role is None:
        parser.error("--role is required")
    if not args.dry_run:
        raise SystemExit(
            "LIVE_PLUG_RUNNER_NOT_ENABLED: use --dry-run, or create a separate live gate"
        )
    print(
        json.dumps(
            dry_run_summary(
                args.role,
                args.headless,
                args.camera_profile,
                args.check_camera_service,
                args.camera_service_url,
                args.check_fci_state,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
