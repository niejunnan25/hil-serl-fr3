from __future__ import annotations

import argparse
import json

from fr3_hil_bridge.demo_export import export_episode_to_lerobot_sidecar, validate_lerobot_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Plug-ZED HIL-SERL demo episode to a LeRobot-compatible sidecar")
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    export = export_episode_to_lerobot_sidecar(args.episode_dir, args.output_dir)
    validation = validate_lerobot_sidecar(export["sidecar"], expected_count=export["transition_count"])
    payload = {
        "ok": validation.get("ok") is True,
        "status": "PLUG_ZED_XBOX_LEROBOT_SIDECAR_EXPORTED",
        "export": export,
        "validation": validation,
        "runtime_execution_performed": False,
        "live_demo_collection": False,
        "motion_command": False,
        "gripper_command": False,
        "policy_rollout": False,
        "remote_training": False,
        "droid_mutation": False,
        "openpi_dependency": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
