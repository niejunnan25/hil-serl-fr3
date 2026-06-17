from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fr3_hil_bridge.demo_recording import EpisodeWriter, make_hil_transition, validate_transition
from fr3_hil_bridge.teleop.xbox_mapping import XboxControllerState, XboxTeleopConfig, map_xbox_to_action


def synthetic_obs(step: int) -> dict:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = step
    return {
        "zed_left": image.copy(),
        "zed_right": image.copy(),
        "state": np.zeros((1, 7), dtype=np.float32),
        "gripper_state": np.array([0.08], dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a mock HIL-SERL Xbox demo episode without robot runtime")
    parser.add_argument("--artifact-root", default="artifacts/plug_zed_xbox_demo_dry_run")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    writer = EpisodeWriter.create(args.artifact_root)
    cfg = XboxTeleopConfig()
    validations = []
    for step in range(args.steps):
        state = XboxControllerState(
            connected=True,
            axes={"LEFTX": 0.0, "LEFTY": 0.0, "RIGHTX": 0.0, "RIGHTY": 0.0, "LT": 0.0, "RT": 0.45},
            buttons={"RB": True, "A": False, "B": False},
            mode="insert_axis",
            seq=step,
            t_mono_ns=time.monotonic_ns(),
            backend="mock",
        )
        mapped = map_xbox_to_action(state, cfg)
        policy_action = np.zeros(7, dtype=np.float32)
        transition = make_hil_transition(
            observations=synthetic_obs(step),
            policy_action=policy_action,
            human_action=mapped.action,
            intervention=True,
            next_observations=synthetic_obs(step + 1),
            reward=float(step == args.steps - 1),
            done=step == args.steps - 1,
            info={"teleop": mapped.info, "step_idx": step},
        )
        validations.append(validate_transition(transition))
        writer.append(transition, sidecar={"step_idx": step, "teleop": mapped.info})

    result = writer.close()
    payload = {
        "ok": all(item.get("ok") for item in validations),
        "status": "PLUG_ZED_XBOX_DEMO_DRY_RUN_WRITTEN",
        "episode": result,
        "transition_validations": validations,
        "transition_count": args.steps,
        "runtime_execution_performed": False,
        "live_demo_collection": False,
        "motion_command": False,
        "gripper_command": False,
        "droid_mutation": False,
        "openpi_dependency": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
