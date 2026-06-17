from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fr3_hil_bridge.demo_recording import EpisodeWriter, make_hil_transition, validate_transition
from fr3_hil_bridge.teleop.xbox_mapping import XboxControllerState, XboxTeleopConfig, map_xbox_to_action


def synthetic_state_obs(step: int) -> dict:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 1] = step
    return {
        "zed_left": image.copy(),
        "zed_right": image.copy(),
        "state": np.zeros((1, 7), dtype=np.float32),
        "gripper_state": np.array([0.08], dtype=np.float32),
        "ee_pose": np.array([0.55, 0.0, 0.53, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }


def mock_controller_state(step: int, mode: str) -> XboxControllerState:
    return XboxControllerState(
        connected=True,
        axes={"LEFTX": 0.08, "LEFTY": -0.10, "RIGHTX": 0.05, "RIGHTY": 0.0, "LT": 0.0, "RT": 0.35},
        buttons={"RB": True, "A": False, "B": False},
        hats={"DPAD": (0, 0)},
        mode=mode,
        seq=step,
        t_mono_ns=time.monotonic_ns(),
        backend="mock",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="No-motion Plug-ZED Xbox state-only actor dry run")
    parser.add_argument("--artifact-root", default="artifacts/plug_zed_xbox_state_only_actor")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mode", choices=["coarse", "fine", "rotation", "insert_axis"], default="insert_axis")
    args = parser.parse_args()

    writer = EpisodeWriter.create(args.artifact_root, task="plug_zed_insertion_state_only")
    cfg = XboxTeleopConfig()
    validations = []
    for step in range(args.steps):
        state = mock_controller_state(step, args.mode)
        mapped = map_xbox_to_action(state, cfg)
        obs = synthetic_state_obs(step)
        next_obs = synthetic_state_obs(step + 1)
        transition = make_hil_transition(
            observations=obs,
            policy_action=np.zeros(7, dtype=np.float32),
            human_action=mapped.action,
            intervention=True,
            next_observations=next_obs,
            reward=0.0,
            done=False,
            info={
                "step_idx": step,
                "teleop": mapped.info,
                "state_only": True,
                "camera_required_for_live_demo": True,
            },
        )
        validations.append(validate_transition(transition))
        writer.append(transition, sidecar={"step_idx": step, "teleop": mapped.info, "state_only": True})

    result = writer.close()
    payload = {
        "ok": all(item.get("ok") for item in validations),
        "status": "PLUG_ZED_XBOX_STATE_ONLY_ACTOR_DRY_RUN_WRITTEN",
        "episode": result,
        "transition_count": args.steps,
        "transition_validations": validations,
        "runtime_execution_performed": False,
        "live_demo_collection": False,
        "fr3_connection": False,
        "fci_state_read": False,
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
