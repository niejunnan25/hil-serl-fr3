from __future__ import annotations

import argparse
import json
import time

from fr3_hil_bridge.teleop.xbox_mapping import XboxControllerState, XboxTeleopConfig, map_xbox_to_action, validate_mapping_result


def mock_state(mode: str, deadman: bool = True) -> XboxControllerState:
    return XboxControllerState(
        connected=True,
        axes={
            "LEFTX": 0.25,
            "LEFTY": -0.35,
            "RIGHTX": 0.20,
            "RIGHTY": 0.0,
            "LT": 0.0,
            "RT": 0.45,
        },
        buttons={"RB": deadman, "A": False, "B": False},
        hats={"DPAD": (0, 0)},
        mode=mode,
        seq=1,
        t_mono_ns=time.monotonic_ns(),
        backend="mock",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview Xbox to normalized 7D Plug-ZED action")
    parser.add_argument("--mode", choices=["coarse", "fine", "rotation", "insert_axis"], default="insert_axis")
    parser.add_argument("--deadman", choices=["pressed", "released"], default="pressed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    state = mock_state(args.mode, deadman=args.deadman == "pressed")
    cfg = XboxTeleopConfig()
    result = map_xbox_to_action(state, cfg)
    payload = {
        "ok": True,
        "status": "PLUG_ZED_XBOX_ACTION_PREVIEW_NO_MOTION",
        "runtime_execution_performed": False,
        "motion_command": False,
        "gripper_command": False,
        "droid_mutation": False,
        "openpi_dependency": False,
        "mapping": result.to_dict(),
        "validation": validate_mapping_result(result),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
