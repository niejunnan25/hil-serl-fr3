from __future__ import annotations

import argparse
import json

from fr3_hil_bridge.teleop.xbox_backend import mock_probe, pygame_joystick_probe, pygame_sdl_controller_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Xbox controller without robot runtime")
    parser.add_argument("--backend", choices=["mock", "pygame", "sdl"], default="mock")
    parser.add_argument("--duration-sec", type=float, default=2.0)
    args = parser.parse_args()

    if args.backend == "mock":
        payload = mock_probe(args.duration_sec)
    elif args.backend == "pygame":
        payload = pygame_joystick_probe(args.duration_sec)
    else:
        payload = pygame_sdl_controller_probe(args.duration_sec)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
