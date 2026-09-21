#!/usr/bin/env python3
"""Build an offline RoboMeter reward cache; never open the robot or cameras."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.config import load_config
from hilserl.reward_provider import RoboMeterClient
from hilserl.reward_seed import prepare_seed_cache
from hilserl.seed_dataset import iter_seed_transitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    spec = config.reward_spec()
    if spec is None or not config.seed_dataset_sha256:
        raise ValueError("A pinned fixed-XYZ reward configuration is required")
    provider = RoboMeterClient(spec, config.reward_url, batch_size=config.reward_batch_size,
                              timeout=config.reward_timeout_seconds)
    try:
        source = iter_seed_transitions(config.path(config.demo_dir),
            expected_action_contract=config.action_contract, expected_manifest_sha256=config.seed_dataset_sha256,
            expected_image_profile=config.image_profile, expected_action_max_z_step=config.action_max_z_step,
            expected_position_target_mode=config.position_target_mode)
        result = prepare_seed_cache(source, config.path(args.output), config.seed_dataset_sha256, spec, provider)
        print(json.dumps(result, indent=2))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
