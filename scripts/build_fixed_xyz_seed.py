#!/usr/bin/env python3
"""Derive audited fixed-XYZ demonstrations without starting any device process."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.seed_dataset import build_seed_dataset, validate_seed_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[], help="Source run directory; repeat for multiple runs")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory; never overwritten")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and not args.run:
        parser.error("building requires at least one --run")
    manifest = validate_seed_dataset(args.output) if args.validate_only else build_seed_dataset(args.run, args.output)
    print(json.dumps({"directory": str(args.output.resolve()), "action_contract": manifest["action_contract"],
                      "counts": manifest["counts"], "omitted_counts": manifest["omitted_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
