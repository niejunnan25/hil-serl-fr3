#!/usr/bin/env python3
"""Rebuild audited seed policy images from exact indexed H.264 source frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.recrop_dataset import build_recropped_dataset, validate_recropped_dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original audited 128px seed directory")
    parser.add_argument("--output", type=Path, required=True, help="New dataset directory; never overwritten")
    parser.add_argument("--image-profile", type=Path, help="Frozen image profile JSON")
    parser.add_argument("--expected-source-manifest-sha256", help="SHA256 of the original seed manifest")
    parser.add_argument("--validate-only", action="store_true", help="Audit the existing output and its original sources")
    parser.add_argument("--verify-pixels", action="store_true", help="During validation, also re-decode every mapped source image")
    args = parser.parse_args()
    if args.validate_only:
        manifest = validate_recropped_dataset(args.output, verify_pixels=args.verify_pixels)
    else:
        if not (args.source and args.image_profile and args.expected_source_manifest_sha256):
            parser.error("building requires --source, --image-profile and --expected-source-manifest-sha256")
        if args.verify_pixels:
            parser.error("--verify-pixels is used with --validate-only")
        manifest = build_recropped_dataset(args.source, args.output,
                                          image_profile=json.loads(args.image_profile.read_bytes()),
                                          expected_source_manifest_sha256=args.expected_source_manifest_sha256)
    print(json.dumps({"directory": str(args.output.resolve()), "counts": manifest["counts"],
                      "image_profile": manifest["image_profile"],
                      "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
                      "mapped_images": manifest["image_rebuild"]["frame_mapping"]["rows"],
                      "unique_images": manifest["image_rebuild"]["unique_images"],
                      "source_encoding": manifest["image_rebuild"]["source_encoding"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
