#!/usr/bin/env python3
"""Build reward-classifier training images from recorded episodes.

The recorder already stores, for every step, the exact image the classifier sees
at deploy time (`observations.side_classifier`, already preprocessed with the
run's image profile). Every episode also carries a human 0/1 verdict. Together
those two facts let the classifier dataset be derived from demos that were
collected anyway, with no separate image-collection pass.

Labelling rule (heuristic, and recorded as such in the manifest):
  * episode complete, verdict_source == "human", outcome == 1
      -> the last `--positive-tail` steps are positives (plug seated),
         the remaining steps are negatives (approach).
  * episode complete, verdict_source == "human", outcome == 0
      -> every step is a negative.
  * anything incomplete or without a human verdict is skipped, never guessed.

Positives are the scarce class, so negatives are subsampled to a fixed ratio and
one contact sheet per class is written next to the manifest so the labels can be
eyeballed before training.

Usage:
  python scripts/build_classifier_data_from_runs.py \
      --run artifacts/runs/<collect-run> --output-dir data/classifier_front \
      --positive-tail 15 --negative-ratio 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hilserl import storage


class BuildError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_episodes(run_dirs):
    """Yield (identifier, episode_dir, metadata) for every episode under the runs."""
    for run in run_dirs:
        run = Path(run)
        for meta_path in sorted(run.glob("recordings/*/episodes/*/episode.json")):
            meta = json.loads(meta_path.read_text())
            identifier = f"{run.name}-{meta_path.parent.name}"
            yield identifier, meta_path.parent, meta


def episode_steps(episode_dir: Path):
    return sorted(episode_dir.glob("*.npz"))


def classifier_image(step_path: Path) -> np.ndarray:
    raw = storage.read_step(step_path)
    value = raw["observations"]["side_classifier"]
    value = np.asarray(value).reshape(-1, value.shape[-3], value.shape[-2], value.shape[-1])
    if value.shape[0] != 1:
        raise BuildError(f"{step_path}: expected a single classifier image, got {value.shape}")
    image = value[0]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise BuildError(f"{step_path}: unexpected classifier image {image.shape} {image.dtype}")
    return image, raw.get("image_profile")


def write_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path, format="PNG", optimize=False)


def contact_sheet(files, path: Path, columns: int = 10, tile: int = 96) -> None:
    if not files:
        return
    sample = files[:: max(1, len(files) // (columns * 6))][: columns * 6]
    rows = (len(sample) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), (16, 16, 16))
    for index, item in enumerate(sample):
        with Image.open(item) as image:
            thumb = image.convert("RGB").resize((tile, tile), Image.LANCZOS)
        sheet.paste(thumb, ((index % columns) * tile, (index // columns) * tile))
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Run directory to read; repeat for multiple runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-tail", type=int, default=15,
                        help="Trailing steps of a successful episode treated as seated (default 15)")
    parser.add_argument("--negative-ratio", type=float, default=3.0,
                        help="Maximum negatives per positive (default 3.0)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"{args.output_dir} already exists and is not empty; choose a new directory")

    positives = []
    negatives = []
    negatives_available = 0
    skipped = []
    profiles = set()
    per_episode = []

    for identifier, episode_dir, meta in discover_episodes(args.run):
        if not meta.get("complete") or meta.get("verdict_source") != "human" or meta.get("outcome") not in (0, 1):
            skipped.append({"episode": identifier, "reason": "not a complete human-verdict episode",
                            "outcome": meta.get("outcome"), "complete": meta.get("complete"),
                            "verdict_source": meta.get("verdict_source")})
            continue
        steps = episode_steps(episode_dir)
        if not steps:
            skipped.append({"episode": identifier, "reason": "no step files"})
            continue
        success = meta["outcome"] == 1
        tail = min(args.positive_tail, len(steps))
        episode_positive = episode_negative = 0
        for index, step_path in enumerate(steps):
            image, profile = classifier_image(step_path)
            if profile:
                profiles.add(profile)
            seated = success and index >= len(steps) - tail
            record = (identifier, step_path.stem, image)
            if seated:
                positives.append(record)
                episode_positive += 1
            else:
                negatives.append(record)
                episode_negative += 1
        per_episode.append({"episode": identifier, "outcome": meta["outcome"],
                            "steps": len(steps), "positive": episode_positive,
                            "negative": episode_negative,
                            "termination_reason": meta.get("termination_reason")})

    if not positives:
        parser.error("no positives found; check that the runs contain complete human-verdict successes")
    if len(profiles) > 1:
        raise BuildError(f"runs mix image profiles {sorted(profiles)}; build them separately")

    negatives_available = len(negatives)
    rng = np.random.default_rng(args.seed)
    limit = int(len(positives) * args.negative_ratio)
    if len(negatives) > limit:
        keep = rng.choice(len(negatives), size=limit, replace=False)
        negatives = [negatives[i] for i in sorted(keep)]

    (args.output_dir / "positive").mkdir(parents=True)
    (args.output_dir / "negative").mkdir(parents=True)
    written = {"positive": [], "negative": []}
    for label, records in (("positive", positives), ("negative", negatives)):
        for identifier, step, image in records:
            name = f"{identifier.replace('/', '_')}_{step}.png"
            path = args.output_dir / label / name
            write_image(path, image)
            written[label].append({"file": name, "episode": identifier, "step": step})

    manifest = {
        "source_runs": [str(Path(run).resolve()) for run in args.run],
        "image_profile": sorted(profiles),
        "classifier_key": "side_classifier",
        "label_rule": {
            "positive": f"complete human-verdict success, last {args.positive_tail} steps",
            "negative": "other steps of successful episodes, plus every step of complete human-verdict failures",
            "negative_ratio": args.negative_ratio,
            "caveat": ("tail-of-episode is a heuristic for the seated state; review "
                       "contact_sheet_positive.png before training"),
        },
        "counts": {"positive": len(positives), "negative": len(negatives),
                   "negatives_available": negatives_available},
        "per_episode": per_episode,
        "skipped": skipped,
        "files": written,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    contact_sheet([args.output_dir / "positive" / item["file"] for item in written["positive"]],
                  args.output_dir / "contact_sheet_positive.png")
    contact_sheet([args.output_dir / "negative" / item["file"] for item in written["negative"]],
                  args.output_dir / "contact_sheet_negative.png")

    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "counts": manifest["counts"],
                      "episodes_used": len(per_episode), "episodes_skipped": len(skipped),
                      "image_profile": manifest["image_profile"],
                      "manifest_sha256": _sha256(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
