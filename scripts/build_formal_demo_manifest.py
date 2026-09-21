#!/usr/bin/env python3
"""Build a learner demo manifest from audited SERL19 full trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--quarantine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-rank", type=int, default=21)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--limit-success", type=int, default=0)
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    quarantine = json.loads(Path(args.quarantine).read_text(encoding="utf-8"))
    quarantined = {entry["path"] for entry in quarantine.get("quarantine", [])}

    candidate_rank = {
        entry["path"]: entry
        for entry in quarantine.get("candidate_after_first20", [])
        if entry.get("rank", 0) >= args.min_rank
    }
    serl19_by_path = {
        r["path"]: r for r in metadata.get("records", [])
        if r.get("root_label") == "serl19_demos" and r.get("path") not in quarantined
    }
    successes = []
    for path, candidate in candidate_rank.items():
        if "success" not in Path(path).name:
            continue
        merged = dict(serl19_by_path.get(path, {}))
        merged.update(candidate)
        merged["label"] = "success"
        successes.append(merged)
    successes = sorted(successes, key=lambda r: r.get("rank", 10**9))
    if args.limit_success > 0:
        successes = successes[: args.limit_success]
    failures = []
    if args.include_failures:
        failures = sorted(
            [
                dict(r, label="fail")
                for r in serl19_by_path.values()
                if "fail" in Path(r.get("path", "")).name
            ],
            key=lambda r: r.get("path", ""),
        )

    entries = []
    for r in successes + failures:
        entries.append({
            "path": r["path"],
            "name": Path(r["path"]).name,
            "label": r.get("label"),
            "rank": r.get("rank"),
            "len": r.get("len"),
            "reward_sum": r.get("reward_sum"),
            "done_last": r.get("done_last"),
            "allowed_for": "rlpd_demo_buffer",
            "source": "serl19_full_trajectory",
        })

    report = {
        "generated_from": {
            "metadata": str(args.metadata),
            "quarantine": str(args.quarantine),
        },
        "selection_rule": {
            "min_rank": args.min_rank,
            "include_failures": args.include_failures,
            "limit_success": args.limit_success,
            "quarantined_count": len(quarantined),
        },
        "demo_paths": [entry["path"] for entry in entries],
        "entries": entries,
        "counts": {
            "entries": len(entries),
            "success": sum(1 for e in entries if e["label"] == "success"),
            "fail": sum(1 for e in entries if e["label"] == "fail"),
            "transitions": sum(int(e.get("len") or 0) for e in entries),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    if report["counts"]["success"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
