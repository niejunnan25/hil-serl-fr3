#!/usr/bin/env python3
"""Audit converted archives against source NPZs and exercise negative controls.

Offline only. Mutations use disposable HDF5 copies; source data and exported
episodes are never edited. Official schema checks and source equality checks
are deliberately separate: a valid pose is not necessarily the recorded action.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import h5py
import numpy as np

from export_oopsie import CAMS, UNKNOWN_ANNOTATOR, read_json, read_step, sha_file, json_write


def same(actual, expected, label):
    if not np.array_equal(actual, expected):
        raise AssertionError(f"source_mismatch:{label}")


def semantic_check(h5_path, rows, meta):
    with h5py.File(h5_path, "r") as f:
        checks = {
            "actions/cartesian_position": [r["controller_commands"][0]["pose"] for r in rows],
            "observations/robot_states/cartesian_position": [r["raw_state"]["values"]["tcp_pose"] for r in rows],
            "observations/robot_states/joint_position": [r["raw_state"]["values"]["q"] for r in rows],
            "observations/robot_states/gripper_position": [[r["raw_state"]["values"]["gripper_pose"]] for r in rows],
            "hilserl/is_human": [r["source_action"] == "human" for r in rows],
            "hilserl/policy_revision": [r["policy"].get("revision", -1) for r in rows],
            "hilserl/next_cartesian_position": [r["raw_next_state"]["values"]["tcp_pose"] for r in rows],
        }
        for key in ("terminated", "truncated", "observed_reward", "global_step"):
            checks[f"hilserl/{key}"] = [r[key] for r in rows]
        for key, expected in checks.items():
            same(f[key][:], np.asarray(expected), key)
        same(f["actions/gripper_binary"][:], np.ones((len(rows), 1)), "closed_gripper_target")
        same(f["hilserl/episode_end"][:], np.arange(len(rows)) == len(rows)-1, "episode_boundary")
        for edge in ("sample_started", "step_started", "step_ended"):
            for clock in ("monotonic_ns", "unix_ns"):
                same(f[f"hilserl/time/{edge}_{clock}"][:], [r[edge][clock] for r in rows], f"{edge}_{clock}")
        for cam, key in CAMS.items():
            for prefix, field in (("", "frame_references"), ("next_", "next_frame_references")):
                for clock in ("capture_id", "monotonic_ns", "unix_ns"):
                    same(f[f"hilserl/time/{prefix}{cam}_{clock}"][:], [r[field][key][clock] for r in rows], f"{prefix}{cam}_{clock}")
        stored = json.loads(f["hilserl"].attrs["episode_metadata"])
        assert stored == meta, "source episode metadata changed"
        assert f[f"episode_annotations/{UNKNOWN_ANNOTATOR}"].attrs["success"] == meta["outcome"]
        assert f.attrs["submission_ready"] == False
        action = f["actions/cartesian_position"][:]
        observed_next = f["hilserl/next_cartesian_position"][:]
        errors_mm = np.linalg.norm(action[:, :3] - observed_next[:, :3], axis=1)*1000
        return {"all_checked_source_fields_exact": True,
                "action_vs_next_state_position_mm_median": float(np.median(errors_mm)),
                "action_vs_next_state_position_mm_max": float(np.max(errors_mm)),
                "max_action_quaternion_norm_error": float(np.max(abs(np.linalg.norm(action[:, 3:], axis=1)-1))),
                "human_steps": sum(r["source_action"] == "human" for r in rows),
                "policy_steps": sum(r["source_action"] == "policy" for r in rows),
                "source_final_terminated": rows[-1]["terminated"],
                "source_final_truncated": rows[-1]["truncated"],
                "source_final_reason": rows[-1]["termination_reason"]}


def negative_controls(source, rows, meta, scratch):
    from oopsie_data_tools.utils.validation.validation_utils import validate_h5_file
    from oopsie_data_tools.utils.validation.errors import EpisodeValidationError
    outcomes = []
    cases = ("zero_quaternion", "missing_state_row", "missing_video", "next_state_used_as_action", "quaternion_wrong_order")
    scratch.mkdir(exist_ok=True)
    for case in cases:
        with tempfile.TemporaryDirectory(prefix=case+"-", dir=scratch) as td:
            td = Path(td)
            candidate = td/source.name
            shutil.copy2(source, candidate)
            for video in source.parent.glob("*.mp4"):
                os.link(video, td/video.name)
            with h5py.File(candidate, "r+") as f:
                if case == "zero_quaternion":
                    f["actions/cartesian_position"][0, 3:] = 0
                elif case == "missing_state_row":
                    values = f["observations/robot_states/joint_position"][:-1]
                    del f["observations/robot_states/joint_position"]
                    f.create_dataset("observations/robot_states/joint_position", data=values)
                elif case == "missing_video":
                    group = f["observations/video_paths"]
                    del group["front_cam"]
                    group.create_dataset("front_cam", data="does-not-exist.mp4")
                elif case == "next_state_used_as_action":
                    f["actions/cartesian_position"][:] = f["hilserl/next_cartesian_position"][:]
                elif case == "quaternion_wrong_order":
                    values = f["actions/cartesian_position"][:]
                    values[:, 3:] = values[:, [6, 3, 4, 5]]
                    f["actions/cartesian_position"][:] = values
            official_error = semantic_error = None
            try:
                validate_h5_file(str(candidate), strict_annotation_check=True)
            except EpisodeValidationError as exc:
                official_error = str(exc)
            if case in ("next_state_used_as_action", "quaternion_wrong_order"):
                try:
                    semantic_check(candidate, rows, meta)
                except AssertionError as exc:
                    semantic_error = str(exc)
                assert semantic_error is not None, f"failed to catch semantic mutation: {case}"
            else:
                assert official_error is not None, f"official validator did not reject: {case}"
            outcomes.append({"case": case, "detected": True, "official_rejection": official_error,
                             "source_semantic_rejection": semantic_error})
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--toolkit", type=Path, required=True)
    p.add_argument("--negative-controls", action="store_true")
    args = p.parse_args()
    sys.path.insert(0, str(args.toolkit.resolve()))
    results = []
    negative = None
    for h5 in sorted((args.output/"episodes").glob("*/*.h5")):
        cert = read_json(h5.parent/"verification.json")
        ep = args.root/cert["source"]["episode"]
        meta = read_json(ep/"episode.json")
        rows = []
        for npz in sorted(ep.glob("*.npz")):
            raw, digest = read_step(npz)
            assert cert["source_hashes"]["steps"][npz.name] == digest, "source changed"
            rows.append(raw)
        assert len(rows) == cert["steps"] == meta["steps"]
        stats = semantic_check(h5, rows, meta)
        videos = {}
        for cam, mapping in cert["video_checks"].items():
            with h5py.File(h5, "r") as f:
                vp = h5.parent/f[f"observations/video_paths/{cam}"].asstr()[()]
            assert sha_file(vp) == mapping["output_sha256"], "exported video changed"
            cap = cv2.VideoCapture(str(vp))
            info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "bytes": vp.stat().st_size}
            cap.release()
            assert info["frames"] == len(rows) and info["width"] == 1280 and info["height"] == 720
            assert abs(info["fps"]-mapping["fps"]) < 1e-6
            for i, m in enumerate(mapping["source_frame_map"]):
                assert m["capture_id"] == rows[i]["frame_references"][CAMS[cam]]["capture_id"]
            videos[cam] = info
        results.append({"episode": h5.parent.name, "steps": len(rows), "outcome": meta["outcome"], **stats, "videos": videos})
        if args.negative_controls and negative is None:
            negative = negative_controls(h5, rows, meta, args.output/"_negative_tests")
    assert results, "no exported episodes"
    report = {"all_passed": True, "episodes_checked": len(results), "steps_checked": sum(x["steps"] for x in results),
              "negative_controls": negative, "episodes": results}
    json_write(args.output/"semantic-audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
