#!/usr/bin/env python3
"""Offline, resumable HIL-SERL -> Oopsie local-validation conversion.

Never contacts a robot or uploads. Original archives are read-only. Explicitly
versioned fixed-XYZ train/eval episodes are supported; others are inventoried.
Videos use one *recorded input-frame reference* per step at the configured nominal
rate. All real timestamps are retained; playback time is not wall-clock time.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np

VERSION = "hilserl-oopsie-local-v1"
TOOLKIT_COMMIT = "cca4e6b23f97d2732bd28eaa472d509828e17e4e"
CAMS = {"front_cam": "side_policy", "wrist_cam": "wrist_1"}
LOCAL_LAB = "LOCAL_VALIDATION_UNREGISTERED"
UNKNOWN_OPERATOR = "source_operator_identity_unrecorded"
UNKNOWN_ANNOTATOR = "imported_human_verdict_identity_unrecorded"


class Reject(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Reject(reason)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def json_write(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_step(path):
    data_bytes = path.read_bytes()
    with np.load(io.BytesIO(data_bytes), allow_pickle=False) as data:
        def unpack(v):
            if isinstance(v, dict):
                if set(v) == {"__array__"}:
                    return data[v["__array__"]].copy()
                return {k: unpack(x) for k, x in v.items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return v
        raw = unpack(json.loads(str(data["__tree__"])))
    return raw, sha_bytes(data_bytes)


def ns(value):
    return value["monotonic_ns"]


def finite(value, shape, label):
    a = np.asarray(value)
    require(a.shape == shape and np.isfinite(a).all(), f"invalid_array:{label}:{a.shape}")
    return a


def check_pose(value, label):
    a = finite(value, (7,), label)
    require(abs(np.linalg.norm(a[3:]) - 1) < 1e-4, f"invalid_quaternion:{label}")
    return a


def inventory(root):
    entries = []
    for mp in sorted((root / "artifacts/runs").glob("*/recordings/*/manifest.json")):
        m = read_json(mp)
        capture = mp.parent
        for ep in sorted((capture / "episodes").glob("*/episode.json")):
            e = read_json(ep)
            mode = e.get("mode", m.get("metadata", {}).get("mode"))
            entry = {"source": str(ep.parent.relative_to(root)),
                     "selection_id": f"{capture.parents[1].name}/{capture.name}/{e['id']}",
                     "mode": mode, "steps": e.get("steps", 0), "outcome": e.get("outcome"),
                     "capture_status": m.get("status"), "eligible": False}
            if m.get("metadata", {}).get("synthetic"):
                reason = "synthetic_recording"
            elif m.get("status") == "recording":
                reason = "capture_still_recording"
            elif not e.get("complete") or e.get("status") != "complete":
                reason = "incomplete_episode"
            elif mode not in ("train", "eval"):
                reason = "human_collect_mode_retained_as_demonstrations"
            elif e.get("action_contract", m.get("metadata", {}).get("action_contract")) != "fixed-xyz-v1":
                reason = "legacy_or_missing_action_contract_requires_separate_mapping"
            elif e.get("verdict_source") != "human" or e.get("outcome") not in (0, 1):
                reason = "missing_final_human_verdict"
            else:
                reason = None
                entry["eligible"] = True
            entry["reason"] = reason
            entries.append(entry)
    return entries


def load_episode(root, entry):
    directory = root / entry["source"]
    capture = directory.parents[1]
    meta_path = directory / "episode.json"
    meta_bytes = meta_path.read_bytes()
    meta = json.loads(meta_bytes)
    manifest = read_json(capture / "manifest.json")
    cfg_path = Path(manifest["metadata"]["config_file"]).resolve()
    require(cfg_path.is_relative_to(root), "config_outside_project")
    cfg_bytes = cfg_path.read_bytes()
    cfg = json.loads(cfg_bytes)
    n = meta["steps"]
    require(n > 0 and n == meta["committed_steps"], "step_count_mismatch")
    files = sorted(directory.glob("*.npz"))
    require([p.name for p in files] == [f"{i:06d}.npz" for i in range(n)], "raw_step_files_missing_or_extra")
    rows, hashes = [], {}
    image_profile = meta["image_profile"]
    for i, path in enumerate(files):
        r, h = read_step(path)
        hashes[path.name] = h
        require(r.get("complete_transition") is True, f"incomplete_transition:{i}")
        require(r["episode_step"] == i and r["episode_id"] == meta["id"], f"wrong_step_identity:{i}")
        require(r.get("action_contract") == "fixed-xyz-v1", f"wrong_action_contract:{i}")
        require(r.get("image_profile") == image_profile == cfg["image_profile"], f"image_profile_mismatch:{i}")
        action = finite(r["actions"], (7,), f"actions:{i}")
        learning = finite(r["learning_action"], (3,), f"learning_action:{i}")
        require(np.array_equal(action[:3], learning) and np.all(action[3:] == 0), f"fixed_axis_contract_violation:{i}")
        require(r["source_action"] in ("human", "policy"), f"unknown_action_source:{i}")
        commands = r["controller_commands"]
        require(len(commands) == 1 and commands[0].get("kind") == "pose", f"unsupported_command_sequence:{i}")
        require(commands[0].get("request_sent") is True and commands[0].get("returned") is True,
                f"command_not_confirmed:{i}")
        check_pose(commands[0]["pose"], f"pose_command:{i}")
        require(not (r["terminated"] or r["truncated"]) or i == n-1, f"premature_episode_termination:{i}")
        require(ns(r["sample_started"]) <= ns(r["step_started"]) <= ns(commands[0]["sent"])
                <= ns(commands[0]["response"]) <= ns(r["step_ended"]), f"invalid_step_time_order:{i}")
        for state_key in ("raw_state", "raw_next_state"):
            state = r[state_key]
            check_pose(state["values"]["tcp_pose"], f"{state_key}.pose:{i}")
            finite(state["values"]["q"], (7,), f"{state_key}.q:{i}")
            finite(state["values"]["dq"], (7,), f"{state_key}.dq:{i}")
            health = state["capture"].get("server_health", {})
            require(health.get("controller_running") is True and
                    all(health.get(k) is False for k in ("state_stale", "jacobian_stale", "gripper_state_stale")),
                    f"feedback_health_not_confirmed:{i}:{state_key}")
        for obs_key in ("observations", "next_observations"):
            finite(r[obs_key]["state"], (1, 19), f"{obs_key}.state:{i}")
            for cam in CAMS.values():
                a = r[obs_key][cam]
                require(a.ndim == 4 and a.shape[0] == 1 and a.shape[-1] == 3 and a.dtype == np.uint8,
                        f"invalid_image_array:{i}:{cam}")
        if rows:
            require(all(np.array_equal(rows[-1]["next_observations"][k], r["observations"][k])
                        for k in r["observations"]), f"noncontiguous_observations:{i}")
            require(ns(r["step_started"]) > ns(rows[-1]["step_started"]), f"nonmonotonic_action_time:{i}")
        rows.append(r)
    # The UI may end a collection between steps. In that case the NPZ still
    # correctly says nonterminal; episode.json records the later manual end.
    manual_boundary = (meta["termination_reason"] == "manual" and rows[-1]["termination_reason"] is None
                       and not rows[-1]["terminated"] and not rows[-1]["truncated"])
    require(rows[-1]["termination_reason"] == meta["termination_reason"] or manual_boundary,
            "final_termination_reason_mismatch")
    require(ns(meta["collection_ended"]) == ns(rows[-1]["step_ended"]), "collection_end_not_at_last_step")
    require(sum(r["source_action"] == "human" for r in rows) == meta["human_steps"], "human_count_mismatch")
    require(meta_path.read_bytes() == meta_bytes and cfg_path.read_bytes() == cfg_bytes, "source_changed_during_read")
    source_hash = sha_bytes(json.dumps(hashes, sort_keys=True).encode() + meta_bytes + cfg_bytes)
    return capture, meta, cfg, rows, {"steps": hashes, "episode": sha_bytes(meta_bytes),
                                    "config": sha_bytes(cfg_bytes), "combined": source_hash}


def resolve_frames(capture, rows, camera):
    path = capture / "video" / camera / "frames.jsonl"
    needed = {r[key][camera]["capture_id"]: r[key][camera] for r in rows
              for key in ("frame_references", "next_frame_references")}
    matches = {}
    first_frame = {}
    with path.open() as stream:
        for line in stream:
            f = json.loads(line)
            first_frame.setdefault(f["segment"], f["frame"])
            if f["capture_id"] in needed:
                require(f["capture_id"] not in matches, "duplicate_capture_id_in_video_index")
                require(f["monotonic_ns"] == needed[f["capture_id"]]["monotonic_ns"], "frame_timestamp_mismatch")
                matches[f["capture_id"]] = f
    require(len(matches) == len(needed), f"missing_video_frame_references:{camera}")
    selected = [matches[r["frame_references"][camera]["capture_id"]] for r in rows]
    require(all(a["frame"] <= b["frame"] for a, b in zip(selected, selected[1:])), "nonmonotonic_video_frames")
    for f in matches.values():
        require((capture / "video" / camera / f"{f['segment']:06d}.mp4").is_file(), "missing_video_segment")
    return selected, first_frame, sha_file(path)


def encode_video(capture, camera, selected, first_frame, rows, profile, output, fps, previews, project_root):
    from hilserl.image_profile import preprocess_image
    cap = None
    proc = None
    current_segment = None
    current_local = -1
    frame = None
    checks, sampled_pixels, crop_errors = [], [], []
    chosen_previews = {0, len(rows)//2, len(rows)-1}
    error_log = output.with_suffix(".encoder.log")
    with error_log.open("wb") as log:
        try:
            for i, f in enumerate(selected):
                segment = f["segment"]
                wanted = f["frame"] - first_frame[segment]
                if segment != current_segment:
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(str(capture / "video" / camera / f"{segment:06d}.mp4"))
                    require(cap.isOpened(), f"cannot_decode_source_segment:{camera}:{segment}")
                    require(cap.set(cv2.CAP_PROP_POS_FRAMES, wanted), "cannot_seek_source_video")
                    current_local = wanted - 1
                    current_segment = segment
                while current_local < wanted:
                    require(cap.grab(), f"source_video_missing_decoded_frame:{camera}:{f['frame']}")
                    current_local += 1
                ok, frame = cap.retrieve()
                require(ok and frame is not None, "source_frame_decode_failed")
                require(frame.shape == (720,1280,3), f"unexpected_source_resolution:{frame.shape}")
                if proc is None:
                    command = ["ffmpeg","-nostdin","-hide_banner","-loglevel","error","-n",
                        "-f","rawvideo","-pixel_format","bgr24","-video_size","1280x720","-framerate",str(fps),
                        "-i","pipe:0","-an","-c:v","libx264","-preset","fast","-crf","19","-threads","2",
                        "-pix_fmt","yuv420p","-movflags","+faststart",str(output)]
                    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                expected = rows[i]["observations"][camera][0]
                recrop = preprocess_image(frame, camera, profile)
                mae = float(np.abs(recrop.astype(np.float32)-expected).mean())
                require(mae < 15, f"source_video_does_not_match_model_observation:{camera}:{i}:mae={mae:.3f}")
                crop_errors.append(mae)
                checks.append({"episode_step":i, "capture_id":f["capture_id"], "source_global_frame":f["frame"],
                               "source_segment":segment, "segment_frame":wanted,
                               "decoded_source_sha256":sha_bytes(frame.tobytes()), "model_crop_mae":mae})
                sampled_pixels.append(frame[::8,::8].copy())
                proc.stdin.write(frame.tobytes())
                if i in chosen_previews:
                    cv2.imwrite(str(previews / f"{i:06d}-{camera}-source.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY,90])
                    cv2.imwrite(str(previews / f"{i:06d}-{camera}-model.png"), expected[...,::-1])
            proc.stdin.close()
            require(proc.wait(timeout=120) == 0, f"ffmpeg_encoder_failed:{error_log}")
        finally:
            if cap is not None:
                cap.release()
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
    # Decode every output frame; source-vs-output correspondence is checked even
    # though H.264 encoding is lossy. No pixel-equality claim is made for MP4s.
    verify = cv2.VideoCapture(str(output))
    errors = []
    try:
        for i, expected in enumerate(sampled_pixels):
            ok, frame = verify.read()
            require(ok, f"output_video_missing_frame:{camera}:{i}")
            mae = float(np.abs(frame[::8,::8].astype(np.float32)-expected).mean())
            require(mae < 12, f"output_frame_correspondence_failed:{camera}:{i}:mae={mae}")
            errors.append(mae)
        require(not verify.read()[0], "output_video_has_extra_frames")
    finally:
        verify.release()
    return {"frames":len(selected), "width":1280,"height":720,"fps":fps,
            "model_crop_mae_max":max(crop_errors), "output_decode_mae_max":max(errors),
            "source_frame_map":checks, "output_sha256":sha_file(output),
            "quality_note":"Decoded archived CRF23 source, re-encoded CRF19; not recovered lossless raw sensor pixels."}


def write_h5(path, meta, cfg, rows, videos, source, args):
    from oopsie_data_tools.utils.conversion_utils import write_root_attrs,write_actions,write_video_paths,write_episode_annotations,write_additional_data
    from oopsie_data_tools.utils.robot_profile.robot_profile import RobotProfile, AdditionalDataSource
    n = len(rows)
    profile = RobotProfile(policy_name="HIL-SERL SAC; per-step revision and human takeover in hilserl group",
        robot_name="Franka FR3", gripper_name="two_finger_gripper_model_unconfirmed",
        is_biarm=False,uses_mobile_base=False,control_freq=float(cfg["control_hz"]),camera_names=list(CAMS),
        robot_state_keys=["cartesian_position","joint_position","gripper_position"],
        robot_state_joint_names=[f"fr3_joint{i}" for i in range(1,8)],
        action_space=["cartesian_position","gripper_binary"],orientation_representation="quat",
        robot_state_orientation_representation="quat",controller="cartesian_impedance",
        additional_data={"tcp_wrench":AdditionalDataSource(sensor="Franka bridge feedback",
                         sensor_info={"source_fields":["tcp_force","tcp_torque"],"units":["N","Nm"],
                                      "frame":"native bridge frame; not transformed by converter"}),
                         "joint_velocity":AdditionalDataSource(sensor="Franka joint feedback",sensor_info={"units":"rad/s"})})
    states = {"cartesian_position":np.stack([r["raw_state"]["values"]["tcp_pose"] for r in rows]).astype(np.float64),
              "joint_position":np.stack([r["raw_state"]["values"]["q"] for r in rows]).astype(np.float64),
              "gripper_position":np.array([r["raw_state"]["values"]["gripper_pose"] for r in rows],dtype=np.float64).reshape(n,1)}
    actions = {"cartesian_position":np.stack([r["controller_commands"][0]["pose"] for r in rows]).astype(np.float64),
               "gripper_binary":np.ones((n,1),dtype=np.float64)}
    with h5py.File(path,"w") as f:
        write_root_attrs(f,episode_id=path.stem,language_instruction=args.instruction,
                         lab_id=LOCAL_LAB,operator_name=UNKNOWN_OPERATOR,robot_profile=profile,
                         timestamp=meta["started"]["unix_ns"]/1e9)
        f.attrs["local_validation_only"] = True
        f.attrs["submission_ready"] = False
        f.attrs["metadata_unresolved"] = json.dumps(["registered lab_id","operator/annotator identity","task wording review",
                                                    "gripper model","acceptance of mixed human/policy rollouts","irregular timing disclosure"])
        f.attrs["gripper_binary_encoding"] = "1=closed,0=open; constant desired closed state confirmed by user; not repeated close-command events"
        f.attrs["gripper_state_units"] = "original bridge scalar; preserved without assuming metres"
        f.attrs["cartesian_pose_convention"] = "absolute robot-base-frame [x,y,z,qx,qy,qz,qw]; xyz in metres, quaternion dimensionless"
        f.attrs["joint_position_units"] = "radians; measured state, not inferred action targets"
        f.attrs["time_axis"] = "one recorded input frame per action; nominal-rate step playback; see hilserl/time for actual wall clock"
        rs = f.require_group("observations/robot_states")
        for key,a in states.items(): rs.create_dataset(key,data=a)
        write_actions(f,actions,profile.action_space)
        write_video_paths(f,{k:str(v) for k,v in videos.items()},path)
        write_additional_data(f,profile,{
            "tcp_wrench":np.stack([np.concatenate([r["raw_state"]["values"]["tcp_force"],r["raw_state"]["values"]["tcp_torque"]]) for r in rows]),
            "joint_velocity":np.stack([r["raw_state"]["values"]["dq"] for r in rows])})
        label_time=datetime.fromtimestamp(meta["verdict_time"]["unix_ns"]/1e9,tz=timezone.utc).isoformat()
        write_episode_annotations(f,annotator_name=UNKNOWN_ANNOTATOR,success=float(meta["outcome"]),timestamp=label_time)
        ann=f[f"episode_annotations/{UNKNOWN_ANNOTATOR}"]
        ann.attrs["source"]="human"
        ann.attrs["label_provenance"]="Imported unchanged from episode.json; original annotator identity was not recorded. No AI-generated failure labels."
        h=f.create_group("hilserl")
        h.attrs["converter_version"]=VERSION
        h.attrs["toolkit_commit"]=TOOLKIT_COMMIT
        h.attrs["converter_sha256"]=sha_file(__file__)
        h.attrs["source"]=json.dumps(source)
        h.attrs["episode_metadata"]=json.dumps(meta)
        h.attrs["config_snapshot"]=json.dumps(cfg)
        h.attrs["state_layout"]="gripper[0:1],force[1:4],reset-relative XYZ/Euler[4:10],torque[10:13],body twist[13:19]"
        h.create_dataset("is_human",data=np.array([r["source_action"]=="human" for r in rows]))
        h.create_dataset("episode_step",data=np.arange(n,dtype=np.int64))
        for key in ("terminated", "truncated", "observed_reward", "global_step"):
            h.create_dataset(key,data=np.array([r[key] for r in rows]))
        h.create_dataset("episode_end",data=np.arange(n)==n-1)
        h["episode_end"].attrs["source"]="Derived from complete archive boundary; separate from original terminated/truncated and human success label."
        h.create_dataset("termination_reason",data=np.array([r["termination_reason"] or "" for r in rows],dtype=object),dtype=h5py.string_dtype())
        h.create_dataset("policy_revision",data=np.array([r["policy"].get("revision",-1) for r in rows],dtype=np.int64))
        for key,values in {
            "original_state": [r["observations"]["state"][0] for r in rows],
            "next_original_state": [r["next_observations"]["state"][0] for r in rows],
            "next_cartesian_position": [r["raw_next_state"]["values"]["tcp_pose"] for r in rows],
            "next_joint_position": [r["raw_next_state"]["values"]["q"] for r in rows],
            "normalized_selected_action": [r["learning_action"] for r in rows],
            "normalized_policy_proposal": [r["policy_action"] for r in rows],
            "normalized_controller_input": [r["controller_input_action"] for r in rows],
            "jacobian": [r["raw_state"]["values"]["jacobian"] for r in rows],
        }.items():h.create_dataset(key,data=np.stack(values),compression="gzip")
        h.create_dataset("raw_transition_id",data=np.array([r["id"] for r in rows],dtype=object),dtype=h5py.string_dtype())
        h.create_dataset("controller_commands_json",data=np.array([json.dumps(plain(r["controller_commands"])) for r in rows],dtype=object),dtype=h5py.string_dtype())
        tg=h.create_group("time")
        for key in ("sample_started","step_started","step_ended"):
            for clock in ("monotonic_ns","unix_ns"):
                tg.create_dataset(f"{key}_{clock}",data=np.array([r[key][clock] for r in rows],dtype=np.int64))
        for prefix in ("raw_state","raw_next_state"):
            for edge in ("request","response"):
                for clock in ("monotonic_ns","unix_ns"):
                    tg.create_dataset(f"{prefix}_{edge}_{clock}",data=np.array([r[prefix]["capture"][edge][clock] for r in rows],dtype=np.int64))
        for camera,key in CAMS.items():
            for clock in ("monotonic_ns","unix_ns","capture_id"):
                tg.create_dataset(f"{camera}_{clock}",data=np.array([r["frame_references"][key][clock] for r in rows],dtype=np.int64))
                tg.create_dataset(f"next_{camera}_{clock}",data=np.array([r["next_frame_references"][key][clock] for r in rows],dtype=np.int64))
        for clock in ("monotonic_ns","unix_ns"):
            tg.create_dataset(f"command_sent_{clock}",data=np.array([r["controller_commands"][0]["sent"][clock] for r in rows],dtype=np.int64))
        ages=np.stack([(np.array([ns(r["step_started"])-ns(r["frame_references"][cam]) for r in rows])/1e6) for cam in CAMS.values()],axis=1)
        h.create_dataset("input_image_age_ms",data=ages)
        h.create_dataset("diagnostic_image_age_over_200ms",data=np.max(ages,axis=1)>200)
        h.attrs["quality_threshold_note"]="200 ms is diagnostic only; no rows are removed or relabeled."
    # Exact numerical mapping check, independent of the official format validator.
    with h5py.File(path,"r") as f:
        for key,a in states.items():require(np.array_equal(f[f"observations/robot_states/{key}"][:],a),f"state_roundtrip:{key}")
        for key,a in actions.items():require(np.array_equal(f[f"actions/{key}"][:],a),f"action_roundtrip:{key}")
        require(f[f"episode_annotations/{UNKNOWN_ANNOTATOR}"].attrs["success"]==meta["outcome"],"human_label_changed")
        require(np.array_equal(f["hilserl/original_state"][:],np.stack([r["observations"]["state"][0] for r in rows])),"proprio_roundtrip")
        require(np.array_equal(f["hilserl/next_cartesian_position"][:],np.stack([r["raw_next_state"]["values"]["tcp_pose"] for r in rows])),"next_state_roundtrip")
    return {"states":{k:list(a.shape) for k,a in states.items()},"actions":{k:list(a.shape) for k,a in actions.items()},
            "all_numeric_mappings_exact":True,"original_human_outcome_preserved":True}


def convert_one(args, entry):
    from oopsie_data_tools.utils.validation.validation_utils import validate_h5_file
    capture,meta,cfg,rows,hashes=load_episode(args.root,entry)
    identity=entry["selection_id"].replace("/","__")
    dest=args.output / "episodes" / identity
    if dest.exists():
        require(args.resume,"output_exists_use_resume")
        cert=read_json(dest/"verification.json")
        require(cert["source_hashes"]["combined"]==hashes["combined"] and cert["converter_version"]==VERSION
                and cert["instruction"]==args.instruction,"existing_output_source_or_converter_mismatch")
        # VERSION defines compatible mappings. Preserve the original producer
        # SHA in the certificate when a newer checker adds supported inputs.
        with h5py.File(dest/f"{identity}.h5", "r") as existing:
            require(existing["hilserl"].attrs["converter_sha256"]==cert["converter_sha256"], "producer_hash_mismatch")
            for camera, verification in cert["video_checks"].items():
                video = dest/existing[f"observations/video_paths/{camera}"].asstr()[()]
                require(sha_file(video)==verification["output_sha256"], "existing_output_video_changed")
        validate_h5_file(str(dest/f"{identity}.h5"),strict_annotation_check=True)
        return {"source":entry["source"],"status":"resumed","directory":str(dest),"steps":len(rows),"outcome":meta["outcome"],
                "human_steps":meta["human_steps"],"stale_input_steps":cert["stale_input_steps_diagnostic"]}
    stage=args.output / "_staging" / identity
    require(not stage.exists(),"staging_exists_requires_review")
    stage.mkdir(parents=True)
    (stage/"previews").mkdir()
    try:
        require(shutil.disk_usage(args.output).free>8*2**30,"disk_reserve_below_8GiB")
        videos,checks={},{}
        for name,cam in CAMS.items():
            selected,first,index_sha=resolve_frames(capture,rows,cam)
            videos[name]=stage/f"{identity}_{name}.mp4"
            checks[name]=encode_video(capture,cam,selected,first,rows,meta["image_profile"],videos[name],float(cfg["control_hz"]),stage/"previews",args.root)
            checks[name]["source_index_sha256"]=index_sha
        source={"episode":entry["source"],"capture_status":entry["capture_status"],"hashes":hashes}
        h5=stage/f"{identity}.h5"
        numeric=write_h5(h5,meta,cfg,rows,videos,source,args)
        require(validate_h5_file(str(h5),strict_annotation_check=True),"official_validation_failed")
        cert={"converter_version":VERSION,"converter_sha256":sha_file(__file__),"instruction":args.instruction,
              "toolkit_commit":TOOLKIT_COMMIT,"local_validation_only":True,
              "official_validation_passed":True,"submission_ready":False,"source_hashes":hashes,
              "source":source,"steps":len(rows),"human_steps":meta["human_steps"],"outcome":meta["outcome"],
              "numerical_checks":numeric,"video_checks":checks,
              "wall_collection_seconds":(ns(meta["collection_ended"])-ns(meta["started"]))/1e9,
              "step_playback_seconds":len(rows)/cfg["control_hz"],
              "stale_input_steps_diagnostic":[i for i,r in enumerate(rows) if max((ns(r["step_started"])-ns(r["frame_references"][c]))/1e6 for c in CAMS.values())>200],
              "source_metadata_notes":["Outcome imported from human label; identity unrecorded.","Task text drafted from user description.","Closed gripper target from user-confirmed invariant.","Nominal-rate step video; true timing retained."]}
        json_write(stage/"verification.json",cert)
        json_write(stage/"source-episode.json",meta)
        json_write(stage/"source-config.json",cfg)
        (stage/"LOCAL_VALIDATION_ONLY.txt").write_text("Not a registered submission. Lab/person identities and gripper model unresolved.\nHuman outcome imported; task instruction is a draft. Videos use a nominal step clock, not wall clock.\nNo upload performed.\n")
        dest.parent.mkdir(parents=True,exist_ok=True)
        stage.rename(dest)
        return {"source":entry["source"],"status":"converted","directory":str(dest),"steps":len(rows),"outcome":meta["outcome"],
                "human_steps":meta["human_steps"],"stale_input_steps":cert["stale_input_steps_diagnostic"]}
    except Exception:
        # Keep only this converter's partial output for diagnosis, outside episodes/.
        rejected=args.output/"_rejected"/identity
        rejected.parent.mkdir(parents=True,exist_ok=True)
        if stage.exists() and not rejected.exists():stage.rename(rejected)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--toolkit",type=Path,required=True)
    parser.add_argument("--episode",action="append",help="run/capture/episode; repeat for pilot selection; omitted means batch")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--instruction",default="Insert the already-grasped plug into the designated socket.")
    args=parser.parse_args()
    args.root=args.root.resolve();args.output=args.output.resolve();args.toolkit=args.toolkit.resolve()
    require(args.output.is_relative_to(args.root/"artifacts"),"output_must_be_inside_project_artifacts")
    require(not args.output.is_relative_to(args.root/"artifacts/runs"),"output_must_not_be_inside_source_runs")
    sys.path[:0]=[str(args.toolkit),str(args.root)]
    args.output.mkdir(parents=True,exist_ok=True)
    entries=inventory(args.root)
    json_write(args.output/"inventory.json",entries)
    if args.episode:
        known={e["selection_id"]:e for e in entries}
        require(all(x in known for x in args.episode),"unknown_episode_selection")
        selected=[known[x] for x in args.episode]
    else:selected=entries
    results=[]
    begin=time.monotonic()
    for i,entry in enumerate(selected):
        if not entry["eligible"]:
            results.append({**entry,"status":"excluded"})
            continue
        try:
            result=convert_one(args,entry)
        except Exception as exc:
            result={"source":entry["source"],"status":"rejected","error_type":type(exc).__name__,"reason":str(exc)}
        results.append(result)
        print(json.dumps({"index":i+1,"of":len(selected),**result},ensure_ascii=False),flush=True)
        json_write(args.output/"results.json",results)
    summary={"converter_version":VERSION,"toolkit_commit":TOOLKIT_COMMIT,"local_validation_only":True,"submission_ready":False,
             "inventoried_episodes":len(entries),"selected":len(selected),"counts":dict(Counter(r["status"] for r in results)),
             "exclusion_reasons":dict(Counter(r.get("reason") for r in results if r["status"] in ("excluded","rejected"))),
             "successful_episodes":sum(r.get("outcome")==1 for r in results if r["status"] in ("converted","resumed")),
             "failed_episodes":sum(r.get("outcome")==0 for r in results if r["status"] in ("converted","resumed")),
             "converted_steps":sum(r.get("steps",0) for r in results if r["status"] in ("converted","resumed")),
             "elapsed_seconds":time.monotonic()-begin}
    json_write(args.output/"results.json",results)
    json_write(args.output/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return int(any(r["status"]=="rejected" for r in results))


if __name__=="__main__":
    raise SystemExit(main())
