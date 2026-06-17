#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np


ROOT = Path(os.environ.get("SERL_FR3_ROOT", "/home/robot/serl_projects/hil-serl-fr3"))
HOST = "127.0.0.1"
PORT = int(os.environ.get("SERL_FR3_ROBOT_PORT", "5017"))
BASE_URL = f"http://{HOST}:{PORT}/"
PYNPUT_BACKEND = "dummy"
TARGET_PORTS = [50051, 5017, 54817, 8000, 8001, 8002, 8010, 8014]


def ensure_import_paths() -> None:
    for path in [
        ROOT,
        ROOT / "upstream" / "hil-serl",
        ROOT / "upstream" / "hil-serl" / "serl_robot_infra",
        ROOT / "upstream" / "hil-serl" / "examples",
    ]:
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def assert_port_free(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(f"PORT_OCCUPIED {host}:{port}")


def post_json(path: str, payload: dict | None = None) -> tuple[int, dict]:
    req = Request(
        BASE_URL + path,
        data=json.dumps(payload or {}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=2.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def get_json(path: str) -> tuple[int, dict]:
    with urlopen(BASE_URL + path, timeout=2.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def check_versions() -> dict[str, str]:
    wanted = [
        "jax",
        "jaxlib",
        "jax-cuda12-plugin",
        "numpy",
        "scipy",
        "flax",
        "optax",
        "orbax-checkpoint",
        "opencv-python",
        "serl-launcher",
        "serl-robot-infra",
        "agentlace",
        "gymnasium",
    ]
    versions = {"python": sys.version.split()[0]}
    for package in wanted:
        versions[package] = metadata.version(package)
    return versions


def check_pip() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout.strip()


def check_imports() -> list[str]:
    modules = [
        "jax",
        "jax.numpy",
        "flax",
        "optax",
        "orbax.checkpoint",
        "gymnasium",
        "cv2",
        "serl_launcher",
        "agentlace",
        "franka_env.envs.franka_env",
        "examples.experiments.ram_insertion.config",
        "examples.experiments.usb_pickup_insertion.config",
        "experiments.config",
        "experiments.usb_pickup_insertion.wrapper",
        "fr3_experiments.plug_insertion.config",
        "fr3_experiments.plug_insertion.zed_config",
        "fr3_hil_bridge.plug_bridge",
        "fr3_hil_bridge.image_adapter",
        "fr3_hil_bridge.zed_camera_client",
        "fr3_hil_bridge.fci_state_client",
    ]
    loaded = []
    for module in modules:
        importlib.import_module(module)
        loaded.append(module)
    return loaded


def check_jax_devices() -> list[str]:
    import jax

    return [str(device) for device in jax.devices()]


def check_fake_franka_env() -> dict:
    from franka_env.envs.franka_env import DefaultEnvConfig, FrankaEnv

    class NoMotionConfig(DefaultEnvConfig):
        SERVER_URL = BASE_URL
        REALSENSE_CAMERAS = {}
        DISPLAY_IMAGE = False
        TARGET_POSE = np.array([0.45, 0.0, 0.30, 0.0, 0.0, 0.0])
        GRASP_POSE = TARGET_POSE.copy()
        RESET_POSE = TARGET_POSE.copy()
        REWARD_THRESHOLD = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
        ACTION_SCALE = np.array([0.01, 0.05, 1.0])
        ABS_POSE_LIMIT_LOW = np.array([0.35, -0.20, 0.20, -3.14, -3.14, -3.14])
        ABS_POSE_LIMIT_HIGH = np.array([0.65, 0.20, 0.45, 3.14, 3.14, 3.14])
        COMPLIANCE_PARAM = {}
        RESET_PARAM = {}
        PRECISION_PARAM = {}
        MAX_EPISODE_LENGTH = 5

    env = FrankaEnv(hz=10, fake_env=True, save_video=False, config=NoMotionConfig())
    return {
        "action_shape": list(env.action_space.shape),
        "observation_keys": sorted(env.observation_space.spaces.keys()),
        "currpos_len": int(len(env.currpos)),
    }


def space_keys(space: object) -> list[str]:
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        return sorted(spaces.keys())
    return []


def check_local_plug_env() -> dict:
    module = importlib.import_module("fr3_experiments.plug_insertion.config")
    cfg = module.TrainConfig()
    env = cfg.get_environment(fake_env=True, save_video=False, classifier=False)
    return {
        "module": getattr(module, "__file__", None),
        "env_type": type(env).__name__,
        "action_shape": list(env.action_space.shape),
        "observation_keys": space_keys(env.observation_space),
        "image_keys": list(cfg.image_keys),
        "classifier_keys": list(cfg.classifier_keys),
        "proprio_keys": list(cfg.proprio_keys),
        "note": "constructed only; reset/step intentionally not called",
    }


def check_experiment_shape() -> dict[str, list[str]]:
    required = ["README.md", "config.py", "ports.env"]
    result = {}
    for name in ["plug_insertion", "plug_zed_insertion", "usb_pickup_insertion", "ram_insertion"]:
        exp_dir = ROOT / "experiments" / name
        missing = [item for item in required if not (exp_dir / item).exists()]
        if missing:
            raise RuntimeError(f"{name} missing {missing}")
        result[name] = sorted(path.name for path in exp_dir.iterdir())
    return result


def check_droid_guard() -> dict:
    patterns = [
        re.compile(r"(?m)^\s*import\s+droid\b"),
        re.compile(r"(?m)^\s*from\s+droid\b"),
        re.compile(r"/home/robot/droid"),
    ]
    scan_roots = [ROOT / "fr3_experiments", ROOT / "fr3_hil_bridge", ROOT / "tools", ROOT / "tests"]
    matches: list[dict[str, object]] = []
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*.py"):
            rel_path = path.relative_to(ROOT)
            if str(rel_path) == "tools/no_motion_validate.py" or path.name.startswith("._"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(
                        {
                            "file": str(rel_path),
                            "pattern": pattern.pattern,
                            "offset": match.start(),
                        }
                    )
    py_path = os.environ.get("PYTHONPATH", "")
    runtime_hits = [item for item in sys.path if item and "/home/robot/droid" in item]
    if "/home/robot/droid" in py_path:
        runtime_hits.append("PYTHONPATH:/home/robot/droid")
    if matches or runtime_hits:
        raise RuntimeError({"DROID_GUARD_FAILED": {"static": matches, "runtime": runtime_hits}})
    return {
        "static_scan_roots": [str(path.relative_to(ROOT)) for path in scan_roots if path.exists()],
        "static_matches": [],
        "runtime_matches": [],
        "status": "DROID_GUARD_OK",
    }


def check_plug_bridge_validation() -> dict:
    from fr3_hil_bridge.plug_bridge import (
        DEFAULT_MAX_FUTURE_SECONDS,
        NO_MOTION_STEP_REJECTED,
        ActionRejected,
        PlugFr3Bridge,
        validate_action_7d,
    )

    now = 10.0
    cases = {
        "valid_zero_7d": validate_action_7d(np.zeros(7), timestamp=now, now=now).to_dict(),
        "current_timestamp_string": validate_action_7d(
            np.zeros(7), timestamp=str(now), now=now
        ).to_dict(),
        "bad_timestamp": validate_action_7d(np.zeros(7), timestamp="not-a-time", now=now).to_dict(),
        "future": validate_action_7d(
            np.zeros(7),
            timestamp=now + DEFAULT_MAX_FUTURE_SECONDS + 0.01,
            now=now,
        ).to_dict(),
        "old_openpi_droid_8d": validate_action_7d(np.zeros(8), timestamp=now, now=now).to_dict(),
        "wrong_shape": validate_action_7d(np.zeros(6), timestamp=now, now=now).to_dict(),
        "non_finite": validate_action_7d([0, 0, 0, np.nan, 0, 0, 0], timestamp=now, now=now).to_dict(),
        "out_of_range": validate_action_7d([2, 0, 0, 0, 0, 0, 0], timestamp=now, now=now).to_dict(),
        "stale": validate_action_7d(np.zeros(7), timestamp=0.0, now=now).to_dict(),
    }
    expected = {
        "valid_zero_7d": "ACCEPT_7D_ACTION",
        "current_timestamp_string": "ACCEPT_7D_ACTION",
        "bad_timestamp": "REJECT_BAD_TIMESTAMP",
        "future": "REJECT_FUTURE_ACTION",
        "old_openpi_droid_8d": "REJECT_OLD_OPENPI_DROID_8D_ACTION",
        "wrong_shape": "REJECT_ACTION_SHAPE",
        "non_finite": "REJECT_NON_FINITE_ACTION",
        "out_of_range": "REJECT_ACTION_OUT_OF_RANGE",
        "stale": "REJECT_STALE_ACTION",
    }
    for name, reason in expected.items():
        if cases[name]["reason"] != reason:
            raise RuntimeError(f"BAD_ACTION_VALIDATION {name}: {cases[name]}")

    bridge = PlugFr3Bridge(BASE_URL, no_motion=True)
    health = bridge.connect_state_only()
    state = bridge.get_state()
    safety = bridge.get_safety_state()
    try:
        bridge.step_7d(np.zeros(7))
    except ActionRejected as exc:
        step_rejection = exc.validation.to_dict()
    else:
        raise RuntimeError("NO_MOTION_STEP_WAS_NOT_REJECTED")
    if step_rejection["reason"] != NO_MOTION_STEP_REJECTED:
        raise RuntimeError(f"BAD_STEP_REJECTION {step_rejection}")
    return {
        "cases": cases,
        "max_future_seconds": DEFAULT_MAX_FUTURE_SECONDS,
        "connect_state_only": health,
        "state_pose_len": len(state.get("pose", [])),
        "safety": safety,
        "step_rejection": step_rejection,
    }


def check_image_adapter_validation() -> dict:
    from fr3_hil_bridge.image_adapter import (
        ZED_IMAGE_KEYS,
        resize_with_pad_224,
        validate_expected_keys,
        validate_hilserl_image,
        validate_zed_observation,
        validate_uint8_hwc,
    )

    valid_128 = np.zeros((128, 128, 3), dtype=np.uint8)
    wrong_shape = np.zeros((64, 128, 3), dtype=np.uint8)
    wrong_dtype = np.zeros((128, 128, 3), dtype=np.float32)
    images = {"side_policy": valid_128, "wrist_1": valid_128, "wrist_2": valid_128}
    zed_images = {"zed_left": valid_128, "zed_right": valid_128.copy()}

    result = {
        "implementation": "fr3_hil_bridge/image_adapter.py",
        "keys_ok": validate_expected_keys(images, ["side_policy", "wrist_1", "wrist_2"]),
        "missing_key": validate_expected_keys(images, ["side_policy", "wrist_1", "wrist_2", "side_classifier"]),
        "valid_hilserl": validate_hilserl_image(valid_128, key="side_policy").to_dict(),
        "wrong_shape": validate_hilserl_image(wrong_shape, key="side_policy").to_dict(),
        "wrong_dtype": validate_uint8_hwc(wrong_dtype, key="side_policy").to_dict(),
        "zed_image_keys": list(ZED_IMAGE_KEYS),
        "zed_valid": validate_zed_observation(zed_images),
        "zed_missing_key": validate_zed_observation({"zed_left": valid_128}),
        "zed_wrong_shape": validate_zed_observation({"zed_left": valid_128, "zed_right": wrong_shape}),
        "openpi_224_shape": list(resize_with_pad_224(np.zeros((80, 120, 3), dtype=np.uint8)).shape),
    }
    if result["keys_ok"]["valid"] is not True:
        raise RuntimeError(f"IMAGE_KEYS_SHOULD_PASS {result['keys_ok']}")
    if result["missing_key"]["reason"] != "REJECT_MISSING_IMAGE_KEY":
        raise RuntimeError(f"MISSING_KEY_SHOULD_REJECT {result['missing_key']}")
    if result["valid_hilserl"]["reason"] != "ACCEPT_HILSERL_128_IMAGE":
        raise RuntimeError(f"HILSERL_IMAGE_SHOULD_PASS {result['valid_hilserl']}")
    if result["wrong_shape"]["reason"] != "REJECT_HILSERL_IMAGE_SHAPE":
        raise RuntimeError(f"WRONG_SHAPE_SHOULD_REJECT {result['wrong_shape']}")
    if result["wrong_dtype"]["reason"] != "REJECT_IMAGE_DTYPE":
        raise RuntimeError(f"WRONG_DTYPE_SHOULD_REJECT {result['wrong_dtype']}")
    if result["zed_valid"]["reason"] != "ACCEPT_HILSERL_IMAGES":
        raise RuntimeError(f"ZED_IMAGES_SHOULD_PASS {result['zed_valid']}")
    if result["zed_missing_key"]["reason"] != "REJECT_MISSING_IMAGE_KEY":
        raise RuntimeError(f"ZED_MISSING_KEY_SHOULD_REJECT {result['zed_missing_key']}")
    if result["zed_wrong_shape"]["reason"] != "REJECT_IMAGE_VALUE":
        raise RuntimeError(f"ZED_WRONG_SHAPE_SHOULD_REJECT {result['zed_wrong_shape']}")
    if result["openpi_224_shape"] != [224, 224, 3]:
        raise RuntimeError(f"OPENPI_224_BAD_SHAPE {result['openpi_224_shape']}")
    return result


def check_zed_plug_config() -> dict[str, object]:
    from fr3_experiments.plug_insertion.zed_config import ZED_CAMERA_PROFILE, ZedTrainConfig

    cfg = ZedTrainConfig()
    contract = cfg.observation_contract().to_dict()
    validation = cfg.validate_synthetic_observation()
    if contract["camera_profile"] != ZED_CAMERA_PROFILE:
        raise RuntimeError(f"ZED_PROFILE_BAD {contract}")
    if contract["image_keys"] != ["zed_left", "zed_right"]:
        raise RuntimeError(f"ZED_IMAGE_KEYS_BAD {contract}")
    if contract["classifier_keys"] != ["zed_left", "zed_right"]:
        raise RuntimeError(f"ZED_CLASSIFIER_KEYS_BAD {contract}")
    if contract["runtime_enabled"] is not False:
        raise RuntimeError(f"ZED_RUNTIME_SHOULD_BE_DISABLED {contract}")
    if validation.get("valid") is not True:
        raise RuntimeError(f"ZED_SYNTHETIC_OBSERVATION_BAD {validation}")
    try:
        cfg.get_environment(fake_env=True)
    except RuntimeError as exc:
        env_rejection = str(exc).split(":", 1)[0]
    else:
        raise RuntimeError("ZED_ENV_CONSTRUCTION_SHOULD_BE_GATE_BLOCKED")
    return {
        "train_config": "fr3_experiments.plug_insertion.zed_config.ZedTrainConfig",
        "contract": contract,
        "synthetic_observation": validation,
        "env_rejection": env_rejection,
    }


def check_plug_runner_dry_run() -> dict[str, dict]:
    env = os.environ.copy()
    env.setdefault("PYNPUT_BACKEND", PYNPUT_BACKEND)
    env.setdefault("SERL_FR3_ROOT", str(ROOT))

    outputs: dict[str, dict] = {}
    for script_name, role in [
        ("run_plug_learner.py", "learner"),
        ("run_plug_actor.py", "actor"),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / script_name),
                "--dry-run",
                "--headless",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=True,
        )
        payload = json.loads(result.stdout)
        if payload.get("role") != role:
            raise RuntimeError(f"PLUG_RUNNER_ROLE_MISMATCH {script_name}: {payload}")
        if payload.get("dry_run") is not True:
            raise RuntimeError(f"PLUG_RUNNER_NOT_DRY_RUN {script_name}: {payload}")
        if payload.get("train_config") != "fr3_experiments.plug_insertion.config.TrainConfig":
            raise RuntimeError(f"PLUG_RUNNER_BAD_CONFIG {script_name}: {payload}")
        if payload.get("uses_upstream_config_mapping") is not False:
            raise RuntimeError(f"PLUG_RUNNER_CONFIG_MAPPING_REQUIRED {script_name}: {payload}")
        if payload.get("no_reset_step_fci_camera_or_gripper") is not True:
            raise RuntimeError(f"PLUG_RUNNER_TOUCHES_LIVE_SURFACE {script_name}: {payload}")
        if payload.get("pynput_backend") != PYNPUT_BACKEND:
            raise RuntimeError(f"PLUG_RUNNER_HEADLESS_NOT_SCOPED {script_name}: {payload}")
        if result.stderr:
            payload["stderr_lines"] = result.stderr.splitlines()[:20]
        outputs[script_name] = payload

    for script_name, role in [
        ("run_plug_learner.py", "learner"),
        ("run_plug_actor.py", "actor"),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / script_name),
                "--dry-run",
                "--headless",
                "--camera-profile",
                "zed_stereo",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=True,
        )
        payload = json.loads(result.stdout)
        if payload.get("role") != role:
            raise RuntimeError(f"PLUG_ZED_RUNNER_ROLE_MISMATCH {script_name}: {payload}")
        if payload.get("camera_profile") != "zed_stereo":
            raise RuntimeError(f"PLUG_ZED_RUNNER_PROFILE_MISMATCH {script_name}: {payload}")
        if payload.get("train_config") != "fr3_experiments.plug_insertion.zed_config.ZedTrainConfig":
            raise RuntimeError(f"PLUG_ZED_RUNNER_BAD_CONFIG {script_name}: {payload}")
        if payload.get("image_keys") != ["zed_left", "zed_right"]:
            raise RuntimeError(f"PLUG_ZED_RUNNER_BAD_IMAGE_KEYS {script_name}: {payload}")
        zed_validation = payload.get("zed_observation_validation")
        if not isinstance(zed_validation, dict) or zed_validation.get("valid") is not True:
            raise RuntimeError(f"PLUG_ZED_RUNNER_BAD_OBSERVATION_VALIDATION {script_name}: {payload}")
        if payload.get("no_reset_step_fci_camera_or_gripper") is not True:
            raise RuntimeError(f"PLUG_ZED_RUNNER_TOUCHES_LIVE_SURFACE {script_name}: {payload}")
        if result.stderr:
            payload["stderr_lines"] = result.stderr.splitlines()[:20]
        outputs[f"{script_name}:zed_stereo"] = payload
    return outputs


def port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def check_target_ports_closed() -> dict[int, bool]:
    # Give the mock server context a short moment to release 5017.
    time.sleep(0.2)
    status = {port: port_open(HOST, port) for port in TARGET_PORTS}
    open_ports = [port for port, is_open in status.items() if is_open]
    if open_ports:
        raise RuntimeError(f"TARGET_PORTS_LEFT_OPEN {open_ports}")
    return status


@contextlib.contextmanager
def mock_server():
    assert_port_free(HOST, PORT)
    mock_path = ROOT / "fr3_hil_bridge" / "mock_server" / "serl_contract_mock.py"
    proc = subprocess.Popen(
        [sys.executable, str(mock_path), "--host", HOST, "--port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(50):
            try:
                status, payload = get_json("healthz")
                if status == 200 and payload.get("ok") is True:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("mock server did not become ready")

        reader_lines: list[str] = []

        def reader() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                reader_lines.append(line.rstrip())

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        yield reader_lines
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def main() -> None:
    os.environ.setdefault("PYNPUT_BACKEND", PYNPUT_BACKEND)
    ensure_import_paths()

    report: dict[str, object] = {
        "root": str(ROOT),
        "base_url": BASE_URL,
        "pynput_backend": os.environ.get("PYNPUT_BACKEND"),
        "versions": check_versions(),
        "pip_check": check_pip(),
        "imports": check_imports(),
        "jax_devices": check_jax_devices(),
        "droid_guard": check_droid_guard(),
    }
    with mock_server() as mock_log:
        status, state = post_json("getstate")
        if status != 200 or len(state.get("pose", [])) != 7:
            raise RuntimeError(f"bad getstate response: {status} {state}")
        pose_status, pose_payload = post_json("pose", {"arr": [0.0] * 7})
        if pose_status != 409 or pose_payload.get("error") != "NO_MOTION_MOCK_REJECTED":
            raise RuntimeError(f"pose endpoint was not rejected: {pose_status} {pose_payload}")
        report["mock_getstate"] = state
        report["mock_pose_rejection"] = pose_payload
        report["fake_franka_env"] = check_fake_franka_env()
        report["local_plug_config_import"] = importlib.import_module(
            "fr3_experiments.plug_insertion.config"
        ).__file__
        report["plug_fake_env"] = check_local_plug_env()
        report["plug_zed_config"] = check_zed_plug_config()
        report["plug_bridge_validation"] = check_plug_bridge_validation()
        report["plug_runner_dry_run"] = check_plug_runner_dry_run()
        report["image_adapter_validation"] = check_image_adapter_validation()
        report["mock_log"] = mock_log[:20]
    report["experiments"] = check_experiment_shape()
    report["post_validation_process_ports"] = check_target_ports_closed()
    print(json.dumps(report, indent=2, sort_keys=True))
    print("NO_MOTION_VALIDATION_OK")


if __name__ == "__main__":
    main()
