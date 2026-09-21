"""The project profile is the only default configuration for every entry point."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import sys

from hilserl.profile_names import PROFILE_NAMES

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_IMAGE_KEYS = ("side_classifier", "wrist_1")


def validate_classifier_input_contract(input_contract, image_key, image_profile):
    """Validate recorded classifier pixels before restoring any model weights."""
    if image_key not in CLASSIFIER_IMAGE_KEYS:
        raise ValueError(f"Unknown classifier_image_key: {image_key}")
    from hilserl.image_profile import observation_image_schema
    schema = observation_image_schema(image_profile)[image_key]
    expected = dict(image_key=image_key, image_shape=schema["shape"][1:],
                    image_dtype=schema["dtype"], image_profile=image_profile)
    if input_contract is None:
        if image_key != "side_classifier":
            raise ValueError("Classifier without input_contract only supports historical side_classifier input")
        return expected
    if not isinstance(input_contract, dict):
        raise ValueError("Classifier input_contract must be an object")
    for field, value in expected.items():
        if input_contract.get(field) != value:
            raise ValueError(f"Classifier input_contract mismatch for {field}: "
                             f"checkpoint={input_contract.get(field)!r}, runtime={value!r}")
    return expected


@dataclass(frozen=True)
class Config:
    root: Path = ROOT
    python: str = "/home/robot/miniconda3/envs/hilserl-fr3/bin/python"
    data_dir: str = "artifacts/runs"
    demo_dir: str = "demos/serl19_vice_fresh_20260911_183435"
    # Missing fields in historical run snapshots must retain the 7D model.
    # The project profile explicitly selects the new contract for fresh runs.
    action_contract: str = "legacy-hybrid-7d-v1"
    image_profile: str = "full-frame128-v1"
    seed_dataset_sha256: str | None = None
    classifier_ckpt: str = "classifier_ckpt"
    classifier_image_key: str = "side_classifier"
    server_url: str = "http://172.16.0.1:5000/"
    port: int = 5588
    broadcast_port: int = 5589
    console_port: int = 8765
    seed: int = 0
    batch_size: int = 256
    replay_buffer_capacity: int = 200000
    checkpoint_keep: int = 0
    cta_ratio: int = 2
    learner_steps: int = 100000
    actor_steps: int = 100000
    training_starts: int = 100
    steps_per_update: int = 50
    max_episode_steps: int = 190
    control_hz: float = 10.0
    random_reset: bool = True
    random_xy_range: float = 0.006
    random_rz_range: float = 0.0
    reset_strict: bool = True
    action_max_step: float = 0.008
    # None preserves historical isotropic action mapping; new runs opt in explicitly.
    action_max_z_step: float | None = None
    position_target_mode: str = "measured-relative-v1"
    action_max_rotation_step: float = 0.045
    safety_force_max: float = 45.0
    safety_dq_max: float = 0.35
    safety_relz_abs_max: float = 0.35
    reset_clear_z: float = 0.2095
    classifier_threshold: float = 0.78
    depth_threshold: float = 0.08
    classifier_streak: int = 3
    min_free_gib: float = 8.0
    video_fps: int = 30
    video_segment_seconds: int = 60
    startup_timeout_seconds: int = 900
    discount: float = 0.98
    reward_mode: str = "sparse"
    reward_url: str = "http://127.0.0.1:20008"
    reward_model_id: str = ""
    reward_task: str = "Insert the plug into the socket."
    reward_image_key: str = "side_policy"
    reward_scale: float = 1.0
    reward_max_frames: int = 8
    reward_batch_size: int = 8
    reward_timeout_seconds: float = 120.0
    reward_seed_cache: str = ""

    def reward_spec(self):
        if self.reward_mode == "sparse":
            return None
        from hilserl.reward_provider import MODE, RewardSpec
        if self.reward_mode != MODE or self.action_contract != "fixed-xyz-v1":
            raise ValueError("Episode rewards require robometer-episode and fixed-xyz-v1")
        return RewardSpec(model_id=self.reward_model_id, task=self.reward_task,
            image_key=self.reward_image_key, image_profile=self.image_profile,
            gamma=self.discount, scale=self.reward_scale, max_frames=self.reward_max_frames).validate()

    def path(self, value):
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def output_root(self):
        return self.path(self.data_dir)

    @property
    def control_root(self):
        return self.root / "artifacts" / "control"

    def snapshot(self):
        value = asdict(self)
        value["root"] = str(self.root)
        return value

    def digest(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()

    def validate(self):
        if type(self.discount) not in (int, float) or not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0,1]")
        if self.reward_spec() is not None:
            from urllib.parse import urlparse
            address = urlparse(self.reward_url)
            if address.scheme not in {"http", "https"} or not address.hostname:
                raise ValueError("reward_url must be an HTTP service URL")
            if (type(self.reward_batch_size) is not int or not 1 <= self.reward_batch_size <= 64
                    or type(self.reward_timeout_seconds) not in (int, float)
                    or not 0 < self.reward_timeout_seconds < float("inf")):
                raise ValueError("Invalid reward batch size or timeout")
        if type(self.checkpoint_keep) is not int or self.checkpoint_keep < 0:
            raise ValueError("checkpoint_keep must be a nonnegative integer (0 retains all)")
        if self.action_contract not in {"legacy-hybrid-7d-v1", "fixed-xyz-v1"}:
            raise ValueError("Unknown action_contract")
        if self.classifier_image_key not in CLASSIFIER_IMAGE_KEYS:
            raise ValueError("Unknown classifier_image_key")
        if self.image_profile not in PROFILE_NAMES:
            raise ValueError("Unknown image_profile")
        if self.image_profile != "full-frame128-v1" and self.action_contract != "fixed-xyz-v1":
            raise ValueError("Cropped policy images require fixed-xyz-v1")
        if self.seed_dataset_sha256 is not None and (not isinstance(self.seed_dataset_sha256, str)
                or len(self.seed_dataset_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.seed_dataset_sha256)):
            raise ValueError("Invalid seed_dataset_sha256")
        for key in ("port", "broadcast_port", "console_port"):
            v = getattr(self, key)
            if type(v) is not int or not 1 <= v <= 65535:
                raise ValueError(f"Invalid {key}")
        if len({self.port, self.broadcast_port, self.console_port}) != 3:
            raise ValueError("Service ports must be distinct")
        for key in ("batch_size", "replay_buffer_capacity", "cta_ratio", "learner_steps", "actor_steps", "training_starts", "steps_per_update",
                    "max_episode_steps", "classifier_streak", "video_fps", "video_segment_seconds", "startup_timeout_seconds"):
            v = getattr(self, key)
            if type(v) is not int or v <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("random_reset", "reset_strict"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        for key in ("control_hz", "random_xy_range", "random_rz_range", "action_max_step", "min_free_gib",
                    "action_max_rotation_step", "safety_force_max", "safety_dq_max", "safety_relz_abs_max", "reset_clear_z"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not 0 <= value < float("inf"):
                raise ValueError(f"Invalid {key}")
        if self.control_hz == 0 or self.action_max_step == 0 or self.batch_size % 2:
            raise ValueError("Control frequency/action bound must be positive; batch size must be even")
        if self.action_max_z_step is not None:
            if (self.action_contract != "fixed-xyz-v1" or self.action_max_step != .008
                    or type(self.action_max_z_step) not in (int, float)
                    or not .008 <= self.action_max_z_step <= .016):
                raise ValueError("Base Z bound requires fixed-xyz-v1, XY=8 mm and Z in [8, 16] mm")
        if self.position_target_mode not in {"measured-relative-v1", "command-relative-v1"}:
            raise ValueError("Unknown position_target_mode")
        if self.position_target_mode == "command-relative-v1" and self.action_contract != "fixed-xyz-v1":
            raise ValueError("Command-relative targets require fixed-xyz-v1")
        if not 0 <= self.classifier_threshold <= 1:
            raise ValueError("classifier_threshold must be in [0,1]")
        return self

    def environment(self, role, *, run_dir, recording_dir=None):
        # Role values are generated here, rather than inherited from historical shell launchers.
        env = os.environ.copy()
        owned_prefixes = ("HILSERL_", "FRANKA_", "ACTOR_", "LEARNER_", "AUTO_REWARD_", "RESET_", "MANUAL_", "DROP_", "XBOX_")
        for key in list(env):
            if key.startswith(owned_prefixes) or key == "_PATCH_RESET_CLEAR_Z":
                env.pop(key)
        values = dict(
            AGENTLACE_PORT=self.port, AGENTLACE_BROADCAST_PORT=self.broadcast_port,
            HILSERL_CLASSIFIER_CKPT=self.path(self.classifier_ckpt), HILSERL_SERVER_URL=self.server_url,
            HILSERL_ACTION_CONTRACT=self.action_contract, HILSERL_SEED_DATASET=self.path(self.demo_dir),
            HILSERL_POSITION_TARGET_MODE=self.position_target_mode,
            HILSERL_IMAGE_PROFILE=self.image_profile,
            HILSERL_CLASSIFIER_IMAGE_KEY=self.classifier_image_key,
            HILSERL_BATCH_SIZE=self.batch_size, HILSERL_CTA_RATIO=self.cta_ratio,
            HILSERL_REPLAY_BUFFER_CAPACITY=self.replay_buffer_capacity,
            HILSERL_CHECKPOINT_KEEP=self.checkpoint_keep,
            HILSERL_MAX_STEPS=self.learner_steps if role == "learner" else self.actor_steps,
            HILSERL_ACTOR_STEPS=self.actor_steps, HILSERL_TRAINING_STARTS=self.training_starts,
            HILSERL_STEPS_PER_UPDATE=self.steps_per_update, HILSERL_MAX_EPISODE_STEPS=self.max_episode_steps,
            HILSERL_CONTROL_HZ=self.control_hz, HILSERL_RANDOM_RESET=int(self.random_reset),
            HILSERL_RANDOM_XY_RANGE=self.random_xy_range, HILSERL_RANDOM_RZ_RANGE=self.random_rz_range,
            HILSERL_VIDEO_FPS=self.video_fps, HILSERL_VIDEO_SEGMENT_SECONDS=self.video_segment_seconds,
            HILSERL_MIN_FREE_GIB=self.min_free_gib, HILSERL_RUN_DIR=run_dir,
            HILSERL_CONFIG_SNAPSHOT=Path(run_dir) / "config.json", HILSERL_CONTROL_DIR=Path(run_dir) / "control",
            MANUAL_RESET=1, MANUAL_SUCCESS=1, RESET_STRICT=int(self.reset_strict),
            XBOX_LOCK_ROTATION=1, ACTOR_LOCK_ROTATION=1, FRANKA_LOCK_ROTATION=1,
            FRANKA_ACTION_MAX_STEP=self.action_max_step, ACTOR_SAFETY_DQ_MAX=self.safety_dq_max,
            ACTOR_SAFETY_FORCE_MAX=self.safety_force_max, ACTOR_RELZ_ABS_MAX=self.safety_relz_abs_max,
            ACTOR_REQUIRE_LEARNER_PARAMS=1, ACTOR_LEARNER_PARAMS_TIMEOUT=60,
            ACTOR_LEARNER_PARAMS_SNAPSHOT_PERIOD=1.0,
            ACTOR_LEARNER_PARAMS_SNAPSHOT_TIMEOUT_MS=10000,
            LEARNER_INITIAL_NETWORK_BROADCAST_PERIOD=2.0,
            LEARNER_INITIAL_NETWORK_GRACE_SECONDS=180.0,
            FRANKA_SAFETY_DQ_MAX=self.safety_dq_max, FRANKA_SAFETY_FORCE_MAX=self.safety_force_max,
            FRANKA_CLEARERR_ON_POSE=0, FRANKA_ACTION_MAX_ROT_STEP=self.action_max_rotation_step,
            HILSERL_RESET_CLEAR_Z=self.reset_clear_z, RESET_JOINT_LIMIT_MARGIN_DEG=12.0, DROP_GRIPPER_MIN=0.4,
            RESET_MOTION_EFFECTIVE_SPEED=0.002, RESET_MOTION_MAX_TIMEOUT=90.0, RESET_MOTION_SETTLE_TIMEOUT=8.0,
            ACTOR_SUCCESS_CREDIT_HORIZON=0 if self.action_contract == "fixed-xyz-v1" else 24,
            ACTOR_SUCCESS_CREDIT_MIN_ACTION_NORM=0.05, ACTOR_SUCCESS_CREDIT_ONLINE=0,
            AUTO_REWARD_CLS_THRESHOLD=self.classifier_threshold, AUTO_REWARD_STATE6_MIN=self.depth_threshold,
            AUTO_REWARD_STREAK=self.classifier_streak,
            XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 if role == "learner" else 0.15,
            TF_FORCE_GPU_ALLOW_GROWTH="true", WANDB_MODE="disabled", SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1,
            SDL_VIDEODRIVER="dummy", PYTHONDONTWRITEBYTECODE=1, TELEOP_SCRIPTS=self.root / "scripts",
        )
        if recording_dir:
            values["HILSERL_RECORDING_DIR"] = recording_dir
        if self.action_max_z_step is not None:
            values["HILSERL_ACTION_MAX_Z_STEP"] = self.action_max_z_step
        if self.seed_dataset_sha256 is not None:
            values["HILSERL_SEED_DATASET_SHA256"] = self.seed_dataset_sha256
        env.update({k: str(v) for k, v in values.items()})
        env["HILSERL_DISCOUNT"] = str(self.discount)
        spec = self.reward_spec()
        if spec is not None:
            env.update(HILSERL_REWARD_SPEC=json.dumps(asdict(spec), sort_keys=True),
                       HILSERL_REWARD_URL=self.reward_url,
                       HILSERL_REWARD_BATCH_SIZE=str(self.reward_batch_size),
                       HILSERL_REWARD_TIMEOUT=str(self.reward_timeout_seconds),
                       HILSERL_REWARD_SEED_CACHE=str(self.path(self.reward_seed_cache)) if self.reward_seed_cache else "")
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(key, None)
        env["no_proxy"] = env["NO_PROXY"] = "172.16.0.1,localhost,127.0.0.1,::1"
        env.setdefault("DISPLAY", ":1")
        env.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")
        packages = list(self.path(self.python).parents[1].glob("lib/python*/site-packages"))
        if packages:
            env["CUDA_ROOT"] = str(packages[0] / "nvidia/cuda_nvcc")
        # Prefer this checkout's Agentlace over an editable install in a shared
        # environment, which may still point at the active main worktree.
        python_paths = [self.root, self.root / "upstream/agentlace",
                        self.root / "upstream/hil-serl/serl_robot_infra",
                        self.root / "upstream/hil-serl/serl_launcher", self.root / "upstream/hil-serl/examples"]
        env["PYTHONPATH"] = os.pathsep.join(map(str, python_paths)) + os.pathsep + env.get("PYTHONPATH", "")
        return env


def load_config(path=None, **overrides):
    if path is not None and not Path(path).is_file():
        raise FileNotFoundError(path)
    path = Path(path) if path else ROOT / "config" / "hilserl.json"
    value = json.loads(path.read_text()) if path.exists() else {}
    allowed = {field.name for field in fields(Config)} - {"root"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    value.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**value).validate()
