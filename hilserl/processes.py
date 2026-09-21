"""Process ownership and launch orchestration. Importing this module never loads JAX."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.parse import urlparse
import uuid

from hilserl.config import Config
from hilserl.files import atomic_json, read_json, stamp


def flag(args, name, default=None):
    for i, arg in enumerate(args):
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
        if arg == name:
            return args[i + 1] if i + 1 < len(args) else default
    return default


def enabled(args, name):
    return name in args or flag(args, name, "false").lower() in {"1", "true"}


def process_start(pid, proc=Path("/proc")):
    try:
        return (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def discover(root, proc=Path("/proc")):
    if not proc.is_dir():
        return []
    found = []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            cwd = Path(os.readlink(path / "cwd")).resolve()
            args = (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if not args or not Path(args[0]).name.startswith("python"):
                continue
            if cwd != Path(root).resolve() and str(Path(root).resolve() / "_run_actor.py") not in args:
                continue
            env = dict(x.split("=", 1) for x in (path / "environ").read_bytes().decode().split("\0") if "=" in x)
            role = None
            if any(Path(arg).name == "_run_actor.py" for arg in args):
                if enabled(args, "--learner"):
                    role = "learner"
                elif enabled(args, "--actor"):
                    role = env.get("HILSERL_MODE", "eval" if flag(args, "--eval_checkpoint_step", "0") != "0" else "train")
            elif any(Path(arg).name in {"eval_rollout.py", "collect_insert_demo_serl_aligned.py", "record_gello_demos_serl.py",
                                       "gello_demo_recorder.py", "xbox_demo_recorder.py", "hybrid_teleop.py", "teach_session.py"} for arg in args):
                role = "legacy_device"
            elif flag(args, "-m") == "experiments.plug_insertion.run_actor":
                role = "legacy_device"
            elif any("from scripts.collect_insert_demo_serl_aligned import main" in arg for arg in args):
                role = "legacy_device"
            if role:
                found.append(dict(pid=int(path.name), start_time=process_start(path.name, proc), role=role, args=args, env=env,
                                  checkpoint=flag(args, "--checkpoint_path"), run_dir=env.get("HILSERL_RUN_DIR")))
        except (OSError, UnicodeError):
            continue
    return found


def public_process(process):
    return {key: process.get(key) for key in ("pid", "start_time", "role", "checkpoint", "run_dir")}


def listening_ports(pid=None):
    if not Path("/proc/net/tcp").exists():
        return set()
    inodes = None
    if pid is not None:
        inodes = set()
        try:
            for fd in (Path("/proc") / str(pid) / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith("socket:["):
                        inodes.add(target[8:-1])
                except OSError:
                    pass
        except OSError:
            return set()
    ports = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        if not Path(table).exists():
            continue
        for line in Path(table).read_text().splitlines()[1:]:
            cells = line.split()
            if cells[3] == "0A" and (inodes is None or cells[9] in inodes):
                ports.add(int(cells[1].rsplit(":", 1)[1], 16))
    return ports


@contextmanager
def lease(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("另一个 HIL-SERL 进程已占用此操作，请使用现有任务。") from exc
        yield handle


def device_lease(root):
    return lease(Path(root) / "artifacts/control/device.lock")


def checkpoint_choice(value):
    path = Path(value).expanduser().resolve()
    if path.name.startswith("checkpoint_") and path.name[11:].isdigit() and path.exists():
        return path.parent, int(path.name[11:])
    checkpoints = [p for p in path.glob("checkpoint_*") if p.name[11:].isdigit()]
    if not checkpoints:
        raise ValueError(f"No checkpoint found: {path}")
    selected = max(checkpoints, key=lambda p: int(p.name[11:]))
    return path, int(selected.name[11:])


class CancelledLaunch(Exception):
    pass


_LAUNCH_PENDING_PHASES = {"checking", "waiting_learner", "waiting_actor"}
_ACTOR_READY_PHASES = {
    "waiting_reset", "waiting_controller", "resetting", "collecting",
    "awaiting_label", "paused", "operator_prompt",
}


class Manager:
    def __init__(self, config: Config):
        self.config = config
        self.children = {}
        self.launch_state = {"phase": "idle"}
        self._thread_lock = threading.Lock()
        self._controller_check_lock = threading.Lock()
        self._controller_state_lock = threading.RLock()

    def _launch_info(self):
        value = read_json(self.config.control_root / "launch.json") or self.launch_state
        if value.get("phase") in _LAUNCH_PENDING_PHASES and (
            not value.get("owner_start") or process_start(value.get("owner_pid")) != value["owner_start"]
        ):
            return dict(value, phase="fault", error="启动入口已退出；已创建的进程和日志保留。")
        return value

    def _publish_launch(self, phase, **fields):
        self.launch_state.update(phase=phase, **fields)
        atomic_json(self.config.control_root / "launch.json", self.launch_state)

    def _check_cancel(self):
        request = self.launch_state["request_id"]
        if (self.config.control_root / "cancel" / f"{request}.json").exists():
            raise CancelledLaunch()

    def _cancel_launch(self):
        value = self._launch_info()
        if value.get("phase") not in _LAUNCH_PENDING_PHASES:
            return False
        directory = self.config.control_root / "cancel"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / f"{value['request_id']}.json", dict(request_id=value["request_id"], **stamp()))
        return True

    @staticmethod
    def _require_alive(process):
        if not process.get("start_time") or process_start(process["pid"]) != process["start_time"]:
            raise RuntimeError("进程身份已变化或进程已退出，请重新查看状态。")

    def _learner_config(self, process):
        self._require_alive(process)
        runtime = self._learner_runtime(process)
        if runtime and (runtime.get("stop_requested") or runtime.get("phase") in
                        {"stop_requested", "closing", "stopped", "fault"}):
            raise RuntimeError("Learner 正在停止或已异常，请等待退出后再启动。")
        if (process["role"] != "learner" or flag(process["args"], "--exp_name") != "plug_insertion"
                or process["env"].get("HILSERL_MODE", "train") != "train" or not process.get("checkpoint")):
            raise RuntimeError("现有 Learner 的任务、模式或 checkpoint 不匹配。")
        if not Path(process["checkpoint"]).is_absolute():
            raise RuntimeError("现有 Learner 的 checkpoint 必须是绝对路径。")
        record = self._record(process) or {}
        if record.get("config_file"):
            return self._snapshot_config(record["config_file"], record.get("config_sha256"))
        return replace(self.config,
            action_contract=process["env"].get("HILSERL_ACTION_CONTRACT", "legacy-hybrid-7d-v1"),
            image_profile=process["env"].get("HILSERL_IMAGE_PROFILE", "full-frame128-v1"),
            port=int(process["env"].get("AGENTLACE_PORT", "5588")),
            broadcast_port=int(process["env"].get("AGENTLACE_BROADCAST_PORT", "5589")),
            classifier_ckpt=process["env"].get("HILSERL_CLASSIFIER_CKPT", self.config.classifier_ckpt),
            classifier_image_key=process["env"].get("HILSERL_CLASSIFIER_IMAGE_KEY", "side_classifier"),
            seed=int(flag(process["args"], "--seed", "42"))).validate()

    def _snapshot_config(self, filename, digest=None):
        snapshot = read_json(Path(filename))
        if not isinstance(snapshot, dict) or Path(snapshot.get("root", "")).resolve() != self.config.root.resolve():
            raise RuntimeError("训练配置快照缺失或属于其他项目，不能继续本轮训练。")
        if digest and hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest() != digest:
            raise RuntimeError("训练配置快照的校验值不一致，不能继续本轮训练。")
        return Config(**dict(snapshot, root=Path(snapshot["root"]))).validate()

    def _resume_details(self, value):
        checkpoint, step = checkpoint_choice(value)
        run = checkpoint.parent.resolve()
        if checkpoint.name != "checkpoints" or run.parent != self.config.output_root.resolve():
            raise ValueError("续训仅支持本控制台管理的训练目录。")
        metadata = read_json(run / "run.json") or {}
        if (metadata.get("mode") != "train" or not metadata.get("checkpoint")
                or Path(metadata["checkpoint"]).resolve() != checkpoint):
            raise ValueError("所选 checkpoint 不属于有效的训练 run。")
        cfg = self._snapshot_config(run / "config.json", metadata.get("config_sha256"))
        latest = max((int(p.name[11:]) for p in checkpoint.glob("checkpoint_*") if p.name[11:].isdigit()), default=-1)
        selected = checkpoint / f"checkpoint_{step}"
        receipt = read_json(selected / "_CHECKPOINT_METADATA") if selected.is_dir() else None
        if step != latest or not (selected.is_file() or isinstance(receipt, dict) and receipt.get("commit_timestamp_nsecs")):
            raise ValueError("续训需要本轮最新且已完成提交的 checkpoint，不能覆盖后续训练。")
        if step + 1 >= cfg.learner_steps:
            raise ValueError("本轮已达到配置的训练步数上限。")
        return run, checkpoint, step, cfg

    def _evaluation_config(self, checkpoint):
        """Evaluation inherits the selected model's contract, never today's default."""
        run = Path(checkpoint).resolve().parent
        metadata = read_json(run / "run.json") or {}
        if metadata:
            if not metadata.get("checkpoint") or Path(metadata["checkpoint"]).resolve() != Path(checkpoint).resolve():
                raise ValueError("checkpoint 与其运行记录不匹配。")
            return self._snapshot_config(run / "config.json", metadata.get("config_sha256"))
        raise ValueError("所选 checkpoint 缺少运行配置，无法确定动作维度；请选择控制台记录的模型。")

    def _resume_candidate(self, launch):
        runs = []
        if launch.get("run_dir"):
            runs.append(Path(launch["run_dir"]))
        runs.extend(sorted(self.config.output_root.glob("*"), reverse=True))
        for run in dict.fromkeys(runs):
            try:
                run, checkpoint, step, _ = self._resume_details(run / "checkpoints")
                return dict(checkpoint=str(checkpoint / f"checkpoint_{step}"), step=step, run_dir=str(run))
            except (OSError, ValueError, RuntimeError, TypeError):
                continue
        return None

    def _validate_pair(self, actor, learner):
        cfg = self._learner_config(learner)
        self._require_alive(actor)
        same = (
            actor["role"] == "train"
            and flag(actor["args"], "--exp_name") == "plug_insertion"
            and flag(actor["args"], "--ip", "localhost") in {"127.0.0.1", "localhost", "::1"}
            and actor.get("checkpoint") is not None
            and Path(actor["checkpoint"]).is_absolute()
            and Path(actor["checkpoint"]).resolve() == Path(learner["checkpoint"]).resolve()
            and int(flag(actor["args"], "--seed", "42")) == cfg.seed
            and int(actor["env"].get("AGENTLACE_PORT", "5588")) == cfg.port
            and int(actor["env"].get("AGENTLACE_BROADCAST_PORT", "5589")) == cfg.broadcast_port
            and actor["env"].get("HILSERL_ACTION_CONTRACT", "legacy-hybrid-7d-v1") == cfg.action_contract
            and actor["env"].get("HILSERL_IMAGE_PROFILE", "full-frame128-v1") == cfg.image_profile
            and actor["env"].get("HILSERL_CLASSIFIER_IMAGE_KEY", "side_classifier") == cfg.classifier_image_key
            and cfg.path(actor["env"].get("HILSERL_CLASSIFIER_CKPT", self.config.classifier_ckpt))
                == cfg.path(cfg.classifier_ckpt)
        )
        if not same:
            raise RuntimeError("现有 Actor 与 Learner 的任务、端口、seed、classifier 或 checkpoint 不一致。")
        if not {cfg.port, cfg.broadcast_port} <= listening_ports(learner["pid"]):
            raise RuntimeError("现有 Learner 的服务端口未就绪，不能复用运行对。")
        self._validate_fingerprints(actor, learner)
        return cfg

    @staticmethod
    def _record(process):
        """Read the immutable launch metadata for a discovered process.

        Old entry points do not create these records, so the absence of a record
        remains compatible.  When present, the record is accepted only for the
        same PID/start time; a reused PID must never inherit an old fingerprint.
        """
        run_dir = process.get("run_dir")
        if not run_dir:
            return None
        role = "learner" if process.get("role") == "learner" else "actor"
        value = read_json(Path(run_dir) / f"{role}.json")
        if not value:
            return None
        if value.get("pid") != process.get("pid") or value.get("start_time") != process.get("start_time"):
            return None
        return value

    def _validate_fingerprints(self, actor=None, learner=None):
        records = []
        if learner is not None:
            records.append(("Learner", self._record(learner)))
        if actor is not None:
            records.append(("Actor", self._record(actor)))
        records = [(name, record) for name, record in records if record]
        if not records:
            return

        # Console administration does not change policy parameters, environment
        # observations, or training data. Retain the full audit manifest, but
        # do not invalidate a live Learner merely when its control panel changes.
        administrative = {"hilserl/processes.py", "hilserl/webserver.py", "hilserl/__main__.py", "hilserl/controller.py"}
        current_source = {k: v for k, v in self._source_hashes().items() if k not in administrative}
        config_digests = []
        for name, record in records:
            source = record.get("source_sha256")
            if source and {k: v for k, v in source.items() if k not in administrative} != current_source:
                raise RuntimeError(f"现有 {name} 的 source fingerprint 与当前代码不一致，请重新启动。")
            digest = record.get("config_sha256")
            if not digest and record.get("config_file"):
                snapshot = read_json(Path(record["config_file"]))
                if snapshot:
                    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
            if digest:
                config_digests.append((name, digest))
        if len({digest for _, digest in config_digests}) > 1:
            raise RuntimeError("现有 Actor 与 Learner 的 config fingerprint 不一致，请重新启动。")

    def _wait_actor_ready(self, actor, *, timeout):
        """Wait for the application-level Actor policy handshake.

        A listening Learner port only proves that the transport is bound.  The
        Actor owns the readiness state and publishes the first non-starting
        phase after its policy callback has fired, so launch must wait for that
        state before advertising ``running``.
        """
        control = Path(actor["run_dir"]) / "control" / "status.json"
        deadline = time.monotonic() + timeout
        while True:
            self._check_cancel()
            state = read_json(control)
            if state and state.get("pid") == actor["pid"] and state.get("start_time") == actor["start_time"]:
                phase = state.get("phase")
                if phase in _ACTOR_READY_PHASES:
                    return state
                if phase == "fault":
                    raise RuntimeError(f"Actor 策略握手失败：{state.get('error', '未知错误')}")
                if phase == "stopped":
                    raise RuntimeError(f"Actor 在策略握手前停止：{state.get('reason', '未知原因')}")
            if process_start(actor["pid"]) != actor["start_time"]:
                raise RuntimeError("Actor 在策略握手完成前退出；进程与日志已保留。")
            if time.monotonic() >= deadline:
                raise RuntimeError("等待 Actor 策略握手超时；进程与日志已保留。")
            time.sleep(.5)

    def _reconcile_launch(self, launch, running):
        """Converge persisted launch state after an Actor exits or faults."""
        if launch.get("phase") == "stopping_learner":
            run_dir = launch.get("run_dir")
            learner_alive = any(
                p.get("role") == "learner" and p.get("run_dir") == run_dir
                for p in running
            )
            if learner_alive:
                return launch
            record = read_json(Path(run_dir) / "learner.json") if run_dir else None
            runtime = self._learner_runtime(record) if record else None
            if runtime and (runtime.get("checkpoint_error") or runtime.get("phase") == "fault"):
                updated = dict(launch, phase="fault", error="Learner 退出时出现错误：" + str(runtime.get("checkpoint_error") or runtime.get("error")))
            else:
                reason = "Learner 已结束并完成保存" if runtime and runtime.get("checkpoint_saved") is True else "Learner 已退出；最终保存结果请查看日志"
                updated = dict(launch, phase="stopped", reason=reason)
            atomic_json(self.config.control_root / "launch.json", updated)
            self.launch_state = dict(updated)
            return updated
        if launch.get("phase") not in {"running", "waiting_actor", "degraded"}:
            return launch
        # ``--learner-only`` deliberately has no Actor.  Its running state is
        # owned by the Learner and must not be degraded merely because the
        # control directory contains no Actor status file.
        if not launch.get("actor_pid") and not launch.get("actor"):
            return launch
        run_dir = launch.get("run_dir")
        if not run_dir:
            return launch
        actor = next((p for p in running if p.get("role") != "learner" and p.get("run_dir") == run_dir), None)
        state = self._operator_state(actor) if actor else None
        if actor and state is None:
            # The process can be alive for a short window before
            # OperatorControl publishes its first status file.  Keep that
            # window as a handshake wait instead of declaring a false fault.
            if launch.get("phase") == "running":
                updated = dict(launch, phase="waiting_actor")
                atomic_json(self.config.control_root / "launch.json", updated)
                self.launch_state = dict(updated)
                return updated
            return launch
        if actor and state and state.get("phase") not in {"fault", "stopped"}:
            if state.get("phase") == "starting" and launch.get("phase") == "running":
                updated = dict(launch, phase="waiting_actor")
                atomic_json(self.config.control_root / "launch.json", updated)
                self.launch_state = dict(updated)
                return updated
            return launch

        persisted = read_json(Path(run_dir) / "control" / "status.json") or state or {}
        phase = persisted.get("phase")
        learner_alive = any(
            p.get("role") == "learner" and p.get("run_dir") == run_dir
            for p in running
        )
        if phase == "stopped" and not learner_alive:
            target = "stopped"
            fields = {"reason": persisted.get("reason", "Actor 已停止")}
        else:
            target = "degraded"
            fields = {
                "error": persisted.get(
                    "error",
                    "Actor 已退出，Learner 仍在运行；当前训练对不再完整。"
                    if learner_alive else "Actor 已退出，当前训练对不再完整。",
                )
            }
        updated = dict(launch, phase=target, **fields)
        if updated != launch:
            atomic_json(self.config.control_root / "launch.json", updated)
            self.launch_state = dict(updated)
        return updated

    def _source_hashes(self):
        root = self.config.root
        files = [root/"_run_actor.py", root/"experiments/config.py", root/"experiments/mappings.py",
                 *sorted((root/"hilserl").rglob("*.py")),
                 *sorted((root/"experiments/plug_insertion").rglob("*.py")),
                 root/"scripts/video_capture.py", root/"scripts/xbox_intervention.py",
                 root/"scripts/zed_capture.py",
                 *sorted((root/"upstream/hil-serl/serl_robot_infra/franka_env/envs").glob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/wrappers").glob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/data").glob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/common").glob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/agents").rglob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/networks").rglob("*.py")),
                 *sorted((root/"upstream/hil-serl/serl_launcher/serl_launcher/vision").rglob("*.py")),
                 root/"upstream/hil-serl/serl_launcher/serl_launcher/utils/launcher.py",
                 *sorted((root/"upstream/agentlace/agentlace").rglob("*.py"))]
        return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}

    def _new_run(self, mode, checkpoint=None, *, config=None):
        cfg = config or self.config
        run = cfg.output_root / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{mode}_{uuid.uuid4().hex[:6]}"
        run.mkdir(parents=True, exist_ok=False)
        for directory in ("logs", "control", "recordings", "attempts"):
            (run / directory).mkdir()
        atomic_json(run / "config.json", cfg.snapshot())
        atomic_json(run / "run.json", dict(id=run.name, mode=mode, started=stamp(),
            checkpoint=str(checkpoint or run/"checkpoints"),config_sha256=cfg.digest(),source_sha256=self._source_hashes()))
        return run

    def preflight(self, *, need_demos=True, check_robot=False, config=None):
        cfg = config or self.config
        errors = []
        for path in (cfg.path(cfg.python), cfg.root / "_run_actor.py"):
            if not path.is_file():
                errors.append(f"文件缺失：{path}")
        if not any(cfg.path(cfg.classifier_ckpt).glob("checkpoint_*")):
            errors.append(f"分类器 checkpoint 缺失：{cfg.path(cfg.classifier_ckpt)}")
        demos = sorted(cfg.path(cfg.demo_dir).glob("*_success.pkl"))
        demo_count = len(demos)
        demo_transition_count = None
        if need_demos:
            if cfg.action_contract == "fixed-xyz-v1":
                from hilserl.seed_dataset import validate_seed_dataset
                try:
                    if not cfg.seed_dataset_sha256:
                        raise ValueError("fixed-xyz-v1 缺少已核验的示范集 SHA-256")
                    seed_manifest = validate_seed_dataset(cfg.path(cfg.demo_dir), expected_action_contract=cfg.action_contract,
                                                          expected_manifest_sha256=cfg.seed_dataset_sha256,
                                                          expected_image_profile=cfg.image_profile)
                    demo_count = seed_manifest["counts"]["episodes"]
                    demo_transition_count = seed_manifest["counts"]["transitions"]
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(f"示教数据验证失败：{exc}")
            elif not demos or any(p.stat().st_size == 0 for p in demos):
                errors.append(f"示教数据缺失或为空：{cfg.path(cfg.demo_dir)}")
        if not shutil.which("ffmpeg"):
            errors.append("未找到 ffmpeg")
        destination = cfg.output_root
        while not destination.exists():
            destination = destination.parent
        free_gib = shutil.disk_usage(destination).free / 2**30
        if free_gib < cfg.min_free_gib:
            errors.append(f"数据盘剩余 {free_gib:.1f} GiB，已到保留容量下限")
        if not os.access(destination, os.W_OK):
            errors.append(f"数据目录不可写：{destination}")
        if check_robot:
            address = urlparse(cfg.server_url)
            try:
                with socket.create_connection((address.hostname, address.port or 80), timeout=3):
                    pass
            except OSError:
                errors.append("机器人服务 TCP 不可达，请先准备既有底层控制器。")
        return dict(ok=not errors, errors=errors, demo_count=demo_count,
                    demo_transition_count=demo_transition_count, free_gib=free_gib,
                    data_dir=str(cfg.output_root), robot_check="TCP only" if check_robot else "not requested")

    def _spawn(self, role, run, checkpoint, cfg, *, mode="train", eval_step=0, eval_episodes=10,
               resume_step=-1, learner_paused=False):
        # Retention is the current storage policy, not a learned-model setting.
        # Historical model/data settings stay frozen; each new attempt records
        # the actual storage policy without rewriting the old run snapshot.
        cfg = replace(cfg, checkpoint_keep=self.config.checkpoint_keep)
        attempt_id = uuid.uuid4().hex
        attempt = run / "attempts" / attempt_id
        attempt.mkdir(parents=True, exist_ok=False)
        recording = run / "recordings" / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{attempt_id[:6]}"
        env = cfg.environment(role, run_dir=run, recording_dir=recording if role=="actor" else None)
        env.update(HILSERL_MODE=mode, HILSERL_ATTEMPT_ID=attempt_id,
                   HILSERL_CONFIG_SNAPSHOT=str(attempt/"config.json"))
        atomic_json(attempt / "config.json", cfg.snapshot())
        relevant = {k:v for k,v in env.items() if k.startswith(("HILSERL_", "FRANKA_", "ACTOR_", "AUTO_REWARD_", "RESET_", "MANUAL_", "DROP_", "XBOX_", "AGENTLACE_"))
                    or k in {"CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION"}}
        atomic_json(attempt/"environment.json", relevant)
        args = [str(cfg.path(cfg.python)), "-u", str(cfg.root/"_run_actor.py"), "--exp_name=plug_insertion", f"--{role}",
                f"--checkpoint_path={checkpoint}", f"--seed={cfg.seed}", "--debug"]
        if role == "learner":
            if resume_step >= 0:
                args.append(f"--resume_checkpoint_step={resume_step}")
            if learner_paused:
                args.append("--learner_paused")
            if cfg.action_contract == "legacy-hybrid-7d-v1":
                for path in sorted(cfg.path(cfg.demo_dir).glob("*_success.pkl")):
                    args.extend(["--demo_path", str(path)])
        else:
            args.append("--ip=127.0.0.1")
            if mode == "eval":
                args.extend([f"--eval_checkpoint_step={eval_step}", f"--eval_n_trajs={eval_episodes}"])
        log_path = run / "logs" / f"{role}_{attempt_id}.log"
        self._check_cancel()
        reader_fd = None
        try:
            if role == "actor" and mode == "eval":
                from hilserl.checkpoint_retention import open_reader_guard
                reader_fd = open_reader_guard(checkpoint, eval_step)
                env["HILSERL_CHECKPOINT_READER_FD"] = str(reader_fd)
            with log_path.open("ab", buffering=0) as log:
                child = subprocess.Popen(args, cwd=cfg.root, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=() if reader_fd is None else (reader_fd,))
        finally:
            if reader_fd is not None:
                # Closing this duplicate preserves the child's inherited flock.
                os.close(reader_fd)
        self.children[child.pid] = child
        record = dict(pid=child.pid, start_time=process_start(child.pid), role=role, mode=mode,
                      attempt_id=attempt_id,run_dir=str(run),checkpoint=str(checkpoint),
                      recording_dir=str(recording) if role=="actor" else None, log=str(log_path),
                      config_file=str(attempt/"config.json"), config_sha256=cfg.digest(),
                      source_sha256=self._source_hashes(),**stamp())
        if role == "learner":
            record["learner_control_protocol"] = 1
            record["resume_checkpoint_step"] = resume_step
            record["initial_paused"] = learner_paused
        atomic_json(attempt/"process.json", record)
        atomic_json(run / f"{role}.json", record)
        return child

    @staticmethod
    def _fresh_checkpoint(path):
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(p.name != ".writer.lock" for p in path.iterdir())):
            raise RuntimeError("新训练必须使用空 checkpoint 目录，不能自动恢复历史权重或 buffers。")

    @staticmethod
    def validate_launch_request(mode="train", *, checkpoint=None, eval_episodes=10, learner_only=False,
                                resume_checkpoint=None, learner_paused=False):
        if mode not in {"train", "eval", "collect"}:
            raise ValueError("Unknown mode")
        if mode == "eval" and (not checkpoint or type(eval_episodes) is not int or eval_episodes <= 0):
            raise ValueError("Evaluation requires a checkpoint and positive episode count")
        if learner_only and mode != "train":
            raise ValueError("learner_only is only available for training")
        if type(learner_paused) is not bool or (learner_paused and mode != "train"):
            raise ValueError("learner_paused is a training-only boolean")
        if resume_checkpoint is not None and (mode != "train" or checkpoint is not None
                or not isinstance(resume_checkpoint, (str, Path)) or not str(resume_checkpoint)):
            raise ValueError("resume_checkpoint requires training and cannot be combined with checkpoint")

    def launch(self, mode="train", *, checkpoint=None, eval_episodes=10, learner_only=False,
               resume_checkpoint=None, learner_paused=False):
        self.validate_launch_request(mode, checkpoint=checkpoint, eval_episodes=eval_episodes,
                                     learner_only=learner_only, resume_checkpoint=resume_checkpoint,
                                     learner_paused=learner_paused)
        with self._thread_lock, lease(self.config.control_root/"launch.lock"):
            if (self._launch_info().get("phase") == "stopping_learner"
                    and any(p["role"] == "learner" for p in discover(self.config.root))):
                raise RuntimeError("请等待 Learner 完成停止并退出后再启动。")
            self.launch_state = dict(phase="checking",mode=mode,request_id=uuid.uuid4().hex,
                                     owner_pid=os.getpid(),owner_start=process_start(os.getpid()))
            self._publish_launch("checking")
            try:
                running = discover(self.config.root)
                learners = [p for p in running if p["role"] == "learner"]
                devices = [p for p in running if p["role"] != "learner"]
                if len(learners)>1 or len(devices)>1:
                    raise RuntimeError("发现多个训练或设备进程，请先在原入口核对。")
                if resume_checkpoint is not None and running:
                    raise RuntimeError("请先结束现有 Actor 和 Learner，再从已保存的 checkpoint 续训。")
                if devices:
                    if mode=="train" and len(learners)==1:
                        self._validate_pair(devices[0],learners[0])
                        if checkpoint and Path(checkpoint).resolve()!=Path(learners[0]["checkpoint"]).resolve():
                            raise RuntimeError("请求的 checkpoint 与当前运行不一致。")
                        result=dict(reused=True,actor=public_process(devices[0]),learner=public_process(learners[0]))
                        self._publish_launch("running",**result)
                        return result
                    raise RuntimeError("机器人与相机已被占用，请先结束当前设备任务。")
                cfg=self.config
                resume_run, resume_step = None, -1
                eval_step = 0
                if mode == "eval":
                    checkpoint, eval_step = checkpoint_choice(checkpoint)
                    cfg = self._evaluation_config(checkpoint)
                if resume_checkpoint is not None:
                    resume_run, checkpoint, resume_step, cfg = self._resume_details(resume_checkpoint)
                learner=learners[0] if mode=="train" and learners else None
                if learner:
                    cfg=self._learner_config(learner)
                    # A running Learner may predate the current protocol or
                    # launcher code.  Refuse to pair a new Actor with it when
                    # immutable launch metadata is available.
                    self._validate_fingerprints(learner=learner)
                    if checkpoint and Path(checkpoint).resolve()!=Path(learner["checkpoint"]).resolve():
                        raise RuntimeError("请求的 checkpoint 与现有 Learner 不一致。")
                    checkpoint=Path(learner["checkpoint"])
                report=self.preflight(need_demos=mode=="train" and learner is None,
                                      check_robot=not learner_only,config=cfg)
                if not report["ok"]:
                    raise RuntimeError("；".join(report["errors"]))
                self._check_cancel()
                if resume_run is not None:
                    run = resume_run
                elif learner and learner.get("run_dir") and (Path(learner["run_dir"])/"run.json").is_file():
                    run=Path(learner["run_dir"])
                else:
                    run=self._new_run(mode,checkpoint,config=cfg)
                checkpoint=Path(checkpoint).resolve() if checkpoint else run/"checkpoints"
                if mode=="train" and learner is None:
                    if resume_step < 0:
                        self._fresh_checkpoint(checkpoint)
                    occupied={cfg.port,cfg.broadcast_port}&listening_ports()
                    if occupied:
                        raise RuntimeError(f"端口已被其他进程占用：{sorted(occupied)}")
                    child=self._spawn("learner",run,checkpoint,cfg, resume_step=resume_step,
                                      learner_paused=learner_paused)
                    learner=dict(pid=child.pid,start_time=process_start(child.pid),checkpoint=str(checkpoint),role="learner",run_dir=str(run))
                if mode=="train":
                    self._publish_launch("waiting_learner",run_dir=str(run),learner=public_process(learner))
                    deadline=time.monotonic()+cfg.startup_timeout_seconds
                    while True:
                        self._check_cancel()
                        self._require_alive(learner)
                        if {cfg.port,cfg.broadcast_port} <= listening_ports(learner["pid"]):
                            break
                        if time.monotonic()>=deadline:
                            raise RuntimeError("等待 Learner 初始化超时；进程与日志已保留。")
                        time.sleep(.5)
                    self._require_alive(learner)
                self._check_cancel()
                if learner_only:
                    result=dict(run_dir=str(run),learner=public_process(learner))
                else:
                    child=self._spawn("actor",run,checkpoint,cfg,mode=mode,eval_step=eval_step,eval_episodes=eval_episodes)
                    actor_start = process_start(child.pid)
                    actor_identity = dict(pid=child.pid, start_time=actor_start, role="train",
                                          run_dir=str(run), env=cfg.environment("actor", run_dir=run))
                    self._publish_launch("waiting_actor", run_dir=str(run),
                                         learner=public_process(learner) if learner else None,
                                         actor_pid=child.pid, actor_start=actor_start)
                    try:
                        self._wait_actor_ready(actor_identity, timeout=cfg.startup_timeout_seconds)
                    except CancelledLaunch:
                        identity=read_json(run/"actor.json")
                        if identity and process_start(identity.get("pid")) == identity.get("start_time"):
                            os.kill(identity["pid"],signal.SIGINT)
                        raise
                    result=dict(run_dir=str(run),actor_pid=child.pid,learner_pid=learner["pid"] if learner else None)
                self._publish_launch("running",**result)
                return result
            except CancelledLaunch:
                self._publish_launch("cancelled")
                return dict(phase="cancelled",message="Actor 启动已取消，已有 Learner 和数据保留。")
            except KeyboardInterrupt:
                self._cancel_launch()
                self._publish_launch("cancelled")
                raise
            except Exception as exc:
                self._publish_launch("fault",error=str(exc))
                raise

    @staticmethod
    def _operator_state(actor):
        if not actor.get("run_dir"):
            return None
        state=read_json(Path(actor["run_dir"])/"control/status.json")
        if (not state or state.get("pid")!=actor["pid"] or state.get("start_time")!=actor["start_time"]
                or state.get("attempt_id")!=actor["env"].get("HILSERL_ATTEMPT_ID")):
            return None
        return state

    @staticmethod
    def _learner_runtime(process):
        if not process or not process.get("run_dir"):
            return None
        state = read_json(Path(process["run_dir"]) / "control/learner_status.json")
        attempt = process.get("attempt_id") or process.get("env", {}).get("HILSERL_ATTEMPT_ID")
        if not attempt:
            record = Manager._record(process)
            attempt = (record or {}).get("attempt_id")
        if not state or not attempt or any(state.get(k) != expected for k, expected in (
                ("pid", process.get("pid")), ("start_time", process.get("start_time")), ("attempt_id", attempt))):
            return None
        return state

    def _previous_actor_record(self, launch):
        """Resolve the last Actor independently of the Learner's launch run.

        Stopping a Learner can replace launch.run_dir after a separate collect
        session. Its retained Actor identity still identifies the right archive.
        """
        actor = launch.get("actor") or {}
        pid = launch.get("actor_pid") or actor.get("pid")
        start = launch.get("actor_start") or actor.get("start_time")
        if not pid or not start:
            return None
        run_dir = actor.get("run_dir") or launch.get("run_dir")
        candidates = [Path(run_dir) / "actor.json"] if run_dir else []
        candidates.extend(self.config.output_root.glob("*/actor.json"))
        for path in dict.fromkeys(candidates):
            try:
                record = read_json(path)
            except (OSError, ValueError):
                continue
            if (record and record.get("pid") == pid and record.get("start_time") == start
                    and record.get("attempt_id") and record.get("run_dir")
                    and Path(record["run_dir"]).resolve() == path.parent.resolve()):
                return record
        return None

    def status(self):
        for pid,child in list(self.children.items()):
            if child.poll() is not None:
                self.children.pop(pid,None)
        running=discover(self.config.root)
        actor=next((p for p in running if p["role"]!="learner"),None)
        state=self._operator_state(actor) if actor else None
        launch=self._reconcile_launch(self._launch_info(), running)
        effective=None
        if actor:
            snapshot=actor.get("env",{}).get("HILSERL_CONFIG_SNAPSHOT")
            if snapshot:
                effective=read_json(Path(snapshot))
            if state is None:
                state=dict(phase="starting" if actor.get("run_dir") else "legacy_running",pid=actor["pid"],
                           prompt="旧入口正在运行，0/1 请继续使用原终端。" if not actor.get("run_dir") else "正在初始化环境和模型")
        else:
            previous = self._previous_actor_record(launch)
            if previous:
                state = self._operator_state(dict(previous, env={
                    "HILSERL_ATTEMPT_ID": previous["attempt_id"],
                }))
                if state and state.get("phase") not in {"fault", "stopped"}:
                    state=dict(state,phase="fault",error="Actor 已退出，当前条目未正常结束；请查看日志与归档。")
                if previous.get("config_file"):
                    effective=read_json(Path(previous["config_file"]))
        directory=self.config.output_root
        while not directory.exists():
            directory=directory.parent
        learner_policy = {"state": "stopped", "reason": "没有运行中的 Learner"}
        learner = next((p for p in running if p["role"] == "learner"), None)
        learner_runtime = self._learner_runtime(learner)
        if not learner and launch.get("run_dir"):
            learner_runtime = self._learner_runtime(read_json(Path(launch["run_dir"]) / "learner.json"))
        if learner:
            actor_error = str((state or {}).get("error", ""))
            actor_phase = (state or {}).get("phase")
            launch_phase = launch.get("phase")
            shutdown = (learner_runtime or {}).get("phase")
            if shutdown in {"stop_requested", "saving_checkpoint", "closing", "stopped", "fault"}:
                phases = {"stop_requested": ("stop_requested", "Learner 已确认停止请求，等待当前更新完成"),
                          "saving_checkpoint": ("saving_checkpoint", "Learner 正在写入最终 checkpoint" if learner_runtime.get("stop_requested") else "训练已暂停，正在保存当前 checkpoint"),
                          "closing": ("closing", "保存结果已记录，正在释放通信资源"),
                          "stopped": ("closing", "Learner 已完成收尾，等待进程退出"),
                          "fault": ("fault", (learner_runtime or {}).get("checkpoint_error") or (learner_runtime or {}).get("error") or "Learner 收尾出现错误")}
                state_name, reason = phases[shutdown]
                learner_policy = {"state": state_name, "reason": reason}
            elif launch_phase == "stopping_learner":
                reason = "等待 Learner 确认停止请求，尚未进入最终保存"
                if time.time_ns() - launch.get("stop_requested_ns", 0) > 15_000_000_000:
                    reason = "Learner 尚未确认停止；请求仍待处理，请查看运行日志"
                learner_policy = {"state": "stop_requested", "reason": reason}
            elif shutdown == "paused":
                learner_policy = {"state": "paused", "reason": learner_runtime.get("pause_reason") or "训练已暂停，参数和数据保留。"}
            elif shutdown == "initializing":
                learner_policy = {"state": "waiting", "reason": "正在初始化或恢复训练状态，尚未更新参数。"}
            elif not actor and launch_phase in {"degraded", "fault"}:
                learner_policy = {
                    "state": "degraded",
                    "reason": "Actor 已退出，Learner 仍在运行；策略链路已断开",
                }
            elif not actor and launch_phase == "stopped":
                learner_policy = {
                    "state": "degraded",
                    "reason": "Actor 已停止，Learner 仍在运行；策略未被控制台确认",
                }
            elif not actor and not launch.get("actor_pid"):
                learner_policy = {
                    "state": "learner_only",
                    "reason": "Learner 独立运行，未启动 Actor",
                }
            elif actor_phase == "fault" and "learner policy" in actor_error.lower():
                learner_policy = {"state": "fault", "reason": "Actor 未收到 Learner 策略参数"}
            elif actor_phase == "starting" and "learner" in str((state or {}).get("prompt", "")).lower():
                learner_policy = {"state": "waiting", "reason": "等待 Actor 完成策略握手"}
            elif actor_phase not in {"fault", "stopped", None}:
                learner_policy = {"state": "connected", "reason": "Actor 已收到策略参数"}
            else:
                learner_policy = {"state": "unknown", "reason": "进程在运行，尚无策略握手结论"}
        receipts = [read_json(self.config.control_root / "gripper.json"), (state or {}).get("gripper")]
        gripper = max((item for item in receipts if item), key=lambda item: item.get("unix_ns", 0), default=None)
        retention = dict(keep_count=self.config.checkpoint_keep,
                         rule="latest_recovery_then_comparable_evaluation_then_recency", last_maintenance=None)
        if launch.get("run_dir"):
            status_path = Path(launch["run_dir"]) / "checkpoints/retention_status.json"
            try:
                retention["last_maintenance"] = read_json(status_path)
            except (OSError, ValueError):
                retention["last_maintenance"] = dict(status="error", error="Checkpoint retention status is unreadable")
        return dict(processes=[public_process(p) for p in running],actor=state,launch=launch,
                    checkpoint_retention=retention,
                    learner_policy=learner_policy,
                    learner_runtime=learner_runtime,
                    resume_candidate=None if learner else self._resume_candidate(launch),
                    gripper=gripper,
                    controller=self._controller_info(),
                    free_gib=shutil.disk_usage(directory).free/2**30,data_dir=str(self.config.output_root),
                    config=self.config.snapshot(),effective_config=effective)

    def command(self,name,value=None,expected=None):
        from hilserl.control import send_command
        cancelled=self._cancel_launch() if name=="stop" else False
        devices=[p for p in discover(self.config.root) if p["role"]!="learner"]
        if not devices and cancelled:
            return dict(phase="cancelling")
        if len(devices)!=1 or not devices[0].get("run_dir"):
            raise ValueError("没有可由控制台操作的 Actor；旧进程请使用原终端。")
        actor=devices[0]
        self._require_alive(actor)
        state=self._operator_state(actor)
        if state is None:
            if name!="stop" or not actor["env"].get("HILSERL_ATTEMPT_ID") or expected is not None:
                raise ValueError("Actor 状态尚未就绪或已经变化。")
            self._require_alive(actor)
            os.kill(actor["pid"],signal.SIGINT)
            return dict(phase="stopping",pid=actor["pid"])
        if name == "stop" and state.get("phase") == "fault":
            # A failed actor no longer polls its command queue.  Signal the
            # still-live process (usually its Agentlace listener thread) so
            # the console can recover the device lease.
            self._require_alive(actor)
            os.kill(actor["pid"], signal.SIGINT)
            return dict(phase="stopping", pid=actor["pid"])
        return send_command(Path(actor["run_dir"])/"control",name,value,expected=expected)

    def gripper(self, operation, *, expected=None):
        """Direct idle I/O, or one fenced request to the Actor that owns the device."""
        from hilserl.control import GRIPPER_PHASES, send_command
        from hilserl.gripper import GripperControl
        if operation not in {"open", "close", "status"}:
            raise ValueError("夹爪操作仅支持 open、close 或 status")
        with lease(self.config.control_root / "launch.lock"):
            if self._launch_info().get("phase") in _LAUNCH_PENDING_PHASES:
                raise ValueError("正在启动，暂不能控制夹爪；请等待 Actor 进入等待复位。")
            devices = [p for p in discover(self.config.root) if p["role"] != "learner"]
            if devices:
                if len(devices) != 1 or not devices[0].get("run_dir"):
                    raise ValueError("设备由其他入口占用；请在原入口控制夹爪。")
                actor = devices[0]
                self._require_alive(actor)
                state = self._operator_state(actor)
                if not state or state.get("phase") not in GRIPPER_PHASES:
                    raise ValueError("请先暂停或停止 Actor；复位和采集中不能手动开合夹爪。")
                if operation == "status":
                    result = GripperControl(actor["env"].get("HILSERL_SERVER_URL", self.config.server_url)).status()
                    return self._save_gripper_receipt(dict(operation="status", after=result, verification="readback"))
                if not state.get("gripper_available"):
                    raise ValueError("当前 Actor 未加载夹爪控制，请停止后重新启动 Actor。")
                item = send_command(Path(actor["run_dir"])/"control", "gripper", operation, expected=expected)
                # Preserve queue timestamp so a completed Actor receipt always
                # supersedes this receipt even if the Actor was polled quickly.
                result = dict(operation=operation, verification="queued", command_id=item["command_id"],
                              command_acknowledged=False, message="已提交给 Actor，等待安全阶段执行",
                              unix_ns=item["unix_ns"], monotonic_ns=item["monotonic_ns"])
                return self._save_gripper_receipt(result)
            if expected and expected.get("attempt_id"):
                raise ValueError("Actor 已退出，旧页面夹爪请求已失效；请刷新后操作。")
            with device_lease(self.config.root):
                control = GripperControl(self.config.server_url)
                try:
                    result = (dict(operation="status", after=control.status(), verification="readback")
                              if operation == "status" else control.command(operation))
                except Exception as exc:
                    self._save_gripper_receipt(dict(getattr(exc, "result", {}), operation=operation,
                                                   verification="error", message=str(exc)))
                    raise
                return self._save_gripper_receipt(result)

    def _save_gripper_receipt(self, result):
        record = dict(stamp(), **result)
        record.setdefault("command_id", uuid.uuid4().hex)
        directory = self.config.control_root / "gripper"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / f"{record['command_id']}.json", record)
        atomic_json(self.config.control_root / "gripper.json", record)
        return record

    def _controller_info(self):
        value = read_json(self.config.control_root / "controller.json") or {
            "phase": "unknown", "message": "尚未检查底层控制服务"}
        if value.get("phase") == "recovering" and process_start(value.get("owner_pid")) != value.get("owner_start"):
            return dict(value, phase="failed", error="恢复入口已退出，远端结果未知；请检查服务状态。")
        if (value.get("phase") == "ready" and
                time.time_ns() - value.get("unix_ns", 0) > 10_000_000_000):
            return dict(value, phase="unknown", message="健康检查已过期，正在重新检查", health=None)
        return value

    def controller_status(self, *, refresh=False):
        """Read-only health refresh, independently of the Actor lifecycle."""
        from hilserl.controller import ControllerService
        value = self._controller_info()
        if value.get("phase") == "recovering":
            return value
        if not refresh and time.time_ns() - value.get("unix_ns", 0) < 2_000_000_000:
            return value
        if not self._controller_check_lock.acquire(blocking=False):
            return value
        try:
            result = ControllerService(self.config.server_url).status()
            # Recover can begin while a GET was in flight. Never overwrite its
            # operation state with a racing health observation.
            with self._controller_state_lock:
                if self._controller_info().get("phase") == "recovering":
                    return self._controller_info()
                return self._save_controller(result)
        finally:
            self._controller_check_lock.release()

    def _save_controller(self, result):
        value = dict(result, **stamp())
        value["message"] = value.get("reason") or value.get("message") or ""
        if value.get("phase") == "failed":
            value["error"] = value.get("error") or value["message"]
        self.config.control_root.mkdir(parents=True, exist_ok=True)
        with self._controller_state_lock:
            atomic_json(self.config.control_root / "controller.json", value)
        return value

    def validate_controller_recovery(self):
        address = urlparse(self.config.server_url)
        if address.hostname != "172.16.0.1" or (address.port or 80) != 5000:
            raise ValueError("此配置不是已安装恢复程序对应的 FR3 控制服务")
        if any(p["role"] != "learner" for p in discover(self.config.root)):
            raise ValueError("请先停止 Actor，再恢复底层控制服务。")
        if self._launch_info().get("phase") in _LAUNCH_PENDING_PHASES:
            raise ValueError("请先取消正在进行的启动，再恢复底层服务。")
        if self._controller_info().get("phase") == "recovering":
            raise ValueError("底层服务正在恢复，请等待。")

    def recover_controller(self):
        """Explicit recovery while no Actor owns the robot; Learner is retained."""
        from hilserl.controller import ControllerService
        with lease(self.config.control_root / "launch.lock"), device_lease(self.config.root):
            self.validate_controller_recovery()
            operation = dict(phase="recovering", owner_pid=os.getpid(), owner_start=process_start(os.getpid()),
                             request_id=uuid.uuid4().hex, message="正在恢复底层服务")
            self._save_controller(operation)
            try:
                result = ControllerService(self.config.server_url).recover()
                final = self._save_controller({**operation, **result})
            except Exception as exc:
                final = self._save_controller(dict(operation, phase="failed", error=str(exc)))
                raise
            finally:
                receipt = self._controller_info()
                directory = self.config.control_root / "controller-operations"
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / f"{operation['request_id']}.json", receipt)
            return final

    def learner_activity(self, command, *, expected):
        from hilserl.learner_control import request_activity
        if command not in {"pause", "resume"}:
            raise ValueError("Learner 仅支持 pause / resume。")
        with self._thread_lock, lease(self.config.control_root / "launch.lock"):
            learners = [p for p in discover(self.config.root) if p["role"] == "learner"]
            if len(learners) != 1:
                raise ValueError("无法唯一确定 Learner")
            process = learners[0]
            self._require_alive(process)
            state = self._learner_runtime(process)
            if (not state or not isinstance(expected, dict)
                    or any(expected.get(k) != state.get(k) for k in ("pid", "start_time", "attempt_id"))):
                raise ValueError("Learner 身份已变化，请刷新控制台后重试。")
            if (state.get("stop_requested") or state.get("phase") in {"stop_requested", "closing", "stopped", "fault"}
                    or self._launch_info().get("phase") == "stopping_learner"):
                raise ValueError("Learner 正在停止，不能再切换暂停状态。")
            request = request_activity(process["run_dir"], command=command,
                **{k: state[k] for k in ("pid", "start_time", "attempt_id")})
            return dict(phase="requested", command=command, request_id=request["request_id"],
                        message="已请求暂停参数更新，等待当前完整更新完成。" if command == "pause" else "已请求继续 Learner；Actor 采集和新数据恢复后继续训练。")

    def stop_learner(self):
        from hilserl.learner_control import request_stop
        with self._thread_lock, lease(self.config.control_root / "launch.lock"):
            running=discover(self.config.root)
            if any(p["role"]!="learner" for p in running) or self._launch_info().get("phase") in _LAUNCH_PENDING_PHASES:
                raise ValueError("请先停止 Actor 或取消启动，再结束 Learner 并保存 checkpoint。")
            learners=[p for p in running if p["role"]=="learner"]
            if len(learners)!=1:
                raise ValueError("无法唯一确定 Learner")
            process=learners[0]
            self._require_alive(process)
            previous = self._launch_info()
            if (previous.get("phase") == "stopping_learner"
                    and (previous.get("learner") or {}).get("pid") == process["pid"]
                    and (previous.get("learner") or {}).get("start_time") == process["start_time"]):
                return dict(pid=process["pid"], phase="stop_requested", repeated=True,
                            message="停止请求已提交，正在等待 Learner 的确认和保存结果")
            record = self._record(process) or {}
            attempt = record.get("attempt_id") or process.get("env", {}).get("HILSERL_ATTEMPT_ID")
            request = None
            if process.get("run_dir") and attempt:
                request = request_stop(process["run_dir"], pid=process["pid"], start_time=process["start_time"], attempt_id=attempt)
            self.launch_state = dict(previous)
            self.launch_state.pop("error", None)
            fields = {"run_dir": process.get("run_dir"), "learner_pid": process["pid"],
                      "learner": public_process(process), "stop_requested_ns": time.time_ns(),
                      "learner_stop_request_id": (request or {}).get("request_id")}
            self._publish_launch("stopping_learner", **fields)
            if record.get("learner_control_protocol") != 1 and self._learner_runtime(process) is None:
                # Legacy compatibility: send once, but never equate dispatch
                # with acknowledgement or a completed save.
                os.kill(process["pid"], signal.SIGINT)
            return dict(pid=process["pid"], phase="stop_requested", message="停止请求已提交，等待 Learner 确认后保存")
