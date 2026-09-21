"""One CLI for the local console, device modes, and archive exports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from hilserl.config import load_config
from hilserl.processes import Manager, lease, process_start
from hilserl.files import atomic_json


def console(config, args):
    from hilserl.webserver import serve
    port = args.port or config.console_port
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def existing():
        try:
            with opener.open(url + "/api/status", timeout=1) as response:
                value = json.load(response)
            if value["config"]["root"] != str(config.root) or value.get("offline") != args.offline:
                raise RuntimeError("Console port belongs to another project or mode")
            return True
        except (OSError, urllib.error.URLError):
            return False

    if args.background:
        with lease(config.control_root / "console-start.lock"):
            if not existing():
                config.control_root.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-m", "hilserl"]
                if args.config:
                    command += ["--config", str(Path(args.config).resolve())]
                if args.data_dir:
                    command += ["--data-dir", args.data_dir]
                command += ["console", "--port", str(port)]
                if args.offline:
                    command.append("--offline")
                with (config.control_root / "console.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(command, cwd=config.root, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                atomic_json(config.control_root / "console.json",
                            dict(pid=child.pid, start_time=process_start(child.pid), port=port, url=url))
                deadline = time.monotonic() + 10
                while not existing():
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Console startup failed; see {config.control_root / 'console.log'}")
                    time.sleep(.2)
        if args.open:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open", url],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"url": url}
    if args.open:
        raise ValueError("Use console --background --open to open the desktop browser")
    serve(config, port=port, offline=args.offline)


def main(argv=None):
    parser = argparse.ArgumentParser(description="HIL-SERL 插插头统一入口")
    parser.add_argument("--config", help="配置 JSON 路径；省略时使用 config/hilserl.json")
    parser.add_argument("--data-dir", help="运行与录制存储目录")
    sub = parser.add_subparsers(dest="command", required=True)
    page = sub.add_parser("console", help="打开本机工作台")
    page.add_argument("--port", type=int)
    page.add_argument("--open", action="store_true")
    page.add_argument("--background", action="store_true")
    page.add_argument("--offline", action="store_true", help="禁用设备启动和控制，用于离线验收")
    train = sub.add_parser("train", help="新建训练或复用当前匹配的 Learner + Actor")
    train.add_argument("--learner-only", action="store_true")
    train.add_argument("--resume-checkpoint", help="显式续训受管理 run 的最新已保存 checkpoint，保留原 step 和 optimizer")
    train.add_argument("--learner-paused", action="store_true", help="启动时保持手动暂停；恢复后仍需 Actor 正在采集")
    evaluate = sub.add_parser("eval", help="独立 checkpoint 评估，不向 Learner 回流")
    evaluate.add_argument("--checkpoint", "--checkpoint_path", required=True)
    evaluate.add_argument("--episodes", "--num_episodes", type=int, default=10)
    sub.add_parser("collect", help="Xbox 人工示教，独立保存全程视频和有效轨迹")
    check = sub.add_parser("check", help="文件、容量和依赖检查；不调用机器人控制接口")
    check.add_argument("--robot-tcp", action="store_true", help="额外只检查机器人服务 TCP")
    sub.add_parser("status", help="查看现有进程和阶段")
    command = sub.add_parser("command", help="标注、复位确认、暂停或停止 Actor")
    command.add_argument("name", choices=["label", "continue", "pause", "stop"])
    command.add_argument("value", type=int, nargs="?")
    sub.add_parser("stop-learner", help="Actor 退出后，结束 Learner 并保存 checkpoint")
    activity = sub.add_parser("learner-activity", help="暂停或继续 Learner 参数更新，保留进程和训练状态")
    activity.add_argument("operation", choices=["pause", "resume"])
    gripper = sub.add_parser("gripper", help="空闲时开合夹爪，或读取开口")
    gripper.add_argument("operation", choices=["open", "close", "status"])
    controller = sub.add_parser("controller", help="检查或显式恢复 FR3 底层控制服务")
    controller.add_argument("operation", choices=["status", "recover"])
    sub.add_parser("replay", help="列出录制；浏览器回放位于 console")
    export = sub.add_parser("export", help="导出已完成、已标注的有效 episode")
    export.add_argument("recording_id", help="run/capture 标识，来自 replay 列表")
    export.add_argument("--outcome", choices=["all", "success", "failure"], default="all")
    export.add_argument("--source", choices=["all", "human", "policy"], default="all")
    export.add_argument("--format", choices=["npz", "serl"], default="npz")
    export.add_argument("--episode", action="append")
    args = parser.parse_args(argv)
    config = load_config(args.config, data_dir=args.data_dir)
    # Console exports use NumPy from the already configured project environment.
    # Merely opening the console never imports JAX or initializes a device.
    configured_python = config.path(config.python)
    if args.command in {"console", "export"} and configured_python.is_file() and configured_python != Path(sys.executable).resolve():
        command = [str(configured_python), str(config.root / "bin/hil-serl"), *(sys.argv[1:] if argv is None else argv)]
        os.execv(str(configured_python), command)
    manager = Manager(config)
    try:
        if args.command == "console":
            result = console(config, args)
        elif args.command == "train":
            result = manager.launch("train", learner_only=args.learner_only,
                                    resume_checkpoint=args.resume_checkpoint, learner_paused=args.learner_paused)
        elif args.command == "eval":
            result = manager.launch("eval", checkpoint=args.checkpoint, eval_episodes=args.episodes)
        elif args.command == "collect":
            result = manager.launch("collect")
        elif args.command == "check":
            result = manager.preflight(check_robot=args.robot_tcp)
        elif args.command == "status":
            result = manager.status()
        elif args.command == "command":
            result = manager.command(args.name, args.value)
        elif args.command == "stop-learner":
            result = manager.stop_learner()
        elif args.command == "learner-activity":
            runtime = manager.status().get("learner_runtime") or {}
            if (type(runtime.get("pid")) is not int or runtime["pid"] <= 0
                    or not isinstance(runtime.get("start_time"), str) or not runtime["start_time"]
                    or not isinstance(runtime.get("attempt_id"), str) or not runtime["attempt_id"]):
                raise RuntimeError("没有身份已确认的 Learner；请先检查 status。")
            expected = {key: runtime[key] for key in ("pid", "start_time", "attempt_id")}
            result = manager.learner_activity(args.operation, expected=expected)
        elif args.command == "gripper":
            result = manager.gripper(args.operation)
        elif args.command == "controller":
            result = manager.controller_status(refresh=True) if args.operation == "status" else manager.recover_controller()
        else:
            from hilserl.archive import Archive
            archive = Archive(config.output_root, min_free_bytes=int(config.min_free_gib * 2**30))
            result = archive.list() if args.command == "replay" else archive.export(
                args.recording_id, outcome=args.outcome, source=args.source,
                format=args.format, episode_ids=args.episode)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.command == "check" and not result["ok"] else 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
