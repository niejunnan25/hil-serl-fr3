"""Loopback console. GET requests never initialize the robot or training stack."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, urlparse

from hilserl.archive import Archive
from hilserl.files import read_json
from hilserl.processes import Manager, discover


def make_server(config, *, port=None, offline=False, manager=None, monitor_controller=False):
    manager = manager or Manager(config)
    archive = Archive(config.output_root, min_free_bytes=int(config.min_free_gib * 2**30))
    token = secrets.token_urlsafe(32)
    launch_lock = threading.Lock()
    web = Path(__file__).with_name("web")

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def send_file(self, path, *, download=False):
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not match:
                    self.json_response({"error": "Unsupported byte range"}, 416)
                    return
                start = int(match[1])
                end = min(int(match[2]) if match[2] else end, end)
                if start > end:
                    self.json_response({"error": "Byte range unavailable"}, 416)
                    return
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(max(0, end - start + 1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = file.read(min(remaining, 256 * 1024))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def current_recording(self):
            actors = [p for p in discover(config.root) if p["role"] != "learner"]
            if len(actors) != 1:
                return None
            value = actors[0].get("env", {}).get("HILSERL_RECORDING_DIR")
            if value:
                path = Path(value).resolve()
                if path.is_relative_to(config.output_root) and path.parent.name == "recordings":
                    return f"{path.parents[1].name}/{path.name}"
            return None

        def do_GET(self):
            try:
                if not self.valid_host():
                    self.json_response({"error": "Invalid local host"}, 403)
                    return
                parsed = urlparse(self.path)
                query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                route = parsed.path
                if route == "/api/status":
                    value = manager.status()
                    if monitor_controller and not offline:
                        value["controller"] = manager.controller_status()
                    value.update(offline=offline, token=token, recording_id=self.current_recording())
                    self.json_response(value)
                elif route == "/api/recordings":
                    self.json_response(archive.list())
                elif route == "/api/recording":
                    value = archive.detail(query["id"])
                    origin = value["manifest"]["started"]["monotonic_ns"]
                    for episode in value["episodes"]:
                        episode["start_seconds"] = (episode["started"]["monotonic_ns"] - origin) / 1e9
                        ended = episode.get("collection_ended", episode.get("finalized", episode["started"]))
                        episode["end_seconds"] = (ended["monotonic_ns"] - origin) / 1e9
                    self.json_response(value)
                elif route == "/api/timeline":
                    frames = archive.timeline(query["id"], query["camera"])
                    origin = read_json(archive.recording(query["id"]) / "manifest.json")["started"]["monotonic_ns"]
                    for frame in frames:
                        frame["time_seconds"] = (frame.pop("monotonic_ns") - origin) / 1e9
                    self.json_response(frames)
                elif route == "/api/video":
                    self.send_file(archive.video_file(query["id"], query["camera"], query["file"]))
                elif route == "/api/preview":
                    camera = query["camera"]
                    if camera not in {"wrist_1", "side_policy"}:
                        raise ValueError("Unknown camera")
                    recording_id = self.current_recording()
                    if not recording_id:
                        raise FileNotFoundError("No active preview")
                    self.send_file(archive.recording(recording_id) / "preview" / f"{camera}.jpg")
                elif route == "/api/logs":
                    output = []
                    runs = {p["run_dir"] for p in discover(config.root) if p.get("run_dir")}
                    latest = manager.status()["launch"].get("run_dir")
                    if latest:
                        runs.add(latest)
                    for run in sorted(runs):
                        log_root = (Path(run) / "logs").resolve()
                        if not log_root.is_dir():
                            continue
                        # A run can contain several Actor attempts.  Reading
                        # only run/actor.json hid the first failed handshake
                        # after a later retry overwrote that pointer.
                        logs = sorted(
                            (path for path in log_root.glob("*.log") if path.is_file()),
                            key=lambda path: path.stat().st_mtime,
                            reverse=True,
                        )[:12]
                        for log in logs:
                            name = log.name
                            role = "actor" if name.startswith("actor_") else "learner" if name.startswith("learner_") else "process"
                            attempt = name.rsplit("_", 1)[-1].removesuffix(".log")
                            with log.open("rb") as file:
                                file.seek(max(0, log.stat().st_size - 16000))
                                output.append(dict(role=f"{role} · attempt {attempt}", text=file.read().decode(errors="replace")))
                    self.json_response(output)
                elif route == "/api/download":
                    name = query["file"]
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.zip", name):
                        raise ValueError("Invalid download")
                    directory = (config.output_root / "exports").resolve()
                    path = (directory / name).resolve()
                    if not path.is_relative_to(directory):
                        raise ValueError("Invalid download")
                    self.send_file(path, download=True)
                elif route in {"/", "/index.html", "/style.css", "/app.js"}:
                    self.send_file(web / ("index.html" if route == "/" else route[1:]))
                else:
                    self.json_response({"error": "Not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except FileNotFoundError as exc:
                self.json_response({"error": str(exc)}, 404)
            except (ValueError, KeyError) as exc:
                self.json_response({"error": str(exc)}, 400)

        def do_POST(self):
            try:
                if not self.valid_host() or self.headers.get("X-HILSERL-Token") != token:
                    self.json_response({"error": "请从本机控制台发起操作。"}, 403)
                    return
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.json_response({"error": "Origin mismatch"}, 403)
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if self.path == "/api/export":
                    result = archive.export(data.pop("id"), **data)
                else:
                    if offline:
                        raise ValueError("离线预览不会启动或控制设备。")
                    if self.path == "/api/start":
                        manager.validate_launch_request(**data)
                        if not launch_lock.acquire(blocking=False):
                            raise ValueError("已有启动请求正在处理中。")
                        def launch():
                            try:
                                manager.launch(**data)
                            except Exception:
                                pass  # Manager publishes the precise fault in status.
                            finally:
                                launch_lock.release()
                        threading.Thread(target=launch, name="launch", daemon=True).start()
                        result = {"phase": "starting"}
                    elif self.path == "/api/command":
                        result = manager.command(data["command"], data.get("value"), expected=data.get("expected"))
                    elif self.path == "/api/stop-learner":
                        result = manager.stop_learner()
                    elif self.path == "/api/learner-activity":
                        if set(data) != {"command", "expected"}:
                            raise ValueError("Learner 暂停请求需要 command 和 expected 身份。")
                        result = manager.learner_activity(data["command"], expected=data["expected"])
                    elif self.path == "/api/gripper":
                        if set(data) - {"operation", "expected"}:
                            raise ValueError("Unknown gripper request fields")
                        result = manager.gripper(data["operation"], expected=data.get("expected"))
                    elif self.path == "/api/controller/status":
                        if data:
                            raise ValueError("底层状态检查不接受自定义参数")
                        result = manager.controller_status(refresh=True)
                    elif self.path == "/api/controller/recover":
                        if data:
                            raise ValueError("底层恢复不接受自定义命令或目标")
                        manager.validate_controller_recovery()
                        if not launch_lock.acquire(blocking=False):
                            raise ValueError("已有启动或恢复操作正在进行")
                        def recover_controller():
                            try:
                                manager.recover_controller()
                            except Exception as exc:
                                if manager._controller_info().get("phase") != "recovering":
                                    manager._save_controller({"phase": "failed", "error": str(exc)})
                            finally:
                                launch_lock.release()
                        threading.Thread(target=recover_controller, name="controller-recover", daemon=True).start()
                        result = {"phase": "starting", "message": "已提交底层服务恢复请求"}
                    else:
                        raise ValueError("Unknown operation")
                self.json_response(result)
            except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
                self.json_response({"error": str(exc)}, 400)

        def log_message(self, fmt, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", config.console_port if port is None else port), Handler)
    server.daemon_threads = True
    return server


def serve(config, *, port=None, offline=False):
    with make_server(config, port=port, offline=offline, monitor_controller=True) as server:
        print(f"HIL-SERL console: http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()
