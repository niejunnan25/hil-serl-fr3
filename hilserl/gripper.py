"""Explicit gripper I/O without constructing an environment or homing the robot.

The Franka bridge exposes POST /getstate and /open_gripper|/close_gripper.
Its gripper_pos is finger separation / 0.08 m. The command response only
acknowledges the HTTP handler; it does not include a ROS action result, and the
bridge's binary command latch can suppress a command. Callers must persist the
returned before/after evidence and must own the device and phase interlocks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


_FULL_WIDTH_M = 0.08
_MOTION_THRESHOLD_M = 0.001
_MAX_RESPONSE_BYTES = 65536
_MAX_STATE_AGE_SECONDS = 2.0


class GripperControlError(RuntimeError):
    """An I/O failure whose result preserves possible command delivery."""

    def __init__(self, message, *, result=None):
        super().__init__(message)
        self.result = result or {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A configured robot endpoint must not redirect a device command.
        return None


class GripperControl:
    """Read width or issue exactly one explicit open/close request.

    Construction performs no I/O. ``command()`` never retries a device command,
    including after a timeout, because the bridge may already have accepted it.
    The subsequent observations are bounded by ``observation_timeout``. Neither
    HTTP acknowledgement nor observed motion establishes grasp success.
    """

    def __init__(self, server_url, *, timeout=2.0, observation_timeout=2.0,
                 poll_interval=0.1):
        address = urllib.parse.urlsplit(server_url)
        if (address.scheme not in {"http", "https"} or not address.hostname
                or address.username or address.password or address.query or address.fragment):
            raise ValueError("夹爪服务地址必须是不带凭据、查询或片段的 HTTP(S) URL")
        try:
            address.port
        except ValueError as exc:
            raise ValueError("夹爪服务地址包含无效端口") from exc
        for name, value in (("timeout", timeout), ("observation_timeout", observation_timeout),
                            ("poll_interval", poll_interval)):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.server_url = server_url.rstrip("/") + "/"
        self.timeout = float(timeout)
        self.observation_timeout = float(observation_timeout)
        self.poll_interval = float(poll_interval)
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())
        self._lock = threading.RLock()

    def _post(self, endpoint, *, timeout=None):
        request = urllib.request.Request(
            self.server_url + endpoint, data=b"{}", method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json, text/plain"})
        try:
            with self._opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                body = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise GripperControlError(f"/{endpoint} 返回内容超过允许大小")
                return response.status, body.decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read(1024).decode("utf-8", errors="replace")
            exc.close()
            raise GripperControlError(
                f"/{endpoint} 返回 HTTP {exc.code}",
                result={"request_endpoint": endpoint,
                        "acknowledgement": {"status_code": exc.code, "body": body}}) from exc
        except (urllib.error.URLError, TimeoutError, OSError, UnicodeError) as exc:
            raise GripperControlError(f"/{endpoint} 请求失败：{exc}") from exc

    def _status(self, *, timeout=None):
        _, body = self._post("getstate", timeout=timeout)
        try:
            state = json.loads(body)
            if not isinstance(state, dict):
                raise ValueError("状态响应必须是 JSON object")
            if "success" in state and state["success"] is not True:
                raise ValueError(str(state.get("error") or "服务报告状态请求失败"))
            position = state["gripper_pos"]
            if (type(position) not in (int, float) or not math.isfinite(position)
                    or not 0 <= position <= 1.25):
                raise ValueError("gripper_pos 必须是 [0, 1.25] 内的有限数值")
        except (ValueError, KeyError, TypeError) as exc:
            raise GripperControlError(f"/getstate 夹爪读数无效：{exc}") from exc
        observed = {
            "gripper_pos": float(position),
            "width_m": float(position) * _FULL_WIDTH_M,
            "width_mm": float(position) * _FULL_WIDTH_M * 1000.0,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            # observed_at is the local response time, not a sensor timestamp.
            "sensor_timestamp_available": False,
            "freshness": "unknown",
        }
        for prefix in ("state", "gripper_state"):
            stale_key, age_key = prefix + "_stale", prefix + "_age_seconds"
            if stale_key in state:
                stale = state[stale_key]
                if type(stale) is not bool:
                    raise GripperControlError(f"/getstate 的 {stale_key} 无效", result={"status": observed})
                observed[stale_key] = stale
                if stale:
                    observed["freshness"] = "stale"
                    raise GripperControlError("机器人服务报告缓存状态已过期，禁止夹爪操作", result={"status": observed})
            if age_key in state:
                age = state[age_key]
                if type(age) not in (int, float) or not math.isfinite(age) or age < 0:
                    raise GripperControlError(f"/getstate 的 {age_key} 无效", result={"status": observed})
                observed[age_key] = float(age)
                if age > _MAX_STATE_AGE_SECONDS:
                    observed["freshness"] = "stale"
                    raise GripperControlError(
                        f"机器人服务的 {age_key}={age:.2f}，缓存状态已过期，禁止夹爪操作",
                        result={"status": observed})
        # Arm callbacks can continue while the independent gripper stream is
        # stopped; arm freshness alone cannot establish fresh finger positions.
        if ("gripper_state_age_seconds" in observed
                and observed.get("gripper_state_stale") is False):
            observed["freshness"] = "fresh"
        return observed

    def status(self):
        """Return the bridge's reported width, without issuing a motion command."""
        with self._lock:
            return self._status()

    def command(self, operation):
        """Issue open/close once and return acknowledgement and width evidence.

        ``GripperControlError.result`` retains the same result schema. A null
        command_acknowledged means delivery is uncertain; false means that no
        device request was attempted. An unchanged width remains ``not_verified``
        even when the bridge replies with the literal string "Opened"/"Closed".
        Fresh gripper callback age and a false stale flag are required before
        motion; old bridges without this evidence remain available for reads.
        """
        if operation not in ("open", "close"):
            raise ValueError("夹爪操作仅支持 open 或 close")
        with self._lock:
            return self._command(operation)

    def _command(self, operation):
        started = time.monotonic()
        result = {
            "operation": operation,
            "command_acknowledged": False,
            "motion_observed": False,
            "verification": "not_verified",
            "before": None,
            "after": None,
            "acknowledgement": None,
            "message": "",
            "duration_seconds": 0.0,
        }
        try:
            # Failure to obtain the initial width must not trigger a device command.
            result["before"] = self._status()
            if result["before"]["freshness"] != "fresh":
                raise GripperControlError("夹爪状态新鲜度未确认；请恢复控制服务后重试")
            result["command_acknowledged"] = None
            code, body = self._post(operation + "_gripper")
            result["acknowledgement"] = {"status_code": code, "body": body}
            expected = "Opened" if operation == "open" else "Closed"
            acknowledged = body.strip() == expected
            if not acknowledged:
                try:
                    reply = json.loads(body)
                except (ValueError, TypeError):
                    reply = None
                if isinstance(reply, dict):
                    acknowledged = (reply.get("success") is True
                                    and reply.get("command_acknowledged") is True
                                    and reply.get("operation") == operation)
                    if reply.get("success") is False:
                        raise GripperControlError(str(reply.get("error") or "夹爪服务报告命令失败"))
            if code != 200 or not acknowledged:
                raise GripperControlError("夹爪服务未返回约定的命令应答；下发结果未知")
            result["command_acknowledged"] = True

            deadline = time.monotonic() + self.observation_timeout
            direction = 1 if operation == "open" else -1
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                result["after"] = self._status(timeout=min(self.timeout, remaining))
                if result["after"]["freshness"] != "fresh":
                    raise GripperControlError("夹爪状态新鲜度未确认；命令已应答，但动作完成未确认")
                delta = result["after"]["width_m"] - result["before"]["width_m"]
                if direction * delta >= _MOTION_THRESHOLD_M:
                    result["motion_observed"] = True
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(self.poll_interval, remaining))
            if result["after"] is None:
                raise GripperControlError("命令已应答，但观察窗口内未取得夹爪读数")
            result["width_change_mm"] = result["after"]["width_mm"] - result["before"]["width_mm"]
            verb, motion = ("打开", "增大") if operation == "open" else ("闭合", "缩小")
            width = result["after"]["width_mm"]
            if result["motion_observed"]:
                result["verification"] = "motion_observed"
                result["message"] = f"{verb}请求已应答；观察到开口{motion}，当前 {width:.1f} mm。服务未提供夹持结果。"
            else:
                result["message"] = (
                    f"{verb}请求已应答，但未观察到开口{motion}，当前 {width:.1f} mm；"
                    "动作完成未确认，请核对实际夹爪状态。")
            result["duration_seconds"] = time.monotonic() - started
            return result
        except GripperControlError as exc:
            if exc.result.get("status") is not None:
                result["before" if result["before"] is None else "after"] = exc.result["status"]
            if (exc.result.get("acknowledgement") is not None
                    and exc.result.get("request_endpoint") == operation + "_gripper"):
                result["acknowledgement"] = exc.result["acknowledgement"]
            # Only a read may follow an uncertain command; never repeat the command.
            if result["command_acknowledged"] is None:
                try:
                    result["after"] = self._status()
                except GripperControlError as observation_error:
                    if observation_error.result.get("status") is not None:
                        result["after"] = observation_error.result["status"]
                    result["observation_error"] = str(observation_error)
            result["message"] = str(exc)
            result["duration_seconds"] = time.monotonic() - started
            raise GripperControlError(str(exc), result=result) from exc
