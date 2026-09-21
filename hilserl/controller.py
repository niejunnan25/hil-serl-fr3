"""Read controller health and invoke the installed lifecycle helper once.

The caller owns the device/launch lease and the operator authorization. This
module never sends pose, reset, recovery-topic, or gripper commands. An SSH
timeout does not cancel a possibly delivered remote recovery, so it must never
cause an automatic retry.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request


_SSH_HOST = "fr3-laptop"
_SCRIPT = "/home/robot/serl_projects/hilserl-fr3-control/controller_lifecycle.py"
_HTTP_TIMEOUT_SECONDS = 2.0
_SSH_TIMEOUT_SECONDS = 90
_MAX_STATE_AGE_SECONDS = 2.0
_MAX_RESPONSE_BYTES = 65536
_STREAMS = ("state", "jacobian", "gripper_state")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _health_error(health):
    """Validate evidence, rather than inferring readiness from HTTP liveness."""
    required = ["controller_running", "pose_subscribers"]
    required += [prefix + suffix for prefix in _STREAMS
                 for suffix in ("_age_seconds", "_stale")]
    missing = [key for key in required if key not in health]
    if missing:
        return "legacy", "底层服务缺少健康字段，请加载新版服务：" + ", ".join(missing)
    if "success" in health and health["success"] is not True:
        return "supported", str(health.get("error") or "底层服务报告不可用")
    if health["controller_running"] is not True:
        return "supported", "底层控制器未运行或运行状态无效"
    subscribers = health["pose_subscribers"]
    if type(subscribers) is not int or subscribers <= 0:
        return "supported", "底层控制器没有有效的 pose subscriber"
    for prefix in _STREAMS:
        stale_key, age_key = prefix + "_stale", prefix + "_age_seconds"
        if health[stale_key] is not False:
            label = {"state": "机械臂", "jacobian": "控制器 Jacobian", "gripper_state": "夹爪"}[prefix]
            age = health.get(age_key)
            detail = f"（{age:.1f} 秒未更新）" if type(age) in (int, float) and math.isfinite(age) else ""
            return "supported", f"{label}反馈已过期或未确认{detail}"
        age = health[age_key]
        if (type(age) not in (int, float) or not math.isfinite(age)
                or not 0 <= age <= _MAX_STATE_AGE_SECONDS):
            return "supported", f"底层服务的 {age_key} 无效或超过 2 秒"
    for flag in ("ready", "motion_available"):
        if flag in health and health[flag] is not True:
            mode = health.get("robot_mode_name")
            return "supported", "机器人当前模式暂不可控制" + (f"（{mode}）" if mode else "")
    return "supported", None


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _receipt(stdout):
    """The installed helper emits its JSON receipt on its last nonempty line."""
    value = _text(stdout)
    if len(value.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        return None, "SSH 回执超过允许大小"
    try:
        lines = value.rstrip().splitlines()
        result = json.loads(lines[-1] if lines else "")
        if not isinstance(result, dict):
            raise ValueError("回执必须是 JSON object")
    except (ValueError, TypeError) as exc:
        return None, f"SSH 未返回有效的 JSON 回执：{exc}"
    return result, None


class ControllerService:
    """Control only the fixed, installed FR3 lifecycle helper.

    ``status()`` performs a single GET /health with a two-second timeout.
    ``recover()`` requires the Manager's device and launch leases; it invokes
    SSH once, then reads health. Success requires the helper's ready receipt,
    valid receipt health, and a fresh HTTP health response. No method accepts a
    command, URL path, or arbitrary remote execution target from a UI request.
    """

    def __init__(self, server_url, ssh_host=_SSH_HOST, script=_SCRIPT):
        address = urllib.parse.urlsplit(server_url)
        if (address.scheme not in {"http", "https"} or not address.hostname
                or address.username or address.password or address.query or address.fragment):
            raise ValueError("控制服务地址必须是不带凭据、查询或片段的 HTTP(S) URL")
        try:
            address.port
        except ValueError as exc:
            raise ValueError("控制服务地址包含无效端口") from exc
        if ssh_host != _SSH_HOST or script != _SCRIPT:
            raise ValueError("控制服务恢复仅支持已安装的 fr3-laptop lifecycle helper")
        self.server_url = server_url.rstrip("/") + "/"
        self.ssh_host = ssh_host
        self.script = script
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())
        self._recovery_lock = threading.Lock()

    def status(self):
        """Return health evidence only; no robot or gripper state/motion call."""
        result = {
            "phase": "failed", "reason": "", "health": None,
            "availability": "unavailable", "http_status": None,
            # This is observation time, not a sensor timestamp.
            "observed_at": datetime.now(timezone.utc).isoformat(),
        }
        request = urllib.request.Request(
            self.server_url + "health", method="GET",
            headers={"Accept": "application/json"})
        try:
            try:
                response = self._opener.open(request, timeout=_HTTP_TIMEOUT_SECONDS)
            except urllib.error.HTTPError as exc:
                response = exc
            with response:
                result["http_status"] = response.status
                body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(body) > _MAX_RESPONSE_BYTES:
                raise ValueError("/health 返回内容超过允许大小")
            health = json.loads(body.decode("utf-8"))
            if not isinstance(health, dict):
                raise ValueError("/health 必须返回 JSON object")
            result["health"] = health
            result["availability"], reason = _health_error(health)
            if result["http_status"] != 200:
                result["reason"] = f"/health 返回 HTTP {result['http_status']}"
                if reason:
                    result["reason"] += "；" + reason
            elif reason:
                result["reason"] = reason
            else:
                result["phase"] = "ready"
                result["reason"] = "底层控制器运行中，状态流新鲜"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeError) as exc:
            if result["http_status"] in (404, 405):
                result["availability"] = "legacy"
            result["reason"] = f"无法验证底层控制服务：{exc}"
        return result

    def recover(self):
        """Invoke the fixed helper exactly once, retaining uncertain delivery."""
        if not self._recovery_lock.acquire(blocking=False):
            return {"phase": "failed", "reason": "底层控制服务正在恢复，请等待当前操作结束",
                    "health": None, "receipt": None,
                    "transport": {"delivery": "not_sent", "returncode": None,
                                  "timed_out": False, "stderr": ""}}
        try:
            return self._recover()
        finally:
            self._recovery_lock.release()

    def _recover(self):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                   self.ssh_host, "python3", self.script, "recover"]
        transport = {"delivery": "unknown", "returncode": None,
                     "timed_out": False, "stderr": ""}
        receipt = None
        error = None
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True,
                stdin=subprocess.DEVNULL, timeout=_SSH_TIMEOUT_SECONDS)
            transport["returncode"] = completed.returncode
            transport["stderr"] = _text(completed.stderr)[:_MAX_RESPONSE_BYTES]
            receipt, error = _receipt(completed.stdout)
            if completed.returncode != 0:
                error = f"SSH 恢复命令退出码为 {completed.returncode}"
            if receipt is not None:
                transport["delivery"] = "completed"
        except subprocess.TimeoutExpired as exc:
            transport["timed_out"] = True
            transport["stderr"] = _text(exc.stderr)[:_MAX_RESPONSE_BYTES]
            receipt, _ = _receipt(exc.stdout)
            error = "SSH 恢复请求超时，远端可能仍在执行；送达与完成结果未知，请读取状态核实"
        except subprocess.CalledProcessError as exc:
            transport["returncode"] = exc.returncode
            transport["stderr"] = _text(exc.stderr)[:_MAX_RESPONSE_BYTES]
            receipt, _ = _receipt(exc.stdout)
            if receipt is not None:
                transport["delivery"] = "completed"
            error = f"SSH 恢复命令失败，退出码 {exc.returncode}"
        except OSError as exc:
            transport["delivery"] = "not_sent"
            error = f"无法启动 SSH 恢复命令：{exc}"

        # A read is safe after an uncertain command; repeating SSH is not.
        observed = self.status()
        result = {**observed, "phase": "failed", "receipt": receipt,
                  "transport": transport}
        if error:
            result["reason"] = error
            if receipt is not None and receipt.get("error"):
                result["reason"] += "；" + str(receipt["error"])
            return result
        if receipt.get("phase") != "ready" or receipt.get("error"):
            result["reason"] = str(receipt.get("error") or "恢复脚本尚未确认底层服务 ready")
            return result
        receipt_health = receipt.get("health")
        if not isinstance(receipt_health, dict):
            result["reason"] = "恢复回执缺少健康状态，不能确认恢复成功"
            return result
        _, error = _health_error(receipt_health)
        if error:
            result["reason"] = "恢复回执的健康状态无效：" + error
            return result
        if observed["phase"] != "ready":
            result["reason"] = "恢复后实时健康检查未通过：" + observed["reason"]
            return result
        result["phase"] = "ready"
        result["reason"] = "底层控制服务已恢复，实时状态验证通过"
        return result
