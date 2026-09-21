"""Dependency-light admission check for one complete Learner update.

Replay contents prove that data once existed. They do not prove an Actor is
collecting now. The run-owned Actor identity, its live heartbeat and transport
receipts must all agree before another update may start.
Each verified reset starts a new collection window; receipts from the previous
window do not establish a stream that can either train or fail in the new one.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


def _read_object(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            text = stream.read(65537)
        value = json.loads(text) if len(text) <= 65536 else None
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, ValueError):
        return {}


def _process_info(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except (OSError, IndexError):
        return None, None


class TrainingGate:
    def __init__(self, run_dir, data_activity, *, clock=time.monotonic,
                 process_info=_process_info, freshness_seconds=2.0):
        self.run_dir = Path(run_dir).resolve() if run_dir is not None else None
        self.data_activity = data_activity
        self.clock = clock
        self.process_info = process_info
        self.freshness_seconds = freshness_seconds

    def check(self, *, manual_paused=False):
        result = dict(allowed=False, pause_code=None, pause_reason=None, save_checkpoint=False,
                      actor_phase=None, actor_attempt_id=None,
                      actor_heartbeat_age_seconds=None, online_received_count=0,
                      online_data_age_seconds=None)

        def paused(code, reason, *, save=True):
            return dict(result, pause_code=code, pause_reason=reason, save_checkpoint=save)

        if manual_paused:
            return paused("manual_pause", "已手动暂停训练；恢复后仍需 Actor 正在采集并收到新数据。")
        if self.run_dir is None:
            return paused("run_identity_missing", "缺少同轮 run 目录，无法核验 Actor 身份。")
        record = _read_object(self.run_dir / "actor.json")
        status = _read_object(self.run_dir / "control" / "status.json")
        result.update(actor_phase=status.get("phase"), actor_attempt_id=status.get("attempt_id"))
        identity = tuple(record.get(key) for key in ("pid", "start_time", "attempt_id"))
        pid, start, attempt = identity
        if (type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start
                or not isinstance(attempt, str) or not attempt
                or record.get("role") != "actor" or record.get("mode") != "train"
                or record.get("run_dir") != str(self.run_dir)
                or any(status.get(key) != record.get(key) for key in ("pid", "start_time", "attempt_id"))):
            return paused("actor_identity_missing", "没有身份匹配的同轮训练 Actor；训练状态已保留。")
        live_start, process_state = self.process_info(pid)
        if live_start != start or process_state in {None, "Z", "X", "x"}:
            return paused("actor_not_alive", "Actor 已退出；等待在本轮中重新连接。")
        if status.get("phase") != "collecting":
            # Normal episode boundaries can wait arbitrarily long and do not
            # need another large checkpoint. Faults remain durable pause points.
            save = (status.get("phase") in {"fault", "stopped"}
                    or bool(status.get("robot_state_error") or status.get("error")))
            return paused("actor_not_collecting", f"Actor 当前为 {status.get('phase', 'unknown')}，暂停参数更新。", save=save)
        heartbeat = status.get("monotonic_ns")
        now = self.clock()
        if type(heartbeat) is not int:
            return paused("actor_heartbeat_missing", "Actor 缺少有效心跳。")
        heartbeat_age = now - heartbeat / 1e9
        result["actor_heartbeat_age_seconds"] = heartbeat_age if math.isfinite(heartbeat_age) else None
        if not math.isfinite(heartbeat_age) or not 0 <= heartbeat_age <= self.freshness_seconds:
            return paused("actor_heartbeat_stale", "Actor 心跳过期；等待采集恢复。")
        collection_started = status.get("collection_started_monotonic_ns")
        if type(collection_started) is not int or not 0 < collection_started <= heartbeat:
            return paused("collection_window_invalid", "Actor 缺少有效的本次采集开始时间；暂停参数更新。")
        try:
            activity = self.data_activity("actor_env")
            intervention = self.data_activity("actor_env_intvn")
        except Exception as exc:
            return paused("data_activity_unavailable", f"无法核验在线数据接收状态：{type(exc).__name__}: {exc}")
        if not isinstance(activity, dict) or not isinstance(intervention, dict):
            return paused("data_activity_unavailable", "在线数据接收状态不可用。")
        for name, store in (("actor_env", activity), ("actor_env_intvn", intervention)):
            if store.get("error"):
                return paused("data_store_error", f"{name} 数据入库失败，已暂停训练：{store['error']}")
        if activity.get("client_id") != attempt:
            return paused("waiting_online_data", "等待当前 Actor 的第一批在线数据；旧 Actor 的接收记录不放行训练。", save=False)
        count = activity.get("received_count")
        received = activity.get("last_received_monotonic")
        if type(count) is not int or count < 0:
            return paused("data_activity_invalid", "在线数据接收计数无效。")
        result["online_received_count"] = count
        if count == 0 or type(received) not in (int, float):
            return paused("waiting_online_data", "等待当前 Actor 的第一批在线数据；已有 replay 保留。", save=False)
        # The server can receive a batch while we read its activity snapshot.
        data_age = self.clock() - received
        result["online_data_age_seconds"] = data_age if math.isfinite(data_age) else None
        if math.isfinite(received) and received < collection_started / 1e9:
            return paused("waiting_online_data", "等待本次复位后的第一批在线数据；已有 replay 保留。", save=False)
        if not math.isfinite(data_age) or not 0 <= data_age <= self.freshness_seconds:
            return paused("online_data_stale", "已超过 2 秒没有收到在线数据；暂停参数更新。")
        return dict(result, allowed=True)
