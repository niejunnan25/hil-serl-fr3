"""Operator commands shared by the terminal and local console.

The Actor owns status. Commands target an attempt and episode; an old label or
reset acknowledgement cannot operate on the next episode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import sys
import time
import uuid

from hilserl.files import atomic_json, stamp


class StopRequested(Exception):
    pass


GRIPPER_PHASES = frozenset({"waiting_reset", "paused"})


class OperatorControl:
    def __init__(self, directory=None, *, terminal=True):
        self.directory = Path(directory) if directory else None
        self.terminal = terminal and bool(sys.stdin and sys.stdin.isatty())
        self.pending = []
        self.gripper_handler = None
        from hilserl.processes import process_start
        self.state = dict(attempt_id=os.environ.get("HILSERL_ATTEMPT_ID") or uuid.uuid4().hex,
                          pid=os.getpid(), start_time=process_start(os.getpid()), phase="starting",
                          episode_id=None, step=0, revision=0, **stamp())
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "commands").mkdir(exist_ok=True)
        self.publish("starting")

    def publish(self, phase=None, **fields):
        if phase:
            if phase != self.state["phase"]:
                self.state["gate_id"] = None
            self.state["phase"] = phase
        self.state.update(fields, **stamp())
        self.state["revision"] += 1
        if self.directory:
            atomic_json(self.directory / "status.json", self.state)

    def _receive(self):
        if self.directory:
            for path in sorted((self.directory / "commands").glob("*.json")):
                try:
                    value = json.loads(path.read_text())
                    if value.get("attempt_id") == self.state["attempt_id"]:
                        self.pending.append(value)
                finally:
                    path.unlink(missing_ok=True)
        if self.terminal and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":
                raise StopRequested("Terminal input closed")
            line = line.strip().lower()
            command = "label" if line in ("0", "1") else "continue" if not line else line
            self.pending.append(dict(command=command, value=int(line) if line in ("0", "1") else line,
                                     episode_id=self.state["episode_id"], phase=self.state["phase"],
                                     gate_id=self.state.get("gate_id")))

    def raise_if_stop(self):
        self._receive()
        if any(c.get("command") in ("stop", "abort", "quit", "exit") for c in self.pending):
            raise StopRequested("Operator stopped the Actor")

    def poll(self, allowed):
        self.raise_if_stop()
        accepted = None
        keep = []
        for command in self.pending:
            name = command.get("command")
            if name == "gripper":
                self._gripper(command, accept=accepted is None)
                continue
            if command.get("episode_id") != self.state["episode_id"]:
                continue
            if name == "continue" and command.get("phase") != self.state["phase"]:
                continue
            if name == "continue" and (not self.state.get("gate_id") or command.get("gate_id") != self.state["gate_id"]):
                continue
            if name in allowed and accepted is None:
                accepted = command
            elif name in ("stop", "pause"):
                keep.append(command)
        self.pending = keep
        return accepted

    def _gripper(self, command, *, accept):
        """Execute only on the Actor thread at the same idle gate as the click."""
        result = dict(operation=command.get("value"), command_id=command.get("command_id"),
                      command_acknowledged=False, verification="rejected", **stamp())
        current = all(command.get(k) == self.state.get(k) for k in
                      ("attempt_id", "episode_id", "phase", "gate_id"))
        age_ns = time.time_ns() - command.get("unix_ns", 0)
        if (not accept or not current or self.state["phase"] not in GRIPPER_PHASES
                or not self.state.get("gate_id") or not 0 <= age_ns <= 10_000_000_000):
            result["message"] = "任务阶段已经变化或夹爪请求已过期；未发送命令，请重新操作。"
        elif command.get("value") not in {"open", "close"} or self.gripper_handler is None:
            result["message"] = "此 Actor 的夹爪控制未就绪；未发送命令。"
        else:
            try:
                self.publish(gripper=dict(result, verification="sending", message="正在发送夹爪命令"))
                result.update(self.gripper_handler(command["value"]), **stamp())
            except Exception as exc:
                result.update(getattr(exc, "result", {}), verification="error", message=str(exc), **stamp())
        self.publish(gripper=result)

    def wait(self, phase, prompt, allowed, *, check=None):
        self.publish(phase, prompt=prompt, gate_id=uuid.uuid4().hex)
        print(prompt, flush=True)
        while True:
            if check:
                check()
            command = self.poll(allowed)
            if command:
                return command
            time.sleep(0.1)

    def input(self, prompt, *, check=None):
        previous = self.state["phase"]
        self.wait("operator_prompt", prompt, {"continue"}, check=check)
        self.publish(previous, prompt="")
        return ""


def send_command(directory, command, value=None, expected=None):
    directory = Path(directory)
    state = json.loads((directory / "status.json").read_text())
    if expected is not None and any(expected.get(key) != state.get(key) for key in
                                    ("attempt_id", "episode_id", "phase", "gate_id")):
        raise ValueError("任务阶段已经变化，请按当前页面重新操作。")
    allowed = {
        "label": {"collecting", "awaiting_label"},
        "continue": {"waiting_reset", "paused", "operator_prompt"},
        "pause": {"collecting"},
        "gripper": GRIPPER_PHASES,
        "stop": {"starting", "waiting_reset", "waiting_controller", "resetting", "collecting", "awaiting_label", "paused", "operator_prompt",
                 "reward_pending", "waiting_reward", "draining_reward"},
    }
    if command not in allowed or state["phase"] not in allowed[command]:
        raise ValueError(f"{command} is unavailable during {state['phase']}")
    if command == "label" and (type(value) is not int or value not in (0, 1)):
        raise ValueError("Label must be integer 0 or 1")
    if command == "gripper" and value not in {"open", "close"}:
        raise ValueError("夹爪操作仅支持 open 或 close")
    if command in {"continue", "gripper"} and not state.get("gate_id"):
        raise ValueError("复位确认尚未就绪")
    item = dict(command=command, value=value, command_id=uuid.uuid4().hex, attempt_id=state["attempt_id"],
                episode_id=state["episode_id"], phase=state["phase"], gate_id=state.get("gate_id"), **stamp())
    filename = f"{time.time_ns():020d}-{uuid.uuid4().hex}.json"
    atomic_json(directory / "commands" / filename, item)
    return item
