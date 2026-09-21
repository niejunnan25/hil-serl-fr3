#!/usr/bin/env python3
"""GelloDroidAdapter — 将 GELLO 输出适配为 DROID step() 输入 (方案C)"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_SCRIPT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, PARENT_SCRIPT_DIR)

from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS  # noqa: E402

JOINT_SIGNS = np.array([1, -1, 1, 1, 1, -1, 1], dtype=float)

@dataclass
class AdapterConfig:
    gello_port: str = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"
    leader_scale: float = 0.50
    max_step: float = 0.003
    max_total_delta: float = 0.03
    limit_margin: float = 0.02
    hz: float = 20.0
    gripper_open_rad: float = 3.558
    gripper_close_rad: float = 2.829

class GelloDroidAdapter:
    def __init__(self, config=None):
        self.config = config or AdapterConfig()
        self._driver = None
        self._q0 = None
        self._raw0 = None
        self._command = None
        self._step_count = 0
        self._connected = False

    def connect(self):
        from gello.dynamixel.driver import DynamixelDriver
        self._driver = DynamixelDriver(list(range(8)), port=self.config.gello_port, baudrate=57600, max_retries=1, use_fake_fallback=False)
        raw_all0 = np.asarray(self._driver.get_joints(), dtype=float)
        self._raw0 = raw_all0[:7]
        self._q0 = np.zeros(7)
        self._command = self._q0.copy()
        self._connected = True
        return True

    def act(self, obs=None):
        if not self._connected:
            return np.zeros(8), {"error": "not_connected"}
        raw_all = np.asarray(self._driver.get_joints(), dtype=float)
        raw = raw_all[:7]
        raw_gripper = float(raw_all[-1])
        raw_delta = raw - self._raw0
        target = self._q0 + raw_delta * JOINT_SIGNS * self.config.leader_scale
        target = np.clip(target, FR3_LOWER_LIMITS + self.config.limit_margin, FR3_UPPER_LIMITS - self.config.limit_margin)
        step_delta = np.clip(target - self._command, -self.config.max_step, self.config.max_step)
        self._command = self._command + step_delta
        total_delta = float(np.abs(self._command - self._q0).max())
        safe = total_delta <= self.config.max_total_delta
        denom = self.config.gripper_close_rad - self.config.gripper_open_rad
        gripper_norm = 0.0
        if abs(denom) > 1e-6:
            gripper_norm = float(np.clip((raw_gripper - self.config.gripper_open_rad) / denom, 0.0, 1.0))
        action_8d = np.concatenate([self._command, [gripper_norm]])
        self._step_count += 1
        return action_8d, {"step": self._step_count, "total_delta": total_delta, "gripper": gripper_norm, "safe": safe}

    def reset(self, initial_joints=None):
        if self._driver is not None:
            self._raw0 = np.asarray(self._driver.get_joints(), dtype=float)[:7]
        self._q0 = np.array(initial_joints, dtype=float) if initial_joints is not None else np.zeros(7)
        self._command = self._q0.copy()
        self._step_count = 0

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._connected = False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gello-port", default=AdapterConfig.gello_port)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    adapter = GelloDroidAdapter(AdapterConfig(gello_port=args.gello_port, hz=args.hz))
    if not args.dry_run:
        adapter.connect()
    for i in range(int(args.duration * args.hz)):
        action, info = adapter.act()
        if i % int(args.hz) == 0:
            print(f"step {i:04d} safe={info.get('safe', 'N/A')}")
        if not info.get("safe", True):
            print(f"ABORT at step {i}"); break
        time.sleep(1.0 / args.hz)
    adapter.close()
