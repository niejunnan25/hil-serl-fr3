#!/usr/bin/env python3
"""record_gello_demos.py

基于 gello_fr3_desktop_follow.py 的安全机制，添加数据记录功能。
将 GELLO 示教轨迹保存为 .npz 格式，用于后续训练。

安全机制:
  - max_step: 每步最大关节增量 (rad)
  - max_total_delta: 累积最大总偏差 (rad)
  - max_tracking_error: 跟踪误差上限 (rad)

记录字段:
  - joint_poses: (N, 7) 关节位置
  - gripper_states: (N,) 夹爪状态
  - timestamps: (N,) 单调时间戳
  - raw_gello: (N, 8) 原始 GELLO 传感器数据
  - target: (N, 7) 目标关节位置
  - command: (N, 7) 发送给机器人的命令
  - tracking_error: (N, 7) 跟踪误差

用法:
  python record_gello_demos.py --dry-run          # 仅连接 GELLO，不发送命令
  python record_gello_demos.py --live              # 连接机器人并记录
  python record_gello_demos.py --live --output-dir /path/to/save
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# 安全参数默认值 (来自 gello_fr3_desktop_follow.py)
# ---------------------------------------------------------------------------
DEFAULT_MAX_STEP = 0.003
DEFAULT_MAX_TOTAL_DELTA = 0.03
DEFAULT_MAX_TRACKING_ERROR = 0.08
DEFAULT_HZ = 20.0
DEFAULT_DURATION = 30.0
DEFAULT_LEADER_SCALE = 0.50
DEFAULT_OUTPUT_DIR = "/tmp/gello_demos"

# GELLO -> FR3 关节符号映射 (来自 gello_fr3_desktop_follow.py)
DEFAULT_JOINT_SIGNS = [1, -1, 1, 1, 1, -1, 1]

# FR3 关节限位
FR3_LOWER_LIMITS = np.array([-2.8, -1.66, -2.8, -2.97, -2.8, 0.08, -2.8])
FR3_UPPER_LIMITS = np.array([2.8, 1.66, 2.8, -0.17, 2.8, 3.65, 2.8])
FR3_DEFAULT_JOINTS = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])


# ---------------------------------------------------------------------------
# GELLO 设备 (使用 DynamixelDriver)
# ---------------------------------------------------------------------------
class GelloDevice:
    """GELLO 设备接口，使用 DynamixelDriver 读取关节数据。"""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 57600):
        self.port = port
        self.baudrate = baudrate
        self._driver = None

    def connect(self):
        from gello.dynamixel.driver import DynamixelDriver

        self._driver = DynamixelDriver(
            list(range(8)),
            port=self.port,
            baudrate=self.baudrate,
            max_retries=1,
            use_fake_fallback=False,
        )
        # 验证读取
        raw = np.asarray(self._driver.get_joints(), dtype=float)
        assert raw.shape == (8,), f"Expected (8,) but got {raw.shape}"
        print(f"[GELLO] Connected. Initial read: {raw.round(3)}")

    def read(self) -> np.ndarray:
        """读取 GELLO 原始数据 (8D: 7 joints + 1 gripper)。"""
        return np.asarray(self._driver.get_joints(), dtype=float)

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("[GELLO] Disconnected.")


# ---------------------------------------------------------------------------
# FR3 机器人 (使用 Polymetis)
# ---------------------------------------------------------------------------
class FR3Robot:
    """FR3 机器人接口，使用 Polymetis RobotInterface。"""

    def __init__(self, robot_ip: str = "localhost"):
        self.robot_ip = robot_ip
        self._robot = None
        self._gripper = None

    def connect(self):
        import torch
        from polymetis import GripperInterface, RobotInterface

        self._robot = RobotInterface(ip_address=self.robot_ip)
        self._torch = torch
        try:
            self._gripper = GripperInterface(ip_address=self.robot_ip)
        except Exception as e:
            print(f"[FR3] Gripper unavailable: {e}")
        q = np.asarray(self._robot.get_joint_positions(), dtype=float)
        print(f"[FR3] Connected. Current joints: {q.round(3)}")

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self._robot.get_joint_positions(), dtype=float)

    def send_joint_command(self, positions: np.ndarray):
        self._robot.update_desired_joint_positions(
            self._torch.tensor(positions, dtype=self._torch.float32)
        )

    def close(self):
        self._robot = None
        self._gripper = None
        print("[FR3] Disconnected.")


# ---------------------------------------------------------------------------
# 安全检查
# ---------------------------------------------------------------------------
def check_max_step(prev, curr, max_step):
    delta = float(np.abs(curr - prev).max())
    return delta <= max_step, delta


def check_max_total_delta(initial, current, max_total_delta):
    total = float(np.abs(current - initial).max())
    return total <= max_total_delta, total


def check_tracking_error(target, actual, max_tracking_error):
    err = target - actual
    max_err = float(np.abs(err).max())
    return max_err <= max_tracking_error, max_err, err


# ---------------------------------------------------------------------------
# 数据保存
# ---------------------------------------------------------------------------
def save_demo(output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"demo_{ts}.npz")
    np.savez(filepath, **kwargs)
    return filepath


# ---------------------------------------------------------------------------
# 主录制循环
# ---------------------------------------------------------------------------
def run_recording(args):
    is_live = args.live
    hz = args.hz
    duration = args.duration
    leader_scale = args.leader_scale
    joint_signs = np.array(args.joint_signs, dtype=float)
    dt = 1.0 / hz
    total_steps = int(duration * hz)

    # 初始化 GELLO
    gello = GelloDevice(port=args.gello_port, baudrate=57600)
    gello.connect()

    # 初始化机器人 (live 模式)
    robot = None
    if is_live:
        robot = FR3Robot(robot_ip=args.robot_ip)
        robot.connect()

    # 数据收集
    joint_poses_list = []
    gripper_states_list = []
    timestamps_list = []
    raw_gello_list = []
    targets_list = []
    commands_list = []
    tracking_errors_list = []

    # 初始状态 — 用首次 GELLO 读数作为基准
    raw0 = gello.read()
    raw_gello0 = raw0[:7]
    # q0: live 时用机器人实际位置, dry-run 时用 GELLO 映射后的位置
    if is_live:
        q0 = np.asarray(robot.get_joint_positions(), dtype=float)
    else:
        q0 = raw_gello0 * joint_signs * leader_scale
        q0 = np.clip(q0, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)
    command = q0.copy()
    prev_target = q0.copy()

    mode_str = "LIVE" if is_live else "DRY-RUN"
    print(f"\n{'=' * 60}")
    print(f"  Recording mode: {mode_str}")
    print(f"  Hz: {hz}, Duration: {duration}s, Steps: {total_steps}")
    print(f"  Max step: {args.max_step}, Max total delta: {args.max_total_delta}")
    print(f"  Leader scale: {leader_scale}")
    print(f"  Joint signs: {joint_signs.tolist()}")
    print(f"{'=' * 60}\n")

    aborted = False
    abort_reason = "none"
    start_time = time.monotonic()

    try:
        for step in range(total_steps):
            loop_start = time.monotonic()
            timestamp = time.monotonic()

            # 读取 GELLO
            raw_all = gello.read()
            raw = raw_all[:7]
            raw_gripper = float(raw_all[-1])

            # 计算目标 (与 gello_fr3_desktop_follow.py 一致)
            raw_delta = raw - raw_gello0
            target = q0 + raw_delta * joint_signs * leader_scale
            target = np.clip(target, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)

            # 安全步进
            step_ok, step_delta = check_max_step(prev_target, target, args.max_step)
            if not step_ok:
                abort_reason = f"max_step:{step_delta:.4f}"
                print(f"[ABORT] Step {step}: {abort_reason}")
                aborted = True
                break

            total_ok, total_delta = check_max_total_delta(q0, target, args.max_total_delta)
            if not total_ok:
                abort_reason = f"max_total_delta:{total_delta:.4f}"
                print(f"[ABORT] Step {step}: {abort_reason}")
                aborted = True
                break

            # 平滑命令
            command = command + np.clip(target - command, -args.max_step, args.max_step)

            # 跟踪误差 (dry-run 时 actual = command)
            actual = robot.get_joint_positions() if is_live else command.copy()
            err_ok, max_err, err_vec = check_tracking_error(command, actual, args.max_tracking_error)
            if not err_ok:
                abort_reason = f"tracking_error:{max_err:.4f}"
                print(f"[ABORT] Step {step}: {abort_reason}")
                aborted = True
                break

            # 发送命令
            if is_live:
                robot.send_joint_command(command)

            # 记录
            joint_poses_list.append(actual.copy())
            gripper_states_list.append(raw_gripper)
            timestamps_list.append(timestamp)
            raw_gello_list.append(raw_all.copy())
            targets_list.append(target.copy())
            commands_list.append(command.copy())
            tracking_errors_list.append(err_vec.copy())

            prev_target = target.copy()

            if step % max(1, int(hz)) == 0:
                elapsed = time.monotonic() - start_time
                print(f"  Step {step:5d}/{total_steps} | {elapsed:.1f}s | step={step_delta:.4f} track={max_err:.4f}")

            elapsed_loop = time.monotonic() - loop_start
            sleep_time = dt - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        gello.close()
        if robot is not None:
            robot.close()

    # 保存
    n = len(joint_poses_list)
    if n == 0:
        print("[WARN] No data collected.")
        return None

    filepath = save_demo(
        output_dir=args.output_dir,
        joint_poses=np.array(joint_poses_list),
        gripper_states=np.array(gripper_states_list),
        timestamps=np.array(timestamps_list) - timestamps_list[0],
        raw_gello=np.array(raw_gello_list),
        target=np.array(targets_list),
        command=np.array(commands_list),
        tracking_error=np.array(tracking_errors_list),
        meta_hz=hz,
        meta_duration=timestamps_list[-1] - timestamps_list[0],
        meta_leader_scale=leader_scale,
        meta_joint_signs=np.array(joint_signs),
        meta_num_steps=n,
        meta_abort_reason=abort_reason,
    )

    file_size = os.path.getsize(filepath)
    print(f"\n  Saved: {filepath} ({file_size / 1024:.1f} KB)")
    print(f"  Steps: {n}, Abort: {abort_reason}")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record GELLO demos for FR3")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="GELLO only, no robot")
    mode.add_argument("--live", action="store_true", help="GELLO + FR3")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gello-port", default="/dev/ttyUSB0")
    parser.add_argument("--robot-ip", default="localhost")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    parser.add_argument("--joint-signs", default=",".join(str(s) for s in DEFAULT_JOINT_SIGNS))
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    parser.add_argument("--max-total-delta", type=float, default=DEFAULT_MAX_TOTAL_DELTA)
    parser.add_argument("--max-tracking-error", type=float, default=DEFAULT_MAX_TRACKING_ERROR)
    args = parser.parse_args()
    args.joint_signs = [int(s) for s in args.joint_signs.split(",")]
    filepath = run_recording(args)
    if filepath:
        print(f"\nDone: {filepath}")


if __name__ == "__main__":
    main()
