#!/usr/bin/env python3
"""record_gello_demos_serl.py

基于 record_gello_demos.py 迁移至 serl_franka_controllers HTTP API。
所有数据格式、安全机制、GELLO 集成保持不变。

主要变更:
  - FR3Robot 从 Polymetis RobotInterface 迁移到 requests.post HTTP API
  - 新增 --server-url 参数 (默认 http://127.0.0.2:5000/)
  - 新增 --dry-run 标志 (仅 GELLO 模式，不发送机器人命令)
  - 保留 --live 模式 (GELLO + FR3 via HTTP API)

安全机制:
  - max_step: 每步最大关节增量 (rad)
  - max_total_delta: 累积最大总偏差 (rad)
  - max_tracking_error: 跟踪误差上限 (rad)

记录字段 (与原版完全一致):
  - joint_poses: (N, 7) 关节位置
  - gripper_states: (N,) 夹爪状态
  - timestamps: (N,) 单调时间戳
  - raw_gello: (N, 8) 原始 GELLO 传感器数据
  - target: (N, 7) 目标关节位置
  - command: (N, 7) 发送给机器人的命令
  - tracking_error: (N, 7) 跟踪误差

用法:
  python record_gello_demos_serl.py --dry-run
  python record_gello_demos_serl.py --live --server-url http://127.0.0.2:5000/
  python record_gello_demos_serl.py --live --output-dir /path/to/save
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import requests
from scipy.spatial.transform import Rotation

# FK converter: joint angles -> Cartesian pose (used in live mode)
import sys as _sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in _sys.path:
    _sys.path.insert(0, _SCRIPT_DIR)
from fk_converter import forward_kinematics

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
DEFAULT_SERVER_URL = "http://127.0.0.2:5000/"
DEFAULT_FK_BIAS_LIMIT = 0.005
DEFAULT_MAX_CARTESIAN_STEP = 0.005
DEFAULT_MAX_CARTESIAN_ROT_STEP = 0.05

# GELLO -> FR3 关节符号映射 (来自 gello_fr3_desktop_follow.py)
DEFAULT_JOINT_SIGNS = [1, -1, 1, 1, 1, -1, 1]

# FR3 关节限位
from fr3_joint_limits import FR3_LOWER_LIMITS, FR3_UPPER_LIMITS  # noqa: E402
FR3_DEFAULT_JOINTS = np.array([0.0, 0.0, 0.0, -1.571, 0.0, 1.571, 0.0])


# ---------------------------------------------------------------------------
# GELLO 设备 (使用 DynamixelDriver) — 与原版完全一致
# ---------------------------------------------------------------------------
class GelloDevice:
    """GELLO 设备接口，使用 DynamixelDriver 读取关节数据。"""

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 57600):
        self.port = port
        self.baudrate = baudrate
        self._driver = None

    def connect(self):
        try:
            from gello.dynamixel.driver import DynamixelDriver
        except ImportError as e:
            raise ImportError(
                "Cannot import gello.dynamixel.driver – is gello installed?"
            ) from e

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
        if self._driver is None:
            raise RuntimeError("GelloDevice not connected. Call connect() first.")
        return np.asarray(self._driver.get_joints(), dtype=float)

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("[GELLO] Disconnected.")


# ---------------------------------------------------------------------------
# FR3 机器人 (HTTP API — serl_franka_controllers)
# ---------------------------------------------------------------------------
class FR3Robot:
    """FR3 机器人接口，使用 franka_server HTTP API (requests.post)。

    API 端点:
      POST /healthz   -> {"ok": true}
      POST /getstate   -> {"pose":[x,y,z,qx,qy,qz,qw], "q":[7], "dq":[7],
                            "force":[3], "torque":[3], "gripper_pos": float}
      POST /pose       <- {"arr": [x,y,z,qx,qy,qz,qw]}
      POST /open_gripper
      POST /close_gripper
      POST /clearerr
    """

    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url.rstrip("/")
        self._state = None
        # Proxy-safe: trust_env=False so the desktop's HTTP(S)_PROXY env
        # (which lacks a CIDR no_proxy for the internal franka_server IP)
        # never intercepts these calls. Matches relative_teleop._session().
        self._session = requests.Session()
        self._session.trust_env = False

    def connect(self):
        """验证服务器可达。"""
        url = f"{self.server_url}/healthz"
        try:
            resp = self._session.post(url, json={}, timeout=5)
            resp.raise_for_status()
            health = resp.json()
            print(f"[FR3] Connected to {self.server_url} (health: {health})")
        except requests.RequestException as e:
            raise ConnectionError(
                f"[FR3] Cannot reach server at {self.server_url}: {e}"
            ) from e

    def get_state(self) -> dict:
        """POST /getstate -> 完整状态字典。"""
        url = f"{self.server_url}/getstate"
        resp = self._session.post(url, json={}, timeout=5)
        resp.raise_for_status()
        self._state = resp.json()
        return self._state

    def get_joint_positions(self) -> np.ndarray:
        """返回当前关节位置 (7,)。"""
        state = self.get_state()
        return np.asarray(state["q"], dtype=float)

    def get_current_pose(self) -> np.ndarray:
        """返回当前末端位姿 (7,): [x, y, z, qx, qy, qz, qw]。"""
        state = self.get_state()
        return np.asarray(state["pose"], dtype=float)

    def get_gripper_pos(self) -> float:
        """返回当前夹爪位置。"""
        state = self.get_state()
        return float(state.get("gripper_pos", 0.0))

    def send_pose_command(self, pose):
        """POST /pose <- {"arr": [x,y,z,qx,qy,qz,qw]}。"""
        url = f"{self.server_url}/pose"
        payload = {"arr": list(pose)}
        resp = self._session.post(url, json=payload, timeout=5)
        resp.raise_for_status()

    def send_gripper_command(self, action: str):
        """发送夹爪命令: 'open' 或 'close'。"""
        endpoint = "open_gripper" if action == "open" else "close_gripper"
        url = f"{self.server_url}/{endpoint}"
        resp = self._session.post(url, json={}, timeout=5)
        resp.raise_for_status()

    def clear_error(self):
        """POST /clearerr。"""
        url = f"{self.server_url}/clearerr"
        try:
            resp = self._session.post(url, json={}, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[FR3] clear_error failed: {e}")

    def close(self):
        self._state = None
        print(f"[FR3] Disconnected from {self.server_url}.")


# ---------------------------------------------------------------------------
# 安全检查 — 与原版完全一致
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


def check_fk_bias(joints, current_pose, limit=DEFAULT_FK_BIAS_LIMIT):
    """Return whether legacy FK agrees with the server TCP pose."""
    fk_pose = forward_kinematics(joints)
    bias = float(np.linalg.norm(fk_pose[:3] - np.asarray(current_pose, dtype=float)[:3]))
    return bias <= limit, bias, fk_pose


def check_cartesian_pose_step(
    prev_pose,
    curr_pose,
    max_translation=DEFAULT_MAX_CARTESIAN_STEP,
    max_rotation=DEFAULT_MAX_CARTESIAN_ROT_STEP,
):
    """Check a live absolute /pose command step before sending it."""
    prev_pose = np.asarray(prev_pose, dtype=float).reshape(-1)[:7]
    curr_pose = np.asarray(curr_pose, dtype=float).reshape(-1)[:7]
    d_pos = float(np.linalg.norm(curr_pose[:3] - prev_pose[:3]))
    r_prev = Rotation.from_quat(prev_pose[3:7])
    r_curr = Rotation.from_quat(curr_pose[3:7])
    d_rot = float((r_curr * r_prev.inv()).magnitude())
    return d_pos <= max_translation and d_rot <= max_rotation, d_pos, d_rot


# ---------------------------------------------------------------------------
# 数据保存 — 与原版完全一致
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

    # 初始化机器人 (live 模式通过 HTTP API)
    robot = None
    if is_live:
        robot = FR3Robot(server_url=args.server_url)
        robot.connect()

    # 数据收集
    joint_poses_list = []
    gripper_states_list = []
    timestamps_list = []
    raw_gello_list = []
    targets_list = []
    commands_list = []
    tracking_errors_list = []
    poses_list = []

    # 初始状态 — 用首次 GELLO 读数作为基准
    raw0 = gello.read()
    raw_gello0 = raw0[:7]

    # q0: live 时用机器人实际关节位置, dry-run 时用 GELLO 映射
    if is_live:
        assert robot is not None
        q0 = np.asarray(robot.get_joint_positions(), dtype=float)
        current_pose = robot.get_current_pose()
        fk_ok, fk_bias, _fk_q0 = check_fk_bias(q0, current_pose, args.fk_bias_limit)
        if not fk_ok:
            print(
                "[ABORT] FK/EE-frame mismatch before live motion: "
                f"||FK(q0)-currpos||={fk_bias:.4f}m > {args.fk_bias_limit:.4f}m. "
                "No /pose command issued."
            )
            gello.close()
            robot.close()
            return None
    else:
        q0 = raw_gello0 * joint_signs * leader_scale
        q0 = np.clip(q0, FR3_LOWER_LIMITS + 0.02, FR3_UPPER_LIMITS - 0.02)
        current_pose = np.array([0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0])

    command = q0.copy()
    prev_target = q0.copy()

    mode_str = "LIVE" if is_live else "DRY-RUN"
    print(f"\n{'=' * 60}")
    print(f"  Recording mode: {mode_str}")
    if is_live:
        print(f"  Server: {args.server_url}")
    print(f"  Hz: {hz}, Duration: {duration}s, Steps: {total_steps}")
    print(f"  Max step: {args.max_step}, Max total delta: {args.max_total_delta}")
    print(f"  Leader scale: {leader_scale}")
    print(f"  Joint signs: {joint_signs.tolist()}")
    print(f"{'=' * 60}\n")

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
                break

            total_ok, total_delta = check_max_total_delta(q0, target, args.max_total_delta)
            if not total_ok:
                abort_reason = f"max_total_delta:{total_delta:.4f}"
                print(f"[ABORT] Step {step}: {abort_reason}")
                break

            # 平滑命令
            command = command + np.clip(target - command, -args.max_step, args.max_step)

            # Live 模式: 读取实际状态并发送位姿命令
            # target_pose: 由平滑后的 command 通过 FK 计算得到的笛卡尔位姿
            actual = command.copy()
            gripper_pos = 0.0
            if is_live:
                assert robot is not None
                actual = np.asarray(robot.get_joint_positions(), dtype=float)
                gripper_pos = robot.get_gripper_pos()
                # FK: 平滑后的关节 command -> 目标笛卡尔位姿 (7D: x,y,z,qx,qy,qz,qw)
                target_pose = forward_kinematics(command)
                cart_ok, cart_step, rot_step = check_cartesian_pose_step(
                    current_pose,
                    target_pose,
                    args.max_cartesian_step,
                    args.max_cartesian_rot_step,
                )
                if not cart_ok:
                    abort_reason = f"max_cartesian_step:{cart_step:.4f}/rot:{rot_step:.4f}"
                    print(f"[ABORT] Step {step}: {abort_reason}")
                    break
                robot.send_pose_command(target_pose)
                current_pose = target_pose.copy()
            else:
                gripper_pos = float(raw_all[-1])

            # 跟踪误差
            err_ok, max_err, err_vec = check_tracking_error(
                command, actual, args.max_tracking_error
            )
            if not err_ok:
                abort_reason = f"tracking_error:{max_err:.4f}"
                print(f"[ABORT] Step {step}: {abort_reason}")
                break

            # 记录
            joint_poses_list.append(actual.copy())
            gripper_states_list.append(gripper_pos)
            timestamps_list.append(timestamp)
            raw_gello_list.append(raw_all.copy())
            targets_list.append(target.copy())
            commands_list.append(command.copy())
            tracking_errors_list.append(err_vec.copy())
            poses_list.append(current_pose.copy())

            prev_target = target.copy()

            if step % max(1, int(hz)) == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"  Step {step:5d}/{total_steps} | {elapsed:.1f}s "
                    f"| step={step_delta:.4f} track={max_err:.4f}"
                )

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
        poses=np.array(poses_list),
        meta_hz=hz,
        meta_duration=timestamps_list[-1] - timestamps_list[0],
        meta_leader_scale=leader_scale,
        meta_joint_signs=np.array(joint_signs),
        meta_num_steps=n,
        meta_abort_reason=abort_reason,
        meta_server_url=args.server_url if is_live else "dry-run",
    )

    file_size = os.path.getsize(filepath)
    print(f"\n  Saved: {filepath} ({file_size / 1024:.1f} KB)")
    print(f"  Steps: {n}, Abort: {abort_reason}")
    return filepath


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record GELLO demos for FR3 (SERL HTTP API)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="GELLO only, no robot commands")
    mode.add_argument("--live", action="store_true", help="GELLO + FR3 via HTTP API")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                        help=f"franka_server HTTP URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gello-port", default="/dev/ttyUSB0")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--leader-scale", type=float, default=DEFAULT_LEADER_SCALE)
    parser.add_argument("--joint-signs", default=",".join(str(s) for s in DEFAULT_JOINT_SIGNS))
    parser.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP)
    parser.add_argument("--max-total-delta", type=float, default=DEFAULT_MAX_TOTAL_DELTA)
    parser.add_argument("--max-tracking-error", type=float, default=DEFAULT_MAX_TRACKING_ERROR)
    parser.add_argument("--fk-bias-limit", type=float, default=DEFAULT_FK_BIAS_LIMIT)
    parser.add_argument("--max-cartesian-step", type=float, default=DEFAULT_MAX_CARTESIAN_STEP)
    parser.add_argument("--max-cartesian-rot-step", type=float, default=DEFAULT_MAX_CARTESIAN_ROT_STEP)
    args = parser.parse_args()
    args.joint_signs = [int(s) for s in args.joint_signs.split(",")]
    filepath = run_recording(args)
    if filepath:
        print(f"\nDone: {filepath}")


if __name__ == "__main__":
    main()
