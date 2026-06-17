"""
plug_insertion 环境实现
---
继承 FrankaEnv，实现插插头任务的 reset / step / reward 逻辑。
"""
import time
import os
import sys
import copy
import numpy as np
import requests
from collections import OrderedDict
from scipy.spatial.transform import Rotation

import gymnasium as gym
from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat

RESET_STRICT = os.environ.get("RESET_STRICT", "1").strip().lower() in ("1", "true", "yes", "on")
MANUAL_RESET = os.environ.get("MANUAL_RESET", "").strip().lower() in ("1", "true", "yes", "on")

# Ensure project root is on path for scripts/ imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.zed_capture import ZEDCapture
from scripts.video_capture import VideoCapture


class PlugInsertionEnv(FrankaEnv):
    """
    插插头任务环境。

    任务流程:
    1. reset: 机械臂移动到初始位姿，张开夹爪
    2. step:  接收 7D 归一化动作，发送到机器人，返回观测
    3. 判定:  奖励由分类器 + 位姿联合给出
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 插入状态跟踪
        self._insertion_started = False
        self._contact_detected = False

    # ----------------------------------------------------------
    # 相机初始化
    # ----------------------------------------------------------
    def init_cameras(self, name_serial_dict=None):
        """
        初始化 ZED 相机。

        side_classifier 复用 side_policy 相机（同物理设备不同裁剪）。
        """
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if cam_name == "side_classifier":
                # 复用 policy 相机，不重复打开设备
                self.cap["side_classifier"] = self.cap["side_policy"]
            else:
                # Filter out keys ZEDCapture doesn't accept
                zed_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in ("serial_number", "dim", "fps", "exposure")
                }
                cap = VideoCapture(
                    ZEDCapture(name=cam_name, **zed_kwargs)
                )
                self.cap[cam_name] = cap

    # ----------------------------------------------------------
    # Reset: 初始化到抓取/对准起始位置
    # ----------------------------------------------------------
    def reset(self, **kwargs):
        """
        hold-grip 复位 (insert-only): 全程不松爪、保持夹持插头。
        1. 恢复 + 精密参数
        2. 保持夹持, 直上拔出插头到清空高度
        3. 移到 RESET_POSE (插座上方+后退) + RANDOM_RESET 扰动
        4. 掉插头检测 (gripper_pos 远低于握持值 -> 警告)
        """
        # 恢复（如有异常则触发安全恢复）
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)

        # 切换到精密控制参数
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # !! 不张爪 !! 保持夹持插头 (hold-grip): 夹爪全程钳住; HoldGripperWrapper 也会把 episode
        # 内的夹爪动作置 no-op, 所以这里不发任何夹爪指令 (旧版 _send_gripper_command(1.0) 会开爪丢插头)。

        # 直上拔出插头到清空高度 (extract straight up: 保持当前 xy + 朝向, 仅升 z)
        clear = copy.deepcopy(self.currpos)
        clear[2] = max(float(clear[2]), float(self.config.TARGET_POSE[2] + 0.12))
        clear_err = self.interpolate_move(clear, timeout=8.0, name="clear")
        time.sleep(0.3)
        self._require_reset_settled("clear", clear_err)

        # 移到固定 RESET_POSE。Phase C 真机由操作者评估成功, 禁用随机 reset 以避免
        # "每次复位更低/不同" 的可见漂移被 random xy/yaw 掩盖。
        reset_pose = copy.deepcopy(np.asarray(self.config.RESET_POSE, dtype=float))
        reset_err = self.interpolate_move(reset_pose, timeout=8.0, name="reset_pose")
        time.sleep(0.3)
        self._require_reset_settled("reset_pose", reset_err)

        # 关节限位监控 (defense-in-depth, FIX 20260616): nullspace_stiffness=25 已防 j4 漂移,
        # 但若 impedance 启动构型/q_d_nullspace 不安全, 在此提前预警, 避免 joint_velocity_violation 乱颤。
        try:
            _q = np.array(requests.post(self.url + "getstate", timeout=5).json()["q"])
            _LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
            _HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
            _md = np.degrees(np.minimum(_q - _LO, _HI - _q))
            _j = int(np.argmin(_md))
            if _md[_j] < 12.0:
                print("\n[!!! 关节逼近限位] j%d margin=%.1fdeg (<12). 乱颤风险! 请停下复位机械臂构型再继续。\n"
                      % (_j + 1, _md[_j]), flush=True)
        except Exception:
            pass

        # 不调用 super().reset(): 父类 reset 会再次 self.go_to_reset(), 导致同一轮 reset 执行两遍。
        # 这里仅执行父类 reset 的非运动收尾逻辑，保持 teach-style 单一路径复位。
        self.curr_path_length = 0
        self.terminate = False
        self._update_currpos()
        obs = self._get_obs()
        info = {"succeed": False}

        # !! 不张爪 !! 保持夹持

        # 掉插头检测: 握持值≈0.567, 远低于 0.40 视为插头滑脱/丢失
        self._update_currpos()
        held = float(np.asarray(self.curr_gripper_pos).reshape(-1)[0])
        if held < 0.40:
            print("\n[!!! 可能掉插头] gripper_pos=%.3f < 0.40 (握持≈0.567). "
                  "请停下并重新抓取插头再继续。\n" % held, flush=True)

        # 重置任务状态
        self.success = False
        self._insertion_started = False
        self._contact_detected = False

        self._update_currpos()
        obs = self._get_obs()

        return obs, info

    def _require_reset_settled(self, name, err):
        pos_err, rot_err = err
        while RESET_STRICT and (pos_err > 0.012 or rot_err > 0.12):
            msg = (
                f"[reset_warn] {name} did not settle: "
                f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}. "
                "reposition/check contact before continuing."
            )
            print(msg, flush=True)
            if not MANUAL_RESET:
                fatal = msg.replace("[reset_warn]", "[reset_fatal]")
                print(fatal, flush=True)
                raise RuntimeError(fatal)
            answer = input(
                "\n[reset_manual_recovery] reset 未收敛。处理好插头/线缆/接触后按 Enter 重新检查；"
                "输入 abort 退出 actor: "
            ).strip().lower()
            if answer in ("abort", "q", "quit", "exit"):
                fatal = msg.replace("[reset_warn]", "[reset_fatal]")
                print(fatal, flush=True)
                raise RuntimeError(fatal)
            self._update_currpos()
            curr = np.asarray(self.currpos, dtype=float).reshape(7)
            target = getattr(self, "_last_reset_goal", None)
            if target is None:
                print(
                    "[reset_manual_recovery] missing reset target; accepting operator confirmation.",
                    flush=True,
                )
                return
            pos_err = float(np.linalg.norm(target[:3] - curr[:3]))
            rot_err = float(np.linalg.norm(
                (Rotation.from_quat(target[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
            ))
            print(
                f"[reset_manual_recovery] {name} recheck target_z={target[2]:.4f} "
                f"actual_z={curr[2]:.4f} pos_err={pos_err:.4f} rot_err={rot_err:.4f}",
                flush=True,
            )

    # ----------------------------------------------------------
    # 辅助：线性插值移动
    # ----------------------------------------------------------
    def interpolate_move(self, goal: np.ndarray, timeout: float, name: str = "move"):
        """
        线性插值移动到目标位姿。

        Args:
            goal:    6D [x,y,z,roll,pitch,yaw] 或 7D [x,y,z,quaternion]
            timeout: 移动持续时间 (s)
        """
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])

        # Teach-aligned reset motion: each tick recompute delta from current pose and clamp
        # the per-tick target jump. Re-sending a far fixed setpoint near the j5≈0 wrist
        # singularity caused impedance ringing / 乱颤.
        goal = np.asarray(goal, dtype=float).reshape(7)
        self._last_reset_goal = goal.copy()
        deadline = time.time() + float(timeout)
        max_step = 0.0035          # m/tick, close to teach DEFAULT_MAX_STEP and actor z clip
        max_rot_step = 0.05        # rad/tick, equals controller rotational_clip after our fix
        pos_tol = 0.004
        rot_tol = 0.04
        while time.time() < deadline:
            self._update_currpos()
            curr = np.asarray(self.currpos, dtype=float).reshape(7)
            dxyz = goal[:3] - curr[:3]
            drot = (Rotation.from_quat(goal[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
            if np.linalg.norm(dxyz) < pos_tol and np.linalg.norm(drot) < rot_tol:
                break
            n = float(np.linalg.norm(dxyz))
            if n > max_step:
                dxyz = dxyz * (max_step / n)
            nr = float(np.linalg.norm(drot))
            if nr > max_rot_step:
                drot = drot * (max_rot_step / nr)
            nxt = curr.copy()
            nxt[:3] = curr[:3] + dxyz
            nxt[3:] = (Rotation.from_rotvec(drot) * Rotation.from_quat(curr[3:])).as_quat()
            self._send_pos_command(nxt)
            time.sleep(1.0 / self.hz)
        # Hold the exact final target, then poll until the measured TCP catches up.
        # The impedance controller can lag the last setpoint by >0.5s during vertical
        # clear moves; treating that transient as fatal caused false reset failures.
        for _ in range(max(1, int(0.5 * self.hz))):
            self._send_pos_command(goal)
            time.sleep(1.0 / self.hz)
        settle_deadline = time.time() + 4.0
        pos_err = float("inf")
        rot_err = float("inf")
        curr = None
        while time.time() < settle_deadline:
            self._update_currpos()
            curr = np.asarray(self.currpos, dtype=float).reshape(7)
            pos_err = float(np.linalg.norm(goal[:3] - curr[:3]))
            rot_err = float(np.linalg.norm(
                (Rotation.from_quat(goal[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
            ))
            if pos_err <= 0.012 and rot_err <= 0.12:
                break
            self._send_pos_command(goal)
            print(
                f"[reset_settle_wait] {name} target_z={goal[2]:.4f} actual_z={curr[2]:.4f} "
                f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}",
                flush=True,
            )
            time.sleep(0.2)
        if curr is None:
            self._update_currpos()
            curr = np.asarray(self.currpos, dtype=float).reshape(7)
            pos_err = float(np.linalg.norm(goal[:3] - curr[:3]))
            rot_err = float(np.linalg.norm(
                (Rotation.from_quat(goal[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
            ))
        print(
            f"[reset_dbg] {name} target_z={goal[2]:.4f} actual_z={curr[2]:.4f} "
            f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}",
            flush=True,
        )
        if pos_err > 0.012 or rot_err > 0.12:
            print(
                f"[reset_warn] {name} did not settle: pos_err={pos_err:.4f} rot_err={rot_err:.4f}",
                flush=True,
            )
        return pos_err, rot_err

    # ----------------------------------------------------------
    # 复位到初始位姿（支持随机化）
    # ----------------------------------------------------------
    def go_to_reset(self, joint_reset=False):
        """
        回到 reset 位姿。支持关节级复位和 XY 随机扰动。

        Args:
            joint_reset: 是否触发关节级复位（长时间运行后消除漂移）
        """
        if joint_reset:
            print("[PlugInsertion] JOINT RESET triggered")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        if self.randomreset:
            reset_pose = self.resetpos.copy()
            # XY 随机扰动（增强泛化）
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            # yaw 随机扰动
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=3.0)
        else:
            self.interpolate_move(self.resetpos.copy(), timeout=3.0)

        # 切回合规模式（操作阶段使用）
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
