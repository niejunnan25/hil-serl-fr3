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
RESET_JOINT_LIMIT_MARGIN_DEG = float(os.environ.get("RESET_JOINT_LIMIT_MARGIN_DEG", "12.0"))
DROP_GRIPPER_MIN = float(os.environ.get("DROP_GRIPPER_MIN", "0.40"))
RESET_MOTION_EFFECTIVE_SPEED = float(os.environ.get("RESET_MOTION_EFFECTIVE_SPEED", "0.002"))
RESET_MOTION_MAX_TIMEOUT = float(os.environ.get("RESET_MOTION_MAX_TIMEOUT", "90.0"))
RESET_MOTION_SETTLE_TIMEOUT = float(os.environ.get("RESET_MOTION_SETTLE_TIMEOUT", "8.0"))
RESET_POSITION_TOLERANCE = 0.012
RESET_ROTATION_TOLERANCE = 0.12
RESET_STALL_TIMEOUT = 8.0
RESET_HTTP_TIMEOUT = (2.0, 5.0)
RESET_GRIPPER_OPEN_THRESHOLD = 0.85

FR3_RESET_LOWER_LIMITS = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_RESET_UPPER_LIMITS = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])

# Ensure project root is on path for scripts/ imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.zed_capture import ZEDCapture
from scripts.video_capture import VideoCapture
from hilserl.errors import RobotStateUnavailable


def joint_limit_margin_degrees(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("reset requires seven finite joint positions")
    margins = np.degrees(np.minimum(q - FR3_RESET_LOWER_LIMITS, FR3_RESET_UPPER_LIMITS - q))
    joint_index = int(np.argmin(margins))
    return float(margins[joint_index]), joint_index, margins


def is_plug_held(gripper_pos, threshold=DROP_GRIPPER_MIN):
    """Check plausible held width only; a width cannot prove a secure grasp.

    The server reports normalized opening: 1 is fully open, 0 fully closed.
    Retain the calibrated lower bound, and reject the existing open threshold.
    """
    value = np.asarray(gripper_pos, dtype=float).reshape(-1)
    return bool(value.size == 1 and np.isfinite(value[0])
                and float(threshold) <= value[0] < RESET_GRIPPER_OPEN_THRESHOLD)


def sample_reset_pose(config, rng=np.random):
    """Sample reset translation only; orientation is fixed by RESET_POSE."""
    reset_pose = copy.deepcopy(np.asarray(config.RESET_POSE, dtype=float)).reshape(6)
    if getattr(config, "RANDOM_RESET", False):
        reset_pose[:2] += rng.uniform(
            -float(config.RANDOM_XY_RANGE),
            float(config.RANDOM_XY_RANGE),
            size=(2,),
        )
    return reset_pose


def compute_reset_clear_pose(currpos, config):
    """Return the hold-grip vertical clear pose for reset."""
    clear = copy.deepcopy(np.asarray(currpos, dtype=float)).reshape(7)
    configured_clear_z = getattr(config, "RESET_CLEAR_Z", None)
    if configured_clear_z is None:
        configured_clear_z = float(config.TARGET_POSE[2] + 0.12)
    clear[2] = max(float(clear[2]), float(configured_clear_z))
    return clear


class PlugInsertionEnv(FrankaEnv):
    """
    插插头任务环境。

    任务流程:
    1. reset: 保持夹持，逐阶段验证机械臂到达初始位姿；不自动开合夹爪
    2. step:  接收 7D 归一化动作，发送到机器人，返回观测
    3. 判定:  奖励由分类器 + 位姿联合给出
    """

    def __init__(self, recorder=None, operator=None, manual_reward=False, **kwargs):
        self.manual_reward = manual_reward
        self.recorder = recorder
        self.operator = operator
        self._state_capture = None
        self._step_commands = []
        self._command_pose = None
        self.frame_references = {}
        self.last_reset_info = None
        try:
            super().__init__(**kwargs)
        except BaseException:
            self.close()
            raise
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
                zed_kwargs["fps"] = int(os.environ.get("HILSERL_VIDEO_FPS", zed_kwargs.get("fps", 30)))
                cap = VideoCapture(ZEDCapture(name=cam_name, **zed_kwargs),
                                   name=cam_name, recorder=self.recorder, continuous=True)
                self.cap[cam_name] = cap

    def get_im(self):
        import cv2
        images, display, packets = {}, {}, {}
        for key, cap in self.cap.items():
            if id(cap) not in packets:
                packets[id(cap)] = (cap.read(), cap.last_read)
            frame, reference = packets[id(cap)]
            profile = getattr(self.config, "IMAGE_PROFILE", "full-frame128-v1")
            if profile != "full-frame128-v1":
                from hilserl.image_profile import crop_image, preprocess_image
                crop = crop_image(frame, key, profile)
                images[key] = preprocess_image(frame, key, profile)
                resized = images[key][..., ::-1]
                if images[key].shape != self.observation_space["images"][key].shape:
                    raise ValueError("Image profile and environment observation space differ")
            else:
                crop = self.config.IMAGE_CROP[key](frame) if key in self.config.IMAGE_CROP else frame
                resized = cv2.resize(crop, self.observation_space["images"][key].shape[:2][::-1])
                images[key] = resized[..., ::-1].copy()
            display[key] = resized
            display[key + "_full"] = crop
            self.frame_references[key] = dict(reference) if reference else None
        if self.display_image:
            self.img_queue.put(display)
        return images

    def raw_state(self):
        from hilserl.storage import stamp
        fields = {"tcp_pose": "currpos", "tcp_vel": "currvel", "tcp_force": "currforce",
                  "tcp_torque": "currtorque", "q": "q", "dq": "dq", "jacobian": "currjacobian",
                  "gripper_pose": "curr_gripper_pos"}
        return dict(values={name: np.asarray(getattr(self, attr)).copy() for name, attr in fields.items()},
                    capture=self._state_capture or stamp())

    def _update_currpos(self):
        from hilserl.storage import stamp
        before = stamp()
        try:
            state = self._robot_post("getstate").json()
        except ValueError as exc:
            raise RobotStateUnavailable("FR3 状态响应不是有效 JSON，已暂停动作。",
                                        details={"reason": "invalid_json"}) from exc
        health = {key: value for key, value in state.items()
                  if key.endswith(("_age_seconds", "_stale", "_sequence"))
                  or key in {"controller_running", "controller_exit_code", "pose_subscribers"}} if isinstance(state, dict) else {}
        # Legacy bridges return plausible cached numbers after ROS has died.
        # Require independent callback freshness before using a state for any
        # reset, action, or recording; HTTP response time is not sensor time.
        if not isinstance(state, dict) or state.get("controller_running") is not True:
            raise RobotStateUnavailable("FR3 控制器未运行或状态时效未确认；请恢复底层控制服务。",
                                        health=health)
        for prefix in ("state", "jacobian", "gripper_state"):
            age = state.get(prefix + "_age_seconds")
            if (state.get(prefix + "_stale") is not False or type(age) not in (int, float)
                    or not np.isfinite(age) or not 0 <= age <= 2.0):
                raise RobotStateUnavailable(f"FR3 {prefix} 状态缺失或已过期，禁止复位和采集。",
                                            health=health, details={"feedback": prefix})
        fields = {"currpos": "pose", "currvel": "vel", "currforce": "force",
                  "currtorque": "torque", "q": "q", "dq": "dq",
                  "curr_gripper_pos": "gripper_pos"}
        # Validate the entire response before replacing any cached measurements.
        sizes = {"pose": 7, "vel": 6, "force": 3, "torque": 3, "q": 7,
                 "dq": 7, "gripper_pos": 1, "jacobian": 42}
        try:
            values = {key: np.asarray(state[key], dtype=float) for key in sizes}
            for key, size in sizes.items():
                if values[key].size != size or not np.all(np.isfinite(values[key])):
                    raise ValueError(f"invalid {key}: expected {size} finite values")
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotStateUnavailable(f"FR3 状态字段无效，已暂停动作：{exc}",
                                        health=health, details={"reason": "invalid_state"}) from exc
        for attr, key in fields.items():
            setattr(self, attr, values[key].reshape(-1) if key != "gripper_pos" else values[key].reshape(()))
        self.currjacobian = values["jacobian"].reshape(6, 7)
        self._state_capture = dict(request=before, response=stamp(), server_health=health)

    def _robot_post(self, endpoint, **kwargs):
        """One bounded request; never retry an action with unknown delivery."""
        self._check_recording_and_stop()
        try:
            response = requests.post(self.url + endpoint, timeout=RESET_HTTP_TIMEOUT, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            unknown = endpoint != "getstate"
            message = (f"FR3 {endpoint} 通信失败，已暂停动作：{exc}"
                       + ("；命令是否送达未知，不会自动重发。" if unknown else ""))
            raise RobotStateUnavailable(message, endpoint=endpoint, delivery_unknown=unknown,
                                        details={"transport_error": type(exc).__name__}) from exc
        if getattr(response, "status_code", None) == 503:
            try:
                details = response.json()
            except ValueError:
                details = {"error": str(getattr(response, "text", "Service unavailable"))[:1000]}
            if not isinstance(details, dict):
                details = {"error": str(details)}
            health = details.get("health", {})
            if not isinstance(health, dict):
                health = {}
            health.update({key: value for key, value in details.items()
                           if key.endswith(("_age_seconds", "_stale", "_sequence"))
                           or key in {"controller_running", "controller_exit_code", "pose_subscribers"}})
            reason = details.get("error") or details.get("reason") or details.get("message") or "Service unavailable"
            raise RobotStateUnavailable(f"FR3 {endpoint} 暂不可用（HTTP 503）：{reason}",
                                        endpoint=endpoint, status_code=503, details=details,
                                        health=health, delivery_unknown=endpoint != "getstate")
        response.raise_for_status()
        self._check_recording_and_stop()
        return response

    def _recover(self):
        self._robot_post("clearerr")

    def _check_recording_and_stop(self):
        if getattr(self, "recorder", None):
            self.recorder.check()
        if getattr(self, "operator", None):
            self.operator.raise_if_stop()

    def _send_pos_command(self, pos):
        from hilserl.storage import stamp
        self._check_recording_and_stop()
        item = dict(kind="pose", pose=np.asarray(pos, dtype=np.float32).copy(), sent=stamp(),
                    request_sent=False, returned=False)
        self._step_commands.append(item)
        if self.clearerr_on_pose:
            self._recover()
        item.update(request_sent=True, sent=stamp())
        try:
            self._robot_post("pose", json={"arr": np.asarray(pos, dtype=np.float32).tolist()})
        except RobotStateUnavailable as exc:
            item.update(error=exc.as_dict(), delivery_unknown=exc.delivery_unknown, response=stamp())
            raise
        item.update(returned=True, response=stamp())
        self._command_pose = item["pose"].astype(np.float64).copy()

    def _send_gripper_command(self, pos, mode="binary"):
        """Preserve the binary/cooldown contract while recording one HTTP attempt."""
        from hilserl.storage import stamp
        if mode != "binary":
            raise NotImplementedError("Continuous gripper control is optional")
        if time.time() - self.last_gripper_act <= self.gripper_sleep:
            return
        endpoint = ("close_gripper" if pos <= -.5 and self.curr_gripper_pos > .85 else
                    "open_gripper" if pos >= .5 and self.curr_gripper_pos < .85 else None)
        if endpoint is None:
            return
        self._check_recording_and_stop()
        item = dict(kind="gripper", endpoint=endpoint, sent=stamp(), request_sent=True, returned=False)
        self._step_commands.append(item)
        try:
            self._robot_post(endpoint)
        except RobotStateUnavailable as exc:
            item.update(error=exc.as_dict(), delivery_unknown=exc.delivery_unknown, response=stamp())
            raise
        item.update(returned=True, response=stamp())
        self.last_gripper_act = time.time()
        time.sleep(self.gripper_sleep)

    def compute_reward(self, obs):
        # Human collection must not stop on an old pose heuristic or classifier.
        if getattr(self, "manual_reward", False):
            return 0
        return super().compute_reward(obs)

    def _action_target_position(self, xyz_delta):
        if getattr(self.config, "POSITION_TARGET_MODE", "measured-relative-v1") != "command-relative-v1":
            return super()._action_target_position(xyz_delta)
        if self._command_pose is None:
            raise RuntimeError("Command-relative motion requires a confirmed reset target")
        from hilserl.motion_limits import advance_position_target
        return advance_position_target(self._command_pose[:3], xyz_delta, self.currpos[:3],
                                       self.action_max_step, getattr(self.config, "ACTION_MAX_Z_STEP", None))

    def _action_target_orientation(self, rot_delta):
        if self.lock_rotation:
            return self.resetpos[3:].copy()
        return super()._action_target_orientation(rot_delta)

    def _clip_translation_delta(self, delta):
        z_limit = getattr(self.config, "ACTION_MAX_Z_STEP", None)
        if z_limit is None:
            return super()._clip_translation_delta(delta)
        from hilserl.motion_limits import clip_translation_delta
        return clip_translation_delta(delta, self.action_max_step, z_limit)

    def step(self, action):
        self._step_commands = []
        self._check_recording_and_stop()
        base_action = np.asarray(action).copy()
        try:
            # Sampling may take longer than the feedback freshness window. Check
            # again before super().step can send either a gripper or pose action.
            self._update_currpos()
            for attr, limit_attr in (("currforce", "safety_force_max"), ("dq", "safety_dq_max")):
                values = np.asarray(getattr(self, attr), dtype=float)
                value = float(np.linalg.norm(values) if attr == "currforce" else np.max(np.abs(values)))
                if not np.isfinite(value) or value > float(getattr(self, limit_attr)):
                    raise RuntimeError(f"[franka_safety_fatal] {attr}={value} > {getattr(self, limit_attr)}")
            obs, reward, done, truncated, info = super().step(action)
        except BaseException as exc:
            if self.recorder:
                self.recorder.event("step_interrupted", error=type(exc).__name__,
                                    commands=[{**c, **({"pose": c["pose"].tolist()} if "pose" in c else {})}
                                              for c in self._step_commands])
            raise
        reason = "environment_success" if reward else "environment_terminate" if self.terminate else None
        if reason is None and self.curr_path_length >= self.max_episode_length:
            reason, done, truncated = "time_limit", False, True
        info.update(termination_reason=reason, controller_input_action=base_action,
                    controller_commands=self._step_commands, raw_next_state=self.raw_state(),
                    next_frame_references=dict(self.frame_references))
        return obs, reward, done, truncated, info

    def _operator_input(self, prompt):
        if getattr(self, "operator", None):
            return self.operator.input(prompt, check=self.recorder.check if self.recorder else None)
        return input(prompt)

    def close(self):
        if getattr(self, "_hilserl_closed", False):
            return
        self._hilserl_closed = True
        if hasattr(self, "listener"):
            self.listener.stop()
        for cap in set((getattr(self, "cap", None) or {}).values()):
            cap.close()
        if hasattr(self, "img_queue"):
            self.img_queue.put(None)
        if hasattr(self, "displayer"):
            self.displayer.join(timeout=3)

    # ----------------------------------------------------------
    # Reset: 初始化到抓取/对准起始位置
    # ----------------------------------------------------------
    def _reset_progress(self, phase, *, state="running", force=False, **details):
        """Publish measured evidence. Only reset() may mark the whole reset successful."""
        now = time.monotonic()
        if (not getattr(self, "last_reset_info", None)
                or state in ("running", "moving") and self.last_reset_info.get("state") != "running"):
            self._reset_started = now
            self.last_reset_info = dict(success=False, state="running", phases=[])
        info = self.last_reset_info
        info.update(phase=phase, elapsed_seconds=now - self._reset_started)
        phases = info["phases"]
        if not phases or phases[-1]["name"] != phase:
            self._reset_phase_started = now
            phases.append(dict(name=phase))
        phases[-1].update(state=state, elapsed_seconds=now - self._reset_phase_started, **details)
        if state in ("failed", "cancelled"):
            info.update(success=False, state=state, reason=details.get("reason", state))
        snapshot = copy.deepcopy(info)
        if force or now - getattr(self, "_reset_last_publish", float("-inf")) >= 1.0:
            self._reset_last_publish = now
            if getattr(self, "operator", None):
                label = {"precheck": "检查夹爪和关节", "prepare": "准备控制器",
                         "clear": "垂直拔出", "reset_pose": "移动到起始位姿",
                         "verify": "验证复位结果", "ready": "复位完成"}.get(phase, phase)
                self.operator.publish("resetting", reset=snapshot,
                                      prompt=f"复位：{label} · {info['elapsed_seconds']:.1f} 秒")
            if getattr(self, "recorder", None):
                self.recorder.event("reset_progress", reset=snapshot)
        return snapshot

    def _reset_failure(self, name, reason, **details):
        self._reset_progress(name, state="failed", force=True, reason=reason, **details)
        errors = " ".join(f"{key}={value}" for key, value in details.items())
        message = f"[reset_fatal] {name} {reason}; {errors}. 复位失败，未进入采集。"
        print(message, flush=True)
        raise RuntimeError(message)

    def reset(self, **kwargs):
        """Hold the plug, clear vertically, then move to the sampled reset pose.

        Gripper width is a plausibility precheck, not proof of a secure grasp.
        Failed checks and motion stages always stop; RESET_STRICT/MANUAL_RESET
        cannot override measured convergence or grant permission to collect.
        """
        self._reset_started = time.monotonic()
        self._step_commands = []
        self._reset_phase_started = self._reset_started
        self._reset_last_publish = float("-inf")
        self.last_reset_info = dict(success=False, state="running", phases=[],
                                    position_tolerance_m=RESET_POSITION_TOLERANCE,
                                    rotation_tolerance_rad=RESET_ROTATION_TOLERANCE)
        try:
            # Read-only prechecks must precede recovery and every pose command.
            self._reset_progress("precheck", force=True)
            self._update_currpos()
            self._require_plug_held()
            self._require_joint_limit_margin()
            self._reset_progress("precheck", state="passed", force=True,
                                 gripper_position=float(np.asarray(self.curr_gripper_pos).reshape(-1)[0]),
                                 gripper_check="plausible_width_only")

            self._reset_progress("prepare", force=True)
            self._recover()
            self._update_currpos()
            self._send_pos_command(self.currpos)
            time.sleep(0.1)
            self._robot_post("update_param", json=self.config.PRECISION_PARAM)
            self._reset_progress("prepare", state="passed", force=True)

            clear = compute_reset_clear_pose(self.currpos, self.config)
            clear_err = self.interpolate_move(clear, timeout=8.0, name="clear")
            self._require_reset_settled("clear", clear_err)

            reset_pose = sample_reset_pose(self.config)
            if getattr(self.config, "RANDOM_RESET", False):
                print(
                    "[reset_random] "
                    f"x={reset_pose[0]:.4f} y={reset_pose[1]:.4f} z={reset_pose[2]:.4f} "
                    f"yaw={reset_pose[5]:.4f} "
                    f"xy_range={float(self.config.RANDOM_XY_RANGE):.4f} "
                    f"rz_range={float(self.config.RANDOM_RZ_RANGE):.4f}",
                    flush=True,
                )
            else:
                print("[reset_fixed] using configured RESET_POSE", flush=True)
            reset_err = self.interpolate_move(reset_pose, timeout=8.0, name="reset_pose")
            self._require_reset_settled("reset_pose", reset_err)

            self._reset_progress("verify", force=True)
            self._update_currpos()
            self._require_joint_limit_margin()
            self._require_plug_held()
            # Do not call the base reset: it would move the robot for a second time.
            self.curr_path_length = 0
            self.terminate = False
            self.success = False
            self._insertion_started = False
            self._contact_detected = False
            self._check_recording_and_stop()
            obs = self._get_obs()
            self._check_recording_and_stop()
            self._reset_progress("verify", state="passed", force=True)
            self.last_reset_info.update(success=True, state="succeeded")
            evidence = self._reset_progress("ready", state="passed", force=True)
            return obs, {"succeed": False, "reset": evidence}
        except BaseException as exc:
            state = "cancelled" if type(exc).__name__ in ("StopRequested", "KeyboardInterrupt") else "failed"
            phase = self.last_reset_info.get("phase", "precheck")
            reason = self.last_reset_info.get("reason", str(exc))
            try:
                evidence = self._reset_progress(phase, state=state, force=True,
                                                reason=reason, error=type(exc).__name__)
                exc.reset_info = evidence
            except Exception as publish_error:
                # A recorder failure must not hide the original reset failure.
                print(f"[reset_evidence_error] {publish_error}", flush=True)
            raise

    def _require_joint_limit_margin(self, q=None):
        try:
            if q is None:
                q = self._robot_post("getstate").json()["q"]
            margin_deg, joint_index, _margins = joint_limit_margin_degrees(q)
        except Exception as exc:
            # Preserve cancellation instead of relabelling it as a sensor failure.
            if type(exc).__name__ == "StopRequested" or isinstance(exc, RobotStateUnavailable):
                raise
            fatal = f"[joint_limit_fatal] unable to read valid q for reset safety check: {exc}"
            print(fatal, flush=True)
            raise RuntimeError(fatal) from exc
        if margin_deg < RESET_JOINT_LIMIT_MARGIN_DEG:
            fatal = (f"[joint_limit_fatal] j{joint_index + 1} margin={margin_deg:.1f}deg "
                     f"(<{RESET_JOINT_LIMIT_MARGIN_DEG:.1f}). reposition before continuing.")
            print(fatal, flush=True)
            raise RuntimeError(fatal)

    def _require_plug_held(self):
        if is_plug_held(self.curr_gripper_pos):
            return
        phase = (getattr(self, "last_reset_info", None) or {}).get("phase", "precheck")
        outcome = "复位未执行" if phase == "precheck" else "复位已中止，未进入采集"
        fatal = (f"[drop_plug_fatal] gripper_pos={np.asarray(self.curr_gripper_pos).tolist()}; "
                 f"expected plausible held width {DROP_GRIPPER_MIN:.2f} <= value < "
                 f"{RESET_GRIPPER_OPEN_THRESHOLD:.2f}. "
                 f"夹爪全开、空夹或读数异常；先调整夹爪/插头，{outcome}。")
        print(fatal, flush=True)
        raise RuntimeError(fatal)

    def _require_reset_settled(self, name, err):
        pos_err, rot_err = (float(value) for value in err)
        if (not np.isfinite(pos_err) or not np.isfinite(rot_err)
                or pos_err > RESET_POSITION_TOLERANCE or rot_err > RESET_ROTATION_TOLERANCE):
            self._reset_failure(name, "not_converged", position_error_m=pos_err,
                                rotation_error_rad=rot_err)

    # ----------------------------------------------------------
    # 辅助：线性插值移动
    # ----------------------------------------------------------
    def interpolate_move(self, goal: np.ndarray, timeout: float, name: str = "move"):
        """Follow the existing bounded pose targets; require measured convergence.

        HTTP deadlines, a monotonic motion deadline, and progress measured against
        the best remaining error bound a frozen controller without speeding up
        the robot or enlarging any positional/rotational command limits.
        """
        goal = np.asarray(goal, dtype=float)
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        goal = goal.reshape(7)
        self._last_reset_goal = goal.copy()
        max_step = 0.015
        max_z_step = getattr(self.config, "ACTION_MAX_Z_STEP", None)
        if max_z_step is not None:
            max_step = min(max_step, getattr(self, "action_max_step", .008))
        max_rot_step = 0.05
        pos_tol, rot_tol = 0.004, 0.04
        self._reset_progress(name, force=True, target_pose=goal.tolist())
        try:
            if not np.all(np.isfinite(goal)) or np.linalg.norm(goal[3:]) < 1e-9:
                self._reset_failure(name, "invalid_target")
            timing = (float(timeout), RESET_MOTION_EFFECTIVE_SPEED,
                      RESET_MOTION_MAX_TIMEOUT, RESET_MOTION_SETTLE_TIMEOUT, float(self.hz))
            if not all(np.isfinite(value) and value > 0 for value in timing):
                self._reset_failure(name, "invalid_motion_timing")

            def measured():
                self._check_recording_and_stop()
                self._update_currpos()
                curr = np.asarray(self.currpos, dtype=float).reshape(7)
                if not np.all(np.isfinite(curr)) or np.linalg.norm(curr[3:]) < 1e-9:
                    self._reset_failure(name, "invalid_measured_pose")
                self._require_plug_held()
                self._require_joint_limit_margin(self.q)
                pos_err = float(np.linalg.norm(goal[:3] - curr[:3]))
                rot_err = float(np.linalg.norm(
                    (Rotation.from_quat(goal[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
                ))
                # Apply the same measured force/joint-speed limits as env.step.
                for attr, limit_attr in (("currforce", "safety_force_max"), ("dq", "safety_dq_max")):
                    if hasattr(self, attr) and hasattr(self, limit_attr):
                        values = np.asarray(getattr(self, attr), dtype=float)
                        value = float(np.linalg.norm(values) if attr == "currforce" else np.max(np.abs(values)))
                        if not np.isfinite(value) or value > float(getattr(self, limit_attr)):
                            self._reset_failure(name, "safety_limit", measurement=attr,
                                                value=value, limit=float(getattr(self, limit_attr)))
                return curr, pos_err, rot_err

            def bounded_target(curr):
                dxyz = goal[:3] - curr[:3]
                drot = (Rotation.from_quat(goal[3:]) * Rotation.from_quat(curr[3:]).inv()).as_rotvec()
                distance, angle = float(np.linalg.norm(dxyz)), float(np.linalg.norm(drot))
                if name == "clear":
                    # Keep the withdrawal XY reference fixed instead of letting
                    # radial scaling make it follow lateral tracking drift.
                    lateral_ratio = float(np.linalg.norm(dxyz[:2])) / max_step
                    if lateral_ratio >= 1:
                        self._reset_failure(name, "lateral_deviation",
                                            lateral_error_m=float(np.linalg.norm(dxyz[:2])))
                    z_budget = (max_z_step or max_step) * np.sqrt(1 - lateral_ratio**2)
                    dxyz[2] = np.clip(dxyz[2], -z_budget, z_budget)
                elif max_z_step is not None:
                    from hilserl.motion_limits import clip_translation_delta
                    dxyz = clip_translation_delta(goal[:3] - curr[:3], max_step, max_z_step)
                elif distance > max_step:
                    dxyz *= max_step / distance
                if angle > max_rot_step:
                    drot *= max_rot_step / angle
                nxt = curr.copy()
                nxt[:3] += dxyz
                nxt[3:] = (Rotation.from_rotvec(drot) * Rotation.from_quat(curr[3:])).as_quat()
                return nxt

            curr, pos_err, rot_err = measured()
            adaptive_timeout = min(RESET_MOTION_MAX_TIMEOUT, max(
                float(timeout), pos_err / RESET_MOTION_EFFECTIVE_SPEED + 5.0,
                rot_err / (max_rot_step * float(self.hz) * 0.5) + 1.0,
            ))
            started = time.monotonic()
            deadline = started + adaptive_timeout
            last_progress = started
            best_pos_err, best_rot_err = pos_err, rot_err

            def report(curr, pos_err, rot_err, *, state="moving", force=False):
                return self._reset_progress(name, state=state, force=force,
                    target_pose=goal.tolist(), actual_pose=curr.tolist(),
                    position_error_m=pos_err, rotation_error_rad=rot_err,
                    timeout_seconds=adaptive_timeout, no_progress_seconds=time.monotonic() - last_progress)

            def check_progress(curr, pos_err, rot_err):
                nonlocal last_progress, best_pos_err, best_rot_err
                now = time.monotonic()
                # Net improvement in either component counts, including when
                # orientation converges before translation. Jitter does not.
                if pos_err <= best_pos_err - 0.0005 or rot_err <= best_rot_err - 0.005:
                    best_pos_err = min(best_pos_err, pos_err)
                    best_rot_err = min(best_rot_err, rot_err)
                    last_progress = now
                report(curr, pos_err, rot_err)
                if (now - last_progress >= RESET_STALL_TIMEOUT
                        and (pos_err > RESET_POSITION_TOLERANCE or rot_err > RESET_ROTATION_TOLERANCE)):
                    self._reset_failure(name, "stalled", position_error_m=pos_err,
                                        rotation_error_rad=rot_err,
                                        no_progress_seconds=now - last_progress)

            report(curr, pos_err, rot_err, force=True)
            print(f"[reset_motion] {name} timeout={adaptive_timeout:.1f}s "
                  f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}", flush=True)
            while time.monotonic() < deadline:
                curr, pos_err, rot_err = measured()
                check_progress(curr, pos_err, rot_err)
                if pos_err < pos_tol and rot_err < rot_tol:
                    break
                if (time.monotonic() - last_progress >= RESET_STALL_TIMEOUT
                        and pos_err <= RESET_POSITION_TOLERANCE and rot_err <= RESET_ROTATION_TOLERANCE):
                    break
                self._send_pos_command(bounded_target(curr))
                time.sleep(min(1.0 / self.hz, max(0.0, deadline - time.monotonic())))

            curr, pos_err, rot_err = measured()
            check_progress(curr, pos_err, rot_err)
            exact_goal_allowed = pos_err <= max_step * 1.5 and rot_err <= max_rot_step * 1.5
            if not exact_goal_allowed and (pos_err > RESET_POSITION_TOLERANCE or rot_err > RESET_ROTATION_TOLERANCE):
                self._reset_failure(name, "timeout", position_error_m=pos_err,
                                    rotation_error_rad=rot_err, timeout_seconds=adaptive_timeout)

            # Retain the existing short exact-goal hold only when already nearby.
            hold_deadline = time.monotonic() + (0.5 if exact_goal_allowed else 0.0)
            while time.monotonic() < hold_deadline:
                self._send_pos_command(bounded_target(curr))
                time.sleep(min(1.0 / self.hz, max(0.0, hold_deadline - time.monotonic())))
                curr, pos_err, rot_err = measured()
                check_progress(curr, pos_err, rot_err)

            settle_deadline = time.monotonic() + RESET_MOTION_SETTLE_TIMEOUT
            while True:
                curr, pos_err, rot_err = measured()
                check_progress(curr, pos_err, rot_err)
                if pos_err <= RESET_POSITION_TOLERANCE and rot_err <= RESET_ROTATION_TOLERANCE:
                    report(curr, pos_err, rot_err, state="converged", force=True)
                    print(f"[reset_dbg] {name} target_z={goal[2]:.4f} actual_z={curr[2]:.4f} "
                          f"pos_err={pos_err:.4f} rot_err={rot_err:.4f}", flush=True)
                    return pos_err, rot_err
                if time.monotonic() >= settle_deadline:
                    self._reset_failure(name, "not_converged", position_error_m=pos_err,
                                        rotation_error_rad=rot_err, timeout_seconds=adaptive_timeout)
                self._send_pos_command(bounded_target(curr))
                time.sleep(min(0.2, max(0.0, settle_deadline - time.monotonic())))
        except BaseException as exc:
            if self.last_reset_info.get("state") not in ("failed", "cancelled"):
                state = "cancelled" if type(exc).__name__ in ("StopRequested", "KeyboardInterrupt") else "failed"
                self._reset_progress(name, state=state, force=True, reason=str(exc), error=type(exc).__name__)
            raise

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
            self._robot_post("jointreset")
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
        self._robot_post("update_param", json=self.config.COMPLIANCE_PARAM)
