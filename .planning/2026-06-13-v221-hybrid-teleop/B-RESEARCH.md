# Phase B Research — franka_server 命令契约 grounding（2026-06-13）

来源：fan-out 三源读真实代码（desktop follow 脚本 + franka_server 源 / 本地 repo scripts /
upstream hil-serl）。目的：在 plan/实现前钉死 `/pose` 契约，解决 C2 的 pose/joint 歧义。

## 权威契约（从代码确证）

### 运动命令：只有 `/pose`（绝对位姿）
- `POST /pose`，body `{"arr":[x,y,z,qx,qy,qz,qw]}`（key 字面就是 `arr`，7 个 float）。
- **绝对** EE 平衡位姿；base frame `"0"`；位置米；姿态 **单位四元数 scalar-last** `(qx,qy,qz,qw)`。
- server `move(pose)`（franka_server.py:149-157）发布 `geometry_msgs/PoseStamped` 到 ROS
  `/cartesian_impedance_controller/equilibrium_pose`。是 setpoint，不是 delta。
- **没有关节位置命令端点**（无 `/jointpos`；`/jointreset` 只是 homing）。

### 状态读取
- `POST /getstate {}` → `{pose:[7], q:[7], dq:[7], vel:[6], force:[3], torque:[3], jacobian:6x7, gripper_pos:float}`。
  `pose` 由 `O_T_EE` 推出（EE-in-base, m + quat scalar-last）；`q` 是 7 关节角(rad)。
- 也有 `/getpos`(pose)、`/getpos_euler`(xyz+XYZ欧拉)、`/getq`、`/getdq`、`/getvel`、`/getforce`、
  `/gettorque`、`/getjacobian`、`/get_gripper`。
- 错误恢复 / 控制器：`/clearerr`、`/update_param`(PRECISION/COMPLIANCE)、`/jointreset`、`/startimp`、`/set_load`。
- 夹爪：`/move_gripper {"gripper_pos": int 0-255}`、`/open_gripper`、`/close_gripper`、`/close_gripper_slow`。

### 两条 canonical 命令路径（不同空间）
- **(A) gello_fr3_desktop_follow.py** — 纯 **关节空间**，polymetis gRPC，**不走 HTTP**。
  `target=q0+raw_delta*joint_signs*leader_scale` → 限幅 → `update_desired_joint_positions`。
  `joint_signs=[1,-1,1,1,1,-1,1]`, `leader_scale=0.50`, `max_step=0.003`, `max_total_delta=0.03`。
- **(B) record_gello_demos_serl.py** — **HTTP / 绝对位姿空间**（B1 要镜像的）。
  同样算关节目标 `q0+raw_delta*signs*scale`（serl:301-303，q0 从 `/getstate['q']`），
  然后 `target_pose = forward_kinematics(target)`（fk_converter.py:79-118，DH，含 panda_hand
  flange d=0.1034 + π/4），`send_pose_command(target_pose)` → `POST /pose {"arr": pose}`（serl:169-174）。

### RL 期契约（upstream franka_env，C/D 训练用，与 B 必须一致）
- `env.step(7D action∈[-1,1])`：`/getstate` 读 currpos → `nextpos[:3]+=action[:3]*ACTION_SCALE[0]`；
  `nextpos[3:]=(Rotation.from_rotvec(action[3:6]*ACTION_SCALE[1]) * Rotation.from_quat(currpos[3:])).as_quat()`；
  `clip_safety_box` → `POST /pose {"arr": nextpos}`；每次 POST 前 `POST /clearerr`。
- 例 `ACTION_SCALE`：usb_pickup_insertion `[0.015,0.1,1]`；ram_insertion `[0.01,0.06,1]`。10Hz。
  safety box 任务相关、相对 `TARGET_POSE`。gripper 二值（≤-0.5 关 / ≥0.5 开 + 0.6s 去抖）。
- `GelloCartesianDeltaAgent` 产的是**归一化 [-1,1] delta action**（normalize_action 除以 pos_scale=0.1/
  rpy_scale=0.2），**不能直接发 /pose**——这正是 C2 原 bug（把归一化 delta 当绝对位姿 POST）。
  它喂的是 RL env.step（介入路径），不是 raw follow。

## C2 最终结论
原实现把 `GelloCartesianDeltaAgent` 的归一化 delta 直接 POST 到 `/pose`（期望绝对位姿）→ 失控。
正确路径 = record_gello_demos_serl：**关节目标 → FK → 绝对 /pose `{"arr"}`**。A6 fail-closed 是对的；
B1 用真实契约实现。

## 必须真机现场核对（代码定不了，Phase B live verify）
1. `/healthz` 可能 404（upstream server 未定义；只在 client/test 出现）→ 用 `/getstate` 探活。
   （16 shell 的 healthz 检查接受 200 或 404，故不阻塞。）
2. **payload key**：record 用 `'arr'`（对）；`verify_franka_server.py` 用 `'pose'`（错，会被忽略）。
   权威 = `'arr'`。
3. **FK 偏置**：fk_converter 是 repo 内 DH 近似，与 server 的 `O_T_EE`/flange 约定可能差一个静态
   offset → FK 出的绝对位姿可能有偏，首个命令可能跳。**B1a 加启动 bias 门：`‖FK(q0)[:3]-currpos[:3]‖>阈值` 则拒绝**。
4. 四元数约定（scalar-last，代码内一致）live 确认。
5. `/pose` 是否需先 `/startimp`（impedance 控制器在跑）才生效。
6. fr3 fork 的 config 可能覆盖 ACTION_SCALE / ABS_POSE_LIMIT / COMPLIANCE / gripper 后端（Robotiq vs Franka）。

## verbatim
- `record_gello_demos_serl.py:330-331  target_pose = forward_kinematics(target); robot.send_pose_command(target_pose)`
- `record_gello_demos_serl.py:169-174  url=.../pose; payload={'arr': list(pose)}; requests.post(...)`
- `record_gello_demos_serl.py:148-152  POST /getstate -> state['q'](7,), state['pose'](7,)`
- `franka_server.py:354-357  @route('/pose',POST): pos=np.array(request.json['arr']); robot_server.move(pos)`
- `franka_server.py:149-157  move(pose): assert len==7; PoseStamped frame_id='0'; pos=pose[0:3]; quat=Quaternion(pose[3..6]); eepub.publish`
- `franka_env.py:213-227  nextpos[:3]+=xyz_delta*ACTION_SCALE[0]; nextpos[3:]=(from_rotvec(action[3:6]*ACTION_SCALE[1])*from_quat(currpos[3:])).as_quat(); _send_pos_command(clip_safety_box(nextpos))`
