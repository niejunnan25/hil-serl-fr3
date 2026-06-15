# GELLO 跟随路线决策 — 2026-06-15

## 背景
GELLO 笛卡尔速度积分跟随（relative_teleop/gello_demo_recorder 的 gello_twist）真机失败：
摇动大、机器人仅动 2.28cm、3mm 限幅从未触发 → 速度积分增益低 + 用静止机器人 Jacobian 映射本身有损。
用户提示参考 DROID/Pi0.5 的 GELLO 跟随。

## 调查（ultracode workflow + 运行时核查）
- **serl 官方示教** = SpaceMouse 笛卡尔介入（`examples/record_demos.py` + `info["intervene_action"]`），
  与策略同动作空间；非 GELLO、非关节。
- **DROID/gello_software 正解** = Polymetis 关节阻抗 + `update_desired_joint_positions`（关节直接跟随，
  连续夹爪）。但 **laptop（FCI/RT 主机）无 Polymetis**（desktop polymetis-local 有 client，server 必须在
  laptop）→ 路线 A 高成本 + 独立栈 + 与训练 Cartesian 控制器失配。
- **serl 自带 joint_position_controller** 是一次性到位（10s 插值、无流式）→ 路线 D 要改 C++ + rebuild。

## 关键突破：正确 FK
- repo 内 `fk_converter` 错 52cm（DH 错，y 大幅偏）→ 不可用。
- **pinocchio（panda_arm.urdf）FK 正确**：`FK_correct(q)=pinocchio_FK(panda_link8,q) ∘ T_offset`，
  `T_offset=[z 平移 0.10329m, z 旋转 -45°]`（标准 Franka hand F_T_EE，常量）。
  实测 vs server O_T_EE：**位置误差 0.0、四元数完全一致**。
- pinocchio 已装入 hilserl-fr3（py3.10，清代理走清华镜像；`pip install pin`，pin-4.0.0）。

## 决策（用户 2026-06-15 选定）
**Route E + 原计划混合**：
- **Route E（GELLO）**：`FK_correct(q_gello)` 位置式绝对跟随，带 anchor 偏移（接管无跳变、1:1 无损），
  命令 /pose 到**现有 cartesian_impedance 控制器**（与训练同控制器/同动作空间，零失配，无需 Polymetis/无需切栈）。
- **Xbox**：现有笛卡尔介入做精插段。
- **仲裁/接管（用户强调）**：按键切换；**Xbox 激活时 GELLO 完全失效**；切回 GELLO 重锚。
- 复用录制器的 ZED/contract 录制/限幅（3mm/0.1rad）/夹爪。

拒绝：C（真机已死）、A（Polymetis 不在 laptop、高成本、失配）、D（改 C++ rebuild）。
B（serl Xbox 笛卡尔）作为 Xbox 段本就采用。

## 运行时事实
- laptop=robot-HP-ZHAN66，kernel 5.9.1-rt20（RT），serl franka_server 运行中（PID 2640，allow_motion）。
- Polymetis 可用仅在 desktop polymetis-local（py3.8）；laptop 无。
- ZED：外置 2i 36276705→side_policy、腕 ZED-M 13132609→wrist_1（BGR→RGB 已修）。
- 调查代码副本：inv-gello-polymetis/code/（30 文件，分析用，可清理）。
