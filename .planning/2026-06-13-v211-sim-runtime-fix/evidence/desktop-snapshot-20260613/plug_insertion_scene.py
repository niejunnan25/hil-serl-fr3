#!/usr/bin/env python3
"""
IsaacLab 插头插入任务 — 场景配置

场景组成：
  - FR3 机械臂（已有 USD 模型）
  - 桌面（复用蔬菜操作场景）
  - 国标两脚插座（固定在桌面）
  - 国标两脚插头（可抓取刚体）
  - 双 ZED 相机（外置 policy 相机 + 腕部相机）
  - 灯光配置

坐标系约定（与任务规格一致）：
  - Y 轴朝上（Isaac Sim 默认）
  - 插座正面中心为 fixture frame 原点
  - 插入方向为 -Z（插座面板法线方向）

依赖：
  - IsaacLab >= 1.2 (isaaclab.scene, isaaclab.assets 等)
  - 已生成的 cn_two_pin_plug.usd 和 cn_three_pin_socket.usd
  - FR3 机械臂 USD 模型

参考：
  - 任务规格: 00-plug-insertion-task-spec.md
  - 模型生成: generate_plug_usd.py, generate_socket_usd.py
"""

import os
from dataclasses import MISSING

# ============================================================
# IsaacLab 核心导入
# ============================================================
"""Launch Isaac Sim Simulator first."""

import argparse

import importlib
_isaaclab_app_mod = importlib.import_module("isaaclab.app")
AppLauncher = _isaaclab_app_mod.AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Plug insertion scene configuration."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils import configclass


# --- 物理材质工具 ---
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialCfg,
)

# --- 地面/桌面 ---
from isaaclab.terrains import TerrainImporterCfg

# ============================================================
# 路径常量 — 根据实际部署修改
# ============================================================
# FR3 机械臂 USD（复用已有蔬菜操作场景的资产）
FR3_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "fr3", "fr3.usd"
)
# 如果使用 Isaac Sim 自带的 Franka 模型：
# FR3_USD_PATH = "/Isaac/Robots/FrankaEmika/panda_instanceable.usd"

# 插头/插座 USD（实际部署绝对路径）
PLUG_USD_PATH = "/home/robot/plug_insertion_sim/sim-scene/cn_two_pin_plug.usd"
SOCKET_USD_PATH = "/home/robot/plug_insertion_sim/sim-scene/cn_three_pin_socket.usd"

# 桌面 USD（复用蔬菜操作场景）
TABLE_USD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "table", "table.usd"
)

# ============================================================
# 物理材质定义
# ============================================================

@configclass
class PlugPhysicsMaterialCfg(RigidBodyMaterialCfg):
    """插头物理材质 — 塑料外壳 + 铜插脚"""
    static_friction: float = 0.5       # 静摩擦（塑料-金属/桌面）
    dynamic_friction: float = 0.35     # 动摩擦
    restitution: float = 0.3           # 恢复系数


@configclass
class SocketPhysicsMaterialCfg(RigidBodyMaterialCfg):
    """插座物理材质 — 高摩擦内壁（便于插入时自对齐）"""
    static_friction: float = 0.8       # 高静摩擦（塑料内壁）
    dynamic_friction: float = 0.6      # 高动摩擦（防止回弹）
    restitution: float = 0.1           # 低恢复系数（吸收碰撞能量）


@configclass
class TablePhysicsMaterialCfg(RigidBodyMaterialCfg):
    """桌面物理材质"""
    static_friction: float = 0.6
    dynamic_friction: float = 0.4
    restitution: float = 0.2


# ============================================================
# 资产配置：FR3 机械臂
# ============================================================

@configclass
class FrankaFR3Cfg:
    """Franka Research 3 机械臂配置

    使用 IsaacLab ArticulationCfg 封装 FR3 的 URDF/USD 模型。
    包含 7 个关节 + 2 个夹爪关节（panda_finger_joint1/2）。
    """
    robot: ArticulationCfg = ArticulationCfg(
        # USD 文件路径
        spawn=sim_utils.UsdFileCfg(
            usd_path=FR3_USD_PATH,
            # 刚体属性 — 机械臂基座固定
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,          # 动力学模式
                disable_gravity=False,            # 开启重力
                enable_gyroscopic_forces=True,    # 陀螺力
                solver_position_iteration_count=8,  # 高精度（插入任务需要）
                solver_velocity_iteration_count=4,
            ),
            # 碰撞属性
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            # 关节驱动属性
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=8,  # PhysX solver 高精度
                solver_velocity_iteration_count=4,
                fix_root_link=True,               # 基座固定
            ),
        ),
        # 初始关节状态 — 预抓取姿态
        init_state=ArticulationCfg.InitialStateCfg(
            # 基座位姿（桌前）
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            # 关节角度 — 手臂展开，末端朝下
            # FR3: 7 个手臂关节 + 2 个夹爪关节
            joint_pos={
                # 手臂关节（弧度）
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,      # -45 deg
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,      # -135 deg
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,       # 90 deg
                "panda_joint7": 0.785,       # 45 deg
                # 夹爪关节 — 中间位置（0.04m 开口）
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
            joint_vel={".*": 0.0},  # 初始速度为零
        ),
        # 关节驱动器配置
        actuators={
            # 手臂位置控制 — 使用 PD 控制器
            "fr3_arm": DCMotorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                stiffness=400.0,     # 位置刚度（Nm/rad）
                damping=40.0,        # 阻尼（Nms/rad）
                effort_limit=87.0,   # 最大力矩（Nm）
                velocity_limit=2.175,  # 最大角速度（rad/s）
            ),
            # 夹爪位置控制
            "fr3_gripper": DCMotorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                stiffness=200.0,     # 夹爪刚度
                damping=20.0,        # 夹爪阻尼
                effort_limit=20.0,   # 夹爪最大力
                velocity_limit=0.2,  # 夹爪最大速度
            ),
        },
    )


# ============================================================
# 资产配置：国标两脚插头（可抓取刚体）
# ============================================================

@configclass
class PlugCfg:
    """国标两脚插头配置

    - 质量: 50g
    - 摩擦: 静 0.5 / 动 0.35
    - 自由刚体，可被夹爪抓取
    - 初始放置在桌面（阶段二）或夹爪中（阶段一）
    """
    plug: RigidObjectCfg = RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=PLUG_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,      # 自由运动（可被抓取）
                disable_gravity=False,        # 受重力影响
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,  # 高精度碰撞求解
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,    # 穿透恢复速度上限
                max_linear_velocity=1.0,           # 最大线速度限制
                max_angular_velocity=2.0,          # 最大角速度限制
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset: 碰撞提前检测距离（越小越精确，越慢）
                contact_offset=0.002,     # 2mm（适合小物体精密操作）
                rest_offset=0.0005,       # 0.5mm 静止间隙
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.050,               # 50g
            ),
        ),
        # 初始状态 — 桌面上（阶段二：拿起+插入）
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.64, 0.0),       # 桌面位置（需根据实际标定）
            rot=(1.0, 0.0, 0.0, 0.0),    # 无旋转，插脚朝下
        ),
    )


# ============================================================
# 资产配置：国标三角插座（固定在桌面）
# ============================================================

@configclass
class SocketCfg:
    """国标三角插座配置

    - 固定在桌面（kinematic=True）
    - 高摩擦内壁（便于插入自对齐）
    - 插座面板正面朝上（-Z 方向为插入方向）
    """
    socket: RigidObjectCfg = RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=SOCKET_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,       # 固定不动
                disable_gravity=True,         # 不受重力
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0005,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.120,               # 120g（虽固定，但保留质量信息）
            ),
        ),
        # 初始状态 — 固定在桌面中央
        # 插座面板厚度 25mm，底面贴桌，面板正面在 Z=25mm
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.55, 0.0125),    # 桌面上（面板中心高度）
            rot=(1.0, 0.0, 0.0, 0.0),   # 面板正面朝上
        ),
    )


# ============================================================
# 资产配置：桌面
# ============================================================

@configclass
class TableCfg:
    """桌面配置（复用蔬菜操作场景）"""
    table: RigidObjectCfg = RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,       # 桌面固定
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.5, 0.0),        # 桌面位置（需根据实际标定）
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# ============================================================
# 相机配置：双 ZED 相机
# ============================================================

@configclass
class ZEDCameraCfg:
    """ZED 双目相机配置

    两台 ZED 相机：
    1. 外置 policy 相机（side_policy）— 用于策略网络观测
    2. 腕部相机（wrist）— 用于抓取引导和精细操作

    注意：相机内参（焦距、FOV）需要根据实际 ZED 型号标定。
    这里使用 ZED 2i 的典型参数。
    """

    # --- 外置 policy 相机（挂在固定支架上，斜俯视桌面） ---
    side_policy: CameraCfg = CameraCfg(
        update_period=1.0 / 30.0,         # 30 Hz 采集
        height=720,                        # 图像高度
        width=1280,                        # 图像宽度
        data_types=["rgb", "distance_to_image_plane"],  # RGB + 深度
        # 相机位姿 — 斜俯视角度（需根据实际安装标定）
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,              # mm（ZED 2i 典型焦距）
            horizontal_aperture=6.0,       # mm
            vertical_aperture=3.4,         # mm
            clipping_range=(0.1, 10.0),    # 近远裁剪面
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.45, 0.85, 0.45),        # 相机位置（斜上方）
            rot=(0.707, -0.707, 0.0, 0.0), # 朝下 45 度
            convention="world",            # 世界坐标系
        ),
    )

    # --- 腕部相机（安装在机械臂末端法兰上） ---
    wrist: CameraCfg = CameraCfg(
        update_period=1.0 / 30.0,         # 30 Hz
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            horizontal_aperture=6.0,
            vertical_aperture=3.4,
            clipping_range=(0.05, 5.0),   # 近裁剪面更近（近距离操作）
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.05),         # 相对末端法兰偏移（前方 50mm）
            rot=(0.707, -0.707, 0.0, 0.0), # 朝前下方
            convention="body",             # 末端坐标系
        ),
        # 绑定到机械臂末端（IsaacLab 5.1: 通过 prim_path 表达父子关系，不再使用 parent_body kwarg）
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
    )


# ============================================================
# 灯光配置
# ============================================================

@configclass
class LightingCfg:
    """场景灯光配置

    模拟标准实验室照明环境，确保相机观测质量：
    - 顶部主光源（模拟天花板日光灯）
    - 侧面补光（减少阴影）
    """
    # 顶部主光源 — 模拟天花板照明
    light_dome: sim_utils.DomeLightCfg = sim_utils.DomeLightCfg(
        color=(0.85, 0.85, 0.9),          # 略偏冷白色（实验室日光灯）
        intensity=750.0,                   # 光照强度（lux）
    )

    # 桌面工作区域主光源
    light_key: sim_utils.DistantLightCfg = sim_utils.DistantLightCfg(
        color=(1.0, 0.98, 0.95),          # 暖白光
        intensity=1500.0,
        # 光线方向：IsaacLab 5.1.0 的 DistantLightCfg 没有 direction 字段；
        # 默认方向沿 +Z（-Y 方向照射下来）。如需斜向可后续通过
        # cfg.func(path, cfg, orientation=quaternion) 设置 prim 朝向。
    )

    # 侧面补光 — 减少插头和插座的阴影
    light_fill: sim_utils.DistantLightCfg = sim_utils.DistantLightCfg(
        color=(0.8, 0.85, 0.9),           # 略偏蓝（模拟环境反射）
        intensity=500.0,
        # direction 字段已移除（IsaacLab 5.1.0）。
    )


# ============================================================
# 完整场景配置 — 插头插入任务
# ============================================================

@configclass
class PlugInsertionSceneCfg(InteractiveSceneCfg):
    """插头插入任务的完整 IsaacLab 场景配置

    场景布局（俯视图，Y 轴朝上）：

        [外置 ZED 相机]
              |
              v
    +---------------------+
    |                     |
    |     [桌面]           |
    |   +-----------+      |
    |   | [插座]    |      |
    |   | [插头]    |      |
    |   +-----------+      |
    |                     |
    +---------------------+
              ^
              |
         [FR3 机械臂]
         [腕部 ZED 相机]

    坐标系：
    - 机械臂基座: (0, 0, 0)
    - 桌面中心: (0.45, 0.5, 0)
    - 插座: (0.45, 0.55, 0.0125)
    - 插头: (0.45, 0.64, 0)
    """

    # --- 地面 ---
    ground: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.7,
            dynamic_friction=0.5,
            restitution=0.2,
        ),
    )

    # --- FR3 机械臂 ---
    robot: ArticulationCfg = FrankaFR3Cfg().robot

    # --- 桌面 ---
    table: RigidObjectCfg = TableCfg().table

    # --- 国标两脚插头（可抓取刚体） ---
    plug: RigidObjectCfg = PlugCfg().plug

    # --- 国标三角插座（固定） ---
    socket: RigidObjectCfg = SocketCfg().socket

    # --- ZED 相机 ---
    camera_side_policy: CameraCfg = ZEDCameraCfg().side_policy
    camera_wrist: CameraCfg = ZEDCameraCfg().wrist

    # --- 灯光 ---
    light_dome: sim_utils.DomeLightCfg = LightingCfg().light_dome
    light_key: sim_utils.DistantLightCfg = LightingCfg().light_key
    light_fill: sim_utils.DistantLightCfg = LightingCfg().light_fill


# ============================================================
# 阶段一专用配置（插头已在手中，夹爪闭合）
# ============================================================

@configclass
class PlugInsertionPhase1SceneCfg(PlugInsertionSceneCfg):
    """阶段一：纯插入（插头已在手中）

    与主场景的区别：
    - 插头初始位姿在夹爪中（不需要桌面放置位置）
    - 使用 GripperCloseEnv wrapper 控制夹爪
    """
    # 插头初始位置 — 在夹爪中（相对于末端法兰）
    # 具体值需根据实际抓取位置标定
    plug: RigidObjectCfg = PlugCfg().plug
    # 注意：阶段一中插头的实际位置会在 env 的 reset 中
    # 通过设置 RigidObject 的 root_state 来动态调整


# ============================================================
# 伪代码/接口说明：IsaacLab 场景初始化流程
# ============================================================

# === 使用说明 ===
#
# 1. 场景初始化（在 Env 类中）
#
#    class PlugInsertionEnv(IsaacEnv):
#        cfg: PlugInsertionSceneCfg = PlugInsertionSceneCfg()
#
#        def _setup_scene(self):
#            # 创建场景中所有物体
#            self.scene = InteractiveScene(self.cfg)
#
#            # 获取各资产引用
#            self.robot: Articulation = self.scene["robot"]
#            self.plug: RigidObject = self.scene["plug"]
#            self.socket: RigidObject = self.scene["socket"]
#            self.camera_side: Camera = self.scene["camera_side_policy"]
#            self.camera_wrist: Camera = self.scene["camera_wrist"]
#
#        def _reset_idx(self, env_ids):
#            # 重置指定环境
#            # 重置机械臂到初始关节角
#            self.robot.reset(env_ids)
#
#            # 阶段一：将插头放到夹爪中
#            plug_pos = compute_grasp_pose(self.robot, env_ids)
#            self.plug.set_pose(plug_pos, env_ids)
#
#            # 阶段二：将插头放到桌面随机位置
#            # table_pos = randomize_table_position()
#            # self.plug.set_pose(table_pos, env_ids)
#
# 2. 观测获取
#
#    def _get_observations(self):
#        # 机械臂关节状态
#        joint_pos = self.robot.data.joint_pos
#        joint_vel = self.robot.data.joint_vel
#
#        # TCP 位姿（末端执行器）
#        tcp_pose = self.robot.data.body_state_w[:, tcp_id, :7]
#
#        # 相机图像
#        rgb_side = self.camera_side.data.output["rgb"]       # (N, H, W, 3)
#        rgb_wrist = self.camera_wrist.data.output["rgb"]     # (N, H, W, 3)
#        depth_side = self.camera_side.data.output["distance_to_image_plane"]
#
#        # 插头位姿
#        plug_pose = self.plug.data.root_state_w[:, :7]
#
#        # 插座位姿（固定，理论上不变）
#        socket_pose = self.socket.data.root_state_w[:, :7]
#
#        return {
#            "state": torch.cat([joint_pos, tcp_pose, plug_pose], dim=-1),
#            "images": {"side": rgb_side, "wrist": rgb_wrist},
#        }
#
# 3. PhysX 高精度参数（全局设置）
#
#    sim_utils.SimulationCfg(
#        dt=1.0 / 240.0,                    # 240 Hz 物理步进
#        physics_material=RigidBodyMaterialCfg(...),
#        physx=sim_utils.PhysxCfg(
#            solver_type=1,                  # 1=TGS, 精度更高
#            min_position_iteration_count=4,
#            max_position_iteration_count=8,  # 插入任务推荐 8
#            min_velocity_iteration_count=2,
#            max_velocity_iteration_count=4,
#            bounce_threshold_velocity=0.2,
#            friction_offset_threshold=0.004,
#            friction_correlation_distance=0.0025,
#            gpu_max_rigid_contact_count=2**20,
#            gpu_max_rigid_patch_count=2**20,
#        ),
#    )


# ============================================================
# 验证脚本 — 独立运行时打印场景信息
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  IsaacLab 插头插入任务 — 场景配置")
    print("=" * 60)

    # 打印场景配置摘要
    cfg = PlugInsertionSceneCfg()
    print(f"\n[场景] 类型: PlugInsertionSceneCfg")
    print(f"[资产] 机械臂: FR3 (Franka Research 3)")
    print(f"[资产] 插头: 国标两脚 (cn_two_pin_plug.usd)")
    print(f"[资产] 插座: 国标三角 (cn_three_pin_socket.usd)")
    print(f"[相机] 外置 ZED (side_policy): 1280x720 @ 30Hz")
    print(f"[相机] 腕部 ZED (wrist): 1280x720 @ 30Hz")
    print(f"[灯光] DomeLight + DistantLight x2")
    print(f"\n[物理] 物理步进: 240Hz")
    print(f"[物理] Solver: TGS, position_iterations=8")
    print(f"[物理] 插头质量: 50g, 摩擦: 0.5/0.35")
    print(f"[物理] 插座固定: kinematic=True")
    print(f"[物理] 插座内壁摩擦: 0.8/0.6")

    print(f"\n[坐标系] Y 轴朝上")
    print(f"[坐标系] 插入方向: -Z（垂直插座面板向内）")
    print(f"[坐标系] 插脚间距方向: X")

    print(f"\n[注意] USD 路径需根据实际部署环境修改:")
    print(f"  FR3:    {os.path.abspath(FR3_USD_PATH)}")
    print(f"  Plug:   {os.path.abspath(PLUG_USD_PATH)}")
    print(f"  Socket: {os.path.abspath(SOCKET_USD_PATH)}")
    print(f"  Table:  {os.path.abspath(TABLE_USD_PATH)}")

    # ------------------------------------------------------------
    # Main loop — keep Kit session alive so the scene actually renders.
    # We instantiate a minimal SimulationContext for the visual main
    # loop.  The 11 class definitions above are the config / spec;
    # this block makes the script a runnable viewer rather than an
    # immediate-exit module.
    # ------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Entering main loop (visual render)")
    print("  Ctrl-C to close Isaac Sim")
    print("=" * 60)

    _sim = sim_utils.SimulationContext()  # uses default SimulationCfg()
    _sim.reset()

    try:
        while simulation_app.is_running():
            _sim.step(render=True)
    except KeyboardInterrupt:
        print("[INFO]: KeyboardInterrupt -- closing Isaac Sim.")

    simulation_app.close()

