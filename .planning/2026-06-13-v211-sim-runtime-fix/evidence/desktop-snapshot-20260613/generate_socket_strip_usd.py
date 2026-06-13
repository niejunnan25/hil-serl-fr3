#!/usr/bin/env python3
"""国标 公牛/红牛 六口排插 (power strip) USD 生成器.

真实硬件: 中国国标 公牛(BULL) 风格六口排插, 固定在桌面.
  - 6 个插孔: 左 3 个 "双口"(二极, 2 扁孔) + 右 3 个 "三口"(三极, 2 扁孔 + 1 接地).
  - 一个红色电源开关 (公牛标志性红色 rocker).
  - 自带电源线 + 三脚墙插 (插座本身线位三口).
  - 整条排插平放桌面, 插孔朝上 (+Z), 插头从上方 -Z 插入.

坐标系: 直接建成 Z-UP (IsaacLab/USD 默认), 排插底面贴 z=0, 插孔在顶面 +Z.
所以 viewer 里 spawn 时不需要 Y-up->Z-up 旋转 (orientation 用单位四元数).

几何: 不做布尔挖孔 (trimesh 布尔后端在 desktop 不稳); 改用 "深色薄盒嵌在顶面"
表示插孔 —— 视觉上就是顶面上的暗色凹槽, 在相机里清晰可辨.

依赖: pxr (USD Python API), isaaclab conda 环境自带.
"""
import os
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

# ----------------------------------------------------------------------------
# 尺寸 (米) — 公牛六口排插实测量级
# ----------------------------------------------------------------------------
BODY_L = 0.270   # 长 (X)
BODY_W = 0.050   # 宽 (Y)
BODY_H = 0.028   # 高 (Z)

OUTLET_PATCH = (0.034, 0.040, 0.0015)   # 每个插孔面板凹区 (X,Y, 微凸厚度)
N_OUTLETS = 6
OUTLET_X = np.linspace(-0.092, 0.092, N_OUTLETS)  # 6 孔位中心 X, 间距 ~36.8mm

# 扁孔 (火/零线): 1.5 x 6.3 mm; 间距 12.7mm
SLOT_W, SLOT_H, SLOT_D = 0.0015, 0.0063, 0.006
SLOT_SPACING = 0.0127
# 接地孔 (三口顶部): 略宽一点的扁孔
GND_W, GND_H = 0.0018, 0.007

SWITCH_X = 0.122                          # 红色开关在 +X 端
CORD_X = -0.135                           # 电源线从 -X 端引出

TOP_Z = BODY_H                            # 顶面世界 z
COL_BODY = (0.90, 0.90, 0.92)            # 排插本体白
COL_PATCH = (0.82, 0.82, 0.85)           # 插孔面板区 (略深的白)
COL_SLOT = (0.04, 0.04, 0.05)            # 插孔暗槽
COL_SWITCH = (0.85, 0.10, 0.10)          # 红色开关 (公牛/"红牛")
COL_CORD = (0.05, 0.05, 0.05)            # 电源线黑
COL_WALLPLUG = (0.90, 0.90, 0.92)        # 墙插白

# 24-tri unit cube faces (quads)
_FACE_COUNTS = [4] * 6
_FACE_INDICES = [
    0, 3, 2, 1,  4, 5, 6, 7,  0, 1, 5, 4,
    2, 3, 7, 6,  0, 4, 7, 3,  1, 2, 6, 5,
]


def add_box(stage, path, center, extents, color):
    """Define a colored box Mesh at `center` with full `extents` (X,Y,Z)."""
    cx, cy, cz = center
    hx, hy, hz = extents[0] / 2, extents[1] / 2, extents[2] / 2
    verts = [
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*v) for v in verts])
    mesh.GetFaceVertexCountsAttr().Set(_FACE_COUNTS)
    mesh.GetFaceVertexIndicesAttr().Set(_FACE_INDICES)
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    return mesh


def build(output_path):
    stage = Usd.Stage.CreateNew(output_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)   # Z-up
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World/CN_SixOutlet_Strip")

    # --- body (bottom at z=0) ---
    add_box(stage, "/World/CN_SixOutlet_Strip/Body",
            (0.0, 0.0, BODY_H / 2), (BODY_L, BODY_W, BODY_H), COL_BODY)

    layout = []   # (kind, x)
    for i, x in enumerate(OUTLET_X):
        kind = "double" if i < 3 else "triple"   # 左3双口, 右3三口
        layout.append((kind, float(x)))
        # 插孔面板凹区: 微凸薄片, 顶面略高于本体顶面
        add_box(stage, f"/World/CN_SixOutlet_Strip/Patch_{i}",
                (x, 0.0, TOP_Z + OUTLET_PATCH[2] / 2),
                OUTLET_PATCH, COL_PATCH)
        # 暗槽顶面与本体顶面齐平, 向下嵌入 -> 视觉凹孔
        sz = TOP_Z - SLOT_D / 2 + 0.0008
        if kind == "double":
            # 二极: 两条竖直扁孔, 间距 12.7mm
            for dx in (-SLOT_SPACING / 2, SLOT_SPACING / 2):
                add_box(stage, f"/World/CN_SixOutlet_Strip/O{i}_slot_{'L' if dx<0 else 'R'}",
                        (x + dx, 0.0, sz), (SLOT_W, SLOT_H, SLOT_D), COL_SLOT)
        else:
            # 三极: 顶部接地 + 下方两条扁孔 (品字)
            add_box(stage, f"/World/CN_SixOutlet_Strip/O{i}_gnd",
                    (x, 0.009, sz), (GND_W, GND_H, SLOT_D), COL_SLOT)
            for dx in (-SLOT_SPACING / 2, SLOT_SPACING / 2):
                add_box(stage, f"/World/CN_SixOutlet_Strip/O{i}_slot_{'L' if dx<0 else 'R'}",
                        (x + dx, -0.004, sz), (SLOT_W, SLOT_H, SLOT_D), COL_SLOT)

    # --- 红色电源开关 (顶面 +X 端) ---
    add_box(stage, "/World/CN_SixOutlet_Strip/Switch",
            (SWITCH_X, 0.0, TOP_Z + 0.004), (0.016, 0.024, 0.008), COL_SWITCH)

    # --- 电源线 (黑, 从 -X 端引出) + 墙插 ---
    add_box(stage, "/World/CN_SixOutlet_Strip/Cord",
            (CORD_X - 0.03, 0.0, 0.006), (0.06, 0.008, 0.008), COL_CORD)
    add_box(stage, "/World/CN_SixOutlet_Strip/WallPlug",
            (CORD_X - 0.075, 0.0, 0.010), (0.03, 0.024, 0.020), COL_WALLPLUG)
    # 墙插 3 脚 (金属灰)
    for dy in (-0.0095, 0.0095):
        add_box(stage, f"/World/CN_SixOutlet_Strip/WallPin_{'L' if dy<0 else 'R'}",
                (CORD_X - 0.095, dy, 0.010), (0.012, 0.0016, 0.006), (0.6, 0.6, 0.62))
    add_box(stage, "/World/CN_SixOutlet_Strip/WallPin_G",
            (CORD_X - 0.095, 0.0, 0.018), (0.012, 0.0016, 0.006), (0.6, 0.6, 0.62))

    # --- 物理: 固定 (kinematic), 本体碰撞 ---
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.30)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/CN_SixOutlet_Strip/Body"))

    root.GetPrim().SetCustomDataByKey("isaac_sim:asset_type", "power_strip")
    root.GetPrim().SetCustomDataByKey("isaac_sim:brand", "BULL_style_red_switch")
    root.GetPrim().SetCustomDataByKey("isaac_sim:standard", "GB/T 1002-2021")
    root.GetPrim().SetCustomDataByKey("isaac_sim:up_axis", "Z")

    stage.GetRootLayer().Save()

    meta = {
        "model_name": "CN_SixOutlet_PowerStrip",
        "brand_style": "BULL/公牛 (red switch)",
        "standard": "GB/T 1002-2021",
        "units": "meters", "up_axis": "Z",
        "body_mm": [BODY_L * 1000, BODY_W * 1000, BODY_H * 1000],
        "outlets": {"count": N_OUTLETS, "double_pin": 3, "triple_pin": 3,
                    "x_centers_mm": [round(x * 1000, 1) for x in OUTLET_X],
                    "layout": layout},
        "slot_mm": {"width": SLOT_W * 1000, "height": SLOT_H * 1000, "spacing": SLOT_SPACING * 1000},
        "features": ["red_power_switch", "own_3pin_wall_cord", "table_fixed_kinematic"],
        "insertion_axis": "Z_negative (plug inserts downward into top face)",
        "top_world_z_m": TOP_Z,
    }
    mpath = output_path.replace(".usd", "_metadata.json")
    json.dump(meta, open(mpath, "w"), ensure_ascii=False, indent=2)
    print(f"[OK] saved {output_path}")
    print(f"[OK] saved {mpath}")
    print(f"[layout] {layout}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cn_six_outlet_strip.usd")
    build(out)
