#!/usr/bin/env python3
"""Author a proper CN 国标 公牛 GN-109K 六口五孔 power-strip USD + its 三脚 plug USD.

Why this exists: open platforms have no properly-licensed CN 五孔 model, and the
desktop base env has no trimesh/boolean. pxr is only importable AFTER a Kit app
launch. So this script launches a HEADLESS Kit app, then uses pxr to author two
real USD assets with PBR materials + physics:

  cn_gn109k_strip.usd : GN-109K body 204x92x29mm, white, 6 RECESSED 五孔 outlets
                        in a 2x3 grid (real recesses: a white frame grid stands
                        proud of recessed gray floors), each outlet carrying the
                        GB 五孔 pattern (三极 品字 ground+L/N over 两极 L/N), a red
                        总控 master switch, and a side power cord. Kinematic (table-
                        fixed) + box collision.
  cn_gn109k_plug.usd  : the strip's own 三脚 (3-pin 品字) tail-cord plug, the
                        graspable object the FR3 re-inserts into a 五孔 outlet
                        (de-energized self-loop). Dynamic rigid body + collision.

Both are authored Z-UP, meters, bottom resting on z=0.

Run on the GPU host:
    /home/robot/IsaacLab/isaaclab.sh -p generate_gn109k_usd.py
"""
import argparse
import math
import os

# --- launch a headless Kit app so `pxr` becomes importable -------------------
from isaaclab.app import AppLauncher

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_args = _p.parse_args(["--headless"])
_app = AppLauncher(_args).app

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- box faces (quads) ----
_FC = [4] * 6
_FI = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 0, 4, 7, 3, 1, 2, 6, 5]


def make_mat(stage, path, rgb, metallic=0.0, rough=0.5):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def add_box(stage, path, center, size, mat, rot_deg=0.0):
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    local = [
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
    ]
    verts = [(cx + x * ca - y * sa, cy + x * sa + y * ca, cz + z) for (x, y, z) in local]
    m = UsdGeom.Mesh.Define(stage, path)
    m.GetPointsAttr().Set([Gf.Vec3f(*v) for v in verts])
    m.GetFaceVertexCountsAttr().Set(_FC)
    m.GetFaceVertexIndicesAttr().Set(_FI)
    m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    if mat is not None:
        UsdShade.MaterialBindingAPI(m.GetPrim()).Bind(mat)
    return m


# ============================== GN-109K strip ===============================
def build_strip(path):
    L, W, H = 0.204, 0.092, 0.029
    d = 0.0045                      # outlet recess depth (frame stands proud by d)
    base_h = H - d
    stage = Usd.Stage.CreateNew(os.path.join(OUT_DIR, path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/GN109K")          # top-level: required for a valid defaultPrim

    M = lambda n, rgb, me=0.0, ro=0.5: make_mat(stage, f"/GN109K/Looks/{n}", rgb, me, ro)
    m_white = M("White", (0.93, 0.93, 0.94), 0.0, 0.28)   # glossy white plastic
    m_floor = M("Recess", (0.22, 0.22, 0.25), 0.0, 0.55)  # darker matte recess floor
    m_slot = M("Slot", (0.02, 0.02, 0.025), 0.0, 0.18)    # near-black glossy slot
    m_red = M("Red", (0.82, 0.09, 0.09), 0.0, 0.28)       # glossy red switch
    m_cord = M("Cord", (0.04, 0.04, 0.04), 0.0, 0.5)

    R = "/GN109K"
    # base (top at z=base_h = recess floor level)
    add_box(stage, f"{R}/Base", (0, 0, base_h / 2), (L, W, base_h), m_white)

    # outlet grid geometry (2 rows x 3 cols)
    ow, oh = 0.044, 0.036
    cols = [-0.056, 0.0, 0.056]
    rows = [0.023, -0.023]
    xr = L / 2 - 0.0                       # body x half-extent
    yr = W / 2
    col_iv = [(c - ow / 2, c + ow / 2) for c in cols]
    row_iv = [(r - oh / 2, r + oh / 2) for r in rows]

    def frame_bar(name, x0, x1, y0, y1):
        add_box(stage, f"{R}/Frame_{name}",
                ((x0 + x1) / 2, (y0 + y1) / 2, base_h + d / 2),
                (x1 - x0, y1 - y0, d), m_white)

    # vertical bars (full Y) at borders + between columns
    xcuts = [(-xr, col_iv[0][0]), (col_iv[0][1], col_iv[1][0]),
             (col_iv[1][1], col_iv[2][0]), (col_iv[2][1], xr)]
    for i, (x0, x1) in enumerate(xcuts):
        if x1 > x0:
            frame_bar(f"V{i}", x0, x1, -yr, yr)
    # horizontal bars within each column x-span (top border / mid divider / bottom border)
    ytop, ymid, ybot = row_iv[0], (row_iv[1][1], row_iv[0][0]), row_iv[1]
    yseg = [("T", ytop[1], yr), ("M", ymid[0], ymid[1]), ("B", -yr, ybot[0])]
    for ci, (xa, xb) in enumerate(col_iv):
        for tag, y0, y1 in yseg:
            if y1 > y0:
                frame_bar(f"H{ci}{tag}", xa, xb, y0, y1)

    # per-outlet: gray recess floor + GB 五孔 slots
    sw, sh_, sd = 0.0015, 0.0063, 0.004   # flat slot
    gw, gh = 0.0018, 0.0072               # ground slot (taller)
    sp = 0.0127
    slot_top = base_h + 0.0014            # ABOVE the gray floor patch (top base_h+0.0008) so 五孔 slots are visible
    idx = 0
    for cy in rows:
        for cx in cols:
            add_box(stage, f"{R}/Floor_{idx}", (cx, cy, base_h + 0.0004),
                    (ow - 0.004, oh - 0.004, 0.0008), m_floor)
            # 三极 品字: ground (top) + two splayed L/N
            add_box(stage, f"{R}/O{idx}_E", (cx, cy + 0.010, slot_top - sd / 2),
                    (gw, gh, sd), m_slot)
            for dx in (-0.0072, 0.0072):                          # 三极 L/N: 八字 splay
                add_box(stage, f"{R}/O{idx}_t{'L' if dx<0 else 'R'}",
                        (cx + dx, cy + 0.002, slot_top - sd / 2), (sw, sh_, sd), m_slot,
                        rot_deg=(16.0 if dx < 0 else -16.0))
            # 两极: two vertical L/N
            for dx in (-sp / 2, sp / 2):
                add_box(stage, f"{R}/O{idx}_b{'L' if dx<0 else 'R'}",
                        (cx + dx, cy - 0.011, slot_top - sd / 2), (sw, sh_, sd), m_slot)
            idx += 1

    # red 总控 master switch (top, +X end beyond outlets)
    add_box(stage, f"{R}/Switch", (0.090, 0.0, H + 0.003), (0.013, 0.030, 0.007), m_red)
    # side power cord stub (-X end); the actual 三脚 plug is a separate USD
    add_box(stage, f"{R}/Cord", (-0.122, 0.0, 0.006), (0.045, 0.008, 0.008), m_cord)

    # physics: table-fixed (kinematic) + box collision on base
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.55)
    col = UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"{R}/Base"))
    col.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(f"{R}/Base")).CreateApproximationAttr("boundingCube")

    root.GetPrim().SetCustomDataByKey("isaac_sim:model", "BULL_GN-109K")
    root.GetPrim().SetCustomDataByKey("isaac_sim:outlets", "6x五孔 universal 2x3")
    stage.SetDefaultPrim(root.GetPrim())          # required: UsdFileCfg references the default prim
    stage.GetRootLayer().Save()
    print(f"[OK] strip -> {os.path.join(OUT_DIR, path)} (prims under {R}, defaultPrim set)", flush=True)


# ============================== 三脚 plug ===================================
def build_plug(path):
    stage = Usd.Stage.CreateNew(os.path.join(OUT_DIR, path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/GN109K_Plug")     # top-level: required for a valid defaultPrim
    R = "/GN109K_Plug"
    m_white = make_mat(stage, "/GN109K_Plug/Looks/PlugWhite", (0.92, 0.92, 0.93), 0.0, 0.28)
    m_metal = make_mat(stage, "/GN109K_Plug/Looks/PlugMetal", (0.70, 0.70, 0.72), 1.0, 0.22)

    # body (Z-up, bottom z=0); blades protrude +Z (insertion = downward when flipped)
    bx, by, bz = 0.034, 0.027, 0.024
    add_box(stage, f"{R}/Body", (0, 0, bz / 2), (bx, by, bz), m_white)
    # 品字 blades on +Z face
    top = bz
    add_box(stage, f"{R}/PinE", (0.0, 0.009, top + 0.008), (0.0032, 0.0016, 0.016), m_metal)   # ground (top, longer)
    for dy in (-0.0064, 0.0064):
        add_box(stage, f"{R}/Pin{'L' if dy<0 else 'R'}", (0.0, dy + (-0.006), top + 0.006),
                (0.0016, 0.0050, 0.013), m_metal)

    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.045)
    col = UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"{R}/Body"))
    col.CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(f"{R}/Body")).CreateApproximationAttr("convexHull")
    root.GetPrim().SetCustomDataByKey("isaac_sim:model", "GN-109K tail 三脚 plug")
    stage.SetDefaultPrim(root.GetPrim())          # required: UsdFileCfg references the default prim
    stage.GetRootLayer().Save()
    print(f"[OK] plug -> {os.path.join(OUT_DIR, path)} (defaultPrim set)", flush=True)


if __name__ == "__main__":
    build_strip("cn_gn109k_strip.usd")
    build_plug("cn_gn109k_plug.usd")
    print("DONE", flush=True)
    import os as _os
    _os._exit(0)
