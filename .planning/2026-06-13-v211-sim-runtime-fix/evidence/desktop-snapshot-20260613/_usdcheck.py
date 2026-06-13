import argparse, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
app = AppLauncher(p.parse_args(["--headless"])).app
from pxr import Usd, UsdGeom
path = sys.argv[1]
stage = Usd.Stage.Open(path)
dp = stage.GetDefaultPrim()
print("defaultPrim:", dp.GetPath(), "valid:", bool(dp))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
b = cache.ComputeWorldBound(dp).ComputeAlignedRange()
print("bbox min:", tuple(round(v,4) for v in b.GetMin()), "max:", tuple(round(v,4) for v in b.GetMax()))
nmesh = sum(1 for pr in stage.Traverse() if pr.GetTypeName() == "Mesh")
ngeom = sum(1 for pr in stage.Traverse() if pr.GetTypeName() in ("Mesh","Cube","Xform","Material","Shader"))
print("mesh prims:", nmesh, "| sample paths:", [str(pr.GetPath()) for pr in stage.Traverse() if pr.GetTypeName()=="Mesh"][:3])
import os; os._exit(0)
