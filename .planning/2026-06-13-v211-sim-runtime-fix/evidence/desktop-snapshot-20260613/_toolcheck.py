import importlib
for m in ["trimesh","manifold3d","numpy","PIL","pygltflib","shapely","mapbox_earcut","scipy"]:
    try:
        mod = importlib.import_module(m)
        print("OK ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("MISS", m, type(e).__name__)
try:
    import trimesh
    a = trimesh.creation.box((1, 1, 1)); b = trimesh.creation.box((0.4, 0.4, 2))
    c = trimesh.boolean.difference([a, b])
    print("BOOLEAN_DIFF_OK verts", len(c.vertices), "watertight", c.is_watertight)
except Exception as e:
    print("BOOLEAN_FAIL", type(e).__name__, str(e)[:140])
