import sys
import numpy as np
import pyzed.sl as sl
from PIL import Image

serial = int(sys.argv[1]) if len(sys.argv) > 1 else 36276705
out = sys.argv[2] if len(sys.argv) > 2 else "/home/robot/plug_insertion_sim/sim-scene/_zed_real.png"

cam = sl.Camera()
init = sl.InitParameters()
init.set_from_serial_number(serial)
init.camera_resolution = sl.RESOLUTION.HD1080
init.depth_mode = sl.DEPTH_MODE.NONE
err = cam.open(init)
print("open:", err, flush=True)
if err != sl.ERROR_CODE.SUCCESS:
    print("OPEN_FAIL", flush=True); sys.exit(2)
rt = sl.RuntimeParameters()
m = sl.Mat()
ok = 0
for _ in range(60):                       # let auto-exposure settle
    if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(m, sl.VIEW.LEFT); ok += 1
arr = m.get_data()                        # H x W x 4 BGRA
print("grabbed", ok, "shape", arr.shape, flush=True)
Image.fromarray(np.ascontiguousarray(arr[:, :, [2, 1, 0]])).save(out)
print("SAVED", out, flush=True)
cam.close()
