"""Smoke test: sim/obs/ 可 import。"""


def test_camera_wrapper_importable():
    from sim.obs import camera_wrapper

    # camera_wrapper.py 实际暴露 camera_rgb_to_obs 函数（A2/A4 可能加类）。
    assert hasattr(camera_wrapper, "camera_rgb_to_obs")
