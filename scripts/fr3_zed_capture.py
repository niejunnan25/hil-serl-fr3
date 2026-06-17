#!/usr/bin/env python3
"""SERL-style ZED camera adapter for the FR3 desktop vision side.

Interface matches the simple SERL capture convention: read() returns
(success: bool, image: np.ndarray), close() releases the camera.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyzed.sl as sl


def _resolve_resolution(name: str) -> sl.RESOLUTION:
    key = name.upper()
    if not hasattr(sl.RESOLUTION, key):
        raise ValueError(f"unknown ZED resolution {name!r}")
    return getattr(sl.RESOLUTION, key)


@dataclass(frozen=True)
class ZEDCaptureConfig:
    serial_number: int = 36276705
    resolution: str = "HD720"
    fps: int = 15
    view: str = "LEFT"
    channels: str = "RGB"
    depth: bool = False
    grab_retries: int = 30
    retry_sleep_s: float = 0.05
    open_retries: int = 5          # ZED self-calib intermittently returns an open error
                                   # ("POTENTIAL CALIBRATION ISSUE") that clears on retry


class ZEDCapture:
    def __init__(self, config: Optional[ZEDCaptureConfig] = None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("pass either config or keyword overrides, not both")
        self.config = config or ZEDCaptureConfig(**kwargs)
        self._camera = sl.Camera()
        self._image = sl.Mat()
        self._is_open = False
        self._open()

    @staticmethod
    def get_device_serial_numbers() -> list[int]:
        return [int(getattr(d, "serial_number", 0) or 0) for d in sl.Camera.get_device_list()]

    def _open(self) -> None:
        init = sl.InitParameters()
        init.camera_resolution = _resolve_resolution(self.config.resolution)
        init.camera_fps = int(self.config.fps)
        init.depth_mode = sl.DEPTH_MODE.NONE if not self.config.depth else sl.DEPTH_MODE.NEURAL
        init.set_from_serial_number(int(self.config.serial_number))
        err = None
        for _ in range(max(1, int(self.config.open_retries))):
            err = self._camera.open(init)
            if err == sl.ERROR_CODE.SUCCESS:
                self._is_open = True
                return
            # Transient open failures (POTENTIAL CALIBRATION ISSUE from self-calib on an
            # occluded/low-texture scene) clear on retry — close + wait + try again.
            try:
                self._camera.close()
            except Exception:
                pass
            time.sleep(0.7)
        raise RuntimeError(f"failed to open ZED serial {self.config.serial_number}: {err}")

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._is_open:
            return False, np.empty((0, 0, 0), dtype=np.uint8)
        grab_err = None
        for _ in range(int(self.config.grab_retries)):
            grab_err = self._camera.grab()
            if grab_err == sl.ERROR_CODE.SUCCESS:
                break
            time.sleep(float(self.config.retry_sleep_s))
        if grab_err != sl.ERROR_CODE.SUCCESS:
            return False, np.empty((0, 0, 0), dtype=np.uint8)
        view = getattr(sl.VIEW, self.config.view.upper())
        self._camera.retrieve_image(self._image, view)
        frame = self._image.get_data()
        if self.config.channels.upper() == "RGB" and frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3].copy()
        else:
            frame = frame.copy()
        return True, frame

    def close(self) -> None:
        if self._is_open:
            self._camera.close()
            self._is_open = False

    def __enter__(self) -> "ZEDCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=int, default=36276705)
    parser.add_argument("--resolution", default="HD720")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--view", default="LEFT")
    parser.add_argument("--channels", default="RGB", choices=("RGB", "RGBA"))
    args = parser.parse_args()
    print("devices", ZEDCapture.get_device_serial_numbers())
    with ZEDCapture(serial_number=args.serial, resolution=args.resolution, fps=args.fps, view=args.view, channels=args.channels) as cap:
        ok, frame = cap.read()
        print("read", ok, "shape", tuple(frame.shape), "dtype", str(frame.dtype))
        return 0 if ok and frame.size else 1


if __name__ == "__main__":
    raise SystemExit(main())
