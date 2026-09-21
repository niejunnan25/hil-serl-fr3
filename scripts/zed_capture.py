#!/usr/bin/env python3
"""
zed_capture.py – ZED camera adapter for SERL's FrankaEnv framework.

Provides ZEDCapture, a drop-in camera class compatible with the RSCapture
interface that SERL uses.  Wraps the ZED SDK (pyzed) to grab frames from
ZED 2i and ZED-M cameras.

Cameras in hilserl-fr3:
  - ZED 2i external : serial 36276705
  - ZED-M wrist     : serial 13132609

Usage:
    # Quick connection test
    python scripts/zed_capture.py --serial 36276705

    # In code
    from scripts.zed_capture import ZEDCapture
    with ZEDCapture("external", "36276705") as cam:
        ok, frame = cam.read()   # BGR uint8, shape (720, 1280, 3)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None  # type: ignore[assignment]


ZED_EXPOSURE_AUTO = -1
ZED_EXPOSURE_MIN = 0
ZED_EXPOSURE_MAX = 100


def validate_zed_exposure(exposure: Optional[int]) -> Optional[int]:
    """Validate a ZED SDK exposure value.

    ZED uses -1 for auto exposure, otherwise an integer percentage in [0, 100].
    Values like 10500/13000 are RealSense-style microseconds and must not be
    forwarded to the SDK.
    """
    if exposure is None:
        return None
    exposure = int(exposure)
    if exposure == ZED_EXPOSURE_AUTO or ZED_EXPOSURE_MIN <= exposure <= ZED_EXPOSURE_MAX:
        return exposure
    raise ValueError(
        "ZED exposure must be -1 for auto or an integer in [0, 100]; "
        f"got {exposure}. Do not pass RealSense-style microsecond values."
    )


def _resolution_enum(dim: tuple[int, int]):
    """Map (width, height) to the closest sl.RESOLUTION constant."""
    if sl is None:
        raise ImportError("pyzed is not installed")
    w, h = dim
    mapping: dict[tuple[int, int], object] = {
        (672, 376): sl.RESOLUTION.VGA,
        (1280, 720): sl.RESOLUTION.HD720,
        (1920, 1080): sl.RESOLUTION.HD1080,
        (2208, 1242): sl.RESOLUTION.HD2K,
    }
    return mapping.get((w, h), sl.RESOLUTION.HD720)


class ZEDCapture:
    """ZED camera capture compatible with SERL's RSCapture interface.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "external", "wrist").
    serial_number : str
        ZED camera serial number as a string (e.g. "36276705").
    dim : tuple[int, int]
        Requested resolution as (width, height).  Default (1280, 720).
    fps : int
        Requested frame rate.  Default 30.
    """

    def __init__(
        self,
        name: str,
        serial_number: str,
        dim: tuple[int, int] = (1280, 720),
        fps: int = 30,
        exposure: Optional[int] = None,
    ) -> None:
        if sl is None:
            raise ImportError(
                "pyzed is not installed.  Install with:  pip install pyzed"
            )

        self.name = name
        self.serial_number = serial_number
        self.dim = dim
        self.fps = fps
        self.exposure = validate_zed_exposure(exposure)
        self._cam: Optional[sl.Camera] = None
        self._runtime_params: Optional[sl.RuntimeParameters] = None

        self._open()

        # Set exposure if provided (after camera is opened)
        if self.exposure is not None and self._cam is not None:
            status = self._cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, self.exposure)
            if status is not None and status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(
                    f"ZED camera '{self.name}' failed to set exposure "
                    f"{self.exposure}: {status}"
                )

    # ── public API (matches RSCapture) ─────────────────────────────────────

    def read(self) -> tuple[bool, np.ndarray]:
        """Grab one frame from the camera.

        Returns
        -------
        ok : bool
            True if a frame was successfully captured.
        frame : np.ndarray
            BGR uint8 image of shape (H, W, 3).  Empty array on failure.
        """
        if self._cam is None:
            return False, np.empty(0, dtype=np.uint8)

        grab_status = self._cam.grab(self._runtime_params)
        if grab_status != sl.ERROR_CODE.SUCCESS:
            return False, np.empty(0, dtype=np.uint8)

        # Retrieve left-eye image
        zed_mat = sl.Mat()
        self._cam.retrieve_image(zed_mat, sl.VIEW.LEFT)

        # Convert to numpy — ZED SDK returns BGRA by default
        bgra = zed_mat.get_data()  # shape (H, W, 4), dtype uint8
        if bgra is None or bgra.size == 0:
            return False, np.empty(0, dtype=np.uint8)

        # BGRA → BGR (drop alpha channel)
        # Note: ZED SDK returns BGRA format
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return True, bgr

    def close(self) -> None:
        """Release the camera."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None
            self._runtime_params = None

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self) -> ZEDCapture:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── internals ──────────────────────────────────────────────────────────

    def _open(self) -> None:
        """Initialize and open the ZED camera."""
        cam = sl.Camera()

        init_params = sl.InitParameters()
        init_params.set_from_serial_number(int(self.serial_number))
        init_params.camera_resolution = _resolution_enum(self.dim)
        init_params.camera_fps = self.fps
        # Disable verbose SDK logging to keep output clean
        init_params.sdk_verbose = 0

        status = cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            if status == sl.ERROR_CODE.CAMERA_NOT_DETECTED:
                raise RuntimeError(
                    f"ZED camera '{self.name}' (serial {self.serial_number}) "
                    f"not detected.  Check USB connection."
                )
            if status == sl.ERROR_CODE.INVALID_RESOLUTION:
                raise RuntimeError(
                    f"ZED camera '{self.name}': invalid resolution {self.dim}."
                )
            already_opened = getattr(sl.ERROR_CODE, "CAMERA_ALREADY_OPENED", None)
            if already_opened is not None and status == already_opened:
                raise RuntimeError(
                    f"ZED camera '{self.name}' (serial {self.serial_number}) "
                    f"already opened by another process."
                )
            raise RuntimeError(
                f"ZED camera '{self.name}' (serial {self.serial_number}) "
                f"failed to open: {status}"
            )

        self._cam = cam
        self._runtime_params = sl.RuntimeParameters()

    def __repr__(self) -> str:
        state = "open" if self._cam is not None else "closed"
        return (
            f"ZEDCapture(name={self.name!r}, serial={self.serial_number!r}, "
            f"dim={self.dim}, fps={self.fps}, state={state})"
        )


# ── CLI entry point ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ZED camera connection and read one frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--serial", required=True, help="ZED camera serial number (e.g. 36276705)"
    )
    parser.add_argument(
        "--name", default="test", help="Camera label (default: test)"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Image width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Image height (default: 720)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frame rate (default: 30)"
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        default=None,
        help="Save captured frame to this path (e.g. test.png)",
    )
    args = parser.parse_args()

    print(f"Opening ZED camera '{args.name}' (serial {args.serial})...")
    try:
        with ZEDCapture(
            name=args.name,
            serial_number=args.serial,
            dim=(args.width, args.height),
            fps=args.fps,
        ) as cam:
            print(f"  {cam}")
            print("Reading frame...")
            ok, frame = cam.read()

            if not ok:
                print("  FAILED — could not grab frame.", file=sys.stderr)
                sys.exit(1)

            h, w = frame.shape[:2]
            print(f"  OK — frame shape: {w}x{h}, dtype: {frame.dtype}")

            if args.save:
                cv2.imwrite(args.save, frame)
                print(f"  Saved to {args.save}")

    except ImportError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
