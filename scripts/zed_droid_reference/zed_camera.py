"""Local reference port of DROID `scripts/common/zed_camera.py`.

Goals
-----
- Preserve the DROID API surface (``ZedCamera(serial_number | device_path,
  resolution, fps)`` + ``capture_frame`` + ``close`` + ``show_live_preview``)
  so we can diff behavior against the upstream DROID repo on
  ``fr3-desktop-ts:/home/robot/droid`` during audits.
- Strip the DROID-only ``from common.utils import crop_left_right`` import.
  We keep the same call signature in this module (PIL.Image in/out) so the
  reference still exercises the same crop contract, but we provide a local
  pure-Pillow implementation that does not require a ROS/Franka install.
- Fix the upstream ``__main__`` NameError: in
  ``/home/robot/droid/scripts/common/zed_camera.py`` the wrist camera is
  bound to ``zed_wrist_camera`` while ``show_live_preview`` is then called
  with the undefined name ``zed_camera``. Either edit the comments manually
  or hit ``NameError`` at import. We resolve the bug by binding the same
  name we pass to ``show_live_preview``.
- No motion: this module is for static reference, not live capture. All
  tests mock ``pyzed.sl`` so no real camera is ever opened.
"""

import copy
import os

import cv2
import numpy as np

# ``pyzed.sl`` is the only DROID-only native dependency. The import is
# wrapped so that the module can be imported on CI (where the SDK is not
# installed) and so that the unit tests can inject a fake ``sl`` namespace
# before the real module is ever touched.
try:
    import pyzed.sl as sl  # type: ignore
except ImportError:  # pragma: no cover - exercised by tests with a fake ``sl``
    sl = None  # type: ignore[assignment]


__all__ = [
    "ZedCamera",
    "show_live_preview",
    "crop_left_right",
    "_RESOLUTION_BY_NAME",
]


# ---------------------------------------------------------------------------
# Local replacement for ``common.utils.crop_left_right``.
#
# DROID version (paraphrased) takes a ``PIL.Image`` and returns a
# ``PIL.Image`` cropped by a left + right ratio of the width. We keep the
# same signature and semantics so the reference port stays a faithful
# drop-in for the DROID live-preview pipeline.
# ---------------------------------------------------------------------------
def crop_left_right(image, left_ratio: float, right_ratio: float):
    """Crop ``left_ratio`` off the left and ``right_ratio`` off the right.

    Args:
        image: ``PIL.Image`` (any mode; only width is consulted).
        left_ratio: Fraction of width to remove from the left edge.
        right_ratio: Fraction of width to remove from the right edge.

    Returns:
        ``PIL.Image`` of shape ``(width * (1 - left_ratio - right_ratio), height)``.
    """
    if left_ratio < 0 or right_ratio < 0:
        raise ValueError("left_ratio / right_ratio must be non-negative")
    if left_ratio + right_ratio >= 1.0:
        raise ValueError("left_ratio + right_ratio must be < 1.0")

    width, _height = image.size
    left_px = int(width * left_ratio)
    right_px = int(width * right_ratio)
    return image.crop((left_px, 0, width - right_px, _height))


# Resolution lookup is built lazily: it depends on ``sl.RESOLUTION`` which is
# only available when the SDK is importable. We expose the same name as DROID
# (``_RESOLUTION_BY_NAME``) so tests can assert on the mapping shape even when
# the real SDK is missing.
def _build_resolution_map():
    if sl is None:
        # Static fallback used only by tests; values are opaque sentinels.
        return {
            "HD2K": "HD2K",
            "HD1080": "HD1080",
            "HD720": "HD720",
            "VGA": "VGA",
        }
    return {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }


_RESOLUTION_BY_NAME = _build_resolution_map()


def _resolve_resolution(default_resolution):
    resolution_name = os.environ.get("DROID_ZED_RESOLUTION")
    if not resolution_name:
        return default_resolution

    resolution = _RESOLUTION_BY_NAME.get(resolution_name.strip().upper())
    if resolution is None:
        valid = ", ".join(sorted(_RESOLUTION_BY_NAME))
        raise ValueError(
            f"Invalid DROID_ZED_RESOLUTION={resolution_name!r}; "
            f"expected one of: {valid}"
        )
    return resolution


def _resolve_fps(default_fps):
    fps_text = os.environ.get("DROID_ZED_FPS")
    if not fps_text:
        return default_fps

    try:
        fps = int(fps_text)
    except ValueError as exc:
        raise ValueError(
            f"Invalid DROID_ZED_FPS={fps_text!r}; expected a positive integer"
        ) from exc
    if fps <= 0:
        raise ValueError(
            f"Invalid DROID_ZED_FPS={fps_text!r}; expected a positive integer"
        )
    return fps


class ZedCamera:
    """DROID-faithful ZED wrapper. Open in ``__init__`` (DROID contract)."""

    def __init__(
        self,
        serial_number=None,
        device_path=None,
        resolution=None,
        fps=30,
    ):
        if sl is None:
            raise RuntimeError(
                "pyzed.sl is not importable; cannot construct ZedCamera. "
                "Install the ZED SDK or run under a mock."
            )

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        # Default mirrors upstream DROID (HD1080). We resolve the default
        # after construction so the constant lives in one place.
        if resolution is None:
            resolution = sl.RESOLUTION.HD1080
        init_params.camera_resolution = _resolve_resolution(resolution)
        init_params.camera_fps = _resolve_fps(fps)

        # Device input source selection (priority: serial > device_path).
        if serial_number is not None:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init_params.input = input_type
        elif device_path is not None:
            init_params.input = sl.InputType(device_path)

        # Open the camera. DROID prints+exits on failure; we keep the same
        # observable behavior but raise instead of ``exit()`` so callers
        # (and tests) can recover. The error code is preserved in the
        # message for parity with the upstream print.
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Unable to open ZED camera (err={err}). "
                "Upstream DROID would exit() at this point."
            )

        # Pre-allocate output Mats so we never re-bind during capture.
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.frame_count = 0

    def capture_frame(self):
        """Capture a stereo pair.

        Returns:
            ``(left_bgr_np, right_bgr_np)`` of shape ``(H, W, 3)``.
            Returns ``(None, None)`` on grab failure, matching the DROID
            soft-fail contract.
        """
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            left = self.image_left.get_data()[:, :, :3]
            right = self.image_right.get_data()[:, :, :3]
            self.frame_count += 1
            return copy.deepcopy(left), copy.deepcopy(right)
        return None, None

    def close(self):
        self.zed.close()

    def __del__(self):
        # DROID prints a status line; we keep close() idempotent via a
        # hasattr guard so double-close during interpreter shutdown is safe.
        if hasattr(self, "zed") and self.zed is not None:
            try:
                self.zed.close()
            except Exception:
                pass


def show_live_preview(camera, window_name: str = "ZED Preview"):
    """OpenCV live preview. DROID behavior: left-eye only, cropped, 'q' quits."""
    LEFT_RATIO = 0.27
    RIGHT_RATIO = 0.13

    try:
        from PIL import Image  # local import keeps top-level deps minimal

        while True:
            left_img, _ = camera.capture_frame()
            if left_img is None:
                continue
            left_pil = Image.fromarray(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
            cropped_pil = crop_left_right(left_pil, LEFT_RATIO, RIGHT_RATIO)
            cropped_bgr = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, cropped_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        camera.close()


if __name__ == "__main__":
    # FIXED: upstream DROID bound ``zed_wrist_camera`` and then called
    # ``show_live_preview(zed_camera, ...)`` -> NameError. We bind the
    # variable name we actually pass to the preview.
    _WRIST_SERIAL = 13132609
    zed_camera = ZedCamera(serial_number=_WRIST_SERIAL)
    show_live_preview(zed_camera, window_name="Wrist Camera")
