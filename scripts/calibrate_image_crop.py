#!/usr/bin/env python3
"""
calibrate_image_crop.py – Interactive IMAGE_CROP calibration for ZED cameras.

Opens ZED cameras, lets you draw a crop rectangle with mouse, shows real-time
128x128 preview, and outputs IMAGE_CROP dict entries for config.py.

Usage:
    # Single camera
    python scripts/calibrate_image_crop.py --serial 13132609 --name wrist_1

    # Multiple cameras (iterate through list)
    python scripts/calibrate_image_crop.py \
        --camera wrist_1:13132609 \
        --camera side_policy:36276705 \
        --camera side_classifier:36276705

    # Load previous calibration
    python scripts/calibrate_image_crop.py --load calibration_data.json \
        --camera wrist_1:13132609

Controls:
    Mouse drag  – draw/adjust crop rectangle
    'r'         – reset current selection
    's'         – save current selection and print config entry
    'n'         – switch to next camera
    'p'         – switch to previous camera
    '+'/'-'     – adjust target crop size (default 128)
    'g'         – toggle grid overlay
    'q'/Esc     – quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.zed_capture import ZEDCapture, validate_zed_exposure


DEFAULT_ZED_EXPOSURE = 32


# ── Data types ────────────────────────────────────────────────────────────


@dataclass
class CropRegion:
    """A crop region in (y1, y2, x1, x2) format matching IMAGE_CROP lambdas."""

    y1: int = 0
    y2: int = 0
    x1: int = 0
    x2: int = 0

    @property
    def is_valid(self) -> bool:
        return self.y2 > self.y1 and self.x2 > self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    def to_lambda_str(self, camera_name: str) -> str:
        """Generate IMAGE_CROP lambda string for config.py."""
        return f'"{camera_name}": lambda img: img[{self.y1}:{self.y2}, {self.x1}:{self.x2}]'

    def to_dict(self) -> dict:
        return {"y1": self.y1, "y2": self.y2, "x1": self.x1, "x2": self.x2}

    @classmethod
    def from_dict(cls, d: dict) -> "CropRegion":
        return cls(y1=d["y1"], y2=d["y2"], x1=d["x1"], x2=d["x2"])


@dataclass
class CameraConfig:
    """Configuration for a single camera."""

    name: str
    serial: str
    crop: Optional[CropRegion] = None
    exposure: int = DEFAULT_ZED_EXPOSURE


@dataclass
class CalibrationState:
    """Mutable state for the calibration UI."""

    cameras: list[CameraConfig] = field(default_factory=list)
    current_idx: int = 0
    target_size: int = 128
    show_grid: bool = True
    drawing: bool = False
    start_point: Optional[tuple[int, int]] = None
    current_rect: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    saved_crops: dict[str, CropRegion] = field(default_factory=dict)

    @property
    def current_camera(self) -> CameraConfig:
        return self.cameras[self.current_idx]


# ── Drawing helpers ───────────────────────────────────────────────────────


def draw_grid_overlay(image: np.ndarray, grid_spacing: int = 100) -> np.ndarray:
    """Draw a semi-transparent grid overlay on the image."""
    overlay = image.copy()
    h, w = image.shape[:2]
    color = (0, 255, 0)  # green
    alpha = 0.3

    # Vertical lines
    for x in range(0, w, grid_spacing):
        cv2.line(overlay, (x, 0), (x, h), color, 1)

    # Horizontal lines
    for y in range(0, h, grid_spacing):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    # Labels every 200px
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    for x in range(0, w, grid_spacing * 2):
        cv2.putText(overlay, str(x), (x + 2, 15), font, font_scale, color, 1)
    for y in range(0, h, grid_spacing * 2):
        cv2.putText(overlay, str(y), (2, y - 3), font, font_scale, color, 1)

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def draw_crop_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw the crop rectangle on the image."""
    overlay = image.copy()
    x1, y1, x2, y2 = rect

    # Semi-transparent fill
    mask = overlay.copy()
    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 255), -1)
    overlay = cv2.addWeighted(mask, 0.2, overlay, 0.8, 0)

    # Border
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Corner markers
    corner_len = 15
    corner_color = (0, 0, 255)  # red
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for cx, cy in corners:
        dx = corner_len if cx == x1 else -corner_len
        dy = corner_len if cy == y1 else -corner_len
        cv2.line(overlay, (cx, cy), (cx + dx, cy), corner_color, 2)
        cv2.line(overlay, (cx, cy), (cx, cy + dy), corner_color, 2)

    # Size label
    w, h = x2 - x1, y2 - y1
    size_text = f"{w}x{h}"
    if label:
        size_text = f"{label}: {size_text}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), _ = cv2.getTextSize(size_text, font, font_scale, thickness)
    label_y = max(y1 - 10, th + 5)
    cv2.rectangle(
        overlay, (x1, label_y - th - 4), (x1 + tw + 8, label_y + 4), (0, 0, 0), -1
    )
    cv2.putText(
        overlay, size_text, (x1 + 4, label_y), font, font_scale, (0, 255, 255), thickness
    )

    # Crosshair at center
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.drawMarker(
        overlay, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA
    )

    return overlay


def make_crop_preview(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    target_size: int = 128,
) -> np.ndarray:
    """Extract crop region and resize to target_size x target_size."""
    x1, y1, x2, y2 = rect
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def make_side_panel(
    crop_preview: np.ndarray,
    state: CalibrationState,
) -> np.ndarray:
    """Create a side panel with crop preview and info."""
    panel_h = crop_preview.shape[0]
    panel_w = 320
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    # Place crop preview at top
    preview_h, preview_w = crop_preview.shape[:2]
    panel[0:preview_h, 0:preview_w] = crop_preview

    # Info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    color = (200, 200, 200)
    y_offset = preview_h + 20

    cam = state.current_camera
    lines = [
        f"Camera: {cam.name}",
        f"Serial: {cam.serial}",
        f"[{state.current_idx + 1}/{len(state.cameras)}]",
        f"Target size: {state.target_size}x{state.target_size}",
        "",
        "Controls:",
        "  drag   - draw rect",
        "  r      - reset",
        "  s      - save crop",
        "  n/p    - next/prev cam",
        "  +/-    - target size",
        "  g      - toggle grid",
        "  q/Esc  - quit",
    ]

    if cam.crop and cam.crop.is_valid:
        lines.extend(
            [
                "",
                "Current crop:",
                f"  y: [{cam.crop.y1}:{cam.crop.y2}]",
                f"  x: [{cam.crop.x1}:{cam.crop.x2}]",
                f"  size: {cam.crop.width}x{cam.crop.height}",
            ]
        )

    for line in lines:
        if y_offset > panel_h - 10:
            break
        cv2.putText(panel, line, (10, y_offset), font, font_scale, color, 1)
        y_offset += 18

    return panel


# ── Mouse callback ────────────────────────────────────────────────────────


def mouse_callback(event: int, x: int, y: int, flags: int, param: CalibrationState):
    """Handle mouse events for drawing crop rectangles."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param.drawing = True
        param.start_point = (x, y)
        param.current_rect = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE and param.drawing:
        if param.start_point:
            sx, sy = param.start_point
            param.current_rect = (min(sx, x), min(sy, y), max(sx, x), max(sy, y))

    elif event == cv2.EVENT_LBUTTONUP:
        param.drawing = False
        if param.start_point:
            sx, sy = param.start_point
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)

            # Minimum size check
            if x2 - x1 > 10 and y2 - y1 > 10:
                param.current_rect = (x1, y1, x2, y2)
            else:
                param.current_rect = None
            param.start_point = None


# ── Config output ─────────────────────────────────────────────────────────


def print_config_entry(name: str, crop: CropRegion) -> None:
    """Print IMAGE_CROP dict entry ready for config.py."""
    print(f'    "{name}": lambda img: img[{crop.y1}:{crop.y2}, {crop.x1}:{crop.x2}],')


def print_full_config(state: CalibrationState) -> None:
    """Print complete IMAGE_CROP dict for config.py."""
    print("\n" + "=" * 60)
    print("IMAGE_CROP = {")
    for cam in state.cameras:
        crop = state.saved_crops.get(cam.name)
        if crop and crop.is_valid:
            print_config_entry(cam.name, crop)
        else:
            print(f'    "{cam.name}": lambda img: img[:, :],  # NOT CALIBRATED')
    print("}")
    print("=" * 60)


def save_calibration(state: CalibrationState, path: Path) -> None:
    """Save calibration data to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "target_size": state.target_size,
        "cameras": {},
    }
    for cam in state.cameras:
        crop = state.saved_crops.get(cam.name)
        data["cameras"][cam.name] = {
            "serial": cam.serial,
            "exposure": cam.exposure,
            "crop": crop.to_dict() if crop and crop.is_valid else None,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nCalibration saved to: {path}")


def load_calibration(path: Path, state: CalibrationState) -> None:
    """Load calibration data from JSON."""
    if not path.exists():
        print(f"No calibration file found at: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    for cam in state.cameras:
        if cam.name in data.get("cameras", {}):
            cam_data = data["cameras"][cam.name]
            if cam_data.get("crop"):
                crop = CropRegion.from_dict(cam_data["crop"])
                cam.crop = crop
                state.saved_crops[cam.name] = crop
                print(f"  Loaded crop for {cam.name}: [{crop.y1}:{crop.y2}, {crop.x1}:{crop.x2}]")

    if data.get("target_size"):
        state.target_size = data["target_size"]
        print(f"  Loaded target size: {state.target_size}")


# ── Main loop ─────────────────────────────────────────────────────────────


def run_calibration(state: CalibrationState, output_path: Path) -> None:
    """Main calibration loop."""
    window_name = "IMAGE_CROP Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 720)
    cv2.setMouseCallback(window_name, mouse_callback, state)

    print("\n" + "=" * 60)
    print("IMAGE_CROP Calibration Tool")
    print("=" * 60)
    print(f"Cameras: {', '.join(c.name for c in state.cameras)}")
    print(f"Output:  {output_path}")
    print()

    current_cam: Optional[ZEDCapture] = None

    def open_camera(cam_config: CameraConfig) -> ZEDCapture:
        nonlocal current_cam
        if current_cam is not None:
            current_cam.close()
        print(f"\nOpening camera: {cam_config.name} (serial {cam_config.serial})")
        current_cam = ZEDCapture(
            name=cam_config.name,
            serial_number=cam_config.serial,
            dim=(1280, 720),
            fps=30,
            exposure=cam_config.exposure,
        )
        return current_cam

    try:
        cam = open_camera(state.current_camera)

        # Restore previous crop if loaded
        if state.current_camera.crop:
            c = state.current_camera.crop
            state.current_rect = (c.x1, c.y1, c.x2, c.y2)

        while True:
            ok, frame = cam.read()
            if not ok:
                print("  WARNING: Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            # Draw grid if enabled
            display = frame.copy()
            if state.show_grid:
                display = draw_grid_overlay(display)

            # Draw current crop rectangle
            if state.current_rect:
                x1, y1, x2, y2 = state.current_rect
                crop = CropRegion(y1=y1, y2=y2, x1=x1, x2=x2)
                display = draw_crop_rect(display, state.current_rect, state.current_camera.name)
                crop_preview = make_crop_preview(frame, state.current_rect, state.target_size)
            else:
                crop_preview = np.zeros(
                    (state.target_size, state.target_size, 3), dtype=np.uint8
                )

            # Draw saved crop (if different from current)
            saved = state.saved_crops.get(state.current_camera.name)
            if saved and saved.is_valid:
                saved_rect = (saved.x1, saved.y1, saved.x2, saved.y2)
                if saved_rect != state.current_rect:
                    # Draw saved crop in green (dimmer)
                    overlay = display.copy()
                    cv2.rectangle(
                        overlay,
                        (saved.x1, saved.y1),
                        (saved.x2, saved.y2),
                        (0, 200, 0),
                        1,
                    )
                    display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

            # Build side panel
            side_panel = make_side_panel(crop_preview, state)

            # Combine: frame | panel
            combined = np.hstack(
                [
                    cv2.resize(display, (1280, 720)),
                    cv2.resize(side_panel, (320, 720)),
                ]
            )

            # Status bar at top
            status = f"Camera: {state.current_camera.name} | Target: {state.target_size}x{state.target_size} | Grid: {'ON' if state.show_grid else 'OFF'}"
            cv2.putText(
                combined,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )

            cv2.imshow(window_name, combined)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:  # q or Esc
                # Save before quitting
                if state.saved_crops:
                    save_calibration(state, output_path)
                    print_full_config(state)
                break

            elif key == ord("r"):
                state.current_rect = None
                state.current_camera.crop = None
                if state.current_camera.name in state.saved_crops:
                    del state.saved_crops[state.current_camera.name]
                print(f"  Reset crop for {state.current_camera.name}")

            elif key == ord("s"):
                if state.current_rect:
                    x1, y1, x2, y2 = state.current_rect
                    crop = CropRegion(y1=y1, y2=y2, x1=x1, x2=x2)
                    state.current_camera.crop = crop
                    state.saved_crops[state.current_camera.name] = crop
                    save_calibration(state, output_path)
                    print(f"\n  Saved crop for {state.current_camera.name}:")
                    print_config_entry(state.current_camera.name, crop)
                else:
                    print("  No crop region selected. Draw a rectangle first.")

            elif key == ord("n"):
                # Save current before switching
                if state.current_rect:
                    x1, y1, x2, y2 = state.current_rect
                    crop = CropRegion(y1=y1, y2=y2, x1=x1, x2=x2)
                    state.current_camera.crop = crop
                    state.saved_crops[state.current_camera.name] = crop

                state.current_idx = (state.current_idx + 1) % len(state.cameras)
                cam = open_camera(state.current_camera)

                # Restore crop for new camera
                if state.current_camera.crop:
                    c = state.current_camera.crop
                    state.current_rect = (c.x1, c.y1, c.x2, c.y2)
                else:
                    state.current_rect = None

            elif key == ord("p"):
                # Save current before switching
                if state.current_rect:
                    x1, y1, x2, y2 = state.current_rect
                    crop = CropRegion(y1=y1, y2=y2, x1=x1, x2=x2)
                    state.current_camera.crop = crop
                    state.saved_crops[state.current_camera.name] = crop

                state.current_idx = (state.current_idx - 1) % len(state.cameras)
                cam = open_camera(state.current_camera)

                # Restore crop for new camera
                if state.current_camera.crop:
                    c = state.current_camera.crop
                    state.current_rect = (c.x1, c.y1, c.x2, c.y2)
                else:
                    state.current_rect = None

            elif key == ord("g"):
                state.show_grid = not state.show_grid
                print(f"  Grid: {'ON' if state.show_grid else 'OFF'}")

            elif key == ord("+") or key == ord("="):
                state.target_size = min(512, state.target_size + 16)
                print(f"  Target size: {state.target_size}x{state.target_size}")

            elif key == ord("-"):
                state.target_size = max(32, state.target_size - 16)
                print(f"  Target size: {state.target_size}x{state.target_size}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if current_cam is not None:
            current_cam.close()
        cv2.destroyAllWindows()

        # Final output
        if state.saved_crops:
            save_calibration(state, output_path)
            print_full_config(state)


# ── CLI entry point ───────────────────────────────────────────────────────


def parse_camera_arg(arg: str) -> CameraConfig:
    """Parse 'name:serial[:exposure]' format."""
    parts = arg.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid camera format: {arg!r}. Expected name:serial[:exposure]")
    name = parts[0]
    serial = parts[1]
    exposure = int(parts[2]) if len(parts) > 2 else DEFAULT_ZED_EXPOSURE
    exposure = validate_zed_exposure(exposure)
    return CameraConfig(name=name, serial=serial, exposure=exposure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive IMAGE_CROP calibration for ZED cameras.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single camera
  %(prog)s --serial 13132609 --name wrist_1

  # Multiple cameras
  %(prog)s --camera wrist_1:13132609 \\
           --camera side_policy:36276705 \\
           --camera side_classifier:36276705

  # With custom exposure
  %(prog)s --camera wrist_1:13132609:32 \\
           --camera side_policy:36276705:39

  # Load previous calibration
  %(prog)s --load calibration_data.json \\
           --camera wrist_1:13132609 \\
           --camera side_policy:36276705
        """,
    )

    # Single camera mode
    parser.add_argument("--serial", help="Single camera serial number")
    parser.add_argument("--name", default="camera", help="Single camera name (default: camera)")
    parser.add_argument(
        "--exposure",
        type=int,
        default=DEFAULT_ZED_EXPOSURE,
        help=f"ZED exposure percent or -1 auto (default: {DEFAULT_ZED_EXPOSURE})",
    )

    # Multi-camera mode
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME:SERIAL[:EXPOSURE]",
        help="Camera config (repeatable). Overrides --serial/--name.",
    )

    # Options
    parser.add_argument(
        "--output",
        "-o",
        default="calibration_data.json",
        help="Output JSON path (default: calibration_data.json)",
    )
    parser.add_argument(
        "--load",
        "-l",
        metavar="PATH",
        help="Load previous calibration from JSON file",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="Target crop preview size (default: 128)",
    )

    args = parser.parse_args()

    # Build camera list
    cameras: list[CameraConfig] = []
    if args.camera:
        for cam_arg in args.camera:
            cameras.append(parse_camera_arg(cam_arg))
    elif args.serial:
        exposure = validate_zed_exposure(args.exposure)
        cameras.append(
            CameraConfig(name=args.name, serial=args.serial, exposure=exposure)
        )
    else:
        parser.error("Specify --serial or at least one --camera")

    # Build state
    state = CalibrationState(
        cameras=cameras,
        target_size=args.target_size,
    )

    output_path = Path(args.output)

    # Load previous calibration if specified
    if args.load:
        load_calibration(Path(args.load), state)

    # Run
    run_calibration(state, output_path)


if __name__ == "__main__":
    main()
