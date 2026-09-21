#!/usr/bin/env python3
"""Live framing preview for a ZED camera, with mount-composition guides.

Use this while physically moving a camera. `scripts/calibrate_image_crop.py`
picks the crop rectangle afterwards; this tool answers the earlier question of
whether the mount frames the task at all. It shows the live view on the robot
desktop and mirrors the newest frame to disk, so a remote operator can read the
same picture without holding the camera.

Keys: q or Esc quits.

Usage:
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
        python scripts/preview_camera_view.py \
        --serial 36276705 --exposure 39 \
        --profile insert-front-roi160-v1 \
        --mirror-dir /tmp/front-camera-preview
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.image_profile import get_image_profile
from scripts.zed_capture import ZEDCapture

WINDOW = "camera framing preview"

# Horizontal references drawn over the live view. The fixture and the whole
# approach corridor are meant to live between the first and the last line.
BANDS = ((0.30, "corridor top"), (0.55, "fixture centre"), (0.75, "fixture bottom"))


def parse_crop(value):
    parts = [int(p) for p in value.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    return parts


def draw_guides(frame, crop, label, scale_text=0.7):
    out = frame.copy()
    height, width = out.shape[:2]
    for ratio, text in BANDS:
        y = int(height * ratio)
        cv2.line(out, (0, y), (width, y), (0, 200, 255), 1)
        cv2.putText(out, text, (8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, scale_text,
                    (0, 200, 255), 2, cv2.LINE_AA)
    for ratio in (0.25, 0.75):
        x = int(width * ratio)
        cv2.line(out, (x, 0), (x, height), (0, 200, 255), 1)
    if crop:
        x, y, box_w, box_h = crop
        cv2.rectangle(out, (x, y), (x + box_w, y + box_h), (0, 0, 255), 2)
        cv2.putText(out, f"{label} side_policy crop", (x + 8, y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, scale_text, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def write_png(path, image):
    """Write atomically so a reader never sees a half-written frame."""
    temporary = path.with_name(path.name + ".tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write {temporary}")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serial", default="36276705", help="ZED serial (default: external camera)")
    parser.add_argument("--name", default="external", help="Human-readable camera label")
    parser.add_argument("--exposure", type=int, default=39,
                        help="ZED exposure percentage, -1 for auto (default: 39, the side profile value)")
    parser.add_argument("--profile", default=None,
                        help="Image profile whose side_policy crop is drawn as the current ROI")
    parser.add_argument("--crop", type=parse_crop, default=None,
                        help="Crop to draw instead of --profile, as x,y,width,height")
    parser.add_argument("--mirror-dir", type=Path, default=Path("/tmp/front-camera-preview"),
                        help="Directory for latest_raw.png, latest_guide.png and status.json")
    parser.add_argument("--mirror-period", type=float, default=2.0,
                        help="Seconds between mirrored frames (default 2)")
    parser.add_argument("--window-width", type=int, default=1280, help="On-screen window width")
    parser.add_argument("--no-window", action="store_true", help="Mirror only, do not open a window")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N seconds; 0 runs until quit")
    args = parser.parse_args()

    crop, label = args.crop, "manual"
    if crop is None and args.profile:
        profile = get_image_profile(args.profile)
        crop = profile["cameras"]["side_policy"]["crop_xywh"]
        label = profile["name"]

    mirror = args.mirror_dir
    mirror.mkdir(parents=True, exist_ok=True)
    raw_path, guide_path, status_path = (mirror / "latest_raw.png", mirror / "latest_guide.png",
                                         mirror / "status.json")

    window = None
    with ZEDCapture(name=args.name, serial_number=args.serial, dim=(1280, 720), fps=30,
                    exposure=args.exposure) as camera:
        if not args.no_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, args.window_width, int(args.window_width * 9 / 16))
            try:
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass
            window = WINDOW
        print(f"[preview] serial={args.serial} exposure={args.exposure} crop={crop} "
              f"mirror={mirror}", flush=True)

        frames = 0
        started = time.monotonic()
        last_mirror = 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[preview] grab failed", flush=True)
                time.sleep(0.05)
                continue
            frames += 1
            now = time.monotonic()
            guided = draw_guides(frame, crop, label, scale_text=0.9)
            if window is not None:
                cv2.imshow(window, guided)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            else:
                time.sleep(0.02)
            if now - last_mirror >= args.mirror_period:
                last_mirror = now
                write_png(raw_path, frame)
                write_png(guide_path, guided)
                elapsed = now - started
                status_path.write_text(json.dumps({
                    "unix_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "serial": args.serial,
                    "exposure": args.exposure,
                    "crop": crop,
                    "profile": label,
                    "frames": frames,
                    "average_fps": frames / elapsed if elapsed > 0 else None,
                    "mean_brightness": float(frame.mean()),
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "raw_path": str(raw_path),
                }, indent=2) + "\n")
                print(f"[preview] frame={frames} fps={frames / elapsed:.1f} "
                      f"mean_brightness={frame.mean():.1f}", flush=True)
            if args.max_seconds and now - started > args.max_seconds:
                break

    if window is not None:
        cv2.destroyAllWindows()
    print(f"[preview] stopped after {frames} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
