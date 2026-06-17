#!/usr/bin/env python3
"""
collect_classifier_images.py – Collect images for SERL reward classifier training.

SERL requires ~200 positive + ~600 negative images for reward classifier training.
Images come from ZED cameras, cropped and resized to 128x128 RGB.

Usage:
    # Collect positive images (successful insertions)
    python scripts/collect_classifier_images.py --label positive --target-count 200

    # Collect negative images (failure modes)
    python scripts/collect_classifier_images.py --label negative --target-count 600

    # Use wrist camera additionally
    python scripts/collect_classifier_images.py --label positive --wrist-serial 13132609

    # Demo-linked mode (auto-capture during demos)
    python scripts/collect_classifier_images.py --demo-mode

Controls:
    SPACE   - Capture current frame
    p       - Switch to positive label
    n       - Switch to negative label
    q       - Quit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.zed_capture import ZEDCapture


# Default IMAGE_CROP values from experiments/plug_insertion/config.py
DEFAULT_IMAGE_CROP = {
    "wrist_1": lambda img: img[100:500, 300:900],
    "side_policy": lambda img: img[200:500, 350:750],
    "side_classifier": lambda img: img[250:400, 450:650],
}


def parse_crop_json(crop_json: str) -> tuple[int, int, int, int]:
    """Parse IMAGE_CROP JSON string to (y_start, y_end, x_start, x_end).

    Args:
        crop_json: JSON string like '{"y": [250, 400], "x": [450, 650]}'

    Returns:
        Tuple of (y_start, y_end, x_start, x_end)
    """
    try:
        crop = json.loads(crop_json)
        y_start, y_end = crop["y"]
        x_start, x_end = crop["x"]
        return y_start, y_end, x_start, x_end
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing crop JSON: {e}", file=sys.stderr)
        print('Expected format: \'{"y": [250, 400], "x": [450, 650]}\'', file=sys.stderr)
        sys.exit(1)


def get_crop_function(
    crop_arg: str, camera_name: str = "side_classifier"
) -> callable:
    """Get the crop function based on --crop argument.

    Args:
        crop_arg: Either "auto" or a JSON string
        camera_name: Camera name for auto mode lookup

    Returns:
        Crop function that takes an image and returns cropped region
    """
    if crop_arg == "auto":
        if camera_name in DEFAULT_IMAGE_CROP:
            return DEFAULT_IMAGE_CROP[camera_name]
        else:
            print(f"Warning: No default crop for '{camera_name}', using side_classifier", file=sys.stderr)
            return DEFAULT_IMAGE_CROP["side_classifier"]
    else:
        y_start, y_end, x_start, x_end = parse_crop_json(crop_arg)
        return lambda img: img[y_start:y_end, x_start:x_end]


def process_frame(frame: np.ndarray, crop_fn: callable) -> np.ndarray:
    """Process a frame: crop and resize to 128x128.

    Args:
        frame: Raw BGR frame from camera
        crop_fn: Crop function to apply

    Returns:
        Processed 128x128 BGR image
    """
    cropped = crop_fn(frame)
    resized = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
    return resized


def draw_overlay(
    image: np.ndarray,
    label: str,
    count: int,
    target: int,
    filename: str,
) -> np.ndarray:
    """Draw overlay information on the preview image.

    Args:
        image: 128x128 preview image
        label: Current label (positive/negative)
        count: Current count for this label
        target: Target count for this label
        filename: Last saved filename

    Returns:
        Image with overlay drawn
    """
    # Create a slightly larger canvas for overlay
    h, w = image.shape[:2]
    canvas = np.zeros((h + 60, w, 3), dtype=np.uint8)
    canvas[:h, :w] = image

    # Label with color coding
    color = (0, 255, 0) if label == "positive" else (0, 0, 255)
    cv2.putText(
        canvas, f"Label: {label}", (5, h + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )

    # Count
    cv2.putText(
        canvas, f"Count: {count}/{target}", (5, h + 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )

    # Filename
    if filename:
        # Truncate filename if too long
        display_name = filename if len(filename) < 30 else "..." + filename[-27:]
        cv2.putText(
            canvas, display_name, (5, h + 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

    return canvas


def generate_filename(label: str, seq: int, naming: str = "timestamp") -> str:
    """Generate filename based on naming mode.

    Args:
        label: positive or negative
        seq: Sequence number
        naming: "timestamp" or "frame" mode

    Returns:
        Filename like positive_20260610_143022_001.png or frame_000001.png
    """
    if naming == "frame":
        return f"frame_{seq:06d}.png"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{label}_{timestamp}_{seq:03d}.png"


class ImageCollector:
    """Manages image collection from ZED cameras."""

    def __init__(
        self,
        camera_serial: str,
        output_dir: str,
        initial_label: str,
        target_counts: dict[str, int],
        crop_fn: callable,
        wrist_serial: Optional[str] = None,
        naming: str = "timestamp",
    ):
        self.output_dir = Path(output_dir)
        self.crop_fn = crop_fn
        self.target_counts = target_counts
        self.current_label = initial_label
        self.naming = naming

        # Create output directories
        for label in ["positive", "negative"]:
            (self.output_dir / label).mkdir(parents=True, exist_ok=True)

        # Initialize counters
        self.counts = {"positive": 0, "negative": 0}
        self.session_counts = {"positive": 0, "negative": 0}

        # Count existing images
        for label in ["positive", "negative"]:
            existing = list((self.output_dir / label).glob("*.png"))
            self.counts[label] = len(existing)

        # Sequence numbers (continue from existing)
        self.sequences = {"positive": self.counts["positive"], "negative": self.counts["negative"]}

        # Initialize cameras
        print(f"Opening external camera (serial {camera_serial})...")
        self.cam_external = ZEDCapture(
            name="external",
            serial_number=camera_serial,
            dim=(1280, 720),
            fps=30,
        )

        self.cam_wrist = None
        if wrist_serial:
            print(f"Opening wrist camera (serial {wrist_serial})...")
            self.cam_wrist = ZEDCapture(
                name="wrist",
                serial_number=wrist_serial,
                dim=(1280, 720),
                fps=30,
            )

        self.last_filename = ""

    def close(self):
        """Release camera resources."""
        if self.cam_external:
            self.cam_external.close()
        if self.cam_wrist:
            self.cam_wrist.close()

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture and process a frame from the external camera.

        Returns:
            Processed 128x128 frame or None on failure
        """
        ok, frame = self.cam_external.read()
        if not ok:
            return None
        return process_frame(frame, self.crop_fn)

    def capture_wrist_frame(self) -> Optional[np.ndarray]:
        """Capture and process a frame from the wrist camera.

        Returns:
            Processed 128x128 frame or None on failure/no wrist camera
        """
        if self.cam_wrist is None:
            return None
        ok, frame = self.cam_wrist.read()
        if not ok:
            return None
        return process_frame(frame, self.crop_fn)

    def save_image(self, image: np.ndarray, label: Optional[str] = None) -> str:
        """Save an image with the given label.

        Args:
            image: 128x128 BGR image
            label: Label to use (defaults to current_label)

        Returns:
            Saved filename
        """
        label = label or self.current_label
        self.sequences[label] += 1
        filename = generate_filename(label, self.sequences[label], self.naming)
        filepath = self.output_dir / label / filename

        cv2.imwrite(str(filepath), image)

        self.counts[label] += 1
        self.session_counts[label] += 1
        self.last_filename = filename

        return filename

    def get_progress_string(self) -> str:
        """Get progress display string.

        Returns:
            String like "Positive: 45/200 | Negative: 120/600"
        """
        parts = []
        for label in ["positive", "negative"]:
            count = self.counts[label]
            target = self.target_counts[label]
            parts.append(f"{label.capitalize()}: {count}/{target}")
        return " | ".join(parts)

    def get_session_stats(self) -> str:
        """Get per-session statistics.

        Returns:
            Session stats string
        """
        total = sum(self.session_counts.values())
        return (
            f"Session: {total} captured "
            f"(+{self.session_counts['positive']}, "
            f"-{self.session_counts['negative']})"
        )


def run_interactive_mode(collector: ImageCollector) -> None:
    """Run the interactive image collection mode.

    Args:
        collector: ImageCollector instance
    """
    print("\n" + "=" * 60)
    print("HIL-SERL Reward Classifier Image Collection")
    print("=" * 60)
    print("\nControls:")
    print("  SPACE  - Capture current frame")
    print("  p      - Switch to positive label")
    print("  n      - Switch to negative label")
    print("  q      - Quit")
    print("\n" + collector.get_progress_string())
    print()

    window_name = "Classifier Image Collection (128x128)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        # Capture frame
        frame = collector.capture_frame()
        if frame is None:
            print("Warning: Failed to capture frame", file=sys.stderr)
            time.sleep(0.1)
            continue

        # Draw overlay
        preview = draw_overlay(
            frame.copy(),
            collector.current_label,
            collector.counts[collector.current_label],
            collector.target_counts[collector.current_label],
            collector.last_filename,
        )

        # Show preview
        cv2.imshow(window_name, preview)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nQuitting...")
            break
        elif key == ord(" "):  # SPACE
            # Check if target reached for current label
            if collector.counts[collector.current_label] >= collector.target_counts[collector.current_label]:
                print(f"\n⚠️  Target reached for {collector.current_label} ({collector.target_counts[collector.current_label]}). Switch label or continue? (y/n): ", end="", flush=True)
                # Non-blocking check - just warn and continue
                print(f"\n[WARNING] {collector.current_label} target of {collector.target_counts[collector.current_label]} already reached!")
            filename = collector.save_image(frame)
            # Also capture and save wrist image if available
            wrist_frame = collector.capture_wrist_frame()
            if wrist_frame is not None:
                wrist_filename = collector.save_image(wrist_frame, label=collector.current_label)
            print(f"\r{collector.get_progress_string()} | {collector.get_session_stats()} | Saved: {filename}", end="", flush=True)
        elif key == ord("p"):
            collector.current_label = "positive"
            print(f"\nSwitched to positive label")
        elif key == ord("n"):
            collector.current_label = "negative"
            print(f"\nSwitched to negative label")

    cv2.destroyAllWindows()


def run_demo_mode(collector: ImageCollector, capture_interval: float = 0.5) -> None:
    """Run demo-linked mode: auto-capture during demonstrations.

    Args:
        collector: ImageCollector instance
        capture_interval: Seconds between auto-captures
    """
    print("\n" + "=" * 60)
    print("HIL-SERL Demo-Linked Image Collection")
    print("=" * 60)
    print(f"\nAuto-capturing every {capture_interval}s")
    print("Press 'q' to stop capturing, then label the episode")
    print()

    window_name = "Demo Collection (128x128)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    episode_frames = []
    episode_wrist_frames = []
    last_capture_time = 0

    while True:
        # Capture frame
        frame = collector.capture_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # Show preview
        preview = frame.copy()
        cv2.putText(
            preview, f"Recording: {len(episode_frames)} frames", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
        cv2.imshow(window_name, preview)

        # Auto-capture at interval
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
            episode_frames.append(frame)
            # Also capture wrist frame if available
            wrist_frame = collector.capture_wrist_frame()
            if wrist_frame is not None:
                episode_wrist_frames.append(wrist_frame)
            last_capture_time = current_time

        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    if not episode_frames:
        print("No frames captured.")
        return

    # Ask user for episode label
    print(f"\nCaptured {len(episode_frames)} frames in this episode.")
    while True:
        response = input("Was this episode a success (positive) or failure (negative)? [p/n]: ").strip().lower()
        if response in ("p", "positive"):
            label = "positive"
            break
        elif response in ("n", "negative"):
            label = "negative"
            break
        else:
            print("Please enter 'p' for positive or 'n' for negative")

    # Save all frames with the chosen label
    print(f"Saving {len(episode_frames)} frames as {label}...")
    for frame in episode_frames:
        collector.save_image(frame, label)

    # Save wrist frames if available
    if episode_wrist_frames:
        print(f"Saving {len(episode_wrist_frames)} wrist frames as {label}...")
        for frame in episode_wrist_frames:
            collector.save_image(frame, label)

    print(f"Done! {collector.get_progress_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect images for SERL reward classifier training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 200 positive images
  python scripts/collect_classifier_images.py --label positive --target-count 200

  # Collect 600 negative images
  python scripts/collect_classifier_images.py --label negative --target-count 600

  # Use custom crop region
  python scripts/collect_classifier_images.py --label positive --crop '{"y": [200, 400], "x": [400, 600]}'

  # Demo-linked mode
  python scripts/collect_classifier_images.py --demo-mode
        """
    )

    parser.add_argument(
        "--camera-serial",
        default="36276705",
        help="ZED camera serial number (default: 36276705 for external)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/classifier/",
        help="Output directory (default: data/classifier/)",
    )
    parser.add_argument(
        "--label",
        choices=["positive", "negative"],
        default="positive",
        help="Initial label to collect (default: positive)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Target number of images (default: 200 positive, 600 negative)",
    )
    parser.add_argument(
        "--crop",
        default="auto",
        help='IMAGE_CROP as JSON \'{"y": [y1, y2], "x": [x1, x2]}\' or "auto"',
    )
    parser.add_argument(
        "--wrist-serial",
        default=None,
        help="Wrist camera serial number (optional, e.g. 13132609)",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Demo-linked mode: auto-capture and label episodes",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.5,
        help="Seconds between auto-captures in demo mode (default: 0.5)",
    )
    parser.add_argument(
        "--naming",
        choices=["timestamp", "frame"],
        default="timestamp",
        help="File naming mode: 'timestamp' (label_timestamp_seq.png) or 'frame' (frame_000001.png)",
    )

    args = parser.parse_args()

    # Set target counts
    if args.target_count is not None:
        target_counts = {
            "positive": args.target_count,
            "negative": args.target_count,
        }
    else:
        target_counts = {
            "positive": 200,
            "negative": 600,
        }

    # Get crop function
    crop_fn = get_crop_function(args.crop)

    # Initialize collector
    try:
        collector = ImageCollector(
            camera_serial=args.camera_serial,
            output_dir=args.output_dir,
            initial_label=args.label,
            target_counts=target_counts,
            crop_fn=crop_fn,
            wrist_serial=args.wrist_serial,
            naming=args.naming,
        )
    except Exception as e:
        print(f"Error initializing cameras: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.demo_mode:
            run_demo_mode(collector, args.capture_interval)
        else:
            run_interactive_mode(collector)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        collector.close()
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(collector.get_progress_string())
        print(collector.get_session_stats())
        print("=" * 60)


if __name__ == "__main__":
    main()
