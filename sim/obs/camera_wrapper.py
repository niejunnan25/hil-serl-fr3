"""Convert Isaac Lab Camera sensor data to obs dict matching OpenPI schema.

Per spec §2.3 + §3.3: keys/shape/dtype 必须与 schema_audit 100% 一致。
"""
from __future__ import annotations

from typing import Any


def camera_rgb_to_obs(
    camera_data: Any,            # isaaclab Camera sensor.data (lab.sensors.Camera)
    target_key: str,
    target_shape: tuple[int, ...],
    target_dtype: str,
):
    """Pull RGB tensor from Isaac Lab Camera, reshape/cast to target schema.

    isaaclab Camera typically gives `data.output["rgb"]` shape (B, H, W, 3) uint8.
    OpenPI/LeRobot typically expects (3, H, W) uint8 (NCHW). Single env: B=1.
    """
    rgb = camera_data.output["rgb"]  # (B, H, W, 3) torch.uint8
    if rgb.ndim != 4:
        raise ValueError(f"unexpected camera rgb shape {tuple(rgb.shape)}")

    # Take env 0 (single-env Phase 0)
    img = rgb[0]                           # (H, W, 3)
    # Permute to (3, H, W)
    img = img.permute(2, 0, 1).contiguous()
    # If target_shape is (3, H, W), check; else resize
    if tuple(img.shape) != target_shape:
        # Use torchvision/pytorch-native resize when needed
        import torch.nn.functional as F
        # img is (3, H, W); F.interpolate wants (B, C, H, W)
        img_b = img.unsqueeze(0).float()
        img_b = F.interpolate(
            img_b, size=target_shape[1:], mode="bilinear", align_corners=False
        )
        img = img_b.squeeze(0).to(rgb.dtype)
    return img


__all__ = ["camera_rgb_to_obs"]
