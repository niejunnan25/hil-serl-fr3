from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.config import DefaultTrainingConfig

from fr3_hil_bridge.image_adapter import ZED_IMAGE_KEYS
from fr3_hil_bridge.zed_camera_client import DEFAULT_ZED_SERVICE_URL, synthetic_zed_observation
from fr3_hil_bridge.image_adapter import validate_zed_observation


ZED_CAMERA_PROFILE = "zed_stereo"


@dataclass(frozen=True)
class ZedObservationContract:
    camera_profile: str
    service_url: str
    image_keys: list[str]
    classifier_keys: list[str]
    image_shape: list[int]
    image_dtype: str
    runtime_enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_profile": self.camera_profile,
            "service_url": self.service_url,
            "image_keys": list(self.image_keys),
            "classifier_keys": list(self.classifier_keys),
            "image_shape": list(self.image_shape),
            "image_dtype": self.image_dtype,
            "runtime_enabled": self.runtime_enabled,
        }


class ZedTrainConfig(DefaultTrainingConfig):
    image_keys = list(ZED_IMAGE_KEYS)
    classifier_keys = list(ZED_IMAGE_KEYS)
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper-zed"
    camera_profile = ZED_CAMERA_PROFILE
    camera_service_url = DEFAULT_ZED_SERVICE_URL

    def observation_contract(self) -> ZedObservationContract:
        return ZedObservationContract(
            camera_profile=self.camera_profile,
            service_url=self.camera_service_url,
            image_keys=list(self.image_keys),
            classifier_keys=list(self.classifier_keys),
            image_shape=[128, 128, 3],
            image_dtype="uint8",
            runtime_enabled=False,
        )

    def validate_synthetic_observation(self) -> dict[str, Any]:
        return validate_zed_observation(synthetic_zed_observation())

    def get_environment(self, fake_env: bool = False, save_video: bool = False, classifier: bool = False):
        raise RuntimeError(
            "ZED_PLUG_ENV_RUNTIME_NOT_ENABLED: the ZED observation contract is proven, "
            "but live/fake env construction for policy rollout needs a separate actor "
            "integration gate."
        )
