"""
DefaultTrainingConfig — 实验训练配置基类。

各任务 config.py 继承此类并覆盖需要修改的字段。
"""
import numpy as np


class DefaultTrainingConfig:
    """
    SERL / HIL-SERL 训练管线的默认超参数。

    子类（如 plug_insertion.config.TrainConfig）应覆盖：
      - image_keys / classifier_keys / proprio_keys
      - get_environment()
    """

    # -- 观测键（子类必须覆盖） --
    image_keys: list = []
    classifier_keys: list = []
    proprio_keys: list = [
        "tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose",
    ]

    # -- SAC 训练超参数 --
    checkpoint_period: int = 2000
    cta_ratio: int = 2             # critic-to-actor update ratio
    random_steps: int = 0          # 纯随机探索步数（0 = 不做随机探索）
    discount: float = 0.98         # 折扣因子 gamma
    buffer_period: int = 1000      # replay buffer dump 周期（步数）
    encoder_type: str = "resnet-pretrained"
    setup_mode: str = "single-arm-learned-gripper"
    # training hyperparams required by train_rlpd.py (canonical upstream defaults)
    batch_size: int = 256
    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000
    training_starts: int = 100
    steps_per_update: int = 50
    log_period: int = 10

    # -- 环境工厂方法（子类必须覆盖） --
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        构建 Gym 环境管线。

        Args:
            fake_env:   True → 虚拟环境（learner 不连真机）
            save_video: 是否录制调试视频
            classifier: 是否加载奖励分类器
        Returns:
            gym.Env
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_environment()"
        )
