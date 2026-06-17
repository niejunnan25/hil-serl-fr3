#!/usr/bin/env python3
"""train_reward_classifier.py

SERL reward classifier 训练脚本 — 自包含 PyTorch 实现。

使用 ResNet-18 风格的二值图像分类器作为 reward function，
替代 serl_launcher.networks.reward_classifier.load_classifier_func。

⚠️  CHECKPOINT FORMAT WARNING:
本脚本输出 PyTorch .pt 格式 checkpoint，但 SERL 上游的 config.py 和
inference_service.py 通过 load_classifier_func 加载 JAX/Orbax 格式 checkpoint。
两种格式 **不兼容**。

解决方案 (选其一):
  A) 推荐: 使用 train_reward_classifier.sh (调用 SERL 原生 JAX 训练脚本)
     bash scripts/train_reward_classifier.sh
  B) 使用本脚本后，手动将 PyTorch checkpoint 转换为 JAX 格式:
     python scripts/convert_classifier_pt_to_jax.py classifier_ckpt/
  C) 修改 config.py 使用 PyTorch 推理 (不经过 load_classifier_func)

数据流:
  collect_classifier_images.py → data/classifier/{positive,negative}/
  → train_reward_classifier.py → classifier_ckpt/reward_classifier.pt
  → (需要转换) → classifier_ckpt/orbax/
  → config.py load_classifier_func → MultiCameraBinaryRewardClassifierWrapper

目录结构:
  data/classifier/
    positive/   # 正样本 (成功状态, 128x128 RGB PNG)
    negative/   # 负样本 (失败状态, 128x128 RGB PNG)

用法:
  python scripts/train_reward_classifier.py
  python scripts/train_reward_classifier.py --epochs 50 --batch-size 64
  python scripts/train_reward_classifier.py --data-dir data/classifier/ --output-dir classifier_ckpt/

输出:
  - classifier_ckpt/reward_classifier.pt   (PyTorch 模型权重 + 配置)
  - classifier_ckpt/training_history.json  (训练曲线数据)
  - stdout: 每 epoch 指标 + 最终评估报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128
CHANNELS = 3
NUM_CLASSES = 1  # binary: positive=1, negative=0

# ImageNet 标准化 (ResNet 预训练常用)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class ClassifierDataset(Dataset):
    """从 data/classifier/{positive,negative}/ 加载图像。"""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        # positive → label 1
        pos_dir = self.data_dir / "positive"
        if pos_dir.is_dir():
            for f in sorted(pos_dir.iterdir()):
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((f, 1))

        # negative → label 0
        neg_dir = self.data_dir / "negative"
        if neg_dir.is_dir():
            for f in sorted(neg_dir.iterdir()):
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((f, 0))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {data_dir}/{{positive,negative}}/. "
                "Run collect_classifier_images.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, float(label)

    @property
    def num_positive(self) -> int:
        return sum(1 for _, y in self.samples if y == 1)

    @property
    def num_negative(self) -> int:
        return sum(1 for _, y in self.samples if y == 0)


# ---------------------------------------------------------------------------
# 模型: 轻量 ResNet-18 风格
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """基础残差块: 2x [Conv3x3 → BN → ReLU] + skip connection。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class RewardClassifier(nn.Module):
    """ResNet-18 风格二值分类器。

    输入: (B, 3, 128, 128) RGB 图像
    输出: (B, 1) logits (未经 sigmoid)
    """

    def __init__(self):
        super().__init__()

        # Stem: 7x7 conv → BN → ReLU → MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # 4 stages (ResNet-18 layout)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Global average pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, NUM_CLASSES)

        # 权重初始化
        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)  # logits, 不含 sigmoid


# ---------------------------------------------------------------------------
# 数据增强 / 预处理
# ---------------------------------------------------------------------------
def get_train_transform() -> transforms.Compose:
    """训练集变换 (含数据增强)。"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """验证/测试集变换 (无增强)。"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# 训练工具函数
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算二值分类指标。"""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    neg_precision = tn / max(tn + fn, 1)
    neg_recall = tn / max(tn + fp, 1)
    neg_f1 = 2 * neg_precision * neg_recall / max(neg_precision + neg_recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "neg_f1": neg_f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """在给定 DataLoader 上评估模型。"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """打印格式化指标。"""
    print(f"{prefix}Loss: {metrics['loss']:.4f}  |  "
          f"Acc: {metrics['accuracy']:.4f}  |  "
          f"F1: {metrics['f1']:.4f}  |  "
          f"Prec: {metrics['precision']:.4f}  |  "
          f"Rec: {metrics['recall']:.4f}")
    print(f"{prefix}  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    patience: int = 10,
) -> dict:
    """完整训练流程。

    Returns:
        训练历史 dict，含 train/val 每 epoch 指标。
    """
    # -----------------------------------------------------------------------
    # 1. 设备
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 2. 数据
    # -----------------------------------------------------------------------
    full_dataset = ClassifierDataset(data_dir, transform=get_train_transform())
    print(f"\nDataset: {len(full_dataset)} images "
          f"(positive={full_dataset.num_positive}, negative={full_dataset.num_negative})")

    # 类别不平衡 → 计算 pos_weight
    pos_count = max(full_dataset.num_positive, 1)
    neg_count = max(full_dataset.num_negative, 1)
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    print(f"Class balance: pos_weight={pos_weight.item():.3f}")

    # Train/Val split
    torch.manual_seed(seed)
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 验证集用无增强变换 — 创建独立副本
    val_dataset_clean = ClassifierDataset(data_dir, transform=get_val_transform())
    # 用相同的 indices
    val_dataset_clean = torch.utils.data.Subset(val_dataset_clean, val_dataset.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset_clean, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    print(f"Split: train={len(train_dataset)}, val={len(val_dataset_clean)}")

    # -----------------------------------------------------------------------
    # 3. 模型 / 优化器 / 调度器
    # -----------------------------------------------------------------------
    model = RewardClassifier().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: RewardClassifier (ResNet-18 style)")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # -----------------------------------------------------------------------
    # 4. 训练循环
    # -----------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    history: list[dict] = []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'=' * 70}")
    print(f"  Training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"  Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_preds.append(preds.detach().cpu().numpy())
            train_labels.append(labels.cpu().numpy())

        scheduler.step()

        # Train metrics
        train_preds = np.concatenate(train_preds).flatten()
        train_labels = np.concatenate(train_labels).flatten()
        train_metrics = compute_metrics(train_labels, train_preds)
        train_metrics["loss"] = running_loss / len(train_dataset)

        # Val metrics
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{epochs}  ({elapsed:.1f}s, lr={lr:.2e})")
        print(f"  Train: ", end="")
        print_metrics(train_metrics)
        print(f"  Val:   ", end="")
        print_metrics(val_metrics)

        record = {
            "epoch": epoch,
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val": {k: float(v) for k, v in val_metrics.items()},
            "lr": lr,
            "time_s": elapsed,
        }
        history.append(record)

        # Best model checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(output_dir, "reward_classifier.pt")
            save_checkpoint(model, ckpt_path, val_metrics, epoch)
            print(f"  ★ New best (F1={best_val_f1:.4f}), saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
                break

        print()

    # -----------------------------------------------------------------------
    # 5. 最终报告
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Training complete")
    print(f"{'=' * 70}")
    print(f"  Best epoch: {best_epoch}/{epochs}")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {os.path.abspath(os.path.join(output_dir, 'reward_classifier.pt'))}")

    # 保存训练历史
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {os.path.abspath(history_path)}")

    # 最终在验证集上评估 best checkpoint
    best_ckpt = os.path.join(output_dir, "reward_classifier.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
        final_metrics = evaluate(model, val_loader, device)
        print(f"\n  Final evaluation (best checkpoint):")
        print_metrics(final_metrics, prefix="    ")

    print(f"{'=' * 70}\n")
    return {"history": history, "best_val_f1": best_val_f1, "best_epoch": best_epoch}


# ---------------------------------------------------------------------------
# Checkpoint 保存/加载
# ---------------------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    path: str,
    val_metrics: dict,
    epoch: int,
) -> None:
    """保存模型权重 + 元数据。"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "image_size": IMAGE_SIZE,
        "architecture": "RewardClassifier_ResNet18",
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> tuple[nn.Module, dict]:
    """加载模型权重 + 元数据。

    Returns:
        (model, metadata_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = RewardClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return model, metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train SERL reward classifier (binary image classifier).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认训练
  python scripts/train_reward_classifier.py

  # 自定义参数
  python scripts/train_reward_classifier.py --epochs 50 --batch-size 64 --lr 5e-4

  # 指定数据和输出目录
  python scripts/train_reward_classifier.py --data-dir data/classifier/ --output-dir classifier_ckpt/
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/classifier/",
        help="分类器数据目录，含 positive/ 和 negative/ 子目录 (默认: data/classifier/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="classifier_ckpt/",
        help="模型输出目录 (默认: classifier_ckpt/)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="训练轮数 (默认: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="批大小 (默认: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="学习率 (默认: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="权重衰减 (默认: 1e-4)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="验证集比例 (默认: 0.2)",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="早停耐心值 (默认: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader worker 数 (默认: 0)",
    )

    args = parser.parse_args()

    # 路径处理 (相对于项目根)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = str(project_root / data_dir)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = str(project_root / output_dir)

    # 检查数据目录
    if not os.path.isdir(data_dir):
        print(f"Error: data directory not found: {data_dir}", file=sys.stderr)
        print("Run collect_classifier_images.py first to collect training data.", file=sys.stderr)
        sys.exit(1)

    train(
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
