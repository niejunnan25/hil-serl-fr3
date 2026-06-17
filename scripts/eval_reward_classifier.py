#!/usr/bin/env python3
"""eval_reward_classifier.py

SERL reward classifier 评估脚本 — 加载已训练的 PyTorch 分类器，
在 held-out 测试集上评估并输出混淆矩阵 + 分类指标。

数据流:
  train_reward_classifier.py → classifier_ckpt/reward_classifier.pt
  → eval_reward_classifier.py → stdout JSON + confusion_matrix.png

目录结构:
  data/classifier/          (或 --test-dir 指定)
    positive/   # 正样本 (成功状态, 128x128 RGB PNG)
    negative/   # 负样本 (失败状态, 128x128 RGB PNG)

用法:
  python scripts/eval_reward_classifier.py
  python scripts/eval_reward_classifier.py --checkpoint classifier_ckpt/
  python scripts/eval_reward_classifier.py --test-dir data/classifier_test/ --output artifacts/

输出:
  - stdout: JSON 格式的评估指标
  - <output>/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# 常量 (与 train_reward_classifier.py 一致)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128
CHANNELS = 3
NUM_CLASSES = 1

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 模型定义 (与 train_reward_classifier.py 完全一致)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """基础残差块: 2x [Conv3x3 -> BN -> ReLU] + skip connection。"""

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
    """ResNet-18 风格二值分类器。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, NUM_CLASSES)
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
        return self.fc(x)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class ClassifierDataset(Dataset):
    """从 data/classifier/{positive,negative}/ 加载图像。"""

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        pos_dir = self.data_dir / "positive"
        if pos_dir.is_dir():
            for f in sorted(pos_dir.iterdir()):
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((f, 1))

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
# 评估函数
# ---------------------------------------------------------------------------
def get_eval_transform() -> transforms.Compose:
    """评估用变换 (无增强)。"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算二值分类指标 (precision, recall, F1, accuracy)。"""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    neg_precision = tn / max(tn + fn, 1)
    neg_recall = tn / max(tn + fp, 1)
    neg_f1 = 2 * neg_precision * neg_recall / max(neg_precision + neg_recall, 1e-8)

    return {
        "accuracy": round(accuracy, 6),
        "positive": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "negative": {
            "precision": round(neg_precision, 6),
            "recall": round(neg_recall, 6),
            "f1": round(neg_f1, 6),
        },
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "total_samples": total,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """在给定 DataLoader 上评估模型。"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).flatten()

    metrics = compute_metrics(all_labels, all_preds)
    metrics["threshold"] = threshold

    # 额外统计: 预测概率分布
    pos_probs = all_probs[all_labels == 1]
    neg_probs = all_probs[all_labels == 0]
    metrics["prob_stats"] = {
        "positive_mean": round(float(np.mean(pos_probs)), 6) if len(pos_probs) > 0 else None,
        "positive_std": round(float(np.std(pos_probs)), 6) if len(pos_probs) > 0 else None,
        "negative_mean": round(float(np.mean(neg_probs)), 6) if len(neg_probs) > 0 else None,
        "negative_std": round(float(np.std(neg_probs)), 6) if len(neg_probs) > 0 else None,
    }

    return metrics


def save_confusion_matrix_png(metrics: dict, output_path: str) -> None:
    """保存混淆矩阵为 PNG 图片 (纯 matplotlib 实现)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping confusion matrix PNG.", file=sys.stderr)
        return

    cm = metrics["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]],
                        [cm["fn"], cm["tp"]]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Reward Classifier Confusion Matrix")
    plt.colorbar(im, ax=ax)

    classes = ["Negative (0)", "Positive (1)"]
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # 在每个格子中标注数值
    thresh = matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=16)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {output_path}")


# ---------------------------------------------------------------------------
# Checkpoint 加载
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> tuple[nn.Module, dict]:
    """加载模型权重 + 元数据。

    Returns:
        (model, metadata_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 支持目录或文件路径
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "reward_classifier.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        description="Evaluate a trained SERL reward classifier on a held-out test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 默认评估 (使用 classifier_ckpt/ 和 data/classifier/)
  python scripts/eval_reward_classifier.py

  # 指定 checkpoint 和测试集
  python scripts/eval_reward_classifier.py \\
      --checkpoint classifier_ckpt/ \\
      --test-dir data/classifier_test/

  # 输出到指定目录
  python scripts/eval_reward_classifier.py --output artifacts/cls_eval/
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="classifier_ckpt/",
        help="Path to checkpoint dir or .pt file (default: classifier_ckpt/)",
    )
    parser.add_argument(
        "--test-dir", type=str, default=None,
        help="Path to test data dir with positive/ and negative/ subdirs "
             "(default: data/classifier/)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for confusion matrix PNG (default: <checkpoint>/)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for evaluation (default: 64)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0)",
    )

    args = parser.parse_args()

    # 路径处理 (相对于项目根)
    project_root = Path(__file__).resolve().parent.parent

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(project_root / checkpoint_path)

    test_dir = args.test_dir
    if test_dir is None:
        test_dir = str(project_root / "data" / "classifier")
    elif not os.path.isabs(test_dir):
        test_dir = str(project_root / test_dir)

    output_dir = args.output
    if output_dir is None:
        output_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    elif not os.path.isabs(output_dir):
        output_dir = str(project_root / output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" Reward Classifier Evaluation")
    print("=" * 60)
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Test dir:    {test_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Threshold:   {args.threshold}")
    print(f"  Device:      {device}")
    print("=" * 60)

    # Step 1: 加载模型
    print("\n[1/3] Loading model...")
    try:
        model, metadata = load_model_from_checkpoint(checkpoint_path, device)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Architecture: {metadata.get('architecture', 'unknown')}")
    print(f"  Trained epoch: {metadata.get('epoch', 'unknown')}")
    if "val_metrics" in metadata:
        print(f"  Val F1 (during training): {metadata['val_metrics'].get('f1', 'N/A'):.4f}")

    # Step 2: 加载测试数据
    print("\n[2/3] Loading test data...")
    try:
        dataset = ClassifierDataset(test_dir, transform=get_eval_transform())
    except FileNotFoundError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Total samples: {len(dataset)} "
          f"(positive={dataset.num_positive}, negative={dataset.num_negative})")

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    # Step 3: 评估
    print("\n[3/3] Evaluating...")
    metrics = evaluate_model(model, dataloader, device, threshold=args.threshold)

    # 附加元数据
    report = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "test_dir": os.path.abspath(test_dir),
        "device": str(device),
        "dataset": {
            "total": len(dataset),
            "positive": dataset.num_positive,
            "negative": dataset.num_negative,
        },
        "train_metadata": {
            "architecture": metadata.get("architecture"),
            "epoch": metadata.get("epoch"),
        },
        "results": metrics,
    }

    # 保存混淆矩阵 PNG
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    save_confusion_matrix_png(metrics, cm_path)
    report["confusion_matrix_png"] = os.path.abspath(cm_path)

    # 打印结果
    print("\n" + "=" * 60)
    print(" Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Positive class:")
    print(f"    Precision:        {metrics['positive']['precision']:.4f}")
    print(f"    Recall:           {metrics['positive']['recall']:.4f}")
    print(f"    F1:               {metrics['positive']['f1']:.4f}")
    print(f"  Negative class:")
    print(f"    Precision:        {metrics['negative']['precision']:.4f}")
    print(f"    Recall:           {metrics['negative']['recall']:.4f}")
    print(f"    F1:               {metrics['negative']['f1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}")
    ps = metrics["prob_stats"]
    if ps["positive_mean"] is not None:
        print(f"  Prob stats (positive): mean={ps['positive_mean']:.4f}, std={ps['positive_std']:.4f}")
    if ps["negative_mean"] is not None:
        print(f"  Prob stats (negative): mean={ps['negative_mean']:.4f}, std={ps['negative_std']:.4f}")
    print("=" * 60)

    # JSON 输出到 stdout
    print("\n--- JSON Report ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
