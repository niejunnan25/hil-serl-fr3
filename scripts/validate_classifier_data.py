#!/usr/bin/env python3
"""validate_classifier_data.py

分类器训练数据验证工具 — 验证 reward classifier 图像数据集格式。

数据流: ZED → PNG → train_reward_classifier.py

目录结构:
  <classifier_data>/
    positive/   # 正样本 (成功状态)
    negative/   # 负样本 (失败状态)

用法:
  python validate_classifier_data.py /path/to/classifier_data/
  python validate_classifier_data.py /path/to/classifier_data/ --min-positive 300
  python validate_classifier_data.py /path/to/classifier_data/ --strict
"""

import argparse
import hashlib
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# 检查结果
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    is_warning: bool = False


@dataclass
class ClassReport:
    class_name: str  # "positive" or "negative"
    checks: list[CheckResult] = field(default_factory=list)
    total_images: int = 0
    valid_images: int = 0
    corrupted: list[str] = field(default_factory=list)
    duplicates: dict[str, list[str]] = field(default_factory=dict)  # hash -> [paths]
    wrong_size: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add_pass(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=True, message=message))

    def add_fail(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=False, message=message))

    def add_warning(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=True, message=message, is_warning=True))


@dataclass
class ValidationReport:
    data_dir: str
    positive: ClassReport = field(default_factory=lambda: ClassReport(class_name="positive"))
    negative: ClassReport = field(default_factory=lambda: ClassReport(class_name="negative"))

    @property
    def passed(self) -> bool:
        return self.positive.passed and self.negative.passed

    @property
    def total_checks(self) -> int:
        return len(self.positive.checks) + len(self.negative.checks)

    @property
    def total_pass(self) -> int:
        return sum(1 for c in self.positive.checks + self.negative.checks if c.passed and not c.is_warning)

    @property
    def total_fail(self) -> int:
        return sum(1 for c in self.positive.checks + self.negative.checks if not c.passed and not c.is_warning)

    @property
    def total_warn(self) -> int:
        return sum(1 for c in self.positive.checks + self.negative.checks if c.is_warning)

    def print_report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'=' * 70}")
        print(f"  [{status}] Classifier Data Validation: {self.data_dir}")
        print(f"{'=' * 70}")

        for class_report in [self.positive, self.negative]:
            print(f"\n  --- {class_report.class_name}/ ---")
            for c in class_report.checks:
                if c.is_warning:
                    tag = "WARN"
                elif c.passed:
                    tag = " OK "
                else:
                    tag = "FAIL"
                print(f"    [{tag}] {c.name}: {c.message}")

        print(f"\n  Overall: {self.total_pass} passed, {self.total_fail} failed, {self.total_warn} warnings")
        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# 图像哈希
# ---------------------------------------------------------------------------
def image_hash(file_path: str) -> str | None:
    """计算图像内容的 MD5 哈希。"""
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 单类别验证
# ---------------------------------------------------------------------------
EXPECTED_SHAPE = (128, 128)  # (width, height)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def validate_class(
    class_dir: str,
    report: ClassReport,
    min_count: int,
    strict: bool = False,
) -> None:
    """验证单个类别目录 (positive/ 或 negative/)。"""
    if not os.path.isdir(class_dir):
        report.add_fail("dir_exists", f"目录不存在: {class_dir}")
        return
    report.add_pass("dir_exists", f"目录存在: {class_dir}")

    # 收集所有图像文件
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(Path(class_dir).glob(f"*{ext}"))
        image_files.extend(Path(class_dir).glob(f"*{ext.upper()}"))
    image_files = sorted(set(image_files))

    report.total_images = len(image_files)
    if len(image_files) == 0:
        report.add_fail("has_images", "目录中没有图像文件")
        return
    report.add_pass("has_images", f"找到 {len(image_files)} 个图像文件")

    # 最小数量
    if len(image_files) >= min_count:
        report.add_pass("min_count", f"{len(image_files)} >= {min_count} (满足要求)")
    else:
        report.add_fail("min_count", f"{len(image_files)} < {min_count} (不足)")

    # 逐图像检查
    hash_map: dict[str, list[str]] = defaultdict(list)
    valid_count = 0

    for img_path in image_files:
        img_path_str = str(img_path)
        is_valid = True

        # 图像可读性 & 尺寸
        if Image is not None:
            try:
                img = Image.open(img_path_str)
                img.verify()  # 验证完整性
                img = Image.open(img_path_str)  # verify 后需要重新打开
                w, h = img.size
                mode = img.mode

                if (w, h) != EXPECTED_SHAPE:
                    report.wrong_size.append(img_path_str)
                    is_valid = False

                if mode != "RGB":
                    msg = f"{img_path.name}: mode={mode}, 期望 RGB"
                    if strict:
                        report.add_fail(f"color_mode/{img_path.name}", msg)
                        is_valid = False
                    else:
                        report.add_warning(f"color_mode/{img_path.name}", msg)

            except Exception as e:
                report.corrupted.append(img_path_str)
                report.add_fail(f"corrupted/{img_path.name}", f"无法读取: {e}")
                is_valid = False
        else:
            # 没有 PIL, 只能检查文件大小
            if os.path.getsize(img_path_str) == 0:
                report.corrupted.append(img_path_str)
                report.add_fail(f"empty/{img_path.name}", "文件大小为 0")
                is_valid = False

        # 内容哈希 (去重)
        h = image_hash(img_path_str)
        if h is not None:
            hash_map[h].append(img_path_str)

        if is_valid:
            valid_count += 1

    report.valid_images = valid_count

    # 汇总尺寸问题
    if report.wrong_size:
        report.add_fail("image_size", f"{len(report.wrong_size)} 张图像尺寸不是 {EXPECTED_SHAPE[0]}x{EXPECTED_SHAPE[1]}")
        # 打印前 3 个
        for p in report.wrong_size[:3]:
            if Image is not None:
                try:
                    img = Image.open(p)
                    report.add_fail(f"  size/{Path(p).name}", f"实际 {img.size[0]}x{img.size[1]}")
                except Exception:
                    pass
    else:
        report.add_pass("image_size", f"所有 {valid_count} 张有效图像尺寸为 {EXPECTED_SHAPE[0]}x{EXPECTED_SHAPE[1]}")

    # 汇总损坏
    if report.corrupted:
        report.add_fail("corrupted_count", f"{len(report.corrupted)} 张图像损坏")
    else:
        report.add_pass("corrupted_count", "无损坏图像")

    # 重复检测
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    report.duplicates = duplicates
    if duplicates:
        dup_count = sum(len(paths) for paths in duplicates.values())
        report.add_warning("duplicates", f"{len(duplicates)} 组重复 ({dup_count} 张文件)")
        for h, paths in list(duplicates.items())[:3]:
            names = [Path(p).name for p in paths[:5]]
            report.add_warning(f"  dup/{h[:8]}", f"重复: {', '.join(names)}")
    else:
        report.add_pass("duplicates", "无重复图像")


# ---------------------------------------------------------------------------
# 完整验证
# ---------------------------------------------------------------------------
def validate_classifier_data(
    data_dir: str,
    min_positive: int = 200,
    min_negative: int = 600,
    strict: bool = False,
) -> ValidationReport:
    """验证分类器训练数据目录。"""
    report = ValidationReport(data_dir=os.path.abspath(data_dir))

    if not os.path.isdir(data_dir):
        report.positive.add_fail("parent_dir", f"目录不存在: {data_dir}")
        return report

    positive_dir = os.path.join(data_dir, "positive")
    negative_dir = os.path.join(data_dir, "negative")

    print(f"\nValidating classifier data: {data_dir}")
    print(f"  min positive: {min_positive}, min negative: {min_negative}")
    print(f"  strict: {strict}")

    # PIL 依赖检查
    if Image is None:
        print("[WARN] Pillow (PIL) 未安装, 仅能做文件级检查, 无法验证图像尺寸/完整性")
        print("       安装: pip install Pillow")

    validate_class(positive_dir, report.positive, min_positive, strict)
    validate_class(negative_dir, report.negative, min_negative, strict)

    report.print_report()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="分类器训练数据验证 (positive/ negative/ PNG 图像)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("data_dir", type=str, help="分类器数据根目录 (含 positive/ 和 negative/)")
    parser.add_argument("--min-positive", type=int, default=200, help="正样本最小数量 (默认 200)")
    parser.add_argument("--min-negative", type=int, default=600, help="负样本最小数量 (默认 600)")
    parser.add_argument("--strict", action="store_true", help="严格模式")

    args = parser.parse_args()

    report = validate_classifier_data(
        data_dir=args.data_dir,
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        strict=args.strict,
    )
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
