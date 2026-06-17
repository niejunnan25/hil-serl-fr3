#!/usr/bin/env python3
"""validate_demo_data.py

数据管线格式验证工具 — 验证 NPZ (GELLO 录制) 和 PKL (SERL ReplayBuffer) 格式。

数据流: GELLO → npz → convert_to_pkl → SERL ReplayBuffer

用法:
  python validate_demo_data.py /tmp/gello_demos/              # 自动检测 NPZ 或 PKL 目录
  python validate_demo_data.py /tmp/gello_demos/demo_001.npz  # 验证单个 NPZ
  python validate_demo_data.py /tmp/pkl_demos/ --format pkl   # 强制 PKL 模式
  python validate_demo_data.py /tmp/gello_demos/ --strict     # 严格模式 (warnings → fail)
"""

import argparse
import hashlib
import os
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 检查结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    is_warning: bool = False


@dataclass
class ValidationReport:
    file_path: str
    format_type: str  # "npz" or "pkl"
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed and not c.is_warning)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and not c.is_warning)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.is_warning)

    def add_pass(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=True, message=message))

    def add_fail(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=False, message=message))

    def add_warning(self, name: str, message: str):
        self.checks.append(CheckResult(name=name, passed=True, message=message, is_warning=True))

    def print_report(self):
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'=' * 70}")
        print(f"  [{status}] {self.format_type.upper()} Validation: {self.file_path}")
        print(f"{'=' * 70}")

        for c in self.checks:
            if c.is_warning:
                tag = "WARN"
            elif c.passed:
                tag = " OK "
            else:
                tag = "FAIL"
            print(f"  [{tag}] {c.name}: {c.message}")

        print(f"\n  Summary: {self.pass_count} passed, {self.fail_count} failed, {self.warn_count} warnings")
        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# NPZ 验证
# ---------------------------------------------------------------------------
REQUIRED_NPZ_FIELDS = {
    "joint_poses": "(N, 7) 关节位置",
    "gripper_states": "(N,) 夹爪状态",
    "timestamps": "(N,) 时间戳",
}

OPTIONAL_NPZ_FIELDS = {
    "actions": "(N, 7) 动作",
    "raw_gello": "(N, 8) 原始 GELLO 数据",
    "target": "(N, 7) 目标关节位置",
    "command": "(N, 7) 发送命令",
    "tracking_error": "(N, 7) 跟踪误差",
}


def validate_npz(file_path: str, strict: bool = False) -> ValidationReport:
    """验证单个 NPZ 文件。"""
    report = ValidationReport(file_path=file_path, format_type="npz")

    # 1. 文件存在
    if not os.path.isfile(file_path):
        report.add_fail("file_exists", f"文件不存在: {file_path}")
        return report
    report.add_pass("file_exists", f"文件大小 {os.path.getsize(file_path) / 1024:.1f} KB")

    # 2. 加载
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        report.add_fail("file_loadable", f"加载失败: {e}")
        return report
    report.add_pass("file_loadable", f"字段: {list(data.keys())}")

    # 3. 必要字段
    for field_name, desc in REQUIRED_NPZ_FIELDS.items():
        if field_name in data:
            report.add_pass(f"required/{field_name}", f"存在, shape={data[field_name].shape}, dtype={data[field_name].dtype}")
        else:
            report.add_fail(f"required/{field_name}", f"缺少必要字段 ({desc})")

    # 如果缺少必要字段, 后续检查无法进行
    missing_required = [f for f in REQUIRED_NPZ_FIELDS if f not in data]
    if missing_required:
        data.close()
        return report

    joint_poses = data["joint_poses"]
    gripper_states = data["gripper_states"]
    timestamps = data["timestamps"]
    N = len(joint_poses)

    # 4. 维度检查
    if joint_poses.ndim == 2 and joint_poses.shape[1] == 7:
        report.add_pass("dim/joint_poses", f"shape={joint_poses.shape} (N,7)")
    elif joint_poses.ndim == 1 and len(joint_poses) == 7:
        msg = f"单帧数据 shape={joint_poses.shape}, 建议 reshape (1,7)"
        if strict:
            report.add_fail("dim/joint_poses", msg)
        else:
            report.add_warning("dim/joint_poses", msg)
    else:
        report.add_fail("dim/joint_poses", f"shape={joint_poses.shape}, 期望 (N,7)")

    if gripper_states.ndim == 1:
        report.add_pass("dim/gripper_states", f"shape={gripper_states.shape} (N,)")
    else:
        report.add_fail("dim/gripper_states", f"ndim={gripper_states.ndim}, 期望 1D")

    if timestamps.ndim == 1:
        report.add_pass("dim/timestamps", f"shape={timestamps.shape} (N,)")
    else:
        report.add_fail("dim/timestamps", f"ndim={timestamps.ndim}, 期望 1D")

    # 5. 长度一致性
    n_j = joint_poses.shape[0] if joint_poses.ndim == 2 else 1
    n_g = gripper_states.shape[0]
    n_t = timestamps.shape[0]
    if n_j == n_g == n_t:
        report.add_pass("length_consistency", f"所有必要字段长度一致: N={n_j}")
    else:
        report.add_fail("length_consistency", f"joint_poses={n_j}, gripper_states={n_g}, timestamps={n_t}")

    # 6. NaN / Inf
    for field_name in REQUIRED_NPZ_FIELDS:
        arr = data[field_name]
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        if nan_count > 0:
            report.add_fail(f"nan/{field_name}", f"{nan_count} 个 NaN")
        else:
            report.add_pass(f"nan/{field_name}", "无 NaN")
        if inf_count > 0:
            report.add_fail(f"inf/{field_name}", f"{inf_count} 个 Inf")
        else:
            report.add_pass(f"inf/{field_name}", "无 Inf")

    # 7. 时间戳单调性
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        num_decreasing = int(np.sum(diffs < 0))
        if num_decreasing == 0:
            mean_dt = float(np.mean(diffs))
            hz = 1.0 / mean_dt if mean_dt > 0 else 0
            report.add_pass("timestamps_monotonic", f"单调递增, 平均间隔 {mean_dt:.4f}s ({hz:.1f} Hz)")
        else:
            report.add_fail("timestamps_monotonic", f"{num_decreasing} 处非递增")

    # 8. 夹爪范围 [0, 1]
    g_min, g_max = float(np.min(gripper_states)), float(np.max(gripper_states))
    if g_min >= 0.0 and g_max <= 1.0:
        report.add_pass("gripper_range", f"[{g_min:.3f}, {g_max:.3f}] 在 [0, 1] 内")
    else:
        report.add_fail("gripper_range", f"[{g_min:.3f}, {g_max:.3f}] 超出 [0, 1]")

    # 9. 可选字段
    for field_name, desc in OPTIONAL_NPZ_FIELDS.items():
        if field_name in data:
            arr = data[field_name]
            report.add_pass(f"optional/{field_name}", f"存在, shape={arr.shape}, dtype={arr.dtype}")
            # 可选字段也检查 NaN
            nan_c = int(np.isnan(arr).sum())
            inf_c = int(np.isinf(arr).sum())
            if nan_c > 0:
                report.add_fail(f"optional/{field_name}/nan", f"{nan_c} 个 NaN")
            if inf_c > 0:
                report.add_fail(f"optional/{field_name}/inf", f"{inf_c} 个 Inf")

    # 10. 最小帧数 (convert_to_pkl 需要至少 2 帧)
    if n_j < 2:
        msg = f"仅 {n_j} 帧, convert_to_pkl 需要 >= 2 帧"
        if strict:
            report.add_fail("min_frames", msg)
        else:
            report.add_warning("min_frames", msg)
    else:
        report.add_pass("min_frames", f"{n_j} 帧 (>= 2)")

    data.close()
    return report


# ---------------------------------------------------------------------------
# PKL 验证
# ---------------------------------------------------------------------------
REQUIRED_TRANSITION_KEYS = {
    "observations", "next_observations", "actions", "rewards", "masks", "dones",
}
REQUIRED_OBS_KEYS = {"state", "pixels"}
IMAGE_SHAPE = (3, 128, 128)  # CHW


def validate_pkl(file_path: str, strict: bool = False) -> ValidationReport:
    """验证单个 PKL 文件 (SERL transitions list)。"""
    report = ValidationReport(file_path=file_path, format_type="pkl")

    # 1. 文件存在
    if not os.path.isfile(file_path):
        report.add_fail("file_exists", f"文件不存在: {file_path}")
        return report
    report.add_pass("file_exists", f"文件大小 {os.path.getsize(file_path) / 1024:.1f} KB")

    # 2. 加载
    try:
        with open(file_path, "rb") as f:
            transitions = pickle.load(f)
    except Exception as e:
        report.add_fail("file_loadable", f"加载失败: {e}")
        return report

    if not isinstance(transitions, list):
        report.add_fail("is_list", f"顶层类型是 {type(transitions).__name__}, 期望 list")
        return report

    N = len(transitions)
    report.add_pass("is_list", f"transitions list, 长度={N}")

    if N == 0:
        report.add_fail("non_empty", "空 transitions list")
        return report
    report.add_pass("non_empty", f"{N} 个 transitions")

    # 3. 抽样检查 (最多检查 5 个 + 首尾)
    sample_indices = [0, N - 1]
    if N > 4:
        sample_indices += [N // 4, N // 2, 3 * N // 4]
    sample_indices = sorted(set(i for i in sample_indices if i < N))

    for idx in sample_indices:
        t = transitions[idx]
        prefix = f"t[{idx}]"

        # 顶层键
        for key in REQUIRED_TRANSITION_KEYS:
            if key not in t:
                report.add_fail(f"{prefix}/key/{key}", "缺失")
            else:
                report.add_pass(f"{prefix}/key/{key}", "存在")

        # observations 子结构
        for obs_key in ["observations", "next_observations"]:
            if obs_key not in t:
                continue
            obs = t[obs_key]
            if not isinstance(obs, dict):
                report.add_fail(f"{prefix}/{obs_key}/is_dict", f"类型 {type(obs).__name__}, 期望 dict")
                continue

            for sub_key in REQUIRED_OBS_KEYS:
                if sub_key not in obs:
                    report.add_fail(f"{prefix}/{obs_key}/{sub_key}", "缺失")
                    continue

                val = obs[sub_key]
                if not isinstance(val, np.ndarray):
                    report.add_fail(f"{prefix}/{obs_key}/{sub_key}/is_ndarray", f"类型 {type(val).__name__}")
                    continue

                # state: 8D
                if sub_key == "state":
                    if val.shape == (8,):
                        report.add_pass(f"{prefix}/{obs_key}/state/shape", f"shape={val.shape} (8D)")
                    else:
                        report.add_fail(f"{prefix}/{obs_key}/state/shape", f"shape={val.shape}, 期望 (8,)")

                # pixels: (3, 128, 128)
                if sub_key == "pixels":
                    if val.shape == IMAGE_SHAPE:
                        report.add_pass(f"{prefix}/{obs_key}/pixels/shape", f"shape={val.shape} (3,128,128)")
                    else:
                        report.add_fail(f"{prefix}/{obs_key}/pixels/shape", f"shape={val.shape}, 期望 {IMAGE_SHAPE}")

                    # 零图像警告
                    if np.all(val == 0):
                        report.add_warning(f"{prefix}/{obs_key}/pixels/nonzero", "图像全零 (占位符)")
                    else:
                        report.add_pass(f"{prefix}/{obs_key}/pixels/nonzero", "非零图像")

        # actions: 7D, [-1, 1]
        if "actions" in t:
            act = t["actions"]
            if isinstance(act, np.ndarray):
                if act.shape == (7,):
                    report.add_pass(f"{prefix}/actions/shape", f"shape={act.shape} (7D)")
                else:
                    report.add_fail(f"{prefix}/actions/shape", f"shape={act.shape}, 期望 (7,)")

                a_min, a_max = float(act.min()), float(act.max())
                in_range = a_min >= -1.0 - 1e-5 and a_max <= 1.0 + 1e-5
                if in_range:
                    report.add_pass(f"{prefix}/actions/range", f"[{a_min:.4f}, {a_max:.4f}] 在 [-1, 1]")
                else:
                    report.add_fail(f"{prefix}/actions/range", f"[{a_min:.4f}, {a_max:.4f}] 超出 [-1, 1]")
            else:
                report.add_fail(f"{prefix}/actions/type", f"类型 {type(act).__name__}, 期望 ndarray")

        # rewards: float
        if "rewards" in t:
            r = t["rewards"]
            if isinstance(r, (float, np.floating)):
                report.add_pass(f"{prefix}/rewards/type", f"value={float(r):.4f}")
            else:
                msg = f"类型 {type(r).__name__}, 期望 float/np.float32"
                if strict:
                    report.add_fail(f"{prefix}/rewards/type", msg)
                else:
                    report.add_warning(f"{prefix}/rewards/type", msg)

        # masks: float, 1 - dones
        if "masks" in t and "dones" in t:
            mask_val = float(t["masks"])
            done_val = bool(t["dones"])
            expected_mask = 1.0 - float(done_val)
            if abs(mask_val - expected_mask) < 1e-6:
                report.add_pass(f"{prefix}/masks_consistency", f"mask={mask_val:.1f}, done={done_val} (consistent)")
            else:
                report.add_fail(f"{prefix}/masks_consistency", f"mask={mask_val:.4f} != 1-done={expected_mask:.4f}")

    # 4. 全局 action 统计
    try:
        all_actions = np.array([t["actions"] for t in transitions])
        a_min, a_max = float(all_actions.min()), float(all_actions.max())
        zero_count = int(np.sum(np.linalg.norm(all_actions, axis=1) <= 0.0))
        report.add_pass("global/actions_stats", f"range=[{a_min:.4f}, {a_max:.4f}], zero_actions={zero_count}")
        if zero_count > 0:
            report.add_warning("global/zero_actions", f"{zero_count} 个零动作 transition")
    except Exception as e:
        report.add_warning("global/actions_stats", f"统计失败: {e}")

    # 5. masks == 1 - dones 全局一致性
    try:
        all_masks = np.array([float(t["masks"]) for t in transitions])
        all_dones = np.array([float(bool(t["dones"])) for t in transitions])
        if np.allclose(all_masks, 1.0 - all_dones):
            report.add_pass("global/masks_dones_consistency", "所有 masks == 1 - dones")
        else:
            report.add_fail("global/masks_dones_consistency", "部分 masks != 1 - dones")
    except Exception as e:
        report.add_warning("global/masks_dones_consistency", f"检查失败: {e}")

    # 6. 最后一个 transition 的 done 应为 True
    try:
        last_done = bool(transitions[-1]["dones"])
        if last_done:
            report.add_pass("global/last_done", "最后一个 transition done=True")
        else:
            msg = "最后一个 transition done=False"
            if strict:
                report.add_fail("global/last_done", msg)
            else:
                report.add_warning("global/last_done", msg)
    except Exception as e:
        report.add_warning("global/last_done", f"检查失败: {e}")

    return report


# ---------------------------------------------------------------------------
# 目录批量验证
# ---------------------------------------------------------------------------
def validate_directory(dir_path: str, fmt: str | None = None, strict: bool = False) -> list[ValidationReport]:
    """验证目录中的所有 NPZ 或 PKL 文件。"""
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        print(f"[ERROR] 目录不存在: {dir_path}")
        return []

    # 自动检测格式
    npz_files = sorted(Path(dir_path).glob("**/*.npz"))
    pkl_files = sorted(Path(dir_path).glob("**/*.pkl"))

    if fmt == "npz":
        files = npz_files
    elif fmt == "pkl":
        files = pkl_files
    elif npz_files and not pkl_files:
        files = npz_files
    elif pkl_files and not npz_files:
        files = pkl_files
    elif npz_files and pkl_files:
        print(f"[WARN] 目录中同时存在 NPZ ({len(npz_files)}) 和 PKL ({len(pkl_files)}), 使用 --format 指定")
        files = npz_files  # 默认 NPZ
    else:
        print(f"[ERROR] 目录中没有找到 .npz 或 .pkl 文件: {dir_path}")
        return []

    detected_fmt = fmt or ("pkl" if files and files[0].suffix == ".pkl" else "npz")
    validate_fn = validate_pkl if detected_fmt == "pkl" else validate_npz

    print(f"\nValidating {len(files)} {detected_fmt.upper()} files in {dir_path}")
    print(f"Strict mode: {strict}")

    reports = []
    for f in files:
        r = validate_fn(str(f), strict=strict)
        r.print_report()
        reports.append(r)

    # 汇总
    total_pass = sum(r.pass_count for r in reports)
    total_fail = sum(r.fail_count for r in reports)
    total_warn = sum(r.warn_count for r in reports)
    files_pass = sum(1 for r in reports if r.passed)

    print(f"\n{'=' * 70}")
    print(f"  Overall: {files_pass}/{len(reports)} files passed")
    print(f"  Checks: {total_pass} passed, {total_fail} failed, {total_warn} warnings")
    print(f"{'=' * 70}\n")

    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HIL-SERL 数据管线格式验证 (NPZ / PKL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=str, help="NPZ/PKL 文件或目录路径")
    parser.add_argument("--format", choices=["npz", "pkl"], default=None, help="强制指定格式 (默认自动检测)")
    parser.add_argument("--strict", action="store_true", help="严格模式: warnings 变为 failures")

    args = parser.parse_args()
    input_path = os.path.abspath(args.input)

    if os.path.isfile(input_path):
        # 单文件
        if args.format == "pkl" or (args.format is None and input_path.endswith(".pkl")):
            report = validate_pkl(input_path, strict=args.strict)
        else:
            report = validate_npz(input_path, strict=args.strict)
        report.print_report()
        sys.exit(0 if report.passed else 1)

    elif os.path.isdir(input_path):
        reports = validate_directory(input_path, fmt=args.format, strict=args.strict)
        all_passed = all(r.passed for r in reports) if reports else False
        sys.exit(0 if all_passed else 1)

    else:
        print(f"[ERROR] 路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
