#!/usr/bin/env python3
"""convert_classifier_pt_to_jax.py

将 train_reward_classifier.py 输出的 PyTorch .pt checkpoint 转换为
SERL load_classifier_func 兼容的 JAX/Orbax 格式。

问题:
  train_reward_classifier.py → PyTorch .pt
  config.py / inference_service.py → load_classifier_func (JAX/Orbax)

  两种格式不兼容。本脚本读取 PyTorch 权重，创建 JAX 模型实例并保存为 Orbax。

用法:
  python scripts/convert_classifier_pt_to_jax.py classifier_ckpt/
  python scripts/convert_classifier_pt_to_jax.py classifier_ckpt/reward_classifier.pt \
      --output-dir classifier_ckpt_jax/

输出:
  <output_dir>/default/   # Orbax checkpoint 目录

依赖:
  - PyTorch checkpoint (train_reward_classifier.py 输出)
  - JAX, Flax, Orbax
  - serl_launcher (for load_classifier_func)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def convert_pt_to_jax(pt_path: str, output_dir: str, verbose: bool = True):
    """将 PyTorch .pt checkpoint 转换为 JAX/Orbax 格式。

    Args:
        pt_path: PyTorch checkpoint 文件路径
        output_dir: Orbax 输出目录
        verbose: 打印详细信息
    """
    import torch

    # 1. 加载 PyTorch checkpoint
    if verbose:
        print(f"Loading PyTorch checkpoint: {pt_path}")

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_state_dict = checkpoint["model_state_dict"]
    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}

    if verbose:
        print(f"  Architecture: {metadata.get('architecture', 'unknown')}")
        print(f"  Image size: {metadata.get('image_size', 'unknown')}")
        print(f"  Epoch: {metadata.get('epoch', 'unknown')}")
        print(f"  Val F1: {metadata.get('val_metrics', {}).get('f1', 'unknown')}")
        print(f"  Keys: {list(pt_state_dict.keys())[:10]}...")

    # 2. 构建 JAX 模型并导入权重
    # 使用 SERL 的 reward_classifier 网络定义
    try:
        from serl_launcher.networks.reward_classifier import make_classifier_func

        # 获取样本输入形状
        image_size = metadata.get("image_size", 128)
        sample_shape = (1, 3, image_size, image_size)

        # 创建 JAX 模型
        if verbose:
            print(f"\nCreating JAX model...")

        # SERL 的 make_classifier_func 返回 (init_fn, apply_fn)
        init_fn, apply_fn = make_classifier_func()

        # 初始化 JAX 参数
        import jax
        import jax.numpy as jnp

        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros(sample_shape)
        jax_params = init_fn(key, dummy_input)

        # 3. 映射 PyTorch 权重到 JAX
        if verbose:
            print(f"\nMapping PyTorch -> JAX weights...")

        jax_params = _map_pytorch_to_jax(pt_state_dict, jax_params, verbose=verbose)

        # 4. 保存为 Orbax
        if verbose:
            print(f"\nSaving Orbax checkpoint to: {output_dir}")

        from orbax.checkpoint import PyTreeCheckpointer
        os.makedirs(output_dir, exist_ok=True)

        checkpointer = PyTreeCheckpointer()
        checkpointer.save(output_dir, {
            "params": jax_params,
            "metadata": metadata,
        })

        if verbose:
            print(f"  Saved to: {output_dir}")
            # Verify
            restored = checkpointer.restore(output_dir)
            print(f"  Verification: loaded {len(jax.tree.leaves(restored['params']))} param leaves")

        return output_dir

    except ImportError as e:
        print(f"\n[ERROR] SERL import failed: {e}")
        print("  Cannot create JAX model without serl_launcher.")
        print("  Alternative: use config.py with PyTorch fallback (already implemented).")
        sys.exit(1)


def _map_pytorch_to_jax(pt_state_dict: dict, jax_params: dict, verbose: bool = True) -> dict:
    """映射 PyTorch state_dict 到 JAX 参数树。

    PyTorch 和 JAX 的参数命名和形状可能不同，需要适配。
    """
    import jax.numpy as jnp

    # 常见的 PyTorch -> JAX 权重映射规则:
    # 1. PyTorch Conv2d weight: (out_ch, in_ch, kH, kW) -> JAX Conv: (kH, kW, in_ch, out_ch)
    # 2. PyTorch Linear weight: (out_ch, in_ch) -> JAX Dense: (in_ch, out_ch)
    # 3. PyTorch BN: weight/bias/running_mean/running_var -> JAX: scale/bias/mean/var

    mapped = 0
    unmapped = []

    # 尝试直接映射 (如果 SERL 使用与 PyTorch 相同的参数结构)
    flat_jax = {}
    for path, value in _flatten_params(jax_params, ""):
        flat_jax[path] = value

    flat_pt = {}
    for name, tensor in pt_state_dict.items():
        flat_pt[name] = tensor.numpy()

    # 尝试映射
    for pt_name, pt_weight in flat_pt.items():
        # 尝试不同的 JAX 命名模式
        jax_candidates = [
            pt_name,
            pt_name.replace(".", "/"),
            pt_name.replace("weight", "kernel"),
            pt_name.replace(".", "/").replace("weight", "kernel"),
        ]

        for candidate in jax_candidates:
            if candidate in flat_jax:
                jax_shape = flat_jax[candidate].shape
                pt_shape = pt_weight.shape

                # 处理 Conv2d 转置
                if len(pt_shape) == 4 and len(jax_shape) == 4:
                    if pt_shape != jax_shape and pt_shape == (jax_shape[3], jax_shape[2], jax_shape[0], jax_shape[1]):
                        pt_weight = pt_weight.transpose(2, 3, 1, 0)
                        if verbose:
                            print(f"  Transposed Conv2d: {pt_name} -> {candidate}")

                # 处理 Linear 转置
                if len(pt_shape) == 2 and len(jax_shape) == 2:
                    if pt_shape[0] == jax_shape[1] and pt_shape[1] == jax_shape[0]:
                        pt_weight = pt_weight.T
                        if verbose:
                            print(f"  Transposed Linear: {pt_name} -> {candidate}")

                if pt_weight.shape == jax_shape:
                    flat_jax[candidate] = jnp.array(pt_weight)
                    mapped += 1
                    break
        else:
            unmapped.append(pt_name)

    if verbose:
        print(f"  Mapped: {mapped} parameters")
        if unmapped:
            print(f"  Unmapped: {len(unmapped)} parameters")
            if len(unmapped) <= 5:
                for name in unmapped:
                    print(f"    - {name}")

    # 重建 JAX 参数树
    return _unflatten_params(flat_jax, jax_params)


def _flatten_params(params, prefix: str):
    """展平嵌套参数树。"""
    if isinstance(params, dict):
        for k, v in params.items():
            path = f"{prefix}/{k}" if prefix else k
            yield from _flatten_params(v, path)
    else:
        yield prefix, params


def _unflatten_params(flat: dict, template: dict, prefix: str = ""):
    """从展平的参数重建嵌套树。"""
    if isinstance(template, dict):
        result = {}
        for k, v in template.items():
            path = f"{prefix}/{k}" if prefix else k
            result[k] = _unflatten_params(flat, v, path)
        return result
    else:
        return flat.get(prefix, template)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch reward classifier checkpoint to JAX/Orbax format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        type=str,
        help="PyTorch .pt file or classifier_ckpt/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output Orbax directory (default: classifier_ckpt_jax/)",
    )
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Resolve input path
    input_path = args.input
    if os.path.isdir(input_path):
        pt_path = os.path.join(input_path, "reward_classifier.pt")
        if not os.path.isfile(pt_path):
            print(f"[ERROR] reward_classifier.pt not found in {input_path}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        pt_path = input_path
    else:
        print(f"[ERROR] Not found: {input_path}")
        sys.exit(1)

    # Output dir
    output_dir = args.output_dir
    if output_dir is None:
        if os.path.isdir(args.input):
            output_dir = args.input  # Save in same directory
        else:
            output_dir = os.path.join(os.path.dirname(pt_path), "classifier_ckpt_jax")

    convert_pt_to_jax(pt_path, output_dir, verbose=not args.quiet)
    print("\nDone.")


if __name__ == "__main__":
    main()
