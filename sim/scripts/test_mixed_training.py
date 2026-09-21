"""test_mixed_training.py — mixed sim negative + mock real positive smoke (per PLAN-A12).

⚠️ CRITICAL: 本测试是 SCHEMA SMOKE, 不是 READINESS。
- 通过条件 = 脚本跑通 + 模型 fit + 预测 emit (exit 0)
- 报告含 accuracy / baseline / confusion matrix
- accuracy 数值 ∈ [0, 1] (不验证 ≥ 阈值)

本 plan 不依赖 real pkl, 不做 balanced fixture, 不做 precision/recall ≥ 0.85 验证。
真正 mixed training = phase6-ready gate (用户批准)。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from typing import Any, Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_features(
    sim_pkl_path: str, real_pkl_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load sim (reward=0) + real (80% pos) pkl, return X (N, state_dim) + y (N,).

    Features: live flat state (no images).
    Labels: rewards (0.0 / 1.0).
    """
    def _load(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    sim_transitions = _load(sim_pkl_path)
    real_transitions = _load(real_pkl_path)
    X_sim = np.stack([t["observations"]["state"] for t in sim_transitions]).astype(np.float32)
    y_sim = np.array([float(t["rewards"]) for t in sim_transitions], dtype=np.float32)
    X_real = np.stack([t["observations"]["state"] for t in real_transitions]).astype(np.float32)
    y_real = np.array([float(t["rewards"]) for t in real_transitions], dtype=np.float32)
    X = np.concatenate([X_sim, X_real], axis=0)
    y = np.concatenate([y_sim, y_real], axis=0)
    return X, y


def train_classifier(
    X: np.ndarray, y: np.ndarray, max_iter: int = 200, random_state: int = 0,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train sklearn LogisticRegression on (X, y). Returns (model, X_test, y_test).

    80/20 stratified split; test set is what we report on.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn not installed; pip install scikit-learn")
    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y,
    )
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def compute_report(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute accuracy + baseline (majority class) + confusion matrix.

    Returns:
        dict with keys:
          - accuracy: float
          - baseline_accuracy: float (majority class baseline = max(pos_ratio, 1-pos_ratio))
          - confusion_matrix: list of 4 ints [tn, fp, fn, tp]
          - n_test: int
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    pos_ratio = float((y_true == 1.0).sum() / len(y_true))
    baseline_accuracy = float(max(pos_ratio, 1.0 - pos_ratio))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # sklearn 2x2 confusion matrix: rows=true, cols=pred; [[tn, fp], [fn, tp]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "confusion_matrix": [tn, fp, fn, tp],
        "n_test": int(len(y_true)),
    }


def print_report(report: dict[str, Any]) -> None:
    """Print mixed-training report to stdout."""
    print("\n=== test_mixed_training report ===")
    print(f"  accuracy:           {report['accuracy']:.4f}")
    print(f"  baseline_accuracy:  {report['baseline_accuracy']:.4f}  (majority class)")
    tn, fp, fn, tp = report["confusion_matrix"]
    print(f"  confusion_matrix:   tn={tn} fp={fp} fn={fn} tp={tp}  (rows=true, cols=pred)")
    print(f"  n_test:             {report['n_test']}")
    print(f"  --- smoke pass (script ran end-to-end, NOT accuracy-based) ---")


def main() -> int:
    """CLI entry point. Returns 0 (smoke pass) if script runs end-to-end."""
    parser = argparse.ArgumentParser(description="Mixed sim+real training smoke test")
    parser.add_argument("--sim", required=True, help="Path to sim pkl (negatives)")
    parser.add_argument("--real", required=True, help="Path to mock real pkl (mixed labels)")
    args = parser.parse_args()
    try:
        X, y = load_features(args.sim, args.real)
        model, X_test, y_test = train_classifier(X, y, max_iter=200, random_state=0)
        preds = model.predict(X_test)
        report = compute_report(y_test, preds)
        print_report(report)
        return 0
    except Exception as e:
        print(f"\n[FAIL] test_mixed_training raised: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
