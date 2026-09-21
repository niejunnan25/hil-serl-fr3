"""Plug-insertion VICE success gate."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CLASSIFIER_THRESHOLD = 0.78
DEFAULT_DEPTH_THRESHOLD = 0.08
DEFAULT_STREAK_REQUIRED = 3


def classify_success(
    classifier_prob: float,
    depth: float = 0.0,
    *,
    classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
    depth_threshold: float = DEFAULT_DEPTH_THRESHOLD,
) -> bool:
    """Return true when classifier and insertion-depth gates both pass."""
    return bool(
        classifier_prob >= classifier_threshold
        and depth >= depth_threshold
    )


@dataclass
class SuccessStreak:
    required: int = DEFAULT_STREAK_REQUIRED
    classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD
    depth_threshold: float = DEFAULT_DEPTH_THRESHOLD
    current: int = 0

    def update(self, *, classifier_prob: float, depth: float = 0.0) -> bool:
        if (
            classifier_prob >= self.classifier_threshold
            and depth >= self.depth_threshold
        ):
            self.current += 1
        else:
            self.current = 0
        return self.current >= self.required


__all__ = [
    "DEFAULT_CLASSIFIER_THRESHOLD",
    "DEFAULT_DEPTH_THRESHOLD",
    "DEFAULT_STREAK_REQUIRED",
    "SuccessStreak",
    "classify_success",
]
