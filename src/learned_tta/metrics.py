"""Classification metrics for TTA evaluation."""

from __future__ import annotations

import numpy as np


def classification_metrics(
    probabilities: np.ndarray,
    class_idxs: np.ndarray,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Compute top-k accuracy and NLL from class probabilities."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [num_images, num_classes]")
    if class_idxs.shape != (probabilities.shape[0],):
        raise ValueError("class_idxs must have shape [num_images]")

    metrics: dict[str, float] = {}
    for k in topk:
        effective_k = min(k, probabilities.shape[1])
        top_indices = np.argpartition(
            probabilities,
            kth=probabilities.shape[1] - effective_k,
            axis=1,
        )[:, -effective_k:]
        correct = np.array(
            [class_idx in row for class_idx, row in zip(class_idxs, top_indices, strict=True)]
        )
        metrics[f"top{k}"] = float(correct.mean())

    true_probs = probabilities[np.arange(probabilities.shape[0]), class_idxs]
    metrics["nll"] = float(-np.log(np.clip(true_probs, 1e-45, 1.0)).mean())
    return metrics


def expected_calibration_error(
    probabilities: np.ndarray,
    class_idxs: np.ndarray,
    bins: int = 15,
) -> float:
    """Compute expected calibration error from predicted probabilities."""

    probabilities = np.asarray(probabilities, dtype=np.float32)
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == class_idxs

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        accuracy = correct[mask].mean()
        confidence = confidences[mask].mean()
        ece += float(mask.mean() * abs(accuracy - confidence))
    return ece
