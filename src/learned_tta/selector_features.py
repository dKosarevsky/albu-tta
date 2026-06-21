"""Feature extraction helpers for lightweight selector baselines."""

from __future__ import annotations

import numpy as np


def clean_logit_features(
    logits: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Build per-image clean-pass features from model logits."""

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [images, classes]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")

    probs = _softmax(logits)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    capped_top_k = min(top_k, logits.shape[1])
    confidence = sorted_probs[:, 0]
    second_prob = sorted_probs[:, 1] if logits.shape[1] > 1 else np.zeros_like(confidence)
    margin = confidence - second_prob
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    pred_class = np.argmax(logits, axis=1).astype(np.float32)

    columns = [
        confidence.astype(np.float32),
        margin.astype(np.float32),
        entropy.astype(np.float32),
        pred_class,
    ]
    names = [
        "clean_confidence",
        "clean_margin",
        "clean_entropy",
        "clean_pred_class",
    ]
    for index in range(capped_top_k):
        columns.append(sorted_probs[:, index].astype(np.float32))
        names.append(f"clean_top{index + 1}_prob")

    return np.stack(columns, axis=1).astype(np.float32), names


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
