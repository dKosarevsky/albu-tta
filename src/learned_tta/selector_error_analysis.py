"""Selector error-analysis tables for learned TTA policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from learned_tta.tta_eval import learned_topk_selection, oracle_topk_selection


def build_selector_error_analysis_table(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    k: int,
    confidence_bins: list[float] | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Summarize where a selector fixes, breaks, or misses oracle choices."""

    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if predicted_gain.shape != (len(class_idxs), len(aug_ids)):
        raise ValueError("predicted_gain shape must be [images, augmentations]")
    clean_logits = np.asarray(logits_by_aug[identity_aug_id], dtype=np.float32)
    clean_probs = _softmax(clean_logits)
    clean_confidence = clean_probs.max(axis=1)
    clean_pred = np.argmax(clean_probs, axis=1)
    clean_correct = clean_pred == class_idxs
    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    oracle_aug_ids = oracle_topk_selection(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    tta_probs = _average_selected_probabilities(logits_by_aug, selected_aug_ids)
    tta_correct = np.argmax(tta_probs, axis=1) == class_idxs
    rows = []
    for index, (selected, oracle) in enumerate(zip(selected_aug_ids, oracle_aug_ids, strict=True)):
        rows.append(
            {
                "image_index": index,
                "confidence_bucket": _confidence_bucket(
                    float(clean_confidence[index]),
                    confidence_bins or [0.0, 0.5, 0.75, 0.9, 1.01],
                ),
                "clean_confidence": float(clean_confidence[index]),
                "clean_top1": bool(clean_correct[index]),
                "tta_top1": bool(tta_correct[index]),
                "clean_wrong_tta_right": bool((not clean_correct[index]) and tta_correct[index]),
                "clean_right_tta_wrong": bool(clean_correct[index] and (not tta_correct[index])),
                "oracle_recall": _single_oracle_recall(
                    selected=selected,
                    oracle=oracle,
                    identity_aug_id=identity_aug_id,
                ),
                "selected_count": len(selected),
            }
        )
    table = pd.DataFrame(rows)
    grouped = (
        table.groupby("confidence_bucket", sort=True, observed=False)
        .agg(
            images=("image_index", "count"),
            mean_clean_confidence=("clean_confidence", "mean"),
            clean_top1=("clean_top1", "mean"),
            tta_top1=("tta_top1", "mean"),
            clean_wrong_tta_right=("clean_wrong_tta_right", "sum"),
            clean_right_tta_wrong=("clean_right_tta_wrong", "sum"),
            mean_oracle_recall=("oracle_recall", "mean"),
            mean_selected_count=("selected_count", "mean"),
        )
        .reset_index()
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(output_path, index=False)
    return grouped


def _average_selected_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
) -> np.ndarray:
    rows = []
    for image_index, aug_ids in enumerate(selected_aug_ids):
        probs = [
            _softmax(logits_by_aug[aug_id][image_index : image_index + 1])[0] for aug_id in aug_ids
        ]
        rows.append(np.mean(probs, axis=0))
    return np.asarray(rows, dtype=np.float32)


def _confidence_bucket(confidence: float, bins: list[float]) -> str:
    if len(bins) < 2:
        raise ValueError("confidence_bins must contain at least two values")
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        if lower <= confidence < upper:
            return f"[{lower:.2f}, {upper:.2f})"
    return f"[{bins[-2]:.2f}, {bins[-1]:.2f})"


def _single_oracle_recall(
    selected: list[str],
    oracle: list[str],
    identity_aug_id: str,
) -> float:
    selected_set = set(selected) - {identity_aug_id}
    oracle_set = set(oracle) - {identity_aug_id}
    if not oracle_set:
        return 1.0
    return len(selected_set & oracle_set) / len(oracle_set)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
