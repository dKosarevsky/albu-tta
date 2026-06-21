"""Alternate selector target construction from clean and augmented logits."""

from __future__ import annotations

import numpy as np


def build_selector_targets(
    clean_logits: np.ndarray,
    aug_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build target matrices for selector training ablations."""

    clean_logits = np.asarray(clean_logits, dtype=np.float32)
    aug_logits = np.asarray(aug_logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    _validate_shapes(clean_logits, aug_logits, labels)

    clean_nll = _true_class_nll(clean_logits, labels)
    aug_nll = np.stack(
        [
            _true_class_nll(aug_logits[:, aug_index, :], labels)
            for aug_index in range(aug_logits.shape[1])
        ],
        axis=1,
    )
    clean_true_logits = clean_logits[np.arange(clean_logits.shape[0]), labels]
    aug_true_logits = aug_logits[
        np.arange(aug_logits.shape[0])[:, None],
        np.arange(aug_logits.shape[1])[None, :],
        labels[:, None],
    ]
    clean_margin = _true_class_margin(clean_logits, labels)
    aug_margin = np.stack(
        [
            _true_class_margin(aug_logits[:, aug_index, :], labels)
            for aug_index in range(aug_logits.shape[1])
        ],
        axis=1,
    )
    clean_correct = np.argmax(clean_logits, axis=1) == labels
    aug_correct = np.argmax(aug_logits, axis=2) == labels[:, None]
    return {
        "nll_gain": (clean_nll[:, None] - aug_nll).astype(np.float32),
        "true_logit_gain": (aug_true_logits - clean_true_logits[:, None]).astype(np.float32),
        "margin_gain": (aug_margin - clean_margin[:, None]).astype(np.float32),
        "top1_fix": ((~clean_correct[:, None]) & aug_correct).astype(np.float32),
    }


def _validate_shapes(clean_logits: np.ndarray, aug_logits: np.ndarray, labels: np.ndarray) -> None:
    if clean_logits.ndim != 2:
        raise ValueError("clean_logits must have shape [images, classes]")
    if aug_logits.ndim != 3:
        raise ValueError("aug_logits must have shape [images, augmentations, classes]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [images]")
    if aug_logits.shape[0] != clean_logits.shape[0] or labels.shape[0] != clean_logits.shape[0]:
        raise ValueError("clean_logits, aug_logits, and labels must have the same image count")
    if aug_logits.shape[2] != clean_logits.shape[1]:
        raise ValueError("aug_logits class dimension must match clean_logits")
    if labels.size and (labels.min() < 0 or labels.max() >= clean_logits.shape[1]):
        raise ValueError("labels contain class indexes outside the logits shape")


def _true_class_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + np.max(logits, axis=1)
    return (logsumexp - logits[np.arange(logits.shape[0]), labels]).astype(np.float32)


def _true_class_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    true_logits = logits[np.arange(logits.shape[0]), labels]
    other_logits = logits.copy()
    other_logits[np.arange(logits.shape[0]), labels] = -np.inf
    return (true_logits - np.max(other_logits, axis=1)).astype(np.float32)
