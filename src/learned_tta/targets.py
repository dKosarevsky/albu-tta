"""Selector target generation from teacher logits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class TrueClassLoss:
    """True-class probabilities and negative log likelihoods."""

    prob_true: np.ndarray
    nll_true: np.ndarray


@dataclass(frozen=True, slots=True)
class GainTargets:
    """Per-image, per-augmentation gain targets."""

    aug_ids: list[str]
    gain: np.ndarray


@dataclass(frozen=True, slots=True)
class TargetStats:
    """Per-augmentation target normalization statistics."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True, slots=True)
class SavedSelectorTargets:
    """Loaded selector target artifact."""

    aug_ids: list[str]
    image_ids: list[str]
    gain: np.ndarray
    target_z: np.ndarray
    stats: TargetStats


def compute_true_class_nll(logits: np.ndarray, class_idxs: np.ndarray) -> TrueClassLoss:
    """Compute true-class probability and NLL from logits."""

    logits = np.asarray(logits, dtype=np.float32)
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_images, num_classes]")
    if class_idxs.shape != (logits.shape[0],):
        raise ValueError("class_idxs must have shape [num_images]")

    probabilities = _softmax(logits)
    true_probs = probabilities[np.arange(logits.shape[0]), class_idxs].astype(np.float32)
    nll_true = (-np.log(np.clip(true_probs, 1e-45, 1.0))).astype(np.float32)
    return TrueClassLoss(prob_true=true_probs, nll_true=nll_true)


def compute_gain_targets(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    identity_aug_id: str,
) -> GainTargets:
    """Compute gain as clean NLL minus augmented NLL for each augmentation."""

    if identity_aug_id not in logits_by_aug:
        raise ValueError(f"identity augmentation {identity_aug_id!r} is missing")

    aug_ids = sorted(logits_by_aug)
    clean_nll = compute_true_class_nll(logits_by_aug[identity_aug_id], class_idxs).nll_true
    gain_columns = []
    for aug_id in aug_ids:
        aug_nll = compute_true_class_nll(logits_by_aug[aug_id], class_idxs).nll_true
        gain_columns.append(clean_nll - aug_nll)

    return GainTargets(
        aug_ids=aug_ids,
        gain=np.stack(gain_columns, axis=1).astype(np.float32),
    )


def compute_target_stats(gain: np.ndarray, min_std: float = 1.0) -> TargetStats:
    """Compute per-augmentation standardization stats from public-train gain."""

    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim != 2:
        raise ValueError("gain must have shape [num_images, num_augmentations]")

    mean = gain.mean(axis=0).astype(np.float32)
    std = gain.std(axis=0).astype(np.float32)
    std = np.maximum(std, min_std).astype(np.float32)
    return TargetStats(mean=mean, std=std)


def standardize_gain_targets(gain: np.ndarray, stats: TargetStats) -> np.ndarray:
    """Standardize gain targets with public-train per-augmentation stats."""

    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim != 2:
        raise ValueError("gain must have shape [num_images, num_augmentations]")
    if stats.mean.shape != (gain.shape[1],) or stats.std.shape != (gain.shape[1],):
        raise ValueError("stats shape must match gain augmentation dimension")

    return ((gain - stats.mean) / stats.std).astype(np.float32)


def save_selector_targets(
    path: Path,
    aug_ids: list[str],
    image_ids: list[str],
    gain: np.ndarray,
    target_z: np.ndarray,
    stats: TargetStats,
) -> None:
    """Save selector targets and normalization stats as a compressed numpy artifact."""

    gain = np.asarray(gain, dtype=np.float32)
    target_z = np.asarray(target_z, dtype=np.float32)
    if gain.ndim != 2:
        raise ValueError("gain must have shape [num_images, num_augmentations]")
    if target_z.shape != gain.shape:
        raise ValueError("target_z shape must match gain shape")
    if len(image_ids) != gain.shape[0]:
        raise ValueError("image_ids length must match selector target rows")
    if len(aug_ids) != gain.shape[1]:
        raise ValueError("aug_ids length must match selector target columns")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        aug_ids=np.asarray(aug_ids),
        image_ids=np.asarray(image_ids),
        gain=gain,
        target_z=target_z,
        mean=np.asarray(stats.mean, dtype=np.float32),
        std=np.asarray(stats.std, dtype=np.float32),
    )


def load_selector_targets(path: Path) -> SavedSelectorTargets:
    """Load selector target artifact from disk."""

    with np.load(path) as data:
        return SavedSelectorTargets(
            aug_ids=[str(aug_id) for aug_id in data["aug_ids"].tolist()],
            image_ids=(
                [str(image_id) for image_id in data["image_ids"].tolist()]
                if "image_ids" in data.files
                else []
            ),
            gain=np.asarray(data["gain"], dtype=np.float32),
            target_z=np.asarray(data["target_z"], dtype=np.float32),
            stats=TargetStats(
                mean=np.asarray(data["mean"], dtype=np.float32),
                std=np.asarray(data["std"], dtype=np.float32),
            ),
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
