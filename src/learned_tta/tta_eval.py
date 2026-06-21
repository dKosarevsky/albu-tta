"""TTA selection and evaluation helpers."""

from __future__ import annotations

import random
from typing import cast

import numpy as np

from learned_tta.metrics import classification_metrics, expected_calibration_error
from learned_tta.targets import compute_gain_targets


def average_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[str],
) -> np.ndarray:
    """Average softmax probabilities for a shared list of augmentations."""

    probabilities = [_softmax(logits_by_aug[aug_id]) for aug_id in selected_aug_ids]
    return np.mean(probabilities, axis=0).astype(np.float32)


def weighted_average_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
    aug_ids: list[str],
    predicted_gain: np.ndarray,
) -> np.ndarray:
    """Average per-image softmax probabilities with selector-score softmax weights."""

    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [num_images, augmentations]")
    if predicted_gain.shape != (len(selected_aug_ids), len(aug_ids)):
        raise ValueError("predicted_gain shape must match selected images and aug_ids")

    aug_index = {aug_id: index for index, aug_id in enumerate(aug_ids)}
    rows = []
    for image_index, image_aug_ids in enumerate(selected_aug_ids):
        selection_indices = [aug_index[aug_id] for aug_id in image_aug_ids]
        weights = _softmax_vector(predicted_gain[image_index, selection_indices])
        row_probabilities = np.stack(
            [
                _softmax(logits_by_aug[aug_id][image_index : image_index + 1])[0]
                for aug_id in image_aug_ids
            ],
            axis=0,
        )
        rows.append(np.sum(row_probabilities * weights[:, None], axis=0))
    return np.asarray(rows, dtype=np.float32)


def learned_topk_selection(
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    k: int,
) -> list[list[str]]:
    """Select identity plus per-image top-k non-identity augmentations by predicted gain."""

    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [num_images, augmentations]")
    if predicted_gain.shape[1] != len(aug_ids):
        raise ValueError("predicted_gain width must match aug_ids")

    identity_index = aug_ids.index(identity_aug_id)
    selections = []
    for image_gain in predicted_gain:
        ranked_indices = [
            index for index in np.argsort(image_gain)[::-1].tolist() if index != identity_index
        ][:k]
        selections.append([identity_aug_id, *[aug_ids[index] for index in ranked_indices]])
    return selections


def adaptive_topk_selection(
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    useful_prob: np.ndarray,
    identity_aug_id: str,
    threshold: float,
    max_k: int,
) -> list[list[str]]:
    """Select identity plus useful non-identity augmentations capped by predicted gain."""

    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    useful_prob = np.asarray(useful_prob, dtype=np.float32)
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [num_images, augmentations]")
    if useful_prob.shape != predicted_gain.shape:
        raise ValueError("useful_prob must match predicted_gain shape")
    if predicted_gain.shape[1] != len(aug_ids):
        raise ValueError("predicted_gain width must match aug_ids")
    if max_k < 0:
        raise ValueError("max_k must be non-negative")

    identity_index = aug_ids.index(identity_aug_id)
    selections = []
    for image_gain, image_useful_prob in zip(predicted_gain, useful_prob, strict=True):
        ranked_indices = [
            index
            for index in np.argsort(image_gain)[::-1].tolist()
            if index != identity_index and image_useful_prob[index] > threshold
        ][:max_k]
        selections.append([identity_aug_id, *[aug_ids[index] for index in ranked_indices]])
    return selections


def confidence_adaptive_topk_selection(
    clean_logits: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    low_confidence_threshold: float,
    high_confidence_threshold: float,
    low_confidence_k: int,
    mid_confidence_k: int,
    high_confidence_k: int,
) -> list[list[str]]:
    """Select per-image top-k using clean confidence buckets."""

    clean_logits = np.asarray(clean_logits, dtype=np.float32)
    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if clean_logits.ndim != 2:
        raise ValueError("clean_logits must have shape [images, classes]")
    if predicted_gain.shape != (clean_logits.shape[0], len(aug_ids)):
        raise ValueError("predicted_gain shape must match clean_logits rows and aug_ids")
    if not 0.0 <= low_confidence_threshold <= high_confidence_threshold <= 1.0:
        raise ValueError("confidence thresholds must satisfy 0 <= low <= high <= 1")
    if min(low_confidence_k, mid_confidence_k, high_confidence_k) < 0:
        raise ValueError("confidence-adaptive k values must be non-negative")

    clean_confidence = _softmax(clean_logits).max(axis=1)
    identity_index = aug_ids.index(identity_aug_id)
    selections = []
    for confidence, image_gain in zip(clean_confidence, predicted_gain, strict=True):
        if confidence < low_confidence_threshold:
            k = low_confidence_k
        elif confidence < high_confidence_threshold:
            k = mid_confidence_k
        else:
            k = high_confidence_k
        ranked_indices = [
            index for index in np.argsort(image_gain)[::-1].tolist() if index != identity_index
        ][:k]
        selections.append([identity_aug_id, *[aug_ids[index] for index in ranked_indices]])
    return selections


def fixed_light_tta_selection(aug_ids: list[str], identity_aug_id: str, k: int) -> list[str]:
    """Select identity plus the first k non-identity augmentations."""

    non_identity = [aug_id for aug_id in aug_ids if aug_id != identity_aug_id]
    return [identity_aug_id, *non_identity[:k]]


def random_topk_selection(
    aug_ids: list[str],
    num_images: int,
    identity_aug_id: str,
    k: int,
    seed: int,
) -> list[list[str]]:
    """Select identity plus k random non-identity augmentations per image."""

    rng = random.Random(seed)
    non_identity = [aug_id for aug_id in aug_ids if aug_id != identity_aug_id]
    return [
        [identity_aug_id, *rng.sample(non_identity, k=min(k, len(non_identity)))]
        for _ in range(num_images)
    ]


def oracle_topk_selection(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    identity_aug_id: str,
    k: int,
) -> list[list[str]]:
    """Select identity plus per-image top-k augmentations by true gain."""

    targets = compute_gain_targets(logits_by_aug, class_idxs, identity_aug_id=identity_aug_id)
    return learned_topk_selection(
        aug_ids=targets.aug_ids,
        predicted_gain=targets.gain,
        identity_aug_id=identity_aug_id,
        k=k,
    )


def evaluate_selected_tta(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[str] | list[list[str]],
    class_idxs: np.ndarray,
) -> dict[str, float]:
    """Evaluate TTA for shared or per-image augmentation selections."""

    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    if selected_aug_ids and isinstance(selected_aug_ids[0], str):
        shared_aug_ids = cast(list[str], selected_aug_ids)
        probabilities = average_probabilities(logits_by_aug, shared_aug_ids)
    else:
        per_image_aug_ids = cast(list[list[str]], selected_aug_ids)
        probabilities = _average_per_image(logits_by_aug, per_image_aug_ids)

    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    metrics["forwards_per_image"] = float(_mean_selection_size(selected_aug_ids))
    metrics["relative_compute_vs_all"] = metrics["forwards_per_image"] / len(logits_by_aug)
    return metrics


def evaluate_weighted_selected_tta(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
) -> dict[str, float]:
    """Evaluate per-image selected TTA with selector-score softmax weights."""

    probabilities = weighted_average_probabilities(
        logits_by_aug=logits_by_aug,
        selected_aug_ids=selected_aug_ids,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
    )
    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    metrics["forwards_per_image"] = float(_mean_selection_size(selected_aug_ids))
    metrics["relative_compute_vs_all"] = metrics["forwards_per_image"] / len(logits_by_aug)
    return metrics


def evaluate_clean(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    identity_aug_id: str,
) -> dict[str, float]:
    """Evaluate the teacher without TTA."""

    return evaluate_selected_tta(logits_by_aug, [identity_aug_id], class_idxs)


def evaluate_fixed_light_tta(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    identity_aug_id: str,
    k: int,
) -> dict[str, float]:
    """Evaluate identity plus the first k non-identity augmentation candidates."""

    selected_aug_ids = fixed_light_tta_selection(
        aug_ids=aug_ids,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_random_topk(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    identity_aug_id: str,
    k: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate identity plus random non-identity candidates per image."""

    selected_aug_ids = random_topk_selection(
        aug_ids=aug_ids,
        num_images=len(class_idxs),
        identity_aug_id=identity_aug_id,
        k=k,
        seed=seed,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_all_100_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate uniform probability averaging over every available candidate."""

    selected_aug_ids = aug_ids if aug_ids is not None else sorted(logits_by_aug)
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def global_weighted_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Average augmentation probabilities with one non-negative weight per augmentation."""

    weights = _normalize_nonnegative_weights(np.asarray(weights, dtype=np.float32), "weights")
    if weights.shape != (len(aug_ids),):
        raise ValueError("weights must have shape [augmentations]")

    probabilities = np.stack([_softmax(logits_by_aug[aug_id]) for aug_id in aug_ids], axis=0)
    return np.tensordot(weights, probabilities, axes=(0, 0)).astype(np.float32)


def class_weighted_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    class_weights: np.ndarray,
) -> np.ndarray:
    """Average probabilities with separate non-negative augmentation weights per class."""

    class_weights = np.asarray(class_weights, dtype=np.float32)
    if class_weights.ndim != 2 or class_weights.shape[1] != len(aug_ids):
        raise ValueError("class_weights must have shape [classes, augmentations]")
    if np.any(class_weights < 0.0):
        raise ValueError("class_weights must be non-negative")
    row_sums = class_weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each class must have at least one positive augmentation weight")

    probabilities = np.stack([_softmax(logits_by_aug[aug_id]) for aug_id in aug_ids], axis=0)
    if probabilities.shape[2] != class_weights.shape[0]:
        raise ValueError("class_weights class count must match logits class count")

    normalized = class_weights / row_sums
    scores = np.einsum("anc,ca->nc", probabilities, normalized, optimize=True)
    scores_sum = scores.sum(axis=1, keepdims=True)
    return (scores / np.clip(scores_sum, 1e-45, None)).astype(np.float32)


def evaluate_global_weighted_tta(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    weights: np.ndarray,
    active_threshold: float = 1e-6,
) -> dict[str, float]:
    """Evaluate learned global non-negative TTA aggregation weights."""

    probabilities = global_weighted_probabilities(logits_by_aug, aug_ids, weights)
    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    metrics["forwards_per_image"] = float(_active_weight_count(weights, active_threshold))
    metrics["relative_compute_vs_all"] = metrics["forwards_per_image"] / len(aug_ids)
    return metrics


def evaluate_class_weighted_tta(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    class_weights: np.ndarray,
    active_threshold: float = 1e-6,
) -> dict[str, float]:
    """Evaluate learned class-specific non-negative TTA aggregation weights."""

    probabilities = class_weighted_probabilities(logits_by_aug, aug_ids, class_weights)
    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    metrics["forwards_per_image"] = float(
        _active_weight_count_per_any_class(class_weights, active_threshold)
    )
    metrics["relative_compute_vs_all"] = metrics["forwards_per_image"] / len(aug_ids)
    return metrics


def evaluate_learned_topk_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    k: int,
) -> dict[str, float]:
    """Evaluate learned identity plus top-k selection with uniform averaging."""

    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_learned_adaptive_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    useful_prob: np.ndarray,
    identity_aug_id: str,
    threshold: float,
    max_k: int,
) -> dict[str, float]:
    """Evaluate learned adaptive usefulness-thresholded uniform TTA."""

    selected_aug_ids = adaptive_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        useful_prob=useful_prob,
        identity_aug_id=identity_aug_id,
        threshold=threshold,
        max_k=max_k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_confidence_adaptive_topk_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    low_confidence_threshold: float,
    high_confidence_threshold: float,
    low_confidence_k: int,
    mid_confidence_k: int,
    high_confidence_k: int,
) -> dict[str, float]:
    """Evaluate confidence-bucketed learned top-k selection with uniform averaging."""

    selected_aug_ids = confidence_adaptive_topk_selection(
        clean_logits=logits_by_aug[identity_aug_id],
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        low_confidence_threshold=low_confidence_threshold,
        high_confidence_threshold=high_confidence_threshold,
        low_confidence_k=low_confidence_k,
        mid_confidence_k=mid_confidence_k,
        high_confidence_k=high_confidence_k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_learned_topk_softmax_weighted(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    k: int,
) -> dict[str, float]:
    """Evaluate learned top-k selection with selector-score softmax weights."""

    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    return evaluate_weighted_selected_tta(
        logits_by_aug=logits_by_aug,
        selected_aug_ids=selected_aug_ids,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
    )


def evaluate_oracle_topk_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    identity_aug_id: str,
    k: int,
) -> dict[str, float]:
    """Evaluate private diagnostic oracle top-k selection with uniform averaging."""

    selected_aug_ids = oracle_topk_selection(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        identity_aug_id=identity_aug_id,
        k=k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def oracle_selection_recall(
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
    identity_aug_id: str,
) -> float:
    """Compute mean recall of selected non-identity ids against oracle non-identity ids."""

    if len(selected_aug_ids) != len(oracle_aug_ids):
        raise ValueError("selected_aug_ids and oracle_aug_ids must have matching lengths")

    recalls = []
    for selected, oracle in zip(selected_aug_ids, oracle_aug_ids, strict=True):
        selected_set = set(selected) - {identity_aug_id}
        oracle_set = set(oracle) - {identity_aug_id}
        if not oracle_set:
            recalls.append(1.0)
            continue
        recalls.append(len(selected_set & oracle_set) / len(oracle_set))
    return float(np.mean(recalls))


def select_best_k(
    results_by_k: dict[int, dict[str, float]],
    metric: str,
    higher_is_better: bool,
) -> int:
    """Select the best public-val k, using lower k as the deterministic tie-breaker."""

    if not results_by_k:
        raise ValueError("results_by_k must not be empty")

    def sort_key(item: tuple[int, dict[str, float]]) -> tuple[float, int]:
        k, metrics = item
        metric_value = metrics[metric]
        comparable_value = -metric_value if higher_is_better else metric_value
        return comparable_value, k

    return min(results_by_k.items(), key=sort_key)[0]


def _average_per_image(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
) -> np.ndarray:
    rows = []
    for image_index, image_aug_ids in enumerate(selected_aug_ids):
        row_probabilities = [
            _softmax(logits_by_aug[aug_id][image_index : image_index + 1])[0]
            for aug_id in image_aug_ids
        ]
        rows.append(np.mean(row_probabilities, axis=0))
    return np.asarray(rows, dtype=np.float32)


def _mean_selection_size(selected_aug_ids: list[str] | list[list[str]]) -> float:
    if not selected_aug_ids:
        return 0.0
    if isinstance(selected_aug_ids[0], str):
        return float(len(selected_aug_ids))
    per_image_aug_ids = cast(list[list[str]], selected_aug_ids)
    return float(np.mean([len(selection) for selection in per_image_aug_ids]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _softmax_vector(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    shifted = scores - scores.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _normalize_nonnegative_weights(weights: np.ndarray, name: str) -> np.ndarray:
    if np.any(weights < 0.0):
        raise ValueError(f"{name} must be non-negative")
    weight_sum = weights.sum()
    if weight_sum <= 0.0:
        raise ValueError(f"{name} must contain at least one positive value")
    return weights / weight_sum


def _active_weight_count(weights: np.ndarray, active_threshold: float) -> int:
    return int(np.count_nonzero(np.asarray(weights, dtype=np.float32) > active_threshold))


def _active_weight_count_per_any_class(
    class_weights: np.ndarray,
    active_threshold: float,
) -> int:
    return int(
        np.count_nonzero(np.asarray(class_weights, dtype=np.float32).max(axis=0) > active_threshold)
    )
