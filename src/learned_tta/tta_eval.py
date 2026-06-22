"""TTA selection and evaluation helpers."""

from __future__ import annotations

import random
from typing import TypedDict, cast

import numpy as np

from learned_tta.metrics import classification_metrics, expected_calibration_error
from learned_tta.targets import compute_gain_targets


class ConfidenceBucketPolicy(TypedDict, total=False):
    """Calibrated clean-confidence bucket policy."""

    confidence_bins: list[float]
    bucket_k: list[int]
    metric: str
    rows: list[dict[str, float | int]]


def average_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[str],
) -> np.ndarray:
    """Average softmax probabilities for a shared list of augmentations."""

    probabilities = [_softmax(logits_by_aug[aug_id]) for aug_id in selected_aug_ids]
    return np.mean(probabilities, axis=0).astype(np.float32)


def average_per_image_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
) -> np.ndarray:
    """Average softmax probabilities for variable per-image augmentation selections."""

    reference_shape = _validate_per_image_selection(logits_by_aug, selected_aug_ids)
    totals = np.zeros(reference_shape, dtype=np.float32)
    counts = np.zeros(reference_shape[0], dtype=np.float32)

    image_indices_by_aug = _image_indices_by_aug(selected_aug_ids)
    for aug_id, image_indices in image_indices_by_aug.items():
        indices = np.asarray(image_indices, dtype=np.int64)
        probabilities = _softmax(logits_by_aug[aug_id][indices])
        np.add.at(totals, indices, probabilities)
        np.add.at(counts, indices, 1.0)

    return (totals / counts[:, None]).astype(np.float32)


def weighted_average_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    score_temperature: float = 1.0,
) -> np.ndarray:
    """Average per-image softmax probabilities with selector-score softmax weights."""

    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [num_images, augmentations]")
    if predicted_gain.shape != (len(selected_aug_ids), len(aug_ids)):
        raise ValueError("predicted_gain shape must match selected images and aug_ids")
    if score_temperature <= 0.0:
        raise ValueError("score_temperature must be positive")

    reference_shape = _validate_per_image_selection(logits_by_aug, selected_aug_ids)
    aug_index = {aug_id: index for index, aug_id in enumerate(aug_ids)}
    totals = np.zeros(reference_shape, dtype=np.float32)
    image_indices_by_aug: dict[str, list[int]] = {}
    weights_by_aug: dict[str, list[float]] = {}
    for image_index, image_aug_ids in enumerate(selected_aug_ids):
        unknown_aug_ids = [aug_id for aug_id in image_aug_ids if aug_id not in aug_index]
        if unknown_aug_ids:
            raise ValueError(f"unknown augmentation id in aug_ids: {unknown_aug_ids[0]!r}")
        selection_indices = [aug_index[aug_id] for aug_id in image_aug_ids]
        weights = _softmax_vector(
            predicted_gain[image_index, selection_indices] / float(score_temperature)
        )
        for aug_id, weight in zip(image_aug_ids, weights, strict=True):
            image_indices_by_aug.setdefault(aug_id, []).append(image_index)
            weights_by_aug.setdefault(aug_id, []).append(float(weight))

    for aug_id, image_indices in image_indices_by_aug.items():
        indices = np.asarray(image_indices, dtype=np.int64)
        weights = np.asarray(weights_by_aug[aug_id], dtype=np.float32)
        probabilities = _softmax(logits_by_aug[aug_id][indices])
        np.add.at(totals, indices, probabilities * weights[:, None])
    return totals.astype(np.float32)


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


def confidence_bucket_topk_selection(
    clean_logits: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    confidence_bins: list[float],
    bucket_k: list[int],
) -> list[list[str]]:
    """Select per-image top-k using an arbitrary clean-confidence bucket policy."""

    clean_logits = np.asarray(clean_logits, dtype=np.float32)
    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    bins = _validate_confidence_bins(confidence_bins)
    if clean_logits.ndim != 2:
        raise ValueError("clean_logits must have shape [images, classes]")
    if predicted_gain.shape != (clean_logits.shape[0], len(aug_ids)):
        raise ValueError("predicted_gain shape must match clean_logits rows and aug_ids")
    if len(bucket_k) != len(bins) - 1:
        raise ValueError("bucket_k must contain one k value per confidence interval")
    if min(bucket_k, default=0) < 0:
        raise ValueError("bucket_k values must be non-negative")

    confidence = _softmax(clean_logits).max(axis=1)
    bucket_indices = _confidence_bucket_indices(confidence, bins)
    identity_index = aug_ids.index(identity_aug_id)
    selections = []
    for bucket_index, image_gain in zip(bucket_indices, predicted_gain, strict=True):
        k = bucket_k[int(bucket_index)]
        ranked_indices = [
            index for index in np.argsort(image_gain)[::-1].tolist() if index != identity_index
        ][:k]
        selections.append([identity_aug_id, *[aug_ids[index] for index in ranked_indices]])
    return selections


def calibrate_confidence_bucket_policy(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    confidence_bins: list[float],
    k_grid: list[int],
    metric: str = "top1",
) -> ConfidenceBucketPolicy:
    """Choose the best learned top-k value independently for each confidence bucket."""

    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    predicted_gain = np.asarray(predicted_gain, dtype=np.float32)
    bins = _validate_confidence_bins(confidence_bins)
    if metric not in {"top1", "nll"}:
        raise ValueError("metric must be 'top1' or 'nll'")
    if not k_grid:
        raise ValueError("k_grid must not be empty")
    if min(k_grid) < 0:
        raise ValueError("k_grid values must be non-negative")
    if identity_aug_id not in logits_by_aug:
        raise ValueError("identity_aug_id must exist in logits_by_aug")
    if predicted_gain.shape != (len(class_idxs), len(aug_ids)):
        raise ValueError("predicted_gain shape must match class_idxs rows and aug_ids")

    clean_logits = np.asarray(logits_by_aug[identity_aug_id], dtype=np.float32)
    if clean_logits.shape[0] != len(class_idxs):
        raise ValueError("identity logits and class_idxs must have matching rows")

    confidence = _softmax(clean_logits).max(axis=1)
    bucket_indices = _confidence_bucket_indices(confidence, bins)
    higher_is_better = metric == "top1"
    chosen_k: list[int] = []
    rows: list[dict[str, float | int]] = []

    for bucket_index in range(len(bins) - 1):
        mask = bucket_indices == bucket_index
        image_count = int(mask.sum())
        if image_count == 0:
            chosen_k.append(0)
            rows.append(
                _confidence_policy_row(
                    bucket_index=bucket_index,
                    confidence_min=bins[bucket_index],
                    confidence_max=bins[bucket_index + 1],
                    image_count=0,
                    k=0,
                    metrics={},
                )
            )
            continue

        bucket_logits = {aug_id: logits[mask] for aug_id, logits in logits_by_aug.items()}
        bucket_class_idxs = class_idxs[mask]
        bucket_gain = predicted_gain[mask]
        scored_candidates: list[tuple[int, dict[str, float]]] = []
        for k in k_grid:
            metrics = evaluate_learned_topk_uniform(
                logits_by_aug=bucket_logits,
                class_idxs=bucket_class_idxs,
                aug_ids=aug_ids,
                predicted_gain=bucket_gain,
                identity_aug_id=identity_aug_id,
                k=k,
            )
            scored_candidates.append((k, metrics))
            rows.append(
                _confidence_policy_row(
                    bucket_index=bucket_index,
                    confidence_min=bins[bucket_index],
                    confidence_max=bins[bucket_index + 1],
                    image_count=image_count,
                    k=k,
                    metrics=metrics,
                )
            )

        chosen_k.append(
            min(
                scored_candidates,
                key=lambda candidate: (
                    -candidate[1][metric] if higher_is_better else candidate[1][metric],
                    candidate[0],
                ),
            )[0]
        )

    return {
        "confidence_bins": [float(value) for value in bins],
        "bucket_k": chosen_k,
        "metric": metric,
        "rows": rows,
    }


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
        probabilities = average_per_image_probabilities(logits_by_aug, per_image_aug_ids)

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
    score_temperature: float = 1.0,
) -> dict[str, float]:
    """Evaluate per-image selected TTA with selector-score softmax weights."""

    probabilities = weighted_average_probabilities(
        logits_by_aug=logits_by_aug,
        selected_aug_ids=selected_aug_ids,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        score_temperature=score_temperature,
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


def evaluate_confidence_bucket_topk_uniform(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    confidence_bins: list[float],
    bucket_k: list[int],
) -> dict[str, float]:
    """Evaluate a learned confidence-bucket top-k policy with uniform averaging."""

    selected_aug_ids = confidence_bucket_topk_selection(
        clean_logits=logits_by_aug[identity_aug_id],
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        confidence_bins=confidence_bins,
        bucket_k=bucket_k,
    )
    return evaluate_selected_tta(logits_by_aug, selected_aug_ids, class_idxs)


def evaluate_learned_topk_softmax_weighted(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    k: int,
    score_temperature: float = 1.0,
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
        score_temperature=score_temperature,
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
    return average_per_image_probabilities(logits_by_aug, selected_aug_ids)


def _validate_per_image_selection(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
) -> tuple[int, int]:
    if not logits_by_aug:
        raise ValueError("logits_by_aug must not be empty")
    reference_logits = np.asarray(next(iter(logits_by_aug.values())), dtype=np.float32)
    if reference_logits.ndim != 2:
        raise ValueError("logits must have shape [images, classes]")
    if len(selected_aug_ids) != reference_logits.shape[0]:
        raise ValueError("selected_aug_ids must contain one selection per image")

    for aug_id, logits in logits_by_aug.items():
        if np.asarray(logits).shape != reference_logits.shape:
            raise ValueError(f"logits for {aug_id!r} must match the reference shape")

    for image_aug_ids in selected_aug_ids:
        if not image_aug_ids:
            raise ValueError("per-image augmentation selections must not be empty")
        unknown_aug_ids = [aug_id for aug_id in image_aug_ids if aug_id not in logits_by_aug]
        if unknown_aug_ids:
            raise ValueError(f"unknown augmentation id: {unknown_aug_ids[0]!r}")
    return reference_logits.shape


def _image_indices_by_aug(selected_aug_ids: list[list[str]]) -> dict[str, list[int]]:
    image_indices_by_aug: dict[str, list[int]] = {}
    for image_index, image_aug_ids in enumerate(selected_aug_ids):
        for aug_id in image_aug_ids:
            image_indices_by_aug.setdefault(aug_id, []).append(image_index)
    return image_indices_by_aug


def _mean_selection_size(selected_aug_ids: list[str] | list[list[str]]) -> float:
    if not selected_aug_ids:
        return 0.0
    if isinstance(selected_aug_ids[0], str):
        return float(len(selected_aug_ids))
    per_image_aug_ids = cast(list[list[str]], selected_aug_ids)
    return float(np.mean([len(selection) for selection in per_image_aug_ids]))


def _validate_confidence_bins(confidence_bins: list[float]) -> list[float]:
    bins = [float(value) for value in confidence_bins]
    if len(bins) < 2:
        raise ValueError("confidence_bins must contain at least two edges")
    if bins[0] < 0.0 or bins[0] > 1.0:
        raise ValueError("the first confidence bin edge must be within [0, 1]")
    if any(value < 0.0 or value > 1.0 for value in bins[1:-1]):
        raise ValueError("interior confidence bin edges must be within [0, 1]")
    if bins[-1] < 1.0:
        raise ValueError("the last confidence bin edge must cover confidence 1.0")
    if any(right <= left for left, right in zip(bins, bins[1:], strict=False)):
        raise ValueError("confidence_bins must be strictly increasing")
    return bins


def _confidence_bucket_indices(confidence: np.ndarray, bins: list[float]) -> np.ndarray:
    bucket_indices = np.searchsorted(np.asarray(bins[1:-1], dtype=np.float32), confidence, "right")
    return np.clip(bucket_indices, 0, len(bins) - 2).astype(np.int64)


def _confidence_policy_row(
    bucket_index: int,
    confidence_min: float,
    confidence_max: float,
    image_count: int,
    k: int,
    metrics: dict[str, float],
) -> dict[str, float | int]:
    return {
        "bucket_index": bucket_index,
        "confidence_min": confidence_min,
        "confidence_max": confidence_max,
        "image_count": image_count,
        "k": k,
        "top1": float(metrics.get("top1", 0.0)),
        "top5": float(metrics.get("top5", 0.0)),
        "nll": float(metrics.get("nll", 0.0)),
        "ece": float(metrics.get("ece", 0.0)),
        "forwards_per_image": float(metrics.get("forwards_per_image", 0.0)),
    }


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
