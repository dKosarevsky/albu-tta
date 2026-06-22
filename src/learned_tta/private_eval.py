"""Private split evaluation for learned TTA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.data import load_manifest
from learned_tta.metrics import classification_metrics, expected_calibration_error
from learned_tta.reporting import build_compute_table, build_correction_table, build_metrics_table
from learned_tta.split_policy import PUBLIC_VAL_SPLIT, validate_private_evaluation_split
from learned_tta.stacking import (
    default_aggregator_path,
    evaluate_xgboost_multiclass_stacker,
    load_aggregation_artifact,
    xgboost_multiclass_probabilities,
)
from learned_tta.standard_baselines import (
    evaluate_cached_standard_baselines,
    evaluate_ten_crop_artifact,
    load_ten_crop_logits,
    ten_crop_probabilities,
)
from learned_tta.tta_eval import (
    adaptive_topk_selection,
    average_per_image_probabilities,
    average_probabilities,
    class_weighted_probabilities,
    evaluate_all_100_uniform,
    evaluate_class_weighted_tta,
    evaluate_clean,
    evaluate_fixed_light_tta,
    evaluate_global_weighted_tta,
    evaluate_learned_adaptive_uniform,
    evaluate_learned_topk_softmax_weighted,
    evaluate_learned_topk_uniform,
    evaluate_oracle_topk_uniform,
    evaluate_random_topk,
    fixed_light_tta_selection,
    global_weighted_probabilities,
    learned_topk_selection,
    oracle_topk_selection,
    random_topk_selection,
    weighted_average_probabilities,
)
from learned_tta.tta_tuning import predict_selector_outputs

DEFAULT_GLOBAL_TOP_N_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 100)


@dataclass(frozen=True, slots=True)
class PrivateEvaluationSummary:
    """Summary of private evaluation artifacts."""

    best_k: int
    private_metrics_csv: Path
    compute_csv: Path
    corrections_csv: Path
    global_topn_metrics_csv: Path | None
    metrics_by_strategy: dict[str, dict[str, float]]


def evaluate_private_from_artifacts(
    split: str,
    manifest_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    tuning_path: Path,
    output_dir: Path,
    aug_ids: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    random_seeds: list[int],
    global_aggregator_path: Path | None = None,
    class_aggregator_path: Path | None = None,
    xgboost_aggregator_path: Path | None = None,
    ten_crop_logits_path: Path | None = None,
    device: str | torch.device = "cpu",
    identity_aug_id: str = "aug_000",
) -> PrivateEvaluationSummary:
    """Evaluate private split baselines with the frozen public-val top-k."""

    validate_private_evaluation_split(split, command="evaluate-private")
    tuning = _load_tuning_payload(tuning_path)
    best_k = _required_payload_int(tuning, "best_k")
    records = load_manifest(manifest_path)
    logits_by_aug, class_idxs = _read_split_logits(cache_dir, split=split, aug_ids=aug_ids)
    selector_predictions = predict_selector_outputs(
        checkpoint_path=checkpoint_path,
        records=records,
        output_dim=len(aug_ids),
        aug_ids=aug_ids,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    predicted_gain = selector_predictions.gain
    adaptive_threshold = _optional_payload_float(tuning, "best_adaptive_threshold")
    adaptive_max_k = _optional_payload_int(tuning, "best_adaptive_max_k")
    metrics_by_strategy = _evaluate_private_strategies(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        useful_prob=selector_predictions.useful_prob,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        adaptive_threshold=adaptive_threshold,
        adaptive_max_k=adaptive_max_k,
        random_seeds=random_seeds,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
        ten_crop_logits_path=ten_crop_logits_path,
    )

    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    private_metrics_csv = tables_dir / "private_metrics.csv"
    compute_csv = tables_dir / "compute.csv"
    corrections_csv = tables_dir / "corrections.csv"
    global_topn_metrics_csv: Path | None = None
    build_metrics_table(metrics_by_strategy).to_csv(private_metrics_csv, index=False)
    build_compute_table(metrics_by_strategy).to_csv(compute_csv, index=False)
    _build_private_corrections(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        useful_prob=selector_predictions.useful_prob,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        adaptive_threshold=adaptive_threshold,
        adaptive_max_k=adaptive_max_k,
        random_seeds=random_seeds,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
        ten_crop_logits_path=ten_crop_logits_path,
    ).to_csv(corrections_csv, index=False)
    if global_aggregator_path is not None:
        artifact = load_aggregation_artifact(global_aggregator_path)
        global_topn_metrics_csv = tables_dir / "global_weight_topn_private_metrics.csv"
        _build_global_topn_metrics(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            total_aug_ids=aug_ids,
            artifact_aug_ids=artifact.aug_ids,
            artifact_weights=artifact.weights,
        ).to_csv(global_topn_metrics_csv, index=False)
    return PrivateEvaluationSummary(
        best_k=best_k,
        private_metrics_csv=private_metrics_csv,
        compute_csv=compute_csv,
        corrections_csv=corrections_csv,
        global_topn_metrics_csv=global_topn_metrics_csv,
        metrics_by_strategy=metrics_by_strategy,
    )


def evaluate_private_from_config(
    config_path: Path,
    split: str = "private",
    manifest_path: Path | None = None,
    cache_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    tuning_path: Path | None = None,
    output_dir: Path | None = None,
    candidate_ids: list[str] | None = None,
    global_aggregator_path: Path | None = None,
    class_aggregator_path: Path | None = None,
    xgboost_aggregator_path: Path | None = None,
    ten_crop_logits_path: Path | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    random_seeds: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> PrivateEvaluationSummary:
    """Load config and evaluate private split."""

    config = load_experiment_config(config_path)
    use_config_candidate_ids = candidate_ids is None
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    if random_seeds is None:
        random_seeds = [config.seed + offset for offset in range(5)]
    selector_dir = config.artifacts.selector_dir
    resolved_global_aggregator_path = global_aggregator_path
    resolved_class_aggregator_path = class_aggregator_path
    resolved_xgboost_aggregator_path = xgboost_aggregator_path
    if use_config_candidate_ids:
        resolved_global_aggregator_path = resolved_global_aggregator_path or _existing_path(
            default_aggregator_path(selector_dir, split="public_val", method="global-nonnegative")
        )
        resolved_class_aggregator_path = resolved_class_aggregator_path or _existing_path(
            default_aggregator_path(selector_dir, split="public_val", method="class-nonnegative")
        )
        resolved_xgboost_aggregator_path = resolved_xgboost_aggregator_path or _existing_path(
            default_aggregator_path(selector_dir, split="public_val", method="xgboost-multiclass")
        )
    return evaluate_private_from_artifacts(
        split=split,
        manifest_path=manifest_path or config.artifacts.manifests_dir / f"{split}.csv",
        cache_dir=cache_dir or config.artifacts.teacher_cache_dir,
        checkpoint_path=checkpoint_path or config.artifacts.selector_dir / "selector_best.pt",
        tuning_path=tuning_path or config.artifacts.selector_dir / "public_val_tta_tuning.json",
        output_dir=output_dir or config.artifacts.reports_dir,
        aug_ids=candidate_ids,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seeds=random_seeds,
        global_aggregator_path=resolved_global_aggregator_path,
        class_aggregator_path=resolved_class_aggregator_path,
        xgboost_aggregator_path=resolved_xgboost_aggregator_path,
        ten_crop_logits_path=ten_crop_logits_path,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )


def _evaluate_private_strategies(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    useful_prob: np.ndarray | None,
    identity_aug_id: str,
    best_k: int,
    adaptive_threshold: float | None,
    adaptive_max_k: int | None,
    random_seeds: list[int],
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
    xgboost_aggregator_path: Path | None,
    ten_crop_logits_path: Path | None,
) -> dict[str, dict[str, float]]:
    random_metrics = [
        evaluate_random_topk(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            identity_aug_id=identity_aug_id,
            k=best_k,
            seed=seed,
        )
        for seed in random_seeds
    ]
    metrics = {
        **evaluate_cached_standard_baselines(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            identity_aug_id=identity_aug_id,
            reference_aug_count=len(aug_ids),
        ),
        "clean": evaluate_clean(logits_by_aug, class_idxs, identity_aug_id=identity_aug_id),
        "fixed_light_tta": evaluate_fixed_light_tta(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            identity_aug_id=identity_aug_id,
            k=best_k,
        ),
        "random_topk": _mean_metrics(random_metrics),
        "all_100_uniform": evaluate_all_100_uniform(logits_by_aug, class_idxs, aug_ids=aug_ids),
        "learned_topk_uniform": evaluate_learned_topk_uniform(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=best_k,
        ),
        "learned_topk_softmax_weighted": evaluate_learned_topk_softmax_weighted(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=best_k,
        ),
        "oracle_topk_uniform": evaluate_oracle_topk_uniform(
            logits_by_aug,
            class_idxs,
            identity_aug_id=identity_aug_id,
            k=best_k,
        ),
    }
    if useful_prob is not None and adaptive_threshold is not None and adaptive_max_k is not None:
        metrics["learned_adaptive_uniform"] = evaluate_learned_adaptive_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            useful_prob=useful_prob,
            identity_aug_id=identity_aug_id,
            threshold=adaptive_threshold,
            max_k=adaptive_max_k,
        )
    if global_aggregator_path is not None:
        artifact = load_aggregation_artifact(global_aggregator_path)
        metrics["global_weighted_tta"] = evaluate_global_weighted_tta(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=artifact.aug_ids,
            weights=artifact.weights,
            active_threshold=artifact.active_threshold,
        )
    if class_aggregator_path is not None:
        artifact = load_aggregation_artifact(class_aggregator_path)
        metrics["class_weighted_tta"] = evaluate_class_weighted_tta(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=artifact.aug_ids,
            class_weights=artifact.weights,
            active_threshold=artifact.active_threshold,
        )
    if xgboost_aggregator_path is not None:
        metrics["xgboost_multiclass"] = evaluate_xgboost_multiclass_stacker(
            artifact_path=xgboost_aggregator_path,
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            total_augments=len(aug_ids),
        )
    if ten_crop_logits_path is not None:
        metrics["ten_crop"] = evaluate_ten_crop_artifact(
            ten_crop_logits_path,
            expected_class_idxs=class_idxs,
            reference_aug_count=len(aug_ids),
        )
    return metrics


def _build_global_topn_metrics(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    total_aug_ids: list[str],
    artifact_aug_ids: list[str],
    artifact_weights: np.ndarray,
    top_n_grid: tuple[int, ...] = DEFAULT_GLOBAL_TOP_N_GRID,
) -> pd.DataFrame:
    weights = np.asarray(artifact_weights, dtype=np.float32)
    if weights.shape != (len(artifact_aug_ids),):
        raise ValueError("global aggregator weights must have shape [augmentations]")
    ranked_indices = sorted(
        range(len(artifact_aug_ids)),
        key=lambda index: (-float(weights[index]), index),
    )

    rows = []
    for top_n in _resolve_global_top_n_grid(top_n_grid, total_augments=len(artifact_aug_ids)):
        selected_indices = ranked_indices[:top_n]
        selected_aug_ids = [artifact_aug_ids[index] for index in selected_indices]
        selected_weights = weights[selected_indices]
        probabilities = global_weighted_probabilities(
            logits_by_aug=logits_by_aug,
            aug_ids=selected_aug_ids,
            weights=selected_weights,
        )
        metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
        metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
        rows.append(
            {
                "top_n": top_n,
                **metrics,
                "forwards_per_image": float(top_n),
                "relative_compute_vs_all": float(top_n) / len(total_aug_ids),
                "selected_aug_ids": " ".join(selected_aug_ids),
            }
        )
    return pd.DataFrame(rows)


def _resolve_global_top_n_grid(top_n_grid: tuple[int, ...], total_augments: int) -> list[int]:
    if total_augments < 1:
        raise ValueError("total_augments must be positive")
    resolved: set[int] = set()
    for top_n in top_n_grid:
        if top_n < 1:
            raise ValueError("top_n values must be positive")
        if top_n <= total_augments:
            resolved.add(top_n)
    resolved.add(total_augments)
    return sorted(resolved)


def _read_split_logits(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    logits_by_aug: dict[str, np.ndarray] = {}
    reference_class_idxs: np.ndarray | None = None
    reference_image_ids: list[str] | None = None
    for aug_id in aug_ids:
        paths = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id)
        shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
        class_idxs = shard.metadata["class_idx"].to_numpy(dtype=np.int64)
        image_ids = [str(image_id) for image_id in shard.metadata["image_id"].tolist()]
        if reference_class_idxs is None:
            reference_class_idxs = class_idxs
            reference_image_ids = image_ids
        elif not np.array_equal(reference_class_idxs, class_idxs):
            raise ValueError(f"class_idx order mismatch for split {split} and aug {aug_id}")
        elif reference_image_ids != image_ids:
            raise ValueError(f"image_id order mismatch for split {split} and aug {aug_id}")
        logits_by_aug[aug_id] = shard.logits.astype(np.float32)
    if reference_class_idxs is None:
        raise ValueError("aug_ids must not be empty")
    return logits_by_aug, reference_class_idxs


def _build_private_corrections(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    useful_prob: np.ndarray | None,
    identity_aug_id: str,
    best_k: int,
    adaptive_threshold: float | None,
    adaptive_max_k: int | None,
    random_seeds: list[int],
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
    xgboost_aggregator_path: Path | None,
    ten_crop_logits_path: Path | None,
) -> pd.DataFrame:
    probabilities_by_strategy = _private_probabilities_by_strategy(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        useful_prob=useful_prob,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        adaptive_threshold=adaptive_threshold,
        adaptive_max_k=adaptive_max_k,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
        ten_crop_logits_path=ten_crop_logits_path,
    )
    predictions_by_strategy = {
        strategy: probabilities.argmax(axis=1)
        for strategy, probabilities in probabilities_by_strategy.items()
    }
    clean_predictions = predictions_by_strategy["clean"]
    corrections = build_correction_table(
        clean_correct=clean_predictions == class_idxs,
        predictions_by_strategy=predictions_by_strategy,
        class_idxs=class_idxs,
    )
    random_row = _random_topk_correction_row(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        random_seeds=random_seeds,
        clean_correct=clean_predictions == class_idxs,
    )
    return pd.concat([corrections, random_row], ignore_index=True)


def _random_topk_correction_row(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    identity_aug_id: str,
    best_k: int,
    random_seeds: list[int],
    clean_correct: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for seed in random_seeds:
        selected_aug_ids = random_topk_selection(
            aug_ids=aug_ids,
            num_images=len(class_idxs),
            identity_aug_id=identity_aug_id,
            k=best_k,
            seed=seed,
        )
        probabilities = average_per_image_probabilities(logits_by_aug, selected_aug_ids)
        rows.append(
            build_correction_table(
                clean_correct=clean_correct,
                predictions_by_strategy={"random_topk": probabilities.argmax(axis=1)},
                class_idxs=class_idxs,
            ).iloc[0]
        )
    table = pd.DataFrame(rows)
    averaged = {
        column: float(table[column].mean()) for column in table.columns if column != "strategy"
    }
    return pd.DataFrame([{"strategy": "random_topk", **averaged}])


def _private_probabilities_by_strategy(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    useful_prob: np.ndarray | None,
    identity_aug_id: str,
    best_k: int,
    adaptive_threshold: float | None,
    adaptive_max_k: int | None,
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
    xgboost_aggregator_path: Path | None,
    ten_crop_logits_path: Path | None,
) -> dict[str, np.ndarray]:
    probabilities = {
        "clean_center_crop": average_probabilities(logits_by_aug, [identity_aug_id]),
        "clean": average_probabilities(logits_by_aug, [identity_aug_id]),
        "fixed_light_tta": average_probabilities(
            logits_by_aug,
            fixed_light_tta_selection(
                aug_ids=aug_ids,
                identity_aug_id=identity_aug_id,
                k=best_k,
            ),
        ),
        "all_100_uniform": average_probabilities(logits_by_aug, aug_ids),
    }
    if "aug_005" in logits_by_aug:
        probabilities["center_crop_hflip"] = average_probabilities(
            logits_by_aug,
            [identity_aug_id, "aug_005"],
        )
    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )
    probabilities["learned_topk_uniform"] = average_per_image_probabilities(
        logits_by_aug,
        selected_aug_ids,
    )
    probabilities["learned_topk_softmax_weighted"] = weighted_average_probabilities(
        logits_by_aug=logits_by_aug,
        selected_aug_ids=selected_aug_ids,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
    )
    if useful_prob is not None and adaptive_threshold is not None and adaptive_max_k is not None:
        probabilities["learned_adaptive_uniform"] = average_per_image_probabilities(
            logits_by_aug,
            adaptive_topk_selection(
                aug_ids=aug_ids,
                predicted_gain=predicted_gain,
                useful_prob=useful_prob,
                identity_aug_id=identity_aug_id,
                threshold=adaptive_threshold,
                max_k=adaptive_max_k,
            ),
        )
    probabilities["oracle_topk_uniform"] = average_per_image_probabilities(
        logits_by_aug,
        oracle_topk_selection(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            identity_aug_id=identity_aug_id,
            k=best_k,
        ),
    )
    if global_aggregator_path is not None:
        artifact = load_aggregation_artifact(global_aggregator_path)
        probabilities["global_weighted_tta"] = global_weighted_probabilities(
            logits_by_aug,
            aug_ids=artifact.aug_ids,
            weights=artifact.weights,
        )
    if class_aggregator_path is not None:
        artifact = load_aggregation_artifact(class_aggregator_path)
        probabilities["class_weighted_tta"] = class_weighted_probabilities(
            logits_by_aug,
            aug_ids=artifact.aug_ids,
            class_weights=artifact.weights,
        )
    if xgboost_aggregator_path is not None:
        probabilities["xgboost_multiclass"] = xgboost_multiclass_probabilities(
            artifact_path=xgboost_aggregator_path,
            logits_by_aug=logits_by_aug,
        )
    if ten_crop_logits_path is not None:
        artifact = load_ten_crop_logits(ten_crop_logits_path)
        if not np.array_equal(artifact.class_idxs, class_idxs):
            raise ValueError("10-crop class_idxs do not match private cache order")
        probabilities["ten_crop"] = ten_crop_probabilities(artifact.crop_logits)
    return probabilities


def _load_tuning_payload(tuning_path: Path) -> dict[str, object]:
    with Path(tuning_path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    split = data.get("split")
    if split is not None and split != PUBLIC_VAL_SPLIT:
        raise ValueError(f"tuning artifact split must be {PUBLIC_VAL_SPLIT}; got {split!r}")
    return data


def _load_best_k(tuning_path: Path) -> int:
    return _required_payload_int(_load_tuning_payload(tuning_path), "best_k")


def _required_payload_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"tuning artifact is missing {key!r}")
    return _coerce_payload_int(value, key)


def _optional_payload_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return _coerce_payload_int(value, key)


def _optional_payload_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"tuning artifact {key!r} must be numeric")
    return float(value)


def _coerce_payload_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"tuning artifact {key!r} must be an integer")
    return int(value)


def _existing_path(path: Path) -> Path | None:
    if path.exists():
        return path
    return None


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    frame = pd.DataFrame(metrics)
    return {column: float(frame[column].mean()) for column in frame.columns}
