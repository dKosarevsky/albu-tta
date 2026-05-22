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
from learned_tta.reporting import build_compute_table, build_correction_table, build_metrics_table
from learned_tta.stacking import default_aggregator_path, load_aggregation_artifact
from learned_tta.tta_eval import (
    average_probabilities,
    class_weighted_probabilities,
    evaluate_all_100_uniform,
    evaluate_class_weighted_tta,
    evaluate_clean,
    evaluate_fixed_light_tta,
    evaluate_global_weighted_tta,
    evaluate_learned_topk_softmax_weighted,
    evaluate_learned_topk_uniform,
    evaluate_oracle_topk_uniform,
    evaluate_random_topk,
    fixed_light_tta_selection,
    global_weighted_probabilities,
    learned_topk_selection,
    oracle_topk_selection,
    weighted_average_probabilities,
)
from learned_tta.tta_tuning import predict_selector_scores


@dataclass(frozen=True, slots=True)
class PrivateEvaluationSummary:
    """Summary of private evaluation artifacts."""

    best_k: int
    private_metrics_csv: Path
    compute_csv: Path
    corrections_csv: Path
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
    device: str | torch.device = "cpu",
    identity_aug_id: str = "aug_000",
) -> PrivateEvaluationSummary:
    """Evaluate private split baselines with the frozen public-val top-k."""

    best_k = _load_best_k(tuning_path)
    records = load_manifest(manifest_path)
    logits_by_aug, class_idxs = _read_split_logits(cache_dir, split=split, aug_ids=aug_ids)
    predicted_gain = predict_selector_scores(
        checkpoint_path=checkpoint_path,
        records=records,
        output_dim=len(aug_ids),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    metrics_by_strategy = _evaluate_private_strategies(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        random_seeds=random_seeds,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
    )

    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    private_metrics_csv = tables_dir / "private_metrics.csv"
    compute_csv = tables_dir / "compute.csv"
    corrections_csv = tables_dir / "corrections.csv"
    build_metrics_table(metrics_by_strategy).to_csv(private_metrics_csv, index=False)
    build_compute_table(metrics_by_strategy).to_csv(compute_csv, index=False)
    _build_private_corrections(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
    ).to_csv(corrections_csv, index=False)
    return PrivateEvaluationSummary(
        best_k=best_k,
        private_metrics_csv=private_metrics_csv,
        compute_csv=compute_csv,
        corrections_csv=corrections_csv,
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
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    random_seeds: list[int] | None = None,
    device: str | torch.device = "cpu",
) -> PrivateEvaluationSummary:
    """Load config and evaluate private split."""

    config = load_experiment_config(config_path)
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    if random_seeds is None:
        random_seeds = [config.seed + offset for offset in range(5)]
    selector_dir = config.artifacts.selector_dir
    resolved_global_aggregator_path = global_aggregator_path or _existing_path(
        default_aggregator_path(selector_dir, split="public_val", method="global-nonnegative")
    )
    resolved_class_aggregator_path = class_aggregator_path or _existing_path(
        default_aggregator_path(selector_dir, split="public_val", method="class-nonnegative")
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
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )


def _evaluate_private_strategies(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    best_k: int,
    random_seeds: list[int],
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
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
    return metrics


def _read_split_logits(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    logits_by_aug: dict[str, np.ndarray] = {}
    reference_class_idxs: np.ndarray | None = None
    for aug_id in aug_ids:
        paths = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id)
        shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
        class_idxs = shard.metadata["class_idx"].to_numpy(dtype=np.int64)
        if reference_class_idxs is None:
            reference_class_idxs = class_idxs
        elif not np.array_equal(reference_class_idxs, class_idxs):
            raise ValueError(f"class_idx order mismatch for split {split} and aug {aug_id}")
        logits_by_aug[aug_id] = shard.logits.astype(np.float32)
    if reference_class_idxs is None:
        raise ValueError("aug_ids must not be empty")
    return logits_by_aug, reference_class_idxs


def _build_private_corrections(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    best_k: int,
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
) -> pd.DataFrame:
    probabilities_by_strategy = _private_probabilities_by_strategy(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        best_k=best_k,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
    )
    predictions_by_strategy = {
        strategy: probabilities.argmax(axis=1)
        for strategy, probabilities in probabilities_by_strategy.items()
    }
    clean_predictions = predictions_by_strategy["clean"]
    return build_correction_table(
        clean_correct=clean_predictions == class_idxs,
        predictions_by_strategy=predictions_by_strategy,
        class_idxs=class_idxs,
    )


def _private_probabilities_by_strategy(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    best_k: int,
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
) -> dict[str, np.ndarray]:
    probabilities = {
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
    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )
    probabilities["learned_topk_uniform"] = _average_per_image_probabilities(
        logits_by_aug,
        selected_aug_ids,
    )
    probabilities["learned_topk_softmax_weighted"] = weighted_average_probabilities(
        logits_by_aug=logits_by_aug,
        selected_aug_ids=selected_aug_ids,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
    )
    probabilities["oracle_topk_uniform"] = _average_per_image_probabilities(
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
    return probabilities


def _average_per_image_probabilities(
    logits_by_aug: dict[str, np.ndarray],
    selected_aug_ids: list[list[str]],
) -> np.ndarray:
    rows = []
    for image_index, image_aug_ids in enumerate(selected_aug_ids):
        rows.append(
            average_probabilities(
                {
                    aug_id: logits_by_aug[aug_id][image_index : image_index + 1]
                    for aug_id in image_aug_ids
                },
                selected_aug_ids=image_aug_ids,
            )[0]
        )
    return np.asarray(rows, dtype=np.float32)


def _load_best_k(tuning_path: Path) -> int:
    with Path(tuning_path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    return int(data["best_k"])


def _existing_path(path: Path) -> Path | None:
    if path.exists():
        return path
    return None


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    frame = pd.DataFrame(metrics)
    return {column: float(frame[column].mean()) for column in frame.columns}
