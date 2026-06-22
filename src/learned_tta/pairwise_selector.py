"""Pairwise `(image, augmentation)` selector training helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.data import load_manifest
from learned_tta.reporting import build_metrics_table
from learned_tta.selector_error_analysis import build_selector_error_analysis_table
from learned_tta.selector_features import clean_logit_uncertainty_features, load_selector_features
from learned_tta.targets import load_selector_targets
from learned_tta.tta_eval import (
    ConfidenceBucketPolicy,
    evaluate_clean,
    evaluate_confidence_adaptive_topk_uniform,
    evaluate_confidence_bucket_topk_uniform,
    evaluate_learned_topk_softmax_weighted,
    evaluate_learned_topk_uniform,
    evaluate_oracle_topk_uniform,
)


@dataclass(frozen=True, slots=True)
class PairwiseFeatureBundle:
    """Flattened features and targets for `(image, augmentation)` rows."""

    image_ids: list[str]
    aug_ids: list[str]
    features: np.ndarray
    feature_names: list[str]
    targets: np.ndarray
    target_matrix: np.ndarray
    row_image_indices: np.ndarray
    row_aug_indices: np.ndarray
    class_idxs: np.ndarray

    def score_matrix(self, row_scores: np.ndarray) -> np.ndarray:
        """Reshape row scores back to `[images, augmentations]`."""

        row_scores = np.asarray(row_scores, dtype=np.float32)
        expected_rows = len(self.image_ids) * len(self.aug_ids)
        if row_scores.shape != (expected_rows,):
            raise ValueError("row_scores shape must match flattened pairwise rows")
        return row_scores.reshape(len(self.image_ids), len(self.aug_ids)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class PairwiseTrainingSummary:
    """Summary of a pairwise selector training run."""

    checkpoint_path: Path
    summary_csv: Path
    best_epoch: int
    best_val_nll: float
    best_val_top1: float


@dataclass(frozen=True, slots=True)
class PairwiseComparisonSummary:
    """Summary of a pairwise selector objective comparison."""

    results_csv: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class PairwiseEvaluationSummary:
    """Summary of a pairwise selector inference/evaluation run."""

    metrics_csv: Path
    scores_npz: Path
    error_analysis_csv: Path
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class _PairwiseCheckpoint:
    model_state_dict: Mapping[str, Any]
    input_dim: int
    hidden_dim: int
    aug_ids: list[str]
    feature_names: list[str]
    feature_projection_state: dict[str, Any] | None = None


class PairwiseSelectorMLP(torch.nn.Module):
    """Small MLP that scores one `(image, augmentation)` row."""

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one scalar score per pairwise feature row."""

        return self.net(features).squeeze(-1)


def build_pairwise_feature_bundle(
    manifest_path: Path,
    targets_path: Path,
    cache_dir: Path,
    identity_aug_id: str,
    features_path: Path | None = None,
    target_mode: str = "nll_gain",
    feature_projection_dim: int | None = None,
    feature_projection_seed: int = 0,
    feature_projection_method: str = "random",
    feature_projection_state: Mapping[str, Any] | None = None,
) -> PairwiseFeatureBundle:
    """Build flattened `(image, augmentation)` features from cached teacher outputs."""

    records = load_manifest(manifest_path)
    targets = load_selector_targets(targets_path)
    if not records:
        raise ValueError("manifest must contain at least one row")
    image_ids = [record.image_id for record in records]
    if targets.image_ids != image_ids:
        raise ValueError("selector target image_ids must match manifest image_ids")
    split = records[0].split
    if any(record.split != split for record in records):
        raise ValueError("manifest must contain a single split")
    if identity_aug_id not in targets.aug_ids:
        raise ValueError("identity augmentation must be present in selector targets")

    identity_shard = _read_validated_shard(
        cache_dir=cache_dir,
        split=split,
        aug_id=identity_aug_id,
        image_ids=image_ids,
    )
    class_idxs = np.asarray(identity_shard.metadata["class_idx"].tolist(), dtype=np.int64)
    clean_features, clean_names = clean_logit_uncertainty_features(
        identity_shard.logits,
        class_idxs,
    )
    image_features = [clean_features]
    image_feature_names = list(clean_names)

    if features_path is not None:
        pretrained = load_selector_features(features_path)
        if pretrained.split != split:
            raise ValueError("pretrained feature split must match manifest split")
        if pretrained.image_ids != image_ids:
            raise ValueError("pretrained feature image_ids must match manifest image_ids")
        optional_features, optional_feature_names = _prepare_optional_image_features(
            features=pretrained.features,
            feature_names=pretrained.feature_names,
            projection_dim=feature_projection_dim,
            projection_seed=feature_projection_seed,
            projection_method=feature_projection_method,
            projection_state=feature_projection_state,
        )
        image_features.append(optional_features)
        image_feature_names.extend(optional_feature_names)

    if target_mode == "nll_gain":
        target_matrix = targets.gain.astype(np.float32)
    elif target_mode == "top1_delta":
        target_matrix = _top1_delta_targets(
            cache_dir=cache_dir,
            split=split,
            aug_ids=targets.aug_ids,
            image_ids=image_ids,
            identity_aug_id=identity_aug_id,
        )
    elif target_mode == "marginal_logit_gain":
        logits_by_aug = _read_logits_by_aug(
            cache_dir=cache_dir,
            split=split,
            aug_ids=targets.aug_ids,
            image_ids=image_ids,
        )
        target_matrix = build_pairwise_marginal_logit_gain_targets(
            logits_by_aug=logits_by_aug,
            aug_ids=targets.aug_ids,
            class_idxs=class_idxs,
            identity_aug_id=identity_aug_id,
        )
    else:
        raise ValueError("target_mode must be 'nll_gain', 'top1_delta', or 'marginal_logit_gain'")

    per_image_features = np.concatenate(image_features, axis=1).astype(np.float32)
    return _build_pairwise_bundle_from_image_features(
        image_ids=image_ids,
        aug_ids=targets.aug_ids,
        per_image_features=per_image_features,
        image_feature_names=image_feature_names,
        target_matrix=target_matrix,
        class_idxs=class_idxs,
    )


def build_pairwise_inference_bundle(
    manifest_path: Path,
    cache_dir: Path,
    aug_ids: list[str],
    identity_aug_id: str,
    features_path: Path | None = None,
    feature_projection_dim: int | None = None,
    feature_projection_seed: int = 0,
    feature_projection_method: str = "random",
    feature_projection_state: Mapping[str, Any] | None = None,
) -> PairwiseFeatureBundle:
    """Build flattened pairwise features for inference without selector targets."""

    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest must contain at least one row")
    image_ids = [record.image_id for record in records]
    split = records[0].split
    if any(record.split != split for record in records):
        raise ValueError("manifest must contain a single split")
    if identity_aug_id not in aug_ids:
        raise ValueError("identity augmentation must be present in aug_ids")

    identity_shard = _read_validated_shard(
        cache_dir=cache_dir,
        split=split,
        aug_id=identity_aug_id,
        image_ids=image_ids,
    )
    class_idxs = np.asarray(identity_shard.metadata["class_idx"].tolist(), dtype=np.int64)
    clean_features, clean_names = clean_logit_uncertainty_features(
        identity_shard.logits,
        class_idxs,
    )
    image_features = [clean_features]
    image_feature_names = list(clean_names)

    if features_path is not None:
        pretrained = load_selector_features(features_path)
        if pretrained.split != split:
            raise ValueError("pretrained feature split must match manifest split")
        if pretrained.image_ids != image_ids:
            raise ValueError("pretrained feature image_ids must match manifest image_ids")
        optional_features, optional_feature_names = _prepare_optional_image_features(
            features=pretrained.features,
            feature_names=pretrained.feature_names,
            projection_dim=feature_projection_dim,
            projection_seed=feature_projection_seed,
            projection_method=feature_projection_method,
            projection_state=feature_projection_state,
        )
        image_features.append(optional_features)
        image_feature_names.extend(optional_feature_names)

    per_image_features = np.concatenate(image_features, axis=1).astype(np.float32)
    target_matrix = np.zeros((len(image_ids), len(aug_ids)), dtype=np.float32)
    return _build_pairwise_bundle_from_image_features(
        image_ids=image_ids,
        aug_ids=aug_ids,
        per_image_features=per_image_features,
        image_feature_names=image_feature_names,
        target_matrix=target_matrix,
        class_idxs=class_idxs,
    )


def evaluate_pairwise_selector_from_artifacts(
    manifest_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    identity_aug_id: str = "aug_000",
    features_path: Path | None = None,
    feature_projection_dim: int | None = None,
    feature_projection_seed: int = 0,
    feature_projection_method: str = "random",
    top_k: int = 16,
    batch_size: int = 8192,
    strategy_name: str = "pairwise_topk_uniform",
    score_temperature: float = 1.0,
    confidence_low_threshold: float = 0.75,
    confidence_high_threshold: float = 0.9,
    confidence_low_k: int | None = None,
    confidence_mid_k: int | None = None,
    confidence_high_k: int = 8,
    confidence_policy_path: Path | None = None,
    device: str | torch.device = "cpu",
) -> PairwiseEvaluationSummary:
    """Evaluate a pairwise selector checkpoint on a target-free cached split."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    checkpoint = _load_pairwise_checkpoint(checkpoint_path, device=torch_device)
    model = PairwiseSelectorMLP(
        input_dim=checkpoint.input_dim,
        hidden_dim=checkpoint.hidden_dim,
    )
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(torch_device)

    bundle = build_pairwise_inference_bundle(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        aug_ids=checkpoint.aug_ids,
        identity_aug_id=identity_aug_id,
        features_path=features_path,
        feature_projection_dim=feature_projection_dim,
        feature_projection_seed=feature_projection_seed,
        feature_projection_method=feature_projection_method,
        feature_projection_state=checkpoint.feature_projection_state,
    )
    if bundle.feature_names != checkpoint.feature_names:
        raise ValueError("checkpoint feature_names must match inference feature_names")

    row_scores = predict_pairwise_scores(
        model=model,
        bundle=bundle,
        batch_size=batch_size,
        device=torch_device,
    )
    predicted_gain = bundle.score_matrix(row_scores)
    split = load_manifest(manifest_path)[0].split
    logits_by_aug = _read_logits_by_aug(
        cache_dir=cache_dir,
        split=split,
        aug_ids=bundle.aug_ids,
        image_ids=bundle.image_ids,
    )
    low_k = top_k if confidence_low_k is None else confidence_low_k
    mid_k = top_k if confidence_mid_k is None else confidence_mid_k
    confidence_policy = (
        _load_confidence_bucket_policy(confidence_policy_path)
        if confidence_policy_path is not None
        else None
    )
    metrics_by_strategy = {
        "clean": evaluate_clean(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            identity_aug_id=identity_aug_id,
        ),
        strategy_name: evaluate_learned_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            aug_ids=bundle.aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=top_k,
        ),
        f"{strategy_name}_softmax_weighted": evaluate_learned_topk_softmax_weighted(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            aug_ids=bundle.aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=top_k,
            score_temperature=score_temperature,
        ),
        f"{strategy_name}_confidence_adaptive": evaluate_confidence_adaptive_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            aug_ids=bundle.aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            low_confidence_threshold=confidence_low_threshold,
            high_confidence_threshold=confidence_high_threshold,
            low_confidence_k=low_k,
            mid_confidence_k=mid_k,
            high_confidence_k=confidence_high_k,
        ),
        "oracle_topk_uniform": evaluate_oracle_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            identity_aug_id=identity_aug_id,
            k=top_k,
        ),
    }
    if confidence_policy is not None:
        metrics_by_strategy[f"{strategy_name}_confidence_policy"] = (
            evaluate_confidence_bucket_topk_uniform(
                logits_by_aug=logits_by_aug,
                class_idxs=bundle.class_idxs,
                aug_ids=bundle.aug_ids,
                predicted_gain=predicted_gain,
                identity_aug_id=identity_aug_id,
                confidence_bins=confidence_policy["confidence_bins"],
                bucket_k=confidence_policy["bucket_k"],
            )
        )
    metrics_csv = output_dir / "pairwise_selector_metrics.csv"
    build_metrics_table(metrics_by_strategy).to_csv(metrics_csv, index=False)
    scores_npz = output_dir / "pairwise_selector_scores.npz"
    np.savez_compressed(
        scores_npz,
        image_ids=np.asarray(bundle.image_ids, dtype=object),
        aug_ids=np.asarray(bundle.aug_ids, dtype=object),
        predicted_gain=predicted_gain.astype(np.float32),
    )
    error_analysis_csv = output_dir / "pairwise_selector_error_analysis.csv"
    build_selector_error_analysis_table(
        logits_by_aug=logits_by_aug,
        class_idxs=bundle.class_idxs,
        aug_ids=bundle.aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=top_k,
        output_path=error_analysis_csv,
    )
    return PairwiseEvaluationSummary(
        metrics_csv=metrics_csv,
        scores_npz=scores_npz,
        error_analysis_csv=error_analysis_csv,
        metrics=dict(metrics_by_strategy[strategy_name]),
    )


def train_pairwise_selector_from_artifacts(
    train_manifest_path: Path,
    val_manifest_path: Path,
    train_targets_path: Path,
    val_targets_path: Path,
    cache_dir: Path,
    output_dir: Path,
    identity_aug_id: str = "aug_000",
    train_features_path: Path | None = None,
    val_features_path: Path | None = None,
    feature_projection_dim: int | None = None,
    feature_projection_seed: int = 0,
    feature_projection_method: str = "random",
    top_k_grid: list[int] | None = None,
    batch_size: int = 1024,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    positive_gain_weight: float = 0.0,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 16,
    listwise_loss: str = "topk_ce",
    listwise_target_temperature: float = 1.0,
    hard_example_weight: float = 0.0,
    hard_example_confidence_threshold: float = 0.75,
    target_mode: str = "nll_gain",
    selection_metric: str = "val_tta_nll",
    device: str | torch.device = "cpu",
) -> PairwiseTrainingSummary:
    """Train a lightweight pairwise selector and write a compact summary CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_projection_state = _fit_optional_pairwise_feature_projection(
        features_path=train_features_path,
        projection_dim=feature_projection_dim,
        projection_method=feature_projection_method,
    )
    train_bundle = build_pairwise_feature_bundle(
        manifest_path=train_manifest_path,
        targets_path=train_targets_path,
        cache_dir=cache_dir,
        identity_aug_id=identity_aug_id,
        features_path=train_features_path,
        target_mode=target_mode,
        feature_projection_dim=feature_projection_dim,
        feature_projection_seed=feature_projection_seed,
        feature_projection_method=feature_projection_method,
        feature_projection_state=feature_projection_state,
    )
    val_bundle = build_pairwise_feature_bundle(
        manifest_path=val_manifest_path,
        targets_path=val_targets_path,
        cache_dir=cache_dir,
        identity_aug_id=identity_aug_id,
        features_path=val_features_path,
        target_mode=target_mode,
        feature_projection_dim=feature_projection_dim,
        feature_projection_seed=feature_projection_seed,
        feature_projection_method=feature_projection_method,
        feature_projection_state=feature_projection_state,
    )
    model = PairwiseSelectorMLP(input_dim=train_bundle.features.shape[1], hidden_dim=hidden_dim)
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    row_weights = build_pairwise_hard_example_weights(
        train_bundle,
        hard_example_weight=hard_example_weight,
        confidence_threshold=hard_example_confidence_threshold,
    )
    image_batch_size = max(1, int(batch_size) // max(1, len(train_bundle.aug_ids)))
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        torch.utils.data.DataLoader(
            _PairwiseImageBatchDataset(
                bundle=train_bundle,
                row_weights=row_weights,
            ),
            batch_size=image_batch_size,
            shuffle=True,
        )
    )
    rows: list[dict[str, float | int]] = []
    checkpoint_path = output_dir / "pairwise_selector_best.pt"
    if selection_metric not in {"val_tta_nll", "val_tta_top1"}:
        raise ValueError("selection_metric must be 'val_tta_nll' or 'val_tta_top1'")
    best_metric = float("inf") if selection_metric == "val_tta_nll" else float("-inf")
    best_val_nll = float("inf")
    best_val_top1 = 0.0
    best_epoch = 0
    top_k_values = top_k_grid or [1]
    val_logits_by_aug = _read_logits_by_aug(
        cache_dir=cache_dir,
        split=load_manifest(val_manifest_path)[0].split,
        aug_ids=val_bundle.aug_ids,
        image_ids=val_bundle.image_ids,
    )

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=torch_device,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            positive_gain_weight=positive_gain_weight,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
            listwise_loss=listwise_loss,
            listwise_target_temperature=listwise_target_temperature,
        )
        val_scores = predict_pairwise_scores(
            model=model,
            bundle=val_bundle,
            batch_size=batch_size,
            device=torch_device,
        )
        val_matrix = val_bundle.score_matrix(val_scores)
        best_metrics = _best_topk_metrics(
            logits_by_aug=val_logits_by_aug,
            class_idxs=val_bundle.class_idxs,
            aug_ids=val_bundle.aug_ids,
            predicted_gain=val_matrix,
            identity_aug_id=identity_aug_id,
            top_k_grid=top_k_values,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_tta_best_k": int(best_metrics["k"]),
            "val_tta_top1": float(best_metrics["top1"]),
            "val_tta_top5": float(best_metrics["top5"]),
            "val_tta_nll": float(best_metrics["nll"]),
            "val_tta_ece": float(best_metrics["ece"]),
        }
        rows.append(row)
        current_metric = float(row[selection_metric])
        if _is_better_pairwise_metric(
            current=current_metric,
            best=best_metric,
            selection_metric=selection_metric,
        ):
            best_metric = current_metric
            best_val_nll = float(row["val_tta_nll"])
            best_val_top1 = float(row["val_tta_top1"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": train_bundle.features.shape[1],
                    "hidden_dim": hidden_dim,
                    "aug_ids": train_bundle.aug_ids,
                    "feature_names": train_bundle.feature_names,
                    "feature_projection_state": feature_projection_state,
                    "listwise_weight": listwise_weight,
                    "listwise_top_k": listwise_top_k,
                    "listwise_loss": listwise_loss,
                    "listwise_target_temperature": listwise_target_temperature,
                    "hard_example_weight": hard_example_weight,
                    "hard_example_confidence_threshold": hard_example_confidence_threshold,
                },
                checkpoint_path,
            )

    summary_csv = output_dir / "pairwise_selector_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return PairwiseTrainingSummary(
        checkpoint_path=checkpoint_path,
        summary_csv=summary_csv,
        best_epoch=best_epoch,
        best_val_nll=best_val_nll,
        best_val_top1=best_val_top1,
    )


def train_pairwise_selector_comparison_from_artifacts(
    train_manifest_path: Path,
    val_manifest_path: Path,
    train_targets_path: Path,
    val_targets_path: Path,
    cache_dir: Path,
    output_dir: Path,
    identity_aug_id: str = "aug_000",
    train_features_path: Path | None = None,
    val_features_path: Path | None = None,
    feature_projection_dim: int | None = None,
    feature_projection_seed: int = 0,
    feature_projection_method: str = "random",
    top_k_grid: list[int] | None = None,
    batch_size: int = 1024,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    positive_gain_weight: float = 0.0,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 16,
    listwise_loss: str = "topk_ce",
    listwise_target_temperature: float = 1.0,
    hard_example_weight: float = 0.0,
    hard_example_confidence_threshold: float = 0.75,
    device: str | torch.device = "cpu",
) -> PairwiseComparisonSummary:
    """Train NLL-gain and top-1-delta pairwise selector variants."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        ("pairwise_nll_gain", "nll_gain", "val_tta_nll"),
        ("pairwise_top1_delta", "top1_delta", "val_tta_top1"),
    )
    rows: list[dict[str, object]] = []
    for variant, target_mode, selection_metric in specs:
        variant_dir = output_dir / variant
        summary = train_pairwise_selector_from_artifacts(
            train_manifest_path=train_manifest_path,
            val_manifest_path=val_manifest_path,
            train_targets_path=train_targets_path,
            val_targets_path=val_targets_path,
            cache_dir=cache_dir,
            output_dir=variant_dir,
            identity_aug_id=identity_aug_id,
            train_features_path=train_features_path,
            val_features_path=val_features_path,
            feature_projection_dim=feature_projection_dim,
            feature_projection_seed=feature_projection_seed,
            feature_projection_method=feature_projection_method,
            top_k_grid=top_k_grid,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            positive_gain_weight=positive_gain_weight,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
            listwise_loss=listwise_loss,
            listwise_target_temperature=listwise_target_temperature,
            hard_example_weight=hard_example_weight,
            hard_example_confidence_threshold=hard_example_confidence_threshold,
            target_mode=target_mode,
            selection_metric=selection_metric,
            device=device,
        )
        rows.append(
            {
                "variant": variant,
                "target_mode": target_mode,
                "selection_metric": selection_metric,
                "feature_projection_method": feature_projection_method,
                "listwise_weight": listwise_weight,
                "listwise_top_k": listwise_top_k,
                "listwise_loss": listwise_loss,
                "listwise_target_temperature": listwise_target_temperature,
                "hard_example_weight": hard_example_weight,
                "hard_example_confidence_threshold": hard_example_confidence_threshold,
                "best_epoch": summary.best_epoch,
                "best_val_top1": summary.best_val_top1,
                "best_val_nll": summary.best_val_nll,
                "checkpoint_path": str(summary.checkpoint_path),
                "summary_csv": str(summary.summary_csv),
            }
        )
    results_csv = output_dir / "pairwise_selector_comparison.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    return PairwiseComparisonSummary(results_csv=results_csv, rows=rows)


@torch.no_grad()
def predict_pairwise_scores(
    model: PairwiseSelectorMLP,
    bundle: PairwiseFeatureBundle,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Predict one score per flattened pairwise row."""

    torch_device = torch.device(device)
    model.eval()
    rows = []
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(bundle.features)),
        batch_size=batch_size,
        shuffle=False,
    )
    for (features,) in dataloader:
        scores = model(features.to(torch_device))
        rows.append(scores.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0).astype(np.float32)


def pairwise_policy_loss(
    predicted_gain: torch.Tensor,
    target_gain: torch.Tensor,
    *,
    usefulness_logits: torch.Tensor | None = None,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    positive_gain_weight: float = 0.0,
    row_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute policy-aware pairwise selector loss terms."""

    predicted_gain = predicted_gain.float()
    target_gain = target_gain.float()
    if predicted_gain.shape != target_gain.shape:
        raise ValueError("predicted_gain and target_gain must have the same shape")
    weights = torch.ones_like(target_gain)
    if positive_gain_weight > 0.0:
        weights = weights + (target_gain > usefulness_tau).float() * float(positive_gain_weight)
    if row_weights is not None:
        if row_weights.shape != target_gain.shape:
            raise ValueError("row_weights must match target_gain shape")
        weights = weights * row_weights.float()
    regression_terms = torch.nn.functional.smooth_l1_loss(
        predicted_gain,
        target_gain,
        reduction="none",
    )
    regression_loss = (regression_terms * weights).sum() / torch.clamp(weights.sum(), min=1e-12)
    usefulness_bce = torch.zeros((), dtype=regression_loss.dtype, device=regression_loss.device)
    if usefulness_weight > 0.0:
        logits = predicted_gain if usefulness_logits is None else usefulness_logits.float()
        if logits.shape != target_gain.shape:
            raise ValueError("usefulness_logits must match target_gain shape")
        usefulness_target = (target_gain > usefulness_tau).float()
        usefulness_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            usefulness_target,
        )
    total = regression_loss + float(usefulness_weight) * usefulness_bce
    return {
        "loss": total,
        "regression_loss": regression_loss,
        "usefulness_bce": usefulness_bce,
    }


def pairwise_listwise_topk_loss(
    predicted_gain: torch.Tensor,
    target_gain: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Cross-entropy over target top-k membership for per-image augmentation scores."""

    predicted_gain = predicted_gain.float()
    target_gain = target_gain.float()
    if predicted_gain.shape != target_gain.shape:
        raise ValueError("predicted_gain and target_gain must have the same shape")
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [images, augmentations]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    capped_top_k = min(top_k, target_gain.shape[1])
    target_topk = target_gain.topk(k=capped_top_k, dim=1).indices
    target_distribution = torch.zeros_like(target_gain)
    target_distribution.scatter_(dim=1, index=target_topk, value=1.0 / float(capped_top_k))
    return torch.sum(
        -target_distribution * torch.nn.functional.log_softmax(predicted_gain, dim=1), dim=1
    ).mean()


def pairwise_topk_kl_loss(
    predicted_gain: torch.Tensor,
    target_gain: torch.Tensor,
    *,
    top_k: int,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """KL loss to a soft target distribution over target top-k augmentations."""

    predicted_gain = predicted_gain.float()
    target_gain = target_gain.float()
    if predicted_gain.shape != target_gain.shape:
        raise ValueError("predicted_gain and target_gain must have the same shape")
    if predicted_gain.ndim != 2:
        raise ValueError("predicted_gain must have shape [images, augmentations]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if target_temperature <= 0.0:
        raise ValueError("target_temperature must be positive")

    capped_top_k = min(top_k, target_gain.shape[1])
    target_topk = target_gain.topk(k=capped_top_k, dim=1).indices
    topk_targets = torch.gather(target_gain, dim=1, index=target_topk)
    topk_distribution = torch.nn.functional.softmax(
        topk_targets / float(target_temperature),
        dim=1,
    )
    target_distribution = torch.zeros_like(target_gain)
    target_distribution.scatter_(dim=1, index=target_topk, src=topk_distribution)
    log_predicted = torch.nn.functional.log_softmax(predicted_gain, dim=1)
    log_target = torch.log(torch.clamp(target_distribution, min=1e-12))
    return torch.sum(target_distribution * (log_target - log_predicted), dim=1).mean()


def build_pairwise_marginal_logit_gain_targets(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    class_idxs: np.ndarray,
    identity_aug_id: str,
) -> np.ndarray:
    """Score each augmentation by true-class logit gain when paired with identity."""

    if identity_aug_id not in aug_ids:
        raise ValueError("identity augmentation must be present in aug_ids")
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    clean_logits = np.asarray(logits_by_aug[identity_aug_id], dtype=np.float32)
    if clean_logits.ndim != 2:
        raise ValueError("identity logits must have shape [images, classes]")
    if class_idxs.shape != (clean_logits.shape[0],):
        raise ValueError("class_idxs must have shape [images]")
    if np.any(class_idxs < 0) or np.any(class_idxs >= clean_logits.shape[1]):
        raise ValueError("class_idxs values must fall inside logits class dimension")

    row_indices = np.arange(clean_logits.shape[0])
    clean_true_logits = clean_logits[row_indices, class_idxs]
    columns = []
    for aug_id in aug_ids:
        aug_logits = np.asarray(logits_by_aug[aug_id], dtype=np.float32)
        if aug_logits.shape != clean_logits.shape:
            raise ValueError("all augmentation logits must match identity logits shape")
        if aug_id == identity_aug_id:
            columns.append(np.zeros_like(clean_true_logits, dtype=np.float32))
            continue
        ensemble_logits = (clean_logits + aug_logits) / 2.0
        ensemble_true_logits = ensemble_logits[row_indices, class_idxs]
        columns.append((ensemble_true_logits - clean_true_logits).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def build_pairwise_hard_example_weights(
    bundle: PairwiseFeatureBundle,
    *,
    hard_example_weight: float,
    confidence_threshold: float,
) -> np.ndarray:
    """Build row weights that emphasize clean-wrong or low-confidence images."""

    if hard_example_weight <= 0.0:
        return np.ones_like(bundle.targets, dtype=np.float32)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    confidence_index = bundle.feature_names.index("clean_confidence")
    pred_is_true_index = bundle.feature_names.index("clean_pred_is_true")
    per_image_features = bundle.features.reshape(len(bundle.image_ids), len(bundle.aug_ids), -1)[
        :, 0, :
    ]
    clean_confidence = per_image_features[:, confidence_index]
    clean_pred_is_true = per_image_features[:, pred_is_true_index]
    hard_images = (clean_pred_is_true < 0.5) | (clean_confidence < confidence_threshold)
    weights = np.ones_like(bundle.target_matrix, dtype=np.float32)
    weights[hard_images, :] += float(hard_example_weight)
    return weights.reshape(-1).astype(np.float32)


class _PairwiseImageBatchDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(self, bundle: PairwiseFeatureBundle, row_weights: np.ndarray) -> None:
        num_images = len(bundle.image_ids)
        num_augs = len(bundle.aug_ids)
        self.features = bundle.features.reshape(num_images, num_augs, -1)
        self.targets = bundle.target_matrix
        self.row_weights = row_weights.reshape(num_images, num_augs).astype(np.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index].astype(np.float32)),
            torch.from_numpy(self.targets[index].astype(np.float32)),
            torch.from_numpy(self.row_weights[index].astype(np.float32)),
        )


def _train_one_epoch(
    model: PairwiseSelectorMLP,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    usefulness_tau: float,
    usefulness_weight: float,
    positive_gain_weight: float,
    listwise_weight: float,
    listwise_top_k: int,
    listwise_loss: str,
    listwise_target_temperature: float,
) -> float:
    model.train()
    losses = []
    for features, targets, row_weights in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        row_weights = row_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        batch_size, aug_count, feature_dim = features.shape
        scores = model(features.reshape(batch_size * aug_count, feature_dim)).reshape(
            batch_size,
            aug_count,
        )
        loss_terms = pairwise_policy_loss(
            predicted_gain=scores.reshape(-1),
            target_gain=targets.reshape(-1),
            usefulness_logits=scores.reshape(-1),
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            positive_gain_weight=positive_gain_weight,
            row_weights=row_weights.reshape(-1),
        )
        listwise_term = _pairwise_listwise_loss(
            predicted_gain=scores,
            target_gain=targets,
            top_k=listwise_top_k,
            loss_name=listwise_loss,
            target_temperature=listwise_target_temperature,
        )
        loss = loss_terms["loss"] + float(listwise_weight) * listwise_term
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _is_better_pairwise_metric(current: float, best: float, selection_metric: str) -> bool:
    if selection_metric == "val_tta_top1":
        return current > best
    return current < best


def _pairwise_listwise_loss(
    predicted_gain: torch.Tensor,
    target_gain: torch.Tensor,
    *,
    top_k: int,
    loss_name: str,
    target_temperature: float,
) -> torch.Tensor:
    if loss_name == "topk_ce":
        return pairwise_listwise_topk_loss(
            predicted_gain,
            target_gain,
            top_k=top_k,
        )
    if loss_name == "topk_kl":
        return pairwise_topk_kl_loss(
            predicted_gain,
            target_gain,
            top_k=top_k,
            target_temperature=target_temperature,
        )
    raise ValueError("listwise_loss must be 'topk_ce' or 'topk_kl'")


def _prepare_optional_image_features(
    features: np.ndarray,
    feature_names: list[str],
    projection_dim: int | None,
    projection_seed: int,
    projection_method: str,
    projection_state: Mapping[str, Any] | None,
) -> tuple[np.ndarray, list[str]]:
    features = np.asarray(features, dtype=np.float32)
    if projection_state is not None:
        return apply_pairwise_feature_projection(features, projection_state)
    if projection_dim is None:
        return features, list(feature_names)
    if projection_dim <= 0:
        raise ValueError("feature_projection_dim must be positive")
    if projection_method == "pca_whiten":
        return apply_pairwise_feature_projection(
            features,
            fit_pairwise_feature_projection(
                features,
                projection_dim=projection_dim,
                method=projection_method,
            ),
        )
    if projection_method != "random":
        raise ValueError("feature_projection_method must be 'random' or 'pca_whiten'")
    if projection_dim >= features.shape[1]:
        return features, list(feature_names)
    rng = np.random.default_rng(projection_seed)
    projection = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(float(projection_dim)),
        size=(features.shape[1], projection_dim),
    ).astype(np.float32)
    projected = features @ projection
    projected_names = [f"projected_feature_{index:04d}" for index in range(projection_dim)]
    return projected.astype(np.float32), projected_names


def _fit_optional_pairwise_feature_projection(
    features_path: Path | None,
    projection_dim: int | None,
    projection_method: str,
) -> dict[str, Any] | None:
    if projection_method != "pca_whiten":
        return None
    if projection_dim is None:
        return None
    if features_path is None:
        raise ValueError("pca_whiten feature projection requires --train-features")
    pretrained = load_selector_features(features_path)
    return fit_pairwise_feature_projection(
        pretrained.features,
        projection_dim=projection_dim,
        method=projection_method,
    )


def fit_pairwise_feature_projection(
    features: np.ndarray,
    *,
    projection_dim: int,
    method: str = "pca_whiten",
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit a deterministic feature projection state for optional image features."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape [images, features]")
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if method != "pca_whiten":
        raise ValueError("method must be 'pca_whiten'")
    mean = features.mean(axis=0).astype(np.float32)
    centered = features - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    effective_dim = min(projection_dim, vt.shape[0])
    components = vt[:effective_dim].T.astype(np.float32)
    projected = centered @ components
    scale = np.maximum(projected.std(axis=0), eps).astype(np.float32)
    return {
        "method": "pca_whiten",
        "mean": mean,
        "components": components,
        "scale": scale,
        "feature_names": [f"pca_whiten_feature_{index:04d}" for index in range(effective_dim)],
    }


def apply_pairwise_feature_projection(
    features: np.ndarray,
    projection_state: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Apply a fitted optional image feature projection state."""

    features = np.asarray(features, dtype=np.float32)
    if projection_state.get("method") != "pca_whiten":
        raise ValueError("unsupported feature projection state method")
    mean = np.asarray(projection_state["mean"], dtype=np.float32)
    components = np.asarray(projection_state["components"], dtype=np.float32)
    scale = np.asarray(projection_state["scale"], dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape [images, features]")
    if mean.shape != (features.shape[1],):
        raise ValueError("projection mean must match feature dimension")
    if components.shape[0] != features.shape[1]:
        raise ValueError("projection components must match feature dimension")
    if scale.shape != (components.shape[1],):
        raise ValueError("projection scale must match projected dimension")
    projected = (features - mean) @ components
    projected = projected / scale
    return projected.astype(np.float32), [str(name) for name in projection_state["feature_names"]]


def _build_pairwise_bundle_from_image_features(
    image_ids: list[str],
    aug_ids: list[str],
    per_image_features: np.ndarray,
    image_feature_names: list[str],
    target_matrix: np.ndarray,
    class_idxs: np.ndarray,
) -> PairwiseFeatureBundle:
    aug_onehot = np.eye(len(aug_ids), dtype=np.float32)
    repeated_image_features = np.repeat(per_image_features, repeats=len(aug_ids), axis=0)
    tiled_aug_features = np.tile(aug_onehot, (len(image_ids), 1))
    pairwise_features = np.concatenate([repeated_image_features, tiled_aug_features], axis=1)
    row_image_indices = np.repeat(np.arange(len(image_ids), dtype=np.int64), len(aug_ids))
    row_aug_indices = np.tile(np.arange(len(aug_ids), dtype=np.int64), len(image_ids))
    return PairwiseFeatureBundle(
        image_ids=image_ids,
        aug_ids=aug_ids,
        features=pairwise_features.astype(np.float32),
        feature_names=[
            *image_feature_names,
            *[f"aug_onehot:{aug_id}" for aug_id in aug_ids],
        ],
        targets=target_matrix.reshape(-1).astype(np.float32),
        target_matrix=target_matrix.astype(np.float32),
        row_image_indices=row_image_indices,
        row_aug_indices=row_aug_indices,
        class_idxs=class_idxs,
    )


def _best_topk_metrics(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    predicted_gain: np.ndarray,
    identity_aug_id: str,
    top_k_grid: list[int],
) -> dict[str, float]:
    rows = []
    for k in top_k_grid:
        metrics = evaluate_learned_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=k,
        )
        rows.append({"k": float(k), **metrics})
    return min(rows, key=lambda row: float(row["nll"]))


def _read_validated_shard(
    cache_dir: Path,
    split: str,
    aug_id: str,
    image_ids: list[str],
):
    paths = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id)
    shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
    shard_image_ids = [str(image_id) for image_id in shard.metadata["image_id"].tolist()]
    if shard_image_ids != image_ids:
        raise ValueError("teacher shard image_ids must match manifest image_ids")
    return shard


def _load_pairwise_checkpoint(checkpoint_path: Path, device: torch.device) -> _PairwiseCheckpoint:
    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = cast("dict[str, Any]", raw_checkpoint)
    required = {"model_state_dict", "input_dim", "hidden_dim", "aug_ids", "feature_names"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"pairwise checkpoint is missing required keys: {missing}")
    return _PairwiseCheckpoint(
        model_state_dict=cast("Mapping[str, Any]", checkpoint["model_state_dict"]),
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        aug_ids=[str(aug_id) for aug_id in checkpoint["aug_ids"]],
        feature_names=[str(name) for name in checkpoint["feature_names"]],
        feature_projection_state=cast(
            "dict[str, Any] | None",
            checkpoint.get("feature_projection_state"),
        ),
    )


def _load_confidence_bucket_policy(policy_path: Path) -> ConfidenceBucketPolicy:
    raw_policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if not isinstance(raw_policy, dict):
        raise ValueError("confidence policy JSON must contain an object")
    if "confidence_bins" not in raw_policy or "bucket_k" not in raw_policy:
        raise ValueError("confidence policy must contain confidence_bins and bucket_k")

    confidence_bins = [float(value) for value in raw_policy["confidence_bins"]]
    bucket_k = [int(value) for value in raw_policy["bucket_k"]]
    if len(bucket_k) != len(confidence_bins) - 1:
        raise ValueError("confidence policy bucket_k must match confidence_bins intervals")
    return {"confidence_bins": confidence_bins, "bucket_k": bucket_k}


def _top1_delta_targets(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
    image_ids: list[str],
    identity_aug_id: str,
) -> np.ndarray:
    clean = _read_validated_shard(
        cache_dir=cache_dir,
        split=split,
        aug_id=identity_aug_id,
        image_ids=image_ids,
    )
    clean_top1 = np.asarray(clean.metadata["is_top1"].tolist(), dtype=np.float32)
    columns = []
    for aug_id in aug_ids:
        shard = _read_validated_shard(
            cache_dir=cache_dir,
            split=split,
            aug_id=aug_id,
            image_ids=image_ids,
        )
        aug_top1 = np.asarray(shard.metadata["is_top1"].tolist(), dtype=np.float32)
        columns.append((aug_top1 - clean_top1).astype(np.float32))
    return np.stack(columns, axis=1).astype(np.float32)


def _read_logits_by_aug(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
    image_ids: list[str],
) -> dict[str, np.ndarray]:
    logits_by_aug = {}
    for aug_id in aug_ids:
        shard = _read_validated_shard(
            cache_dir=cache_dir,
            split=split,
            aug_id=aug_id,
            image_ids=image_ids,
        )
        logits_by_aug[aug_id] = np.asarray(shard.logits, dtype=np.float32)
    return logits_by_aug
