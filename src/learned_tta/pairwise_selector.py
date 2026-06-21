"""Pairwise `(image, augmentation)` selector training helpers."""

from __future__ import annotations

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
    evaluate_clean,
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
        image_features.append(pretrained.features)
        image_feature_names.extend(pretrained.feature_names)

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
    else:
        raise ValueError("target_mode must be 'nll_gain' or 'top1_delta'")

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
        image_features.append(pretrained.features)
        image_feature_names.extend(pretrained.feature_names)

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
    top_k: int = 16,
    batch_size: int = 8192,
    strategy_name: str = "pairwise_topk_uniform",
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
        "oracle_topk_uniform": evaluate_oracle_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=bundle.class_idxs,
            identity_aug_id=identity_aug_id,
            k=top_k,
        ),
    }
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
    top_k_grid: list[int] | None = None,
    batch_size: int = 1024,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    positive_gain_weight: float = 0.0,
    target_mode: str = "nll_gain",
    selection_metric: str = "val_tta_nll",
    device: str | torch.device = "cpu",
) -> PairwiseTrainingSummary:
    """Train a lightweight pairwise selector and write a compact summary CSV."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bundle = build_pairwise_feature_bundle(
        manifest_path=train_manifest_path,
        targets_path=train_targets_path,
        cache_dir=cache_dir,
        identity_aug_id=identity_aug_id,
        features_path=train_features_path,
        target_mode=target_mode,
    )
    val_bundle = build_pairwise_feature_bundle(
        manifest_path=val_manifest_path,
        targets_path=val_targets_path,
        cache_dir=cache_dir,
        identity_aug_id=identity_aug_id,
        features_path=val_features_path,
        target_mode=target_mode,
    )
    model = PairwiseSelectorMLP(input_dim=train_bundle.features.shape[1], hidden_dim=hidden_dim)
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = cast(
        "torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]",
        torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(train_bundle.features),
                torch.from_numpy(train_bundle.targets),
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
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
    top_k_grid: list[int] | None = None,
    batch_size: int = 1024,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    hidden_dim: int = 128,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    positive_gain_weight: float = 0.0,
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
            top_k_grid=top_k_grid,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            positive_gain_weight=positive_gain_weight,
            target_mode=target_mode,
            selection_metric=selection_metric,
            device=device,
        )
        rows.append(
            {
                "variant": variant,
                "target_mode": target_mode,
                "selection_metric": selection_metric,
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


def _train_one_epoch(
    model: PairwiseSelectorMLP,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    usefulness_tau: float,
    usefulness_weight: float,
    positive_gain_weight: float,
) -> float:
    model.train()
    losses = []
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(features)
        loss_terms = pairwise_policy_loss(
            predicted_gain=scores,
            target_gain=targets,
            usefulness_logits=scores,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            positive_gain_weight=positive_gain_weight,
        )
        loss = loss_terms["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _is_better_pairwise_metric(current: float, best: float, selection_metric: str) -> bool:
    if selection_metric == "val_tta_top1":
        return current > best
    return current < best


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
    )


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
