"""Learned non-negative aggregation weights for cached TTA predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.tta_eval import evaluate_class_weighted_tta, evaluate_global_weighted_tta

AggregatorMethod = Literal["global-nonnegative", "class-nonnegative"]


@dataclass(frozen=True, slots=True)
class AggregationArtifact:
    """Saved learned TTA aggregation weights."""

    method: str
    aug_ids: list[str]
    weights: np.ndarray
    active_threshold: float
    metrics: dict[str, float]

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "method": self.method,
                    "aug_ids": self.aug_ids,
                    "weights": self.weights.astype(float).tolist(),
                    "active_threshold": self.active_threshold,
                    "metrics": self.metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


@dataclass(frozen=True, slots=True)
class AggregationTrainingSummary:
    """Summary of one learned aggregation training run."""

    path: Path
    method: str
    metrics: dict[str, float]


def load_aggregation_artifact(path: Path) -> AggregationArtifact:
    """Load a saved learned aggregation artifact."""

    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return AggregationArtifact(
        method=str(raw["method"]),
        aug_ids=[str(aug_id) for aug_id in raw["aug_ids"]],
        weights=np.asarray(raw["weights"], dtype=np.float32),
        active_threshold=float(raw["active_threshold"]),
        metrics={str(key): float(value) for key, value in raw["metrics"].items()},
    )


def train_global_nonnegative_weights(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    epochs: int,
    learning_rate: float,
    l1_penalty: float,
    active_threshold: float,
    device: str | torch.device = "cpu",
) -> AggregationArtifact:
    """Train one non-negative augmentation weight vector by public split NLL."""

    torch_device = torch.device(device)
    probabilities = _probability_tensor(logits_by_aug, aug_ids, torch_device)
    labels = torch.as_tensor(
        np.array(class_idxs, dtype=np.int64, copy=True),
        dtype=torch.long,
        device=torch_device,
    )
    raw_weights = torch.zeros(len(aug_ids), dtype=torch.float32, device=torch_device)
    raw_weights.requires_grad_(True)
    optimizer = torch.optim.Adam([raw_weights], lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = F.softplus(raw_weights)
        normalized = weights / weights.sum().clamp_min(1e-12)
        ensembled = torch.einsum("a,nac->nc", normalized, probabilities)
        loss = _nll_loss(ensembled, labels) + l1_penalty * normalized.mean()
        loss.backward()
        optimizer.step()

    weights_np = _normalized_numpy(F.softplus(raw_weights).detach().cpu().numpy())
    metrics = evaluate_global_weighted_tta(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        weights=weights_np,
        active_threshold=active_threshold,
    )
    return AggregationArtifact(
        method="global-nonnegative",
        aug_ids=aug_ids,
        weights=weights_np,
        active_threshold=active_threshold,
        metrics=metrics,
    )


def train_class_nonnegative_weights(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    epochs: int,
    learning_rate: float,
    l1_penalty: float,
    active_threshold: float,
    device: str | torch.device = "cpu",
) -> AggregationArtifact:
    """Train class-specific non-negative augmentation weights by public split NLL."""

    torch_device = torch.device(device)
    probabilities = _probability_tensor(logits_by_aug, aug_ids, torch_device)
    labels = torch.as_tensor(
        np.array(class_idxs, dtype=np.int64, copy=True),
        dtype=torch.long,
        device=torch_device,
    )
    num_classes = probabilities.shape[2]
    raw_weights = torch.zeros(
        (num_classes, len(aug_ids)),
        dtype=torch.float32,
        device=torch_device,
    )
    raw_weights.requires_grad_(True)
    optimizer = torch.optim.Adam([raw_weights], lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = F.softplus(raw_weights)
        normalized = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        scores = torch.einsum("nac,ca->nc", probabilities, normalized)
        scores = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
        loss = _nll_loss(scores, labels) + l1_penalty * normalized.mean()
        loss.backward()
        optimizer.step()

    weights_np = _row_normalized_numpy(F.softplus(raw_weights).detach().cpu().numpy())
    metrics = evaluate_class_weighted_tta(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        class_weights=weights_np,
        active_threshold=active_threshold,
    )
    return AggregationArtifact(
        method="class-nonnegative",
        aug_ids=aug_ids,
        weights=weights_np,
        active_threshold=active_threshold,
        metrics=metrics,
    )


def train_aggregator_from_artifacts(
    split: str,
    cache_dir: Path,
    output_path: Path,
    aug_ids: list[str],
    method: str,
    epochs: int,
    learning_rate: float,
    l1_penalty: float,
    active_threshold: float,
    device: str | torch.device = "cpu",
) -> AggregationTrainingSummary:
    """Train and save learned aggregation weights from cached split logits."""

    if method not in {"global-nonnegative", "class-nonnegative"}:
        raise ValueError(f"unknown aggregator method {method!r}")

    logits_by_aug, class_idxs = _read_split_logits(cache_dir, split=split, aug_ids=aug_ids)
    if method == "global-nonnegative":
        artifact = train_global_nonnegative_weights(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            epochs=epochs,
            learning_rate=learning_rate,
            l1_penalty=l1_penalty,
            active_threshold=active_threshold,
            device=device,
        )
    elif method == "class-nonnegative":
        artifact = train_class_nonnegative_weights(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            epochs=epochs,
            learning_rate=learning_rate,
            l1_penalty=l1_penalty,
            active_threshold=active_threshold,
            device=device,
        )
    else:
        raise AssertionError("validated aggregator method became unreachable")

    artifact.save(output_path)
    return AggregationTrainingSummary(
        path=Path(output_path),
        method=artifact.method,
        metrics=artifact.metrics,
    )


def train_aggregator_from_config(
    config_path: Path,
    split: str = "public_val",
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    output_path: Path | None = None,
    candidate_ids: list[str] | None = None,
    method: str = "global-nonnegative",
    epochs: int = 200,
    learning_rate: float = 0.05,
    l1_penalty: float = 0.0,
    active_threshold: float = 1e-6,
    device: str | torch.device = "cpu",
) -> AggregationTrainingSummary:
    """Load config and train a learned TTA aggregation artifact."""

    config = load_experiment_config(config_path)
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    resolved_output_dir = output_dir or config.artifacts.selector_dir
    resolved_output_path = output_path or _default_aggregator_path(
        output_dir=resolved_output_dir,
        split=split,
        method=method,
    )
    return train_aggregator_from_artifacts(
        split=split,
        cache_dir=cache_dir or config.artifacts.teacher_cache_dir,
        output_path=resolved_output_path,
        aug_ids=candidate_ids,
        method=method,
        epochs=epochs,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        active_threshold=active_threshold,
        device=device,
    )


def default_aggregator_path(output_dir: Path, split: str, method: str) -> Path:
    """Return the conventional learned aggregation artifact path."""

    return _default_aggregator_path(output_dir=output_dir, split=split, method=method)


def _probability_tensor(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    device: torch.device,
) -> torch.Tensor:
    probabilities = []
    for aug_id in aug_ids:
        logits = torch.as_tensor(logits_by_aug[aug_id], dtype=torch.float32, device=device)
        probabilities.append(torch.softmax(logits, dim=1))
    return torch.stack(probabilities, dim=1)


def _nll_loss(probabilities: torch.Tensor, class_idxs: torch.Tensor) -> torch.Tensor:
    true_probabilities = probabilities[
        torch.arange(class_idxs.numel(), device=class_idxs.device),
        class_idxs,
    ]
    return -torch.log(true_probabilities.clamp_min(1e-45)).mean()


def _normalized_numpy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return weights / np.clip(weights.sum(), 1e-12, None)


def _row_normalized_numpy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)


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


def _default_aggregator_path(output_dir: Path, split: str, method: str) -> Path:
    method_slug = method.replace("-", "_")
    return Path(output_dir) / f"{split}_{method_slug}_aggregator.json"
