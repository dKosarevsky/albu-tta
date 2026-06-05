"""Learned non-negative aggregation weights for cached TTA predictions."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.metrics import classification_metrics, expected_calibration_error
from learned_tta.split_policy import validate_public_tuning_split
from learned_tta.tta_eval import evaluate_class_weighted_tta, evaluate_global_weighted_tta

AggregatorMethod = Literal["global-nonnegative", "class-nonnegative", "xgboost-multiclass"]


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
class XGBoostAggregationArtifact:
    """Saved optional XGBoost stacker metadata."""

    method: str
    aug_ids: list[str]
    model_path: Path
    num_classes: int
    feature_count: int
    feature_importance: np.ndarray
    metrics: dict[str, float]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = _portable_model_path(self.model_path, path.parent)
        path.write_text(
            json.dumps(
                {
                    "method": self.method,
                    "aug_ids": self.aug_ids,
                    "model_path": str(model_path),
                    "num_classes": self.num_classes,
                    "feature_count": self.feature_count,
                    "feature_importance": self.feature_importance.astype(float).tolist(),
                    "metrics": self.metrics,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _portable_model_path(model_path: Path, artifact_dir: Path) -> Path:
    model_path = Path(model_path)
    artifact_dir = Path(artifact_dir)
    try:
        return model_path.relative_to(artifact_dir)
    except ValueError:
        return model_path


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


def load_xgboost_aggregation_artifact(path: Path) -> XGBoostAggregationArtifact:
    """Load saved optional XGBoost stacker metadata."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    model_path = Path(str(raw["model_path"]))
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    aug_ids = [str(aug_id) for aug_id in raw["aug_ids"]]
    raw_feature_importance = raw.get("feature_importance")
    feature_importance = (
        np.zeros(len(aug_ids), dtype=np.float32)
        if raw_feature_importance is None
        else np.asarray(raw_feature_importance, dtype=np.float32)
    )
    return XGBoostAggregationArtifact(
        method=str(raw["method"]),
        aug_ids=aug_ids,
        model_path=model_path,
        num_classes=int(raw["num_classes"]),
        feature_count=int(raw["feature_count"]),
        feature_importance=feature_importance,
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
    """Train one sparse non-negative augmentation weight vector by public split NLL."""

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
        loss = _nll_loss(ensembled, labels) + _simplex_sparsity_penalty(
            normalized,
            strength=l1_penalty,
        )
        loss.backward()
        optimizer.step()

    weights_np = _prune_and_normalize_weights(
        _normalized_numpy(F.softplus(raw_weights).detach().cpu().numpy()),
        active_threshold=active_threshold,
    )
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
    """Train sparse class-specific non-negative augmentation weights by public split NLL."""

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
        loss = _nll_loss(scores, labels) + _simplex_sparsity_penalty(
            normalized,
            strength=l1_penalty,
        )
        loss.backward()
        optimizer.step()

    weights_np = _prune_and_normalize_weights(
        _row_normalized_numpy(F.softplus(raw_weights).detach().cpu().numpy()),
        active_threshold=active_threshold,
    )
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


def train_xgboost_multiclass_stacker(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    output_path: Path,
    n_estimators: int,
    learning_rate: float,
) -> XGBoostAggregationArtifact:
    """Train an optional XGBoost second-level stacker over flattened TTA probabilities."""

    xgboost = _require_xgboost()
    labels = np.asarray(class_idxs, dtype=np.int64)
    features = _stacker_feature_matrix(logits_by_aug=logits_by_aug, aug_ids=aug_ids)
    num_classes = _num_classes_from_logits(logits_by_aug=logits_by_aug, aug_ids=aug_ids)
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("class_idxs must be valid for logits class count")
    classifier = xgboost.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=max(1, int(n_estimators)),
        learning_rate=float(learning_rate),
        max_depth=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=20260522,
        n_jobs=1,
    )
    classifier.fit(features, labels)
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float32)
    model_path = Path(output_path).with_suffix(".model.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_path)
    metrics = _probability_metrics(
        probabilities=probabilities,
        class_idxs=labels,
        forwards_per_image=len(aug_ids),
        total_augments=len(aug_ids),
    )
    return XGBoostAggregationArtifact(
        method="xgboost-multiclass",
        aug_ids=aug_ids,
        model_path=model_path,
        num_classes=num_classes,
        feature_count=features.shape[1],
        feature_importance=_xgboost_feature_importance(
            classifier=classifier,
            aug_count=len(aug_ids),
            num_classes=num_classes,
        ),
        metrics=metrics,
    )


def xgboost_multiclass_probabilities(
    artifact_path: Path,
    logits_by_aug: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict stacked probabilities with a saved optional XGBoost artifact."""

    artifact = load_xgboost_aggregation_artifact(artifact_path)
    xgboost = _require_xgboost()
    features = _stacker_feature_matrix(logits_by_aug=logits_by_aug, aug_ids=artifact.aug_ids)
    if features.shape[1] != artifact.feature_count:
        raise ValueError("xgboost feature count does not match saved artifact")
    classifier = xgboost.XGBClassifier()
    classifier.load_model(artifact.model_path)
    probabilities = np.asarray(classifier.predict_proba(features), dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != artifact.num_classes:
        raise ValueError("xgboost probabilities do not match saved artifact class count")
    return probabilities


def evaluate_xgboost_multiclass_stacker(
    artifact_path: Path,
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    total_augments: int,
) -> dict[str, float]:
    """Evaluate a saved optional XGBoost stacker on cached TTA predictions."""

    artifact = load_xgboost_aggregation_artifact(artifact_path)
    probabilities = xgboost_multiclass_probabilities(artifact_path, logits_by_aug)
    return _probability_metrics(
        probabilities=probabilities,
        class_idxs=class_idxs,
        forwards_per_image=len(artifact.aug_ids),
        total_augments=total_augments,
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

    validate_public_tuning_split(split, command="train-aggregator")
    if method not in {"global-nonnegative", "class-nonnegative", "xgboost-multiclass"}:
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
    elif method == "xgboost-multiclass":
        artifact = train_xgboost_multiclass_stacker(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            output_path=output_path,
            n_estimators=epochs,
            learning_rate=learning_rate,
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


def _stacker_feature_matrix(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
) -> np.ndarray:
    probabilities = [_softmax_numpy(logits_by_aug[aug_id]) for aug_id in aug_ids]
    return np.concatenate(probabilities, axis=1).astype(np.float32)


def _num_classes_from_logits(logits_by_aug: dict[str, np.ndarray], aug_ids: list[str]) -> int:
    if not aug_ids:
        raise ValueError("aug_ids must not be empty")
    num_classes = int(np.asarray(logits_by_aug[aug_ids[0]]).shape[1])
    for aug_id in aug_ids[1:]:
        if int(np.asarray(logits_by_aug[aug_id]).shape[1]) != num_classes:
            raise ValueError("all augmentation logits must have the same class count")
    return num_classes


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _probability_metrics(
    probabilities: np.ndarray,
    class_idxs: np.ndarray,
    forwards_per_image: int,
    total_augments: int,
) -> dict[str, float]:
    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    metrics["forwards_per_image"] = float(forwards_per_image)
    metrics["relative_compute_vs_all"] = float(forwards_per_image / total_augments)
    return metrics


def _xgboost_feature_importance(
    classifier: Any,
    aug_count: int,
    num_classes: int,
) -> np.ndarray:
    raw_importance = getattr(classifier, "feature_importances_", None)
    if raw_importance is None:
        return np.zeros(aug_count, dtype=np.float32)

    feature_importance = np.asarray(raw_importance, dtype=np.float32)
    expected_features = aug_count * num_classes
    if feature_importance.shape != (expected_features,):
        raise ValueError("xgboost feature importance length must match feature count")

    per_aug = feature_importance.reshape(aug_count, num_classes).sum(axis=1)
    total = float(per_aug.sum())
    if total > 0.0:
        per_aug = per_aug / total
    return per_aug.astype(np.float32)


def _require_xgboost() -> Any:
    try:
        return importlib.import_module("xgboost")
    except ImportError as error:
        raise RuntimeError(
            "xgboost-multiclass requires the optional 'xgboost' package to be installed"
        ) from error


def _nll_loss(probabilities: torch.Tensor, class_idxs: torch.Tensor) -> torch.Tensor:
    true_probabilities = probabilities[
        torch.arange(class_idxs.numel(), device=class_idxs.device),
        class_idxs,
    ]
    return -torch.log(true_probabilities.clamp_min(1e-45)).mean()


def _simplex_sparsity_penalty(weights: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0:
        return weights.new_tensor(0.0)
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum(dim=-1).mean()
    return float(strength) * entropy


def _normalized_numpy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return weights / np.clip(weights.sum(), 1e-12, None)


def _row_normalized_numpy(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    return weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)


def _prune_and_normalize_weights(weights: np.ndarray, active_threshold: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim == 1:
        return _normalized_numpy(_prune_weight_row(weights, active_threshold))
    if weights.ndim == 2:
        return _row_normalized_numpy(
            np.stack(
                [_prune_weight_row(row, active_threshold) for row in weights],
                axis=0,
            )
        )
    raise ValueError("weights must have shape [augmentations] or [classes, augmentations]")


def _prune_weight_row(weights: np.ndarray, active_threshold: float) -> np.ndarray:
    mask = weights > active_threshold
    if not np.any(mask):
        mask[int(np.argmax(weights))] = True
    return np.where(mask, weights, 0.0).astype(np.float32)


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


def _default_aggregator_path(output_dir: Path, split: str, method: str) -> Path:
    method_slug = method.replace("-", "_")
    return Path(output_dir) / f"{split}_{method_slug}_aggregator.json"
