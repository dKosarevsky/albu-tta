"""Tune learned TTA top-k on public validation data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.selector_model import SelectorCNN
from learned_tta.split_policy import validate_public_tuning_split
from learned_tta.tta_eval import (
    evaluate_learned_adaptive_uniform,
    evaluate_learned_topk_uniform,
    select_best_k,
)


@dataclass(frozen=True, slots=True)
class TTATuningSummary:
    """Summary of one TTA tuning run."""

    split: str
    result_path: Path
    best_k: int
    results_by_k: dict[int, dict[str, float]]
    predicted_gain_shape: tuple[int, int]
    adaptive_results: dict[str, dict[str, float]] | None = None
    best_adaptive_threshold: float | None = None
    best_adaptive_max_k: int | None = None
    predicted_useful_shape: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class SelectorPredictions:
    """Selector predictions after checkpoint unstandardization."""

    gain: np.ndarray
    useful_prob: np.ndarray | None = None


def tune_tta_from_artifacts(
    split: str,
    manifest_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    aug_ids: list[str],
    top_k_grid: list[int],
    image_size: int,
    batch_size: int,
    num_workers: int,
    adaptive_threshold_grid: list[float] | None = None,
    adaptive_max_k_grid: list[int] | None = None,
    device: str | torch.device = "cpu",
    identity_aug_id: str = "aug_000",
) -> TTATuningSummary:
    """Tune learned top-k uniform TTA by public-validation NLL."""

    validate_public_tuning_split(split, command="tune-tta")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(manifest_path)
    logits_by_aug, class_idxs = _read_split_logits(cache_dir, split=split, aug_ids=aug_ids)
    predictions = predict_selector_outputs(
        checkpoint_path=checkpoint_path,
        records=records,
        output_dim=len(aug_ids),
        aug_ids=aug_ids,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    predicted_gain = predictions.gain

    results_by_k = {
        k: evaluate_learned_topk_uniform(
            logits_by_aug=logits_by_aug,
            class_idxs=class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id=identity_aug_id,
            k=k,
        )
        for k in top_k_grid
    }
    best_k = select_best_k(results_by_k, metric="nll", higher_is_better=False)
    adaptive_results: dict[str, dict[str, float]] | None = None
    best_adaptive_threshold: float | None = None
    best_adaptive_max_k: int | None = None
    if (
        predictions.useful_prob is not None
        and adaptive_threshold_grid is not None
        and adaptive_max_k_grid is not None
    ):
        adaptive_results = {
            _adaptive_key(threshold=threshold, max_k=max_k): evaluate_learned_adaptive_uniform(
                logits_by_aug=logits_by_aug,
                class_idxs=class_idxs,
                aug_ids=aug_ids,
                predicted_gain=predicted_gain,
                useful_prob=predictions.useful_prob,
                identity_aug_id=identity_aug_id,
                threshold=threshold,
                max_k=max_k,
            )
            for threshold in adaptive_threshold_grid
            for max_k in adaptive_max_k_grid
        }
        best_key = min(adaptive_results, key=lambda key: adaptive_results[key]["nll"])
        best_adaptive_threshold, best_adaptive_max_k = _parse_adaptive_key(best_key)
    result_path = output_dir / f"{split}_tta_tuning.json"
    result_path.write_text(
        json.dumps(
            {
                "split": split,
                "best_k": best_k,
                "results_by_k": results_by_k,
                "adaptive_results": adaptive_results,
                "best_adaptive_threshold": best_adaptive_threshold,
                "best_adaptive_max_k": best_adaptive_max_k,
                "predicted_gain_shape": list(predicted_gain.shape),
                "predicted_useful_shape": (
                    list(predictions.useful_prob.shape)
                    if predictions.useful_prob is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TTATuningSummary(
        split=split,
        result_path=result_path,
        best_k=best_k,
        results_by_k=results_by_k,
        predicted_gain_shape=predicted_gain.shape,
        adaptive_results=adaptive_results,
        best_adaptive_threshold=best_adaptive_threshold,
        best_adaptive_max_k=best_adaptive_max_k,
        predicted_useful_shape=(
            predictions.useful_prob.shape if predictions.useful_prob is not None else None
        ),
    )


def tune_tta_from_config(
    config_path: Path,
    split: str = "public_val",
    manifest_path: Path | None = None,
    cache_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    output_dir: Path | None = None,
    candidate_ids: list[str] | None = None,
    top_k_grid: list[int] | None = None,
    adaptive_threshold_grid: list[float] | None = None,
    adaptive_max_k_grid: list[int] | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | torch.device = "cpu",
) -> TTATuningSummary:
    """Load experiment config and tune learned top-k TTA."""

    config = load_experiment_config(config_path)
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    return tune_tta_from_artifacts(
        split=split,
        manifest_path=manifest_path or config.artifacts.manifests_dir / f"{split}.csv",
        cache_dir=cache_dir or config.artifacts.teacher_cache_dir,
        checkpoint_path=checkpoint_path or config.artifacts.selector_dir / "selector_best.pt",
        output_dir=output_dir or config.artifacts.selector_dir,
        aug_ids=candidate_ids,
        top_k_grid=top_k_grid or config.selector.top_k_grid,
        adaptive_threshold_grid=adaptive_threshold_grid or config.selector.adaptive_threshold_grid,
        adaptive_max_k_grid=adaptive_max_k_grid or config.selector.adaptive_max_k_grid,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )


@torch.inference_mode()
def predict_selector_scores(
    checkpoint_path: Path,
    records: list[ManifestRecord],
    output_dim: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    aug_ids: list[str] | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict per-augmentation selector scores for clean images."""

    return predict_selector_outputs(
        checkpoint_path=checkpoint_path,
        records=records,
        output_dim=output_dim,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_ids=aug_ids,
        device=device,
    ).gain


@torch.inference_mode()
def predict_selector_outputs(
    checkpoint_path: Path,
    records: list[ManifestRecord],
    output_dim: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    aug_ids: list[str] | None = None,
    device: str | torch.device = "cpu",
) -> SelectorPredictions:
    """Predict gain scores and optional usefulness probabilities for clean images."""

    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model = SelectorCNN(
        output_dim=output_dim,
        usefulness_head=_checkpoint_has_usefulness_head(checkpoint),
    ).to(torch_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        _SelectorImageDataset(records=records, image_size=image_size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    predictions = []
    useful_logits = []
    for images in dataloader:
        outputs = model.forward_heads(images.to(torch_device))
        predictions.append(outputs.gain.cpu().numpy().astype(np.float32))
        if outputs.useful_logits is not None:
            useful_logits.append(outputs.useful_logits.cpu().numpy().astype(np.float32))
    target_z = np.concatenate(predictions, axis=0)
    gain = _unstandardize_checkpoint_scores(target_z, checkpoint, output_dim, aug_ids=aug_ids)
    useful_prob = (
        1.0 / (1.0 + np.exp(-np.concatenate(useful_logits, axis=0))) if useful_logits else None
    )
    return SelectorPredictions(gain=gain, useful_prob=useful_prob)


def _unstandardize_checkpoint_scores(
    target_z: np.ndarray,
    checkpoint: dict[str, object],
    output_dim: int,
    aug_ids: list[str] | None = None,
) -> np.ndarray:
    checkpoint_aug_ids = checkpoint.get("aug_ids")
    if aug_ids is not None:
        if checkpoint_aug_ids is None:
            raise ValueError("checkpoint aug_ids are required when requested aug_ids are provided")
        if not isinstance(checkpoint_aug_ids, list):
            raise ValueError("checkpoint aug_ids must be a list")
        if [str(aug_id) for aug_id in checkpoint_aug_ids] != aug_ids:
            raise ValueError("checkpoint aug_ids must match requested aug_ids")

    if "target_mean" not in checkpoint or "target_std" not in checkpoint:
        return target_z

    mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    if mean.shape != (output_dim,) or std.shape != (output_dim,):
        raise ValueError("checkpoint target stats must match selector output_dim")
    if checkpoint_aug_ids is not None and not isinstance(checkpoint_aug_ids, list):
        raise ValueError("checkpoint aug_ids must be a list")
    if checkpoint_aug_ids is not None and len(checkpoint_aug_ids) != output_dim:
        raise ValueError("checkpoint aug_ids length must match selector output_dim")
    return (target_z * std[None, :] + mean[None, :]).astype(np.float32)


def _checkpoint_has_usefulness_head(checkpoint: dict[str, object]) -> bool:
    if bool(checkpoint.get("usefulness_head", False)):
        return True
    state_dict = checkpoint.get("model_state_dict", {})
    if not isinstance(state_dict, dict):
        return False
    return any(str(key).startswith("useful_head.") for key in state_dict)


def _adaptive_key(threshold: float, max_k: int) -> str:
    return f"threshold={threshold:g},max_k={max_k}"


def _parse_adaptive_key(key: str) -> tuple[float, int]:
    parts = dict(part.split("=", maxsplit=1) for part in key.split(","))
    return float(parts["threshold"]), int(parts["max_k"])


class _SelectorImageDataset(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, records: list[ManifestRecord], image_size: int) -> None:
        self.records = records
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.records[index].path) as image:
            resized = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


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
