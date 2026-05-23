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
from learned_tta.tta_eval import evaluate_learned_topk_uniform, select_best_k


@dataclass(frozen=True, slots=True)
class TTATuningSummary:
    """Summary of one TTA tuning run."""

    split: str
    result_path: Path
    best_k: int
    results_by_k: dict[int, dict[str, float]]
    predicted_gain_shape: tuple[int, int]


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
    device: str | torch.device = "cpu",
    identity_aug_id: str = "aug_000",
) -> TTATuningSummary:
    """Tune learned top-k uniform TTA by public-validation NLL."""

    validate_public_tuning_split(split, command="tune-tta")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    result_path = output_dir / f"{split}_tta_tuning.json"
    result_path.write_text(
        json.dumps(
            {
                "split": split,
                "best_k": best_k,
                "results_by_k": results_by_k,
                "predicted_gain_shape": list(predicted_gain.shape),
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
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict per-augmentation selector scores for clean images."""

    torch_device = torch.device(device)
    model = SelectorCNN(output_dim=output_dim).to(torch_device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        _SelectorImageDataset(records=records, image_size=image_size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    predictions = []
    for images in dataloader:
        predictions.append(model(images.to(torch_device)).cpu().numpy().astype(np.float32))
    target_z = np.concatenate(predictions, axis=0)
    return _unstandardize_checkpoint_scores(target_z, checkpoint, output_dim)


def _unstandardize_checkpoint_scores(
    target_z: np.ndarray,
    checkpoint: dict[str, object],
    output_dim: int,
) -> np.ndarray:
    if "target_mean" not in checkpoint or "target_std" not in checkpoint:
        return target_z

    mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["target_std"], dtype=np.float32)
    if mean.shape != (output_dim,) or std.shape != (output_dim,):
        raise ValueError("checkpoint target stats must match selector output_dim")
    checkpoint_aug_ids = checkpoint.get("aug_ids")
    if checkpoint_aug_ids is not None and not isinstance(checkpoint_aug_ids, list):
        raise ValueError("checkpoint aug_ids must be a list")
    if checkpoint_aug_ids is not None and len(checkpoint_aug_ids) != output_dim:
        raise ValueError("checkpoint aug_ids length must match selector output_dim")
    return (target_z * std[None, :] + mean[None, :]).astype(np.float32)


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
