"""Selector CNN training runner."""

from __future__ import annotations

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
from learned_tta.targets import SavedSelectorTargets, TargetStats, load_selector_targets
from learned_tta.train_selector import (
    CheckpointState,
    evaluate_regression,
    save_checkpoint_if_best,
    train_one_epoch,
)
from learned_tta.tta_eval import evaluate_learned_topk_uniform, select_best_k


@dataclass(frozen=True, slots=True)
class SelectorTrainingSummary:
    """Summary of selector training."""

    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
    best_val_nll: float
    history: list[dict[str, float]]


class SelectorImageTargetDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Clean-image selector dataset paired with precomputed target rows."""

    def __init__(
        self,
        records: list[ManifestRecord],
        targets: SavedSelectorTargets,
        image_size: int,
    ) -> None:
        if len(records) != targets.target_z.shape[0]:
            raise ValueError("manifest row count must match selector target rows")
        self.records = records
        self.targets = targets
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as image:
            resized = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1)
        target_tensor = torch.from_numpy(self.targets.target_z[index].astype(np.float32))
        return image_tensor, target_tensor


def make_selector_dataloader(
    manifest_path: Path,
    targets_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a selector DataLoader from a manifest and saved selector target artifact."""

    dataset = SelectorImageTargetDataset(
        records=load_manifest(manifest_path),
        targets=load_selector_targets(targets_path),
        image_size=image_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


def train_selector_from_artifacts(
    train_manifest_path: Path,
    val_manifest_path: Path,
    train_targets_path: Path,
    val_targets_path: Path,
    output_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    learning_rate: float,
    rank_weight: float,
    val_cache_dir: Path | None = None,
    val_split: str = "public_val",
    aug_ids: list[str] | None = None,
    top_k_grid: list[int] | None = None,
    identity_aug_id: str = "aug_000",
    device: str | torch.device = "cpu",
) -> SelectorTrainingSummary:
    """Train selector CNN from manifest CSVs and saved target artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_targets = load_selector_targets(train_targets_path)
    output_dim = train_targets.target_z.shape[1]
    if aug_ids is None:
        aug_ids = train_targets.aug_ids
    if aug_ids != train_targets.aug_ids:
        raise ValueError("aug_ids must match selector target aug_ids")
    torch_device = torch.device(device)

    model = SelectorCNN(output_dim=output_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataloader = make_selector_dataloader(
        manifest_path=train_manifest_path,
        targets_path=train_targets_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = make_selector_dataloader(
        manifest_path=val_manifest_path,
        targets_path=val_targets_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    checkpoint_path = output_dir / "selector_best.pt"
    checkpoint_state = CheckpointState(best_val_nll=float("inf"), path=checkpoint_path)
    val_logits_by_aug: dict[str, np.ndarray] | None = None
    val_class_idxs: np.ndarray | None = None
    if val_cache_dir is not None:
        if not top_k_grid:
            raise ValueError("top_k_grid must be provided when val_cache_dir is set")
        val_logits_by_aug, val_class_idxs = _read_split_logits(
            cache_dir=val_cache_dir,
            split=val_split,
            aug_ids=aug_ids,
        )
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=torch_device,
            rank_weight=rank_weight,
        )
        val_metrics = evaluate_regression(
            model=model,
            dataloader=val_dataloader,
            device=torch_device,
            rank_weight=rank_weight,
        )
        history_row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_spearman": val_metrics["spearman"],
        }
        checkpoint_metric = val_metrics["loss"]
        if val_logits_by_aug is not None and val_class_idxs is not None:
            tta_metrics = _evaluate_validation_tta(
                model=model,
                dataloader=val_dataloader,
                device=torch_device,
                target_stats=train_targets.stats,
                logits_by_aug=val_logits_by_aug,
                class_idxs=val_class_idxs,
                aug_ids=aug_ids,
                top_k_grid=top_k_grid or [],
                identity_aug_id=identity_aug_id,
            )
            history_row.update(tta_metrics)
            checkpoint_metric = tta_metrics["val_tta_nll"]
        checkpoint_state = save_checkpoint_if_best(
            state=checkpoint_state,
            val_nll=checkpoint_metric,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            aug_ids=train_targets.aug_ids,
            target_stats=train_targets.stats,
        )
        history.append(history_row)

    best_epoch = checkpoint_state.best_epoch or 0
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    return SelectorTrainingSummary(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_val_loss=float(best_row["val_loss"]) if best_row is not None else 0.0,
        best_val_nll=checkpoint_state.best_val_nll,
        history=history,
    )


def train_selector_from_config(
    config_path: Path,
    train_manifest_path: Path | None = None,
    val_manifest_path: Path | None = None,
    train_targets_path: Path | None = None,
    val_targets_path: Path | None = None,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    val_split: str = "public_val",
    candidate_ids: list[str] | None = None,
    top_k_grid: list[int] | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    rank_weight: float = 0.2,
    device: str | torch.device = "cpu",
) -> SelectorTrainingSummary:
    """Load experiment config and train selector from configured artifact locations."""

    config = load_experiment_config(config_path)
    selector_dir = output_dir or config.artifacts.selector_dir
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    return train_selector_from_artifacts(
        train_manifest_path=train_manifest_path
        or config.artifacts.manifests_dir / "public_train.csv",
        val_manifest_path=val_manifest_path or config.artifacts.manifests_dir / "public_val.csv",
        train_targets_path=train_targets_path or selector_dir / "public_train_targets.npz",
        val_targets_path=val_targets_path or selector_dir / "public_val_targets.npz",
        output_dir=selector_dir,
        val_cache_dir=cache_dir or config.artifacts.teacher_cache_dir,
        val_split=val_split,
        aug_ids=candidate_ids,
        top_k_grid=top_k_grid or config.selector.top_k_grid,
        identity_aug_id=config.augmentations.identity_id,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=learning_rate,
        rank_weight=rank_weight,
        device=device,
    )


@torch.inference_mode()
def _evaluate_validation_tta(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    target_stats: TargetStats,
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    aug_ids: list[str],
    top_k_grid: list[int],
    identity_aug_id: str,
) -> dict[str, float]:
    predicted_gain = _predict_gain(
        model=model,
        dataloader=dataloader,
        device=device,
        target_stats=target_stats,
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
    best_metrics = results_by_k[best_k]
    return {
        "val_tta_best_k": float(best_k),
        "val_tta_top1": best_metrics["top1"],
        "val_tta_top5": best_metrics["top5"],
        "val_tta_nll": best_metrics["nll"],
        "val_tta_ece": best_metrics["ece"],
    }


@torch.inference_mode()
def _predict_gain(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    target_stats: TargetStats,
) -> np.ndarray:
    model.eval()
    predictions = []
    for images, _ in dataloader:
        predictions.append(model(images.to(device)).cpu().numpy().astype(np.float32))
    target_z = np.concatenate(predictions, axis=0)
    mean = np.asarray(target_stats.mean, dtype=np.float32)
    std = np.asarray(target_stats.std, dtype=np.float32)
    return (target_z * std[None, :] + mean[None, :]).astype(np.float32)


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
