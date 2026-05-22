"""Selector CNN training runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.selector_model import SelectorCNN
from learned_tta.targets import SavedSelectorTargets, load_selector_targets
from learned_tta.train_selector import (
    CheckpointState,
    evaluate_regression,
    save_checkpoint_if_best,
    train_one_epoch,
)


@dataclass(frozen=True, slots=True)
class SelectorTrainingSummary:
    """Summary of selector training."""

    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
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
    device: str | torch.device = "cpu",
) -> SelectorTrainingSummary:
    """Train selector CNN from manifest CSVs and saved target artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_targets = load_selector_targets(train_targets_path)
    output_dim = train_targets.target_z.shape[1]
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
        checkpoint_state = save_checkpoint_if_best(
            state=checkpoint_state,
            val_nll=val_metrics["loss"],
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            aug_ids=train_targets.aug_ids,
            target_stats=train_targets.stats,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_spearman": val_metrics["spearman"],
            }
        )

    return SelectorTrainingSummary(
        checkpoint_path=checkpoint_path,
        best_epoch=checkpoint_state.best_epoch or 0,
        best_val_loss=checkpoint_state.best_val_nll,
        history=history,
    )


def train_selector_from_config(
    config_path: Path,
    train_manifest_path: Path | None = None,
    val_manifest_path: Path | None = None,
    train_targets_path: Path | None = None,
    val_targets_path: Path | None = None,
    output_dir: Path | None = None,
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
    return train_selector_from_artifacts(
        train_manifest_path=train_manifest_path
        or config.artifacts.manifests_dir / "public_train.csv",
        val_manifest_path=val_manifest_path or config.artifacts.manifests_dir / "public_val.csv",
        train_targets_path=train_targets_path or selector_dir / "public_train_targets.npz",
        val_targets_path=val_targets_path or selector_dir / "public_val_targets.npz",
        output_dir=selector_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=learning_rate,
        rank_weight=rank_weight,
        device=device,
    )
