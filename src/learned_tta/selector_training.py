"""Selector CNN training runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.selector_features import clean_logit_features
from learned_tta.selector_model import SelectorCNN, SelectorMLP
from learned_tta.split_policy import validate_public_tuning_split
from learned_tta.targets import SavedSelectorTargets, TargetStats, load_selector_targets
from learned_tta.train_selector import (
    CheckpointState,
    evaluate_regression,
    save_checkpoint_if_best,
    train_one_epoch,
)
from learned_tta.tta_eval import (
    evaluate_learned_topk_uniform,
    learned_topk_selection,
    oracle_selection_recall,
    oracle_topk_selection,
    select_best_k,
)


@dataclass(frozen=True, slots=True)
class SelectorTrainingSummary:
    """Summary of selector training."""

    checkpoint_path: Path
    history_csv: Path
    best_epoch: int
    best_val_loss: float
    best_val_nll: float
    history: list[dict[str, float]]


@dataclass(frozen=True, slots=True)
class SelectorLossAblationSpec:
    """One selector loss variant for ablation training."""

    variant: str
    rank_weight: float
    usefulness_head: bool
    usefulness_tau: float = 0.01
    usefulness_weight: float = 0.0
    feature_mode: str = "image"
    target_mode: str = "nll_gain"
    model_family: str = "image_cnn"
    listwise_weight: float = 0.0
    listwise_top_k: int = 1


@dataclass(frozen=True, slots=True)
class SelectorLossAblationSummary:
    """Summary of a selector loss ablation run."""

    results_csv: Path
    rows: list[dict[str, object]]


DEFAULT_SELECTOR_LOSS_ABLATIONS = (
    SelectorLossAblationSpec(
        variant="gain_only",
        rank_weight=0.0,
        usefulness_head=False,
        usefulness_weight=0.0,
    ),
    SelectorLossAblationSpec(
        variant="gain_rank",
        rank_weight=0.2,
        usefulness_head=False,
        usefulness_weight=0.0,
    ),
    SelectorLossAblationSpec(
        variant="gain_rank_bce",
        rank_weight=0.2,
        usefulness_head=True,
        usefulness_weight=0.05,
    ),
    SelectorLossAblationSpec(
        variant="gain_listwise_topk",
        rank_weight=0.2,
        usefulness_head=False,
        listwise_weight=0.1,
        listwise_top_k=16,
    ),
    SelectorLossAblationSpec(
        variant="clean_logits_mlp_gain_rank",
        rank_weight=0.2,
        usefulness_head=False,
        feature_mode="clean_logits",
        model_family="mlp",
    ),
    SelectorLossAblationSpec(
        variant="clean_logits_mlp_gain_listwise",
        rank_weight=0.2,
        usefulness_head=False,
        feature_mode="clean_logits",
        model_family="mlp",
        listwise_weight=0.1,
        listwise_top_k=16,
    ),
)


def select_selector_loss_ablation_specs(
    variant_names: tuple[str, ...] | None,
    specs: tuple[SelectorLossAblationSpec, ...] = DEFAULT_SELECTOR_LOSS_ABLATIONS,
) -> tuple[SelectorLossAblationSpec, ...]:
    """Return requested selector ablation specs, preserving request order."""

    if not variant_names:
        return specs
    specs_by_name = {spec.variant: spec for spec in specs}
    unknown = [name for name in variant_names if name not in specs_by_name]
    if unknown:
        available = ", ".join(sorted(specs_by_name))
        requested = ", ".join(unknown)
        raise ValueError(
            f"unknown selector ablation variant(s): {requested}; available: {available}"
        )
    return tuple(specs_by_name[name] for name in variant_names)


def _read_completed_selector_training_summary(output_dir: Path) -> SelectorTrainingSummary | None:
    checkpoint_path = output_dir / "selector_best.pt"
    history_csv = output_dir / "selector_history.csv"
    if not checkpoint_path.exists() or not history_csv.exists():
        return None
    history_df = pd.read_csv(history_csv)
    if history_df.empty:
        return None
    metric_column = "val_tta_nll" if "val_tta_nll" in history_df.columns else "val_loss"
    best_index = history_df[metric_column].astype(float).idxmin()
    best_row = history_df.loc[best_index]
    best_val_loss = (
        float(best_row["val_loss"])
        if "val_loss" in history_df.columns
        else float(best_row[metric_column])
    )
    return SelectorTrainingSummary(
        checkpoint_path=checkpoint_path,
        history_csv=history_csv,
        best_epoch=int(best_row["epoch"]) if "epoch" in history_df.columns else 0,
        best_val_loss=best_val_loss,
        best_val_nll=float(best_row[metric_column]),
        history=history_df.to_dict("records"),
    )


SelectorBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class SelectorImageTargetDataset(torch.utils.data.Dataset[SelectorBatch]):
    """Clean-image selector dataset paired with precomputed target rows."""

    def __init__(
        self,
        records: list[ManifestRecord],
        targets: SavedSelectorTargets,
        image_size: int,
    ) -> None:
        if len(records) != targets.target_z.shape[0]:
            raise ValueError("manifest row count must match selector target rows")
        manifest_image_ids = [record.image_id for record in records]
        if not targets.image_ids:
            raise ValueError("selector targets must include image_ids")
        if targets.image_ids != manifest_image_ids:
            raise ValueError("selector target image_ids must match manifest image_ids")
        self.records = records
        self.targets = targets
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SelectorBatch:
        record = self.records[index]
        with Image.open(record.path) as image:
            resized = image.convert("RGB").resize((self.image_size, self.image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(array).permute(2, 0, 1)
        target_tensor = torch.from_numpy(self.targets.target_z[index].astype(np.float32))
        gain_tensor = torch.from_numpy(self.targets.gain[index].astype(np.float32))
        return image_tensor, target_tensor, gain_tensor


class SelectorFeatureTargetDataset(torch.utils.data.Dataset[SelectorBatch]):
    """Vector-feature selector dataset paired with precomputed target rows."""

    def __init__(
        self,
        features: np.ndarray,
        targets: SavedSelectorTargets,
    ) -> None:
        if features.shape[0] != targets.target_z.shape[0]:
            raise ValueError("feature row count must match selector target rows")
        self.features = features.astype(np.float32)
        self.targets = targets

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> SelectorBatch:
        feature_tensor = torch.from_numpy(self.features[index].astype(np.float32))
        target_tensor = torch.from_numpy(self.targets.target_z[index].astype(np.float32))
        gain_tensor = torch.from_numpy(self.targets.gain[index].astype(np.float32))
        return feature_tensor, target_tensor, gain_tensor


def make_selector_dataloader(
    manifest_path: Path,
    targets_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader[SelectorBatch]:
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


def make_clean_logit_selector_dataloader(
    manifest_path: Path,
    targets_path: Path,
    cache_dir: Path,
    identity_aug_id: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> tuple[torch.utils.data.DataLoader[SelectorBatch], int]:
    """Build a selector DataLoader using clean-logit summary features."""

    records = load_manifest(manifest_path)
    targets = load_selector_targets(targets_path)
    if not records:
        raise ValueError("manifest must contain at least one row")
    manifest_image_ids = [record.image_id for record in records]
    if targets.image_ids != manifest_image_ids:
        raise ValueError("selector target image_ids must match manifest image_ids")
    split = records[0].split
    if any(record.split != split for record in records):
        raise ValueError("manifest must contain a single split")
    paths = teacher_shard_paths(cache_dir, split=split, aug_id=identity_aug_id)
    shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
    shard_image_ids = [str(image_id) for image_id in shard.metadata["image_id"].tolist()]
    if shard_image_ids != manifest_image_ids:
        raise ValueError("clean-logit shard image_ids must match manifest image_ids")
    features, _ = clean_logit_features(shard.logits)
    dataset = SelectorFeatureTargetDataset(features=features, targets=targets)
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        ),
        int(features.shape[1]),
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
    usefulness_head: bool = False,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
    feature_mode: str = "image",
    target_mode: str = "nll_gain",
    model_family: str = "image_cnn",
    val_cache_dir: Path | None = None,
    val_split: str = "public_val",
    aug_ids: list[str] | None = None,
    top_k_grid: list[int] | None = None,
    identity_aug_id: str = "aug_000",
    device: str | torch.device = "cpu",
) -> SelectorTrainingSummary:
    """Train selector CNN from manifest CSVs and saved target artifacts."""

    validate_public_tuning_split(val_split, command="train-selector")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_targets = load_selector_targets(train_targets_path)
    output_dim = train_targets.target_z.shape[1]
    if aug_ids is None:
        aug_ids = train_targets.aug_ids
    if aug_ids != train_targets.aug_ids:
        raise ValueError("aug_ids must match selector target aug_ids")
    torch_device = torch.device(device)

    identity_index = aug_ids.index(identity_aug_id)
    train_dataloader: torch.utils.data.DataLoader[SelectorBatch]
    val_dataloader: torch.utils.data.DataLoader[SelectorBatch]
    if feature_mode == "image":
        if model_family != "image_cnn":
            raise ValueError("image feature_mode requires model_family='image_cnn'")
        model: torch.nn.Module = SelectorCNN(
            output_dim=output_dim,
            usefulness_head=usefulness_head,
        ).to(torch_device)
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
    elif feature_mode == "clean_logits":
        if model_family != "mlp":
            raise ValueError("clean_logits feature_mode requires model_family='mlp'")
        if val_cache_dir is None:
            raise ValueError("cache_dir is required for clean_logits feature_mode")
        train_dataloader, input_dim = make_clean_logit_selector_dataloader(
            manifest_path=train_manifest_path,
            targets_path=train_targets_path,
            cache_dir=val_cache_dir,
            identity_aug_id=identity_aug_id,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_dataloader, _ = make_clean_logit_selector_dataloader(
            manifest_path=val_manifest_path,
            targets_path=val_targets_path,
            cache_dir=val_cache_dir,
            identity_aug_id=identity_aug_id,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        model = SelectorMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            usefulness_head=usefulness_head,
        ).to(torch_device)
    else:
        raise ValueError("feature_mode must be 'image' or 'clean_logits'")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_path = output_dir / "selector_best.pt"
    history_csv = output_dir / "selector_history.csv"
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
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            identity_index=identity_index,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
        )
        val_metrics = evaluate_regression(
            model=model,
            dataloader=val_dataloader,
            device=torch_device,
            rank_weight=rank_weight,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            identity_index=identity_index,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
        )
        history_row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_regression_loss": train_metrics["regression_loss"],
            "train_rank_loss": train_metrics["rank_loss"],
            "train_usefulness_bce": train_metrics["usefulness_bce"],
            "train_listwise_topk_loss": train_metrics["listwise_topk_loss"],
            "feature_mode": feature_mode,
            "target_mode": target_mode,
            "model_family": model_family,
            "val_loss": val_metrics["loss"],
            "val_regression_loss": val_metrics["regression_loss"],
            "val_rank_loss": val_metrics["rank_loss"],
            "val_usefulness_bce": val_metrics["usefulness_bce"],
            "val_listwise_topk_loss": val_metrics["listwise_topk_loss"],
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
            target_kind=train_targets.target_kind,
            higher_is_better=train_targets.higher_is_better,
            usefulness_head=usefulness_head,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
        )
        history.append(history_row)

    best_epoch = checkpoint_state.best_epoch or 0
    best_row = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    pd.DataFrame(history).to_csv(history_csv, index=False)
    return SelectorTrainingSummary(
        checkpoint_path=checkpoint_path,
        history_csv=history_csv,
        best_epoch=best_epoch,
        best_val_loss=float(best_row["val_loss"]) if best_row is not None else 0.0,
        best_val_nll=checkpoint_state.best_val_nll,
        history=history,
    )


def train_selector_loss_ablation_from_artifacts(
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
    val_cache_dir: Path | None = None,
    val_split: str = "public_val",
    aug_ids: list[str] | None = None,
    top_k_grid: list[int] | None = None,
    identity_aug_id: str = "aug_000",
    device: str | torch.device = "cpu",
    specs: tuple[SelectorLossAblationSpec, ...] = DEFAULT_SELECTOR_LOSS_ABLATIONS,
    variant_names: tuple[str, ...] | None = None,
    skip_completed: bool = True,
) -> SelectorLossAblationSummary:
    """Train selector loss variants and write a compact comparison table."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected_specs = select_selector_loss_ablation_specs(variant_names, specs=specs)
    for spec in selected_specs:
        variant_dir = output_dir / spec.variant
        summary = _read_completed_selector_training_summary(variant_dir) if skip_completed else None
        status = "skipped" if summary is not None else "trained"
        if summary is None:
            summary = train_selector_from_artifacts(
                train_manifest_path=train_manifest_path,
                val_manifest_path=val_manifest_path,
                train_targets_path=train_targets_path,
                val_targets_path=val_targets_path,
                output_dir=variant_dir,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                epochs=epochs,
                learning_rate=learning_rate,
                rank_weight=spec.rank_weight,
                usefulness_head=spec.usefulness_head,
                usefulness_tau=spec.usefulness_tau,
                usefulness_weight=spec.usefulness_weight,
                listwise_weight=spec.listwise_weight,
                listwise_top_k=spec.listwise_top_k,
                feature_mode=spec.feature_mode,
                target_mode=spec.target_mode,
                model_family=spec.model_family,
                val_cache_dir=val_cache_dir,
                val_split=val_split,
                aug_ids=aug_ids,
                top_k_grid=top_k_grid,
                identity_aug_id=identity_aug_id,
                device=device,
            )
        rows.append(
            {
                "variant": spec.variant,
                "status": status,
                "rank_weight": spec.rank_weight,
                "usefulness_head": spec.usefulness_head,
                "usefulness_tau": spec.usefulness_tau,
                "usefulness_weight": spec.usefulness_weight,
                "feature_mode": spec.feature_mode,
                "target_mode": spec.target_mode,
                "model_family": spec.model_family,
                "listwise_weight": spec.listwise_weight,
                "listwise_top_k": spec.listwise_top_k,
                "best_epoch": summary.best_epoch,
                "best_val_loss": summary.best_val_loss,
                "best_val_nll": summary.best_val_nll,
                "checkpoint_path": str(summary.checkpoint_path),
                "history_csv": str(summary.history_csv),
            }
        )
    results_csv = output_dir / "selector_loss_ablation.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    return SelectorLossAblationSummary(results_csv=results_csv, rows=rows)


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
    usefulness_head: bool | None = None,
    usefulness_tau: float | None = None,
    usefulness_weight: float | None = None,
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
    resolved_usefulness_head = (
        config.selector.usefulness_head if usefulness_head is None else usefulness_head
    )
    resolved_usefulness_tau = (
        config.selector.usefulness_tau if usefulness_tau is None else usefulness_tau
    )
    resolved_usefulness_weight = (
        config.selector.usefulness_weight if usefulness_weight is None else usefulness_weight
    )
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
        usefulness_head=resolved_usefulness_head,
        usefulness_tau=resolved_usefulness_tau,
        usefulness_weight=resolved_usefulness_weight,
        device=device,
    )


def train_selector_loss_ablation_from_config(
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
    epochs: int = 5,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
    variant_names: tuple[str, ...] | None = None,
    skip_completed: bool = True,
) -> SelectorLossAblationSummary:
    """Load experiment config and train the default selector loss ablation set."""

    config = load_experiment_config(config_path)
    selector_dir = config.artifacts.selector_dir
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    return train_selector_loss_ablation_from_artifacts(
        train_manifest_path=train_manifest_path
        or config.artifacts.manifests_dir / "public_train.csv",
        val_manifest_path=val_manifest_path or config.artifacts.manifests_dir / "public_val.csv",
        train_targets_path=train_targets_path or selector_dir / "public_train_targets.npz",
        val_targets_path=val_targets_path or selector_dir / "public_val_targets.npz",
        output_dir=output_dir or selector_dir / "loss_ablation",
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
        device=device,
        variant_names=variant_names,
        skip_completed=skip_completed,
    )


@torch.inference_mode()
def _evaluate_validation_tta(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[SelectorBatch],
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
    selected_aug_ids = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )
    oracle_aug_ids = oracle_topk_selection(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )
    return {
        "val_tta_best_k": float(best_k),
        "val_tta_top1": best_metrics["top1"],
        "val_tta_top5": best_metrics["top5"],
        "val_tta_nll": best_metrics["nll"],
        "val_tta_ece": best_metrics["ece"],
        "val_tta_oracle_recall": oracle_selection_recall(
            selected_aug_ids,
            oracle_aug_ids,
            identity_aug_id=identity_aug_id,
        ),
    }


@torch.inference_mode()
def _predict_gain(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader[SelectorBatch],
    device: torch.device,
    target_stats: TargetStats,
) -> np.ndarray:
    model.eval()
    predictions = []
    for images, _, _ in dataloader:
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
