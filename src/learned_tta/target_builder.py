"""Build selector target artifacts from teacher cache shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config
from learned_tta.split_policy import validate_selector_target_splits
from learned_tta.targets import (
    compute_selector_target_matrices,
    compute_target_stats,
    save_selector_targets,
    select_selector_target_matrix,
    standardize_gain_targets,
)


@dataclass(frozen=True, slots=True)
class SelectorTargetBuildSummary:
    """Summary of generated selector target artifacts."""

    train_path: Path
    val_path: Path
    aug_ids: list[str]
    train_rows: int
    val_rows: int
    target_kind: str


def build_selector_targets_from_cache(
    cache_dir: Path,
    output_dir: Path,
    train_split: str,
    val_split: str,
    aug_ids: list[str],
    identity_aug_id: str,
    target_kind: str = "gain",
) -> SelectorTargetBuildSummary:
    """Build public-train and public-val selector target artifacts from cached logits."""

    validate_selector_target_splits(train_split=train_split, val_split=val_split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_logits, train_class_idxs, train_image_ids = _read_split_logits(
        cache_dir,
        train_split,
        aug_ids,
    )
    val_logits, val_class_idxs, val_image_ids = _read_split_logits(cache_dir, val_split, aug_ids)

    train_matrices = compute_selector_target_matrices(
        logits_by_aug=train_logits,
        class_idxs=train_class_idxs,
        identity_aug_id=identity_aug_id,
    )
    val_matrices = compute_selector_target_matrices(
        logits_by_aug=val_logits,
        class_idxs=val_class_idxs,
        identity_aug_id=identity_aug_id,
    )
    train_target = select_selector_target_matrix(train_matrices, target_kind)
    val_target = select_selector_target_matrix(val_matrices, target_kind)
    stats = compute_target_stats(train_target)
    train_z = standardize_gain_targets(train_target, stats)
    val_z = standardize_gain_targets(val_target, stats)

    train_path = output_dir / f"{train_split}_targets.npz"
    val_path = output_dir / f"{val_split}_targets.npz"
    save_selector_targets(
        path=train_path,
        aug_ids=train_matrices.aug_ids,
        image_ids=train_image_ids,
        gain=train_matrices.gain,
        target_z=train_z,
        stats=stats,
        target_kind=target_kind,
        higher_is_better=True,
    )
    save_selector_targets(
        path=val_path,
        aug_ids=val_matrices.aug_ids,
        image_ids=val_image_ids,
        gain=val_matrices.gain,
        target_z=val_z,
        stats=stats,
        target_kind=target_kind,
        higher_is_better=True,
    )
    return SelectorTargetBuildSummary(
        train_path=train_path,
        val_path=val_path,
        aug_ids=train_matrices.aug_ids,
        train_rows=train_matrices.gain.shape[0],
        val_rows=val_matrices.gain.shape[0],
        target_kind=target_kind,
    )


def build_selector_targets_from_config(
    config_path: Path,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
    train_split: str = "public_train",
    val_split: str = "public_val",
    candidate_ids: list[str] | None = None,
    target_kind: str = "gain",
) -> SelectorTargetBuildSummary:
    """Load experiment config and build selector targets from cached teacher shards."""

    config = load_experiment_config(config_path)
    resolved_cache_dir = cache_dir or config.artifacts.teacher_cache_dir
    resolved_output_dir = output_dir or config.artifacts.selector_dir
    if candidate_ids is None:
        candidate_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    return build_selector_targets_from_cache(
        cache_dir=resolved_cache_dir,
        output_dir=resolved_output_dir,
        train_split=train_split,
        val_split=val_split,
        aug_ids=candidate_ids,
        identity_aug_id=config.augmentations.identity_id,
        target_kind=target_kind,
    )


def _read_split_logits(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
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

    if reference_class_idxs is None or reference_image_ids is None:
        raise ValueError("aug_ids must not be empty")
    return logits_by_aug, reference_class_idxs, reference_image_ids
