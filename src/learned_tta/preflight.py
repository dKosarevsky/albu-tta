"""Preflight checks for the full ImageNet experiment run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from learned_tta.augmentations import (
    load_augmentation_registry,
    validate_augmentation_registry,
)
from learned_tta.config import load_experiment_config
from learned_tta.imagenet_split import (
    ImageRecord,
    build_stratified_splits,
    discover_imagenet_val,
    load_class_to_idx,
)


@dataclass(frozen=True, slots=True)
class FullRunPreflightSummary:
    """Summary of checks needed before launching the full ImageNet run."""

    project_name: str
    teacher_model_name: str
    imagenet_val_dir: Path
    class_count: int
    image_count: int
    images_per_class: int
    candidate_count: int
    identity_id: str
    split_counts: dict[str, int]
    artifact_dirs: dict[str, Path]


def run_full_run_preflight(
    config_path: Path,
    imagenet_val_dir: Path,
) -> FullRunPreflightSummary:
    """Validate config, augmentation registry, and ImageNet-val layout."""

    config = load_experiment_config(config_path)
    imagenet_val_dir = Path(imagenet_val_dir)
    if not imagenet_val_dir.exists() or not imagenet_val_dir.is_dir():
        raise ValueError(f"ImageNet validation directory does not exist: {imagenet_val_dir}")

    candidates = load_augmentation_registry(config.augmentations.registry_path)
    validate_augmentation_registry(
        candidates=candidates,
        expected_count=config.augmentations.candidate_count,
    )
    if config.selector.output_dim != config.augmentations.candidate_count:
        raise ValueError("selector output_dim must match augmentation candidate_count")
    if not candidates or candidates[0].id != config.augmentations.identity_id:
        raise ValueError("first augmentation candidate must match configured identity_id")
    if max(config.selector.top_k_grid) >= config.augmentations.candidate_count:
        raise ValueError("top_k_grid values must be smaller than candidate_count")

    _validate_class_dir_count(
        imagenet_val_dir=imagenet_val_dir,
        expected_class_count=config.dataset.class_count,
    )
    class_to_idx = load_class_to_idx(config.dataset.class_index, config.project_root)
    _validate_class_index_count(
        class_to_idx=class_to_idx,
        expected_class_count=config.dataset.class_count,
    )
    records = discover_imagenet_val(imagenet_val_dir, class_to_idx=class_to_idx)
    if not records:
        raise ValueError(f"ImageNet validation directory contains no images: {imagenet_val_dir}")
    _validate_images_per_class(
        records_per_class=Counter(record.class_idx for record in records),
        expected_images_per_class=config.dataset.images_per_class,
    )
    _validate_image_count(
        records=records,
        expected_class_count=config.dataset.class_count,
        expected_images_per_class=config.dataset.images_per_class,
    )
    splits = build_stratified_splits(records, config.split)

    return FullRunPreflightSummary(
        project_name=config.project_name,
        teacher_model_name=config.teacher.model_name,
        imagenet_val_dir=imagenet_val_dir,
        class_count=len({record.class_idx for record in records}),
        image_count=len(records),
        images_per_class=config.dataset.images_per_class,
        candidate_count=len(candidates),
        identity_id=config.augmentations.identity_id,
        split_counts={split: len(split_records) for split, split_records in splits.items()},
        artifact_dirs={
            "root": config.artifacts.root,
            "manifests_dir": config.artifacts.manifests_dir,
            "teacher_cache_dir": config.artifacts.teacher_cache_dir,
            "selector_dir": config.artifacts.selector_dir,
            "reports_dir": config.artifacts.reports_dir,
        },
    )


def _validate_class_dir_count(imagenet_val_dir: Path, expected_class_count: int) -> None:
    class_count = sum(1 for path in Path(imagenet_val_dir).iterdir() if path.is_dir())
    if class_count != expected_class_count:
        raise ValueError(f"expected {expected_class_count} classes, found {class_count}")


def _validate_class_index_count(
    class_to_idx: dict[str, int],
    expected_class_count: int,
) -> None:
    if len(class_to_idx) != expected_class_count:
        raise ValueError(
            f"class index expected {expected_class_count} classes, found {len(class_to_idx)}"
        )


def _validate_image_count(
    records: list[ImageRecord],
    expected_class_count: int,
    expected_images_per_class: int,
) -> None:
    expected_image_count = expected_class_count * expected_images_per_class
    if len(records) != expected_image_count:
        raise ValueError(f"expected {expected_image_count} images, found {len(records)}")


def _validate_images_per_class(
    records_per_class: Counter[int],
    expected_images_per_class: int,
) -> None:
    bad_counts = {
        class_idx: count
        for class_idx, count in records_per_class.items()
        if count != expected_images_per_class
    }
    if bad_counts:
        first_class_idx = min(bad_counts)
        raise ValueError(
            f"class {first_class_idx} expected {expected_images_per_class} images, "
            f"found {bad_counts[first_class_idx]}"
        )
