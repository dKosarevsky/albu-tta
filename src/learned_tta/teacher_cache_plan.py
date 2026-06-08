"""Teacher cache work planning and progress accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learned_tta.cache import shard_is_complete, teacher_shard_paths
from learned_tta.config import ExperimentConfig, load_experiment_config
from learned_tta.run_status import (
    CACHE_TEACHER_NUM_WORKERS,
    _candidate_metadata_by_id,
    _expected_augmentation_ids,
    _expected_teacher_cache_metadata,
    _expected_teacher_cache_rows,
    _teacher_cache_metadata_matches,
)

DEFAULT_TEACHER_CACHE_PLAN_SPLITS = ("public_train", "public_val", "private")
FP16_BYTES = 2


@dataclass(frozen=True, slots=True)
class TeacherCacheSplitPlan:
    """Expected and completed teacher-cache work for one split."""

    split: str
    expected_images: int
    expected_candidates: int
    expected_predictions: int
    expected_shards: int
    complete_shards: int
    incomplete_shards: int
    missing_files: int
    stale_or_malformed_shards: int
    logits_bytes_estimate: int
    completed_logits_bytes: int
    next_command: str | None

    @property
    def complete(self) -> bool:
        """Return true when every expected shard for this split is complete."""

        return self.incomplete_shards == 0


@dataclass(frozen=True, slots=True)
class TeacherCachePlan:
    """Expected and completed teacher-cache work across splits."""

    config_path: Path
    cache_dir: Path
    splits: tuple[TeacherCacheSplitPlan, ...]
    total_predictions: int
    expected_shards: int
    complete_shards: int
    missing_files: int
    stale_or_malformed_shards: int
    logits_bytes_estimate: int
    completed_logits_bytes: int

    @property
    def complete(self) -> bool:
        """Return true when all expected teacher-cache splits are complete."""

        return all(split.complete for split in self.splits)

    @property
    def splits_by_name(self) -> dict[str, TeacherCacheSplitPlan]:
        """Return split plans keyed by split name."""

        return {split.split: split for split in self.splits}


def build_teacher_cache_plan(
    config_path: Path,
    *,
    cache_dir: Path | None = None,
    splits: tuple[str, ...] = DEFAULT_TEACHER_CACHE_PLAN_SPLITS,
) -> TeacherCachePlan:
    """Build a read-only teacher-cache plan for the configured experiment."""

    config = load_experiment_config(config_path)
    resolved_cache_dir = (
        Path(cache_dir) if cache_dir is not None else config.artifacts.teacher_cache_dir
    )
    expected_aug_ids = _expected_augmentation_ids(config)
    candidates_by_id = _candidate_metadata_by_id(config)
    split_plans = tuple(
        _build_split_plan(
            config,
            cache_dir=resolved_cache_dir,
            split=split,
            expected_aug_ids=expected_aug_ids,
            candidates_by_id=candidates_by_id,
        )
        for split in splits
    )
    return TeacherCachePlan(
        config_path=config.path,
        cache_dir=resolved_cache_dir,
        splits=split_plans,
        total_predictions=sum(split.expected_predictions for split in split_plans),
        expected_shards=sum(split.expected_shards for split in split_plans),
        complete_shards=sum(split.complete_shards for split in split_plans),
        missing_files=sum(split.missing_files for split in split_plans),
        stale_or_malformed_shards=sum(split.stale_or_malformed_shards for split in split_plans),
        logits_bytes_estimate=sum(split.logits_bytes_estimate for split in split_plans),
        completed_logits_bytes=sum(split.completed_logits_bytes for split in split_plans),
    )


def teacher_cache_plan_to_dict(plan: TeacherCachePlan) -> dict[str, Any]:
    """Return a JSON-serializable teacher-cache plan payload."""

    return {
        "config_path": str(plan.config_path),
        "cache_dir": str(plan.cache_dir),
        "complete": plan.complete,
        "total_predictions": plan.total_predictions,
        "expected_shards": plan.expected_shards,
        "complete_shards": plan.complete_shards,
        "missing_files": plan.missing_files,
        "stale_or_malformed_shards": plan.stale_or_malformed_shards,
        "logits_bytes_estimate": plan.logits_bytes_estimate,
        "completed_logits_bytes": plan.completed_logits_bytes,
        "splits": [_split_plan_to_dict(split) for split in plan.splits],
    }


def _build_split_plan(
    config: ExperimentConfig,
    *,
    cache_dir: Path,
    split: str,
    expected_aug_ids: tuple[str, ...],
    candidates_by_id: dict[str, Any],
) -> TeacherCacheSplitPlan:
    expected_images = _expected_teacher_cache_rows(config, split)
    expected_candidates = len(expected_aug_ids)
    expected_shards = expected_candidates
    missing_files = 0
    complete_shards = 0
    stale_or_malformed_shards = 0

    for aug_id in expected_aug_ids:
        paths = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id)
        expected_paths = (paths.metadata_path, paths.logits_path, paths.run_metadata_path)
        missing_for_shard = sum(not path.exists() for path in expected_paths)
        missing_files += missing_for_shard
        if missing_for_shard:
            continue
        expected_metadata = _expected_teacher_cache_metadata(
            config,
            split=split,
            aug_id=aug_id,
            candidate=candidates_by_id.get(aug_id),
        )
        if not _teacher_cache_metadata_matches(paths.run_metadata_path, expected_metadata):
            stale_or_malformed_shards += 1
            continue
        if not shard_is_complete(
            metadata_path=paths.metadata_path,
            logits_path=paths.logits_path,
            expected_rows=expected_images,
            expected_classes=config.dataset.class_count,
            run_metadata_path=paths.run_metadata_path,
        ):
            stale_or_malformed_shards += 1
            continue
        complete_shards += 1

    incomplete_shards = expected_shards - complete_shards
    expected_predictions = expected_images * expected_candidates
    logits_bytes_per_shard = expected_images * config.dataset.class_count * FP16_BYTES
    next_command = None
    if incomplete_shards:
        next_command = (
            f"uv run python -m learned_tta.cli cache-teacher --split {split} "
            f"--config {config.path} --device cuda "
            f"--num-workers {CACHE_TEACHER_NUM_WORKERS}"
        )
    return TeacherCacheSplitPlan(
        split=split,
        expected_images=expected_images,
        expected_candidates=expected_candidates,
        expected_predictions=expected_predictions,
        expected_shards=expected_shards,
        complete_shards=complete_shards,
        incomplete_shards=incomplete_shards,
        missing_files=missing_files,
        stale_or_malformed_shards=stale_or_malformed_shards,
        logits_bytes_estimate=logits_bytes_per_shard * expected_shards,
        completed_logits_bytes=_completed_logits_bytes(
            cache_dir=cache_dir,
            split=split,
            expected_aug_ids=expected_aug_ids,
            expected_images=expected_images,
            expected_classes=config.dataset.class_count,
        ),
        next_command=next_command,
    )


def _completed_logits_bytes(
    *,
    cache_dir: Path,
    split: str,
    expected_aug_ids: tuple[str, ...],
    expected_images: int,
    expected_classes: int,
) -> int:
    completed_bytes = 0
    expected_shape = (expected_images, expected_classes)
    for aug_id in expected_aug_ids:
        path = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id).logits_path
        if not path.exists():
            continue
        try:
            logits = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            continue
        if logits.shape == expected_shape:
            completed_bytes += int(np.prod(logits.shape)) * int(logits.dtype.itemsize)
    return completed_bytes


def _split_plan_to_dict(split: TeacherCacheSplitPlan) -> dict[str, Any]:
    return {
        "split": split.split,
        "complete": split.complete,
        "expected_images": split.expected_images,
        "expected_candidates": split.expected_candidates,
        "expected_predictions": split.expected_predictions,
        "expected_shards": split.expected_shards,
        "complete_shards": split.complete_shards,
        "incomplete_shards": split.incomplete_shards,
        "missing_files": split.missing_files,
        "stale_or_malformed_shards": split.stale_or_malformed_shards,
        "logits_bytes_estimate": split.logits_bytes_estimate,
        "completed_logits_bytes": split.completed_logits_bytes,
        "next_command": split.next_command,
    }
