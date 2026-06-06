"""Clean teacher baseline sanity checks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_tta.cache import read_teacher_shard, teacher_shard_paths
from learned_tta.config import load_experiment_config


@dataclass(frozen=True, slots=True)
class CleanBaselineReport:
    """Metrics and thresholds for a clean identity-cache sanity check."""

    split: str
    identity_aug_id: str
    metrics: dict[str, float]
    thresholds: dict[str, float]
    passed: bool


@dataclass(frozen=True, slots=True)
class CleanCenterCropSummary:
    """Full-validation clean CenterCrop metrics assembled from identity shards."""

    splits: tuple[str, ...]
    identity_aug_id: str
    overall: dict[str, float]
    by_split: dict[str, dict[str, float]]
    benchmark: dict[str, Any]
    output_path: Path


DEFAULT_CLEAN_CENTER_CROP_SPLITS = ("public_train", "public_val", "private")


def check_clean_baseline_from_config(
    config_path: Path,
    *,
    split: str | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    min_top1: float | None = None,
    min_top5: float | None = None,
    max_nll: float | None = None,
) -> CleanBaselineReport:
    """Check the configured clean identity-cache baseline and write an artifact."""

    config = load_experiment_config(config_path)
    resolved_split = split or config.clean_baseline.split
    resolved_cache_dir = cache_dir or config.artifacts.teacher_cache_dir
    identity_aug_id = config.augmentations.identity_id
    resolved_output_path = output_path or clean_baseline_artifact_path(
        cache_dir=resolved_cache_dir,
        split=resolved_split,
        identity_aug_id=identity_aug_id,
    )
    return check_clean_baseline(
        cache_dir=resolved_cache_dir,
        split=resolved_split,
        identity_aug_id=identity_aug_id,
        min_top1=config.clean_baseline.min_top1 if min_top1 is None else min_top1,
        min_top5=config.clean_baseline.min_top5 if min_top5 is None else min_top5,
        max_nll=config.clean_baseline.max_nll if max_nll is None else max_nll,
        output_path=resolved_output_path,
    )


def summarize_clean_center_crop_baseline_from_config(
    config_path: Path,
    *,
    splits: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
) -> CleanCenterCropSummary:
    """Summarize clean CenterCrop metrics over full ImageNet validation splits."""

    config = load_experiment_config(config_path)
    resolved_splits = tuple(splits or DEFAULT_CLEAN_CENTER_CROP_SPLITS)
    resolved_cache_dir = cache_dir or config.artifacts.teacher_cache_dir
    resolved_output_path = (
        output_path
        if output_path is not None
        else config.artifacts.reports_dir / "tables" / "clean_center_crop_baseline.json"
    )
    return summarize_clean_center_crop_baseline(
        cache_dir=resolved_cache_dir,
        splits=resolved_splits,
        identity_aug_id=config.augmentations.identity_id,
        output_path=resolved_output_path,
    )


def summarize_clean_center_crop_baseline(
    *,
    cache_dir: Path,
    splits: Sequence[str],
    identity_aug_id: str,
    output_path: Path,
) -> CleanCenterCropSummary:
    """Write weighted clean CenterCrop metrics from identity teacher-cache shards."""

    resolved_splits = tuple(str(split) for split in splits)
    _validate_summary_splits(resolved_splits)
    by_split = {
        split: _identity_shard_metrics(
            cache_dir=cache_dir,
            split=split,
            identity_aug_id=identity_aug_id,
        )
        for split in resolved_splits
    }
    overall = _weighted_metrics(by_split)
    benchmark = _summarize_benchmarks(
        cache_dir=cache_dir,
        splits=resolved_splits,
        identity_aug_id=identity_aug_id,
        total_images=int(overall["image_count"]),
    )
    summary = CleanCenterCropSummary(
        splits=resolved_splits,
        identity_aug_id=identity_aug_id,
        overall=overall,
        by_split=by_split,
        benchmark=benchmark,
        output_path=Path(output_path),
    )
    _write_center_crop_summary(summary)
    return summary


def check_clean_baseline(
    *,
    cache_dir: Path,
    split: str,
    identity_aug_id: str,
    min_top1: float,
    min_top5: float,
    max_nll: float,
    output_path: Path,
) -> CleanBaselineReport:
    """Check identity-cache metrics against loose sanity thresholds."""

    paths = teacher_shard_paths(cache_dir, split=split, aug_id=identity_aug_id)
    shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
    metadata = shard.metadata
    if metadata.empty:
        raise ValueError("clean baseline identity shard contains no rows")
    if set(metadata["aug_id"]) != {identity_aug_id}:
        raise ValueError("clean baseline shard aug_id does not match identity_aug_id")

    metrics = {
        "image_count": float(len(metadata)),
        "top1": float(metadata["is_top1"].mean()),
        "top5": float(metadata["is_top5"].mean()),
        "nll": float(metadata["nll_true"].mean()),
    }
    thresholds = {
        "min_top1": float(min_top1),
        "min_top5": float(min_top5),
        "max_nll": float(max_nll),
    }
    _validate_thresholds(metrics=metrics, thresholds=thresholds)
    report = CleanBaselineReport(
        split=split,
        identity_aug_id=identity_aug_id,
        metrics=metrics,
        thresholds=thresholds,
        passed=True,
    )
    _write_report(report, output_path)
    return report


def clean_baseline_artifact_path(
    *,
    cache_dir: Path,
    split: str,
    identity_aug_id: str,
) -> Path:
    """Return the default clean-baseline artifact path for one identity shard."""

    return Path(cache_dir) / f"{split}__{identity_aug_id}.clean_baseline.json"


def _identity_shard_metrics(
    *,
    cache_dir: Path,
    split: str,
    identity_aug_id: str,
) -> dict[str, float]:
    paths = teacher_shard_paths(cache_dir, split=split, aug_id=identity_aug_id)
    shard = read_teacher_shard(paths.metadata_path, paths.logits_path)
    metadata = shard.metadata
    if metadata.empty:
        raise ValueError(f"clean CenterCrop identity shard for {split} contains no rows")
    if set(metadata["split"]) != {split}:
        raise ValueError(f"clean CenterCrop shard split does not match {split}")
    if set(metadata["aug_id"]) != {identity_aug_id}:
        raise ValueError("clean CenterCrop shard aug_id does not match identity_aug_id")
    return {
        "image_count": float(len(metadata)),
        "top1": float(metadata["is_top1"].mean()),
        "top5": float(metadata["is_top5"].mean()),
        "nll": float(metadata["nll_true"].mean()),
    }


def _validate_summary_splits(splits: tuple[str, ...]) -> None:
    if not splits:
        raise ValueError("clean CenterCrop summary requires at least one split")
    if len(set(splits)) != len(splits):
        raise ValueError("clean CenterCrop summary splits must be unique")


def _weighted_metrics(by_split: dict[str, dict[str, float]]) -> dict[str, float]:
    total_images = sum(metrics["image_count"] for metrics in by_split.values())
    if total_images <= 0.0:
        raise ValueError("clean CenterCrop summary contains no images")
    return {
        "image_count": float(total_images),
        "top1": _weighted_average(by_split, "top1", total_images),
        "top5": _weighted_average(by_split, "top5", total_images),
        "nll": _weighted_average(by_split, "nll", total_images),
    }


def _weighted_average(
    by_split: dict[str, dict[str, float]],
    metric_name: str,
    total_images: float,
) -> float:
    return float(
        sum(
            metrics[metric_name] * metrics["image_count"]
            for metrics in by_split.values()
        )
        / total_images
    )


def _summarize_benchmarks(
    *,
    cache_dir: Path,
    splits: tuple[str, ...],
    identity_aug_id: str,
    total_images: int,
) -> dict[str, Any]:
    benchmarks: list[dict[str, Any]] = []
    missing: list[str] = []
    for split in splits:
        benchmark_path = teacher_shard_paths(cache_dir, split, identity_aug_id).benchmark_path
        if not benchmark_path.exists():
            missing.append(split)
            continue
        with benchmark_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"benchmark sidecar must be a JSON object: {benchmark_path}")
        benchmarks.append(payload)

    elapsed_seconds = float(
        sum(float(benchmark.get("elapsed_seconds", 0.0)) for benchmark in benchmarks)
    )
    backend = _single_or_sorted(benchmark.get("backend") for benchmark in benchmarks)
    device = _single_or_sorted(benchmark.get("device") for benchmark in benchmarks)
    return {
        "available_shards": len(benchmarks),
        "missing_shards": missing,
        "backend": backend,
        "device": device,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            float(total_images) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "forwards_per_image": 1.0,
        "candidate_count": 1,
    }


def _single_or_sorted(values: Iterable[Any]) -> Any:
    unique = sorted({str(value) for value in values if value is not None})
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def _validate_thresholds(
    *,
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> None:
    if metrics["top1"] < thresholds["min_top1"]:
        raise ValueError(
            "clean baseline top1 "
            f"{metrics['top1']:.4f} is below minimum {thresholds['min_top1']:.4f}"
        )
    if metrics["top5"] < thresholds["min_top5"]:
        raise ValueError(
            "clean baseline top5 "
            f"{metrics['top5']:.4f} is below minimum {thresholds['min_top5']:.4f}"
        )
    if metrics["nll"] > thresholds["max_nll"]:
        raise ValueError(
            f"clean baseline nll {metrics['nll']:.4f} exceeds maximum "
            f"{thresholds['max_nll']:.4f}"
        )


def _write_report(report: CleanBaselineReport, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "split": report.split,
        "identity_aug_id": report.identity_aug_id,
        "metrics": report.metrics,
        "thresholds": report.thresholds,
        "passed": report.passed,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_center_crop_summary(summary: CleanCenterCropSummary) -> None:
    output_path = Path(summary.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "splits": list(summary.splits),
        "identity_aug_id": summary.identity_aug_id,
        "overall": summary.overall,
        "by_split": summary.by_split,
        "benchmark": summary.benchmark,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
