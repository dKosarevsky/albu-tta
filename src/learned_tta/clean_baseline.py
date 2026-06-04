"""Clean teacher baseline sanity checks."""

from __future__ import annotations

import json
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
