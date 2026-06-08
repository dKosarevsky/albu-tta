"""Lightweight diagnostics from completed teacher-cache metadata shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from learned_tta.cache import teacher_shard_paths


@dataclass(frozen=True, slots=True)
class TeacherCacheDiagnostics:
    """Summary metrics computed from teacher-cache parquet metadata."""

    split: str
    aug_ids: list[str]
    image_count: int
    candidate_count: int
    clean_nll: float
    clean_top1: float
    clean_top5: float
    mean_aug_nll: float
    helpful_fraction: float
    harmful_fraction: float
    best_single_aug_id: str
    best_single_aug_mean_gain: float
    oracle_best_mean_gain: float
    top_augmentations: list[dict[str, float | str]]


def summarize_teacher_cache_diagnostics(
    cache_dir: Path,
    split: str,
    aug_ids: list[str],
    identity_aug_id: str,
    top_n: int = 10,
) -> TeacherCacheDiagnostics:
    """Summarize completed teacher-cache metadata for one split."""

    if identity_aug_id not in aug_ids:
        raise ValueError(f"identity augmentation {identity_aug_id!r} is missing from aug_ids")
    if not aug_ids:
        raise ValueError("aug_ids must not be empty")

    metadata_by_aug = {
        aug_id: _read_metadata(cache_dir=cache_dir, split=split, aug_id=aug_id)
        for aug_id in aug_ids
    }
    image_ids = _image_ids(metadata_by_aug[aug_ids[0]])
    nll_columns = []
    for aug_id in aug_ids:
        metadata = metadata_by_aug[aug_id]
        if _image_ids(metadata) != image_ids:
            raise ValueError(f"image_id order mismatch for split {split} and aug {aug_id}")
        nll_columns.append(metadata["nll_true"].to_numpy(dtype=np.float32))

    nll = np.stack(nll_columns, axis=1).astype(np.float32)
    clean_idx = aug_ids.index(identity_aug_id)
    clean_nll = nll[:, clean_idx]
    gain = (clean_nll[:, None] - nll).astype(np.float32)
    mean_gain = gain.mean(axis=0)
    augmentation_indices = [
        index for index, aug_id in enumerate(aug_ids) if aug_id != identity_aug_id
    ]
    if not augmentation_indices:
        augmentation_indices = list(range(len(aug_ids)))
    ranked_indices = np.asarray(augmentation_indices, dtype=np.int64)[
        np.argsort(-mean_gain[augmentation_indices], kind="stable")
    ]
    best_idx = int(ranked_indices[0])
    top_indices = ranked_indices[:top_n]
    identity_metadata = metadata_by_aug[identity_aug_id]

    return TeacherCacheDiagnostics(
        split=split,
        aug_ids=list(aug_ids),
        image_count=len(image_ids),
        candidate_count=len(aug_ids),
        clean_nll=float(clean_nll.mean()),
        clean_top1=float(identity_metadata["is_top1"].mean()),
        clean_top5=float(identity_metadata["is_top5"].mean()),
        mean_aug_nll=float(nll.mean()),
        helpful_fraction=float((gain > 0.0).mean()),
        harmful_fraction=float((gain < 0.0).mean()),
        best_single_aug_id=aug_ids[best_idx],
        best_single_aug_mean_gain=float(mean_gain[best_idx]),
        oracle_best_mean_gain=float(gain.max(axis=1).mean()),
        top_augmentations=[
            {
                "aug_id": aug_ids[int(index)],
                "mean_gain": float(mean_gain[int(index)]),
                "mean_nll": float(nll[:, int(index)].mean()),
            }
            for index in top_indices
        ],
    )


def teacher_cache_diagnostics_to_dict(summary: TeacherCacheDiagnostics) -> dict[str, Any]:
    """Return a JSON-serializable teacher-cache diagnostics payload."""

    return {
        "split": summary.split,
        "aug_ids": list(summary.aug_ids),
        "image_count": summary.image_count,
        "candidate_count": summary.candidate_count,
        "clean_nll": summary.clean_nll,
        "clean_top1": summary.clean_top1,
        "clean_top5": summary.clean_top5,
        "mean_aug_nll": summary.mean_aug_nll,
        "helpful_fraction": summary.helpful_fraction,
        "harmful_fraction": summary.harmful_fraction,
        "best_single_aug_id": summary.best_single_aug_id,
        "best_single_aug_mean_gain": summary.best_single_aug_mean_gain,
        "oracle_best_mean_gain": summary.oracle_best_mean_gain,
        "top_augmentations": list(summary.top_augmentations),
    }


def _read_metadata(cache_dir: Path, split: str, aug_id: str) -> pd.DataFrame:
    path = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id).metadata_path
    if not path.exists():
        raise ValueError(f"missing teacher cache metadata: {path}")
    metadata = pd.read_parquet(path)
    required_columns = {"image_id", "nll_true", "is_top1", "is_top5"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"teacher cache metadata {path} is missing columns: {missing}")
    return metadata


def _image_ids(metadata: pd.DataFrame) -> list[str]:
    return [str(image_id) for image_id in metadata["image_id"].tolist()]
