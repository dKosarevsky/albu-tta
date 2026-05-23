"""Teacher inference cache helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TeacherShard:
    """In-memory teacher outputs for one split and one augmentation."""

    split: str
    aug_id: str
    image_ids: list[str]
    class_idxs: np.ndarray
    logits: np.ndarray
    run_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TeacherShardPaths:
    """On-disk paths for one teacher cache shard."""

    metadata_path: Path
    logits_path: Path
    run_metadata_path: Path


@dataclass(frozen=True, slots=True)
class LoadedTeacherShard:
    """Loaded teacher cache shard."""

    metadata: pd.DataFrame
    logits: np.ndarray
    run_metadata: dict[str, Any]


def teacher_shard_paths(output_dir: Path, split: str, aug_id: str) -> TeacherShardPaths:
    """Return canonical metadata and logits paths for one shard."""

    output_dir = Path(output_dir)
    filename_prefix = f"{split}__{aug_id}"
    return TeacherShardPaths(
        metadata_path=output_dir / f"{filename_prefix}.parquet",
        logits_path=output_dir / f"{filename_prefix}.logits.npy",
        run_metadata_path=output_dir / f"{filename_prefix}.run.json",
    )


def write_teacher_shard(output_dir: Path, shard: TeacherShard) -> TeacherShardPaths:
    """Write logits as fp16 and metadata as parquet."""

    logits = np.asarray(shard.logits, dtype=np.float32)
    class_idxs = np.asarray(shard.class_idxs, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_images, num_classes]")
    if len(shard.image_ids) != logits.shape[0] or class_idxs.shape != (logits.shape[0],):
        raise ValueError("image_ids, class_idxs, and logits row counts must match")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = teacher_shard_paths(output_dir, shard.split, shard.aug_id)

    metadata = _build_metadata(shard, logits, class_idxs)
    metadata.to_parquet(paths.metadata_path, index=False)
    np.save(paths.logits_path, logits.astype(np.float16))
    paths.run_metadata_path.write_text(
        json.dumps(_json_ready(shard.run_metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def read_teacher_shard(metadata_path: Path, logits_path: Path) -> LoadedTeacherShard:
    """Load one teacher cache shard from disk."""

    return LoadedTeacherShard(
        metadata=pd.read_parquet(metadata_path),
        logits=np.load(logits_path),
        run_metadata=_read_run_metadata(_default_run_metadata_path(metadata_path)),
    )


def shard_is_complete(
    metadata_path: Path,
    logits_path: Path,
    expected_rows: int,
    expected_classes: int,
    run_metadata_path: Path | None = None,
    expected_run_metadata: Mapping[str, Any] | None = None,
) -> bool:
    """Return true when a shard exists and has the expected row and class counts."""

    metadata_path = Path(metadata_path)
    logits_path = Path(logits_path)
    if not metadata_path.exists() or not logits_path.exists():
        return False

    try:
        metadata = pd.read_parquet(metadata_path)
        logits = np.load(logits_path, mmap_mode="r")
    except (OSError, ValueError):
        return False

    shape_matches = len(metadata) == expected_rows and logits.shape == (
        expected_rows,
        expected_classes,
    )
    if not shape_matches:
        return False

    if expected_run_metadata is None:
        return True

    resolved_run_metadata_path = (
        Path(run_metadata_path)
        if run_metadata_path is not None
        else _default_run_metadata_path(metadata_path)
    )
    if not resolved_run_metadata_path.exists():
        return False
    try:
        actual_run_metadata = _read_run_metadata(resolved_run_metadata_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return actual_run_metadata == _json_ready(dict(expected_run_metadata))


def _build_metadata(
    shard: TeacherShard,
    logits: np.ndarray,
    class_idxs: np.ndarray,
) -> pd.DataFrame:
    probabilities = _softmax(logits)
    row_indices = np.arange(logits.shape[0])
    true_probs = probabilities[row_indices, class_idxs].astype(np.float32)
    nll_true = (-np.log(np.clip(true_probs, 1e-45, 1.0))).astype(np.float32)
    predicted = np.argmax(logits, axis=1)
    top_k = min(5, logits.shape[1])
    top5 = np.argpartition(logits, kth=logits.shape[1] - top_k, axis=1)[:, -top_k:]

    return pd.DataFrame(
        {
            "split": [shard.split] * logits.shape[0],
            "aug_id": [shard.aug_id] * logits.shape[0],
            "image_id": shard.image_ids,
            "class_idx": class_idxs,
            "prob_true": true_probs,
            "nll_true": nll_true,
            "is_top1": predicted == class_idxs,
            "is_top5": np.array(
                [class_idx in row for class_idx, row in zip(class_idxs, top5, strict=True)],
                dtype=bool,
            ),
        }
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _default_run_metadata_path(metadata_path: Path) -> Path:
    metadata_path = Path(metadata_path)
    if metadata_path.name.endswith(".parquet"):
        return metadata_path.with_name(f"{metadata_path.name.removesuffix('.parquet')}.run.json")
    return metadata_path.with_suffix(".run.json")


def _read_run_metadata(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("teacher shard run metadata must be a JSON object")
    return data


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
