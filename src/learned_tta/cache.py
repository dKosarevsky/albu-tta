"""Teacher inference cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True, slots=True)
class TeacherShardPaths:
    """On-disk paths for one teacher cache shard."""

    metadata_path: Path
    logits_path: Path


@dataclass(frozen=True, slots=True)
class LoadedTeacherShard:
    """Loaded teacher cache shard."""

    metadata: pd.DataFrame
    logits: np.ndarray


def teacher_shard_paths(output_dir: Path, split: str, aug_id: str) -> TeacherShardPaths:
    """Return canonical metadata and logits paths for one shard."""

    output_dir = Path(output_dir)
    filename_prefix = f"{split}__{aug_id}"
    return TeacherShardPaths(
        metadata_path=output_dir / f"{filename_prefix}.parquet",
        logits_path=output_dir / f"{filename_prefix}.logits.npy",
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
    return paths


def read_teacher_shard(metadata_path: Path, logits_path: Path) -> LoadedTeacherShard:
    """Load one teacher cache shard from disk."""

    return LoadedTeacherShard(
        metadata=pd.read_parquet(metadata_path),
        logits=np.load(logits_path),
    )


def shard_is_complete(
    metadata_path: Path,
    logits_path: Path,
    expected_rows: int,
    expected_classes: int,
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

    return len(metadata) == expected_rows and logits.shape == (expected_rows, expected_classes)


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
