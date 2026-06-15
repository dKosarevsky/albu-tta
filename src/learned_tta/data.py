"""Data loading helpers for teacher inference."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One row from a split manifest."""

    split: str
    image_id: str
    class_idx: int
    class_name: str
    path: Path


@dataclass(frozen=True, slots=True)
class TeacherSample:
    """One preprocessed teacher input sample."""

    image_id: str
    class_idx: int
    aug_id: str
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    """Collated teacher input batch."""

    image_ids: list[str]
    class_idxs: torch.Tensor
    aug_ids: list[str]
    images: torch.Tensor


def load_manifest(path: Path) -> list[ManifestRecord]:
    """Load a split manifest CSV."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return [
            ManifestRecord(
                split=row["split"],
                image_id=row["image_id"],
                class_idx=int(row["class_idx"]),
                class_name=row["class_name"],
                path=Path(row["path"]),
            )
            for row in rows
        ]


class AugmentedImageDataset(torch.utils.data.Dataset[TeacherSample]):
    """Dataset that applies one deterministic candidate before teacher preprocessing."""

    def __init__(
        self,
        records: list[ManifestRecord],
        candidate: Any,
        preprocess: Callable[[Image.Image], torch.Tensor],
        seed: int,
    ) -> None:
        self.records = records
        self.candidate = candidate
        self.preprocess = preprocess
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TeacherSample:
        from learned_tta.augmentations import apply_candidate

        record = self.records[index]
        image = load_rgb_image(record.path)
        augmented = apply_candidate(self.candidate, image, seed=self.seed)
        tensor = self.preprocess(Image.fromarray(augmented, mode="RGB"))
        return TeacherSample(
            image_id=record.image_id,
            class_idx=record.class_idx,
            aug_id=self.candidate.id,
            tensor=tensor,
        )


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 numpy array."""

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def collate_teacher_batch(samples: list[TeacherSample]) -> TeacherBatch:
    """Collate teacher samples into batched tensors and metadata lists."""

    return TeacherBatch(
        image_ids=[sample.image_id for sample in samples],
        class_idxs=torch.tensor([sample.class_idx for sample in samples], dtype=torch.long),
        aug_ids=[sample.aug_id for sample in samples],
        images=torch.stack([sample.tensor for sample in samples]),
    )


def make_teacher_dataloader(
    records: list[ManifestRecord],
    candidate: Any,
    preprocess: Callable[[Image.Image], torch.Tensor],
    seed: int,
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader[TeacherBatch]:
    """Build a DataLoader for one candidate over one manifest split."""

    dataset = AugmentedImageDataset(
        records=records,
        candidate=candidate,
        preprocess=preprocess,
        seed=seed,
    )
    return cast(
        torch.utils.data.DataLoader[TeacherBatch],
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_teacher_batch,
        ),
    )
