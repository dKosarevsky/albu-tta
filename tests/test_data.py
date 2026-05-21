from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from learned_tta.augmentations import AugmentationCandidate
from learned_tta.data import (
    AugmentedImageDataset,
    collate_teacher_batch,
    load_manifest,
    make_teacher_dataloader,
)


@pytest.fixture
def rgb_image_path(tmp_path: Path) -> Path:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :4, 0] = 255
    image[:, 4:, 2] = 255
    path = tmp_path / "sample.JPEG"
    Image.fromarray(image, mode="RGB").save(path)
    return path


@pytest.fixture
def manifest_path(tmp_path: Path, rgb_image_path: Path) -> Path:
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "image_id", "class_idx", "class_name", "path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "public",
                "image_id": "sample",
                "class_idx": 7,
                "class_name": "n00000007",
                "path": str(rgb_image_path),
            }
        )
    return path


@pytest.mark.parametrize(
    "candidate",
    [
        AugmentationCandidate("aug_000", "identity", None, {}),
        AugmentationCandidate("aug_001", "horizontal_flip", "HorizontalFlip", {"p": 1.0}),
    ],
)
def test_augmented_image_dataset_applies_candidate_and_preprocesses(
    manifest_path: Path,
    candidate: AugmentationCandidate,
) -> None:
    records = load_manifest(manifest_path)

    def preprocess(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        return torch.from_numpy(array).permute(2, 0, 1)

    dataset = AugmentedImageDataset(
        records,
        candidate=candidate,
        preprocess=preprocess,
        seed=20260522,
    )
    sample = dataset[0]

    assert sample.image_id == "sample"
    assert sample.class_idx == 7
    assert sample.aug_id == candidate.id
    assert sample.tensor.shape == (3, 8, 8)
    assert sample.tensor.dtype == torch.float32


def test_collate_teacher_batch_stacks_tensors(manifest_path: Path) -> None:
    records = load_manifest(manifest_path)
    candidate = AugmentationCandidate("aug_000", "identity", None, {})
    dataset = AugmentedImageDataset(
        records,
        candidate=candidate,
        preprocess=lambda image: torch.ones((3, image.height, image.width)),
        seed=20260522,
    )

    batch = collate_teacher_batch([dataset[0]])

    assert batch.image_ids == ["sample"]
    assert batch.class_idxs.tolist() == [7]
    assert batch.aug_ids == ["aug_000"]
    assert batch.images.shape == (1, 3, 8, 8)


def test_make_teacher_dataloader_uses_teacher_batch_collate(manifest_path: Path) -> None:
    records = load_manifest(manifest_path)
    candidate = AugmentationCandidate("aug_000", "identity", None, {})

    dataloader = make_teacher_dataloader(
        records,
        candidate=candidate,
        preprocess=lambda image: torch.ones((3, image.height, image.width)),
        seed=20260522,
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(dataloader))

    assert batch.image_ids == ["sample"]
    assert batch.class_idxs.tolist() == [7]
    assert batch.aug_ids == ["aug_000"]
    assert batch.images.shape == (1, 3, 8, 8)
