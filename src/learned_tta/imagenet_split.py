"""Deterministic ImageNet validation split helpers."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One image in an ImageNet-style validation directory."""

    image_id: str
    class_idx: int
    class_name: str
    path: Path


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Per-class split sizes for ImageNet-val."""

    seed: int = 20260522
    public_per_class: int = 25
    private_per_class: int = 25
    public_train_per_class: int = 20
    public_val_per_class: int = 5

    def __post_init__(self) -> None:
        if self.public_train_per_class + self.public_val_per_class != self.public_per_class:
            raise ValueError(
                "public_train_per_class + public_val_per_class must equal public_per_class"
            )

    @property
    def required_per_class(self) -> int:
        return self.public_per_class + self.private_per_class


def discover_imagenet_val(val_root: Path) -> list[ImageRecord]:
    """Discover an ImageNet-val directory laid out as `val/class_name/image.JPEG`."""

    val_root = Path(val_root)
    class_dirs = sorted(path for path in val_root.iterdir() if path.is_dir())
    records: list[ImageRecord] = []

    for class_idx, class_dir in enumerate(class_dirs):
        image_paths = sorted(
            path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        records.extend(
            ImageRecord(
                image_id=image_path.stem,
                class_idx=class_idx,
                class_name=class_dir.name,
                path=image_path,
            )
            for image_path in image_paths
        )

    return records


def build_stratified_splits(
    records: list[ImageRecord],
    config: SplitConfig,
) -> dict[str, list[ImageRecord]]:
    """Build deterministic public/private and public train/validation splits."""

    grouped: dict[int, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_idx].append(record)

    splits: dict[str, list[ImageRecord]] = {
        "public_train": [],
        "public_val": [],
        "public": [],
        "private": [],
    }
    rng = random.Random(config.seed)

    for class_idx in sorted(grouped):
        class_records = sorted(grouped[class_idx], key=lambda record: record.image_id)
        if len(class_records) < config.required_per_class:
            raise ValueError(
                f"class {class_idx} expected at least {config.required_per_class} images, "
                f"found {len(class_records)}"
            )

        shuffled = class_records[:]
        rng.shuffle(shuffled)

        public = shuffled[: config.public_per_class]
        private = shuffled[
            config.public_per_class : config.public_per_class + config.private_per_class
        ]
        public_train = public[: config.public_train_per_class]
        public_val = public[
            config.public_train_per_class : config.public_train_per_class
            + config.public_val_per_class
        ]

        splits["public_train"].extend(public_train)
        splits["public_val"].extend(public_val)
        splits["public"].extend(public)
        splits["private"].extend(private)

    return {
        split_name: sorted(split_records, key=lambda record: (record.class_idx, record.image_id))
        for split_name, split_records in splits.items()
    }


def write_split_manifests(
    splits: dict[str, list[ImageRecord]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write split manifests as CSV files and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for split_name, records in splits.items():
        path = output_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "image_id", "class_idx", "class_name", "path"],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "image_id": record.image_id,
                        "class_idx": record.class_idx,
                        "class_name": record.class_name,
                        "path": str(record.path),
                    }
                )
        written[split_name] = path

    return written
