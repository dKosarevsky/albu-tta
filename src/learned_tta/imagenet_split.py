"""Deterministic ImageNet validation split helpers."""

from __future__ import annotations

import csv
import importlib.resources
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
TIMM_IMAGENET_1K_CLASS_INDEX = "timm-imagenet-1k"


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


def discover_imagenet_val(
    val_root: Path,
    class_to_idx: dict[str, int] | None = None,
) -> list[ImageRecord]:
    """Discover an ImageNet-val directory laid out as `val/class_name/image.JPEG`."""

    val_root = Path(val_root)
    class_dirs = sorted(path for path in val_root.iterdir() if path.is_dir())
    records: list[ImageRecord] = []
    if class_to_idx is not None:
        _validate_class_dirs_match_index(class_dirs, class_to_idx)
        class_dirs = [
            path
            for path, _class_idx in sorted(
                ((path, class_to_idx[path.name]) for path in class_dirs),
                key=lambda item: item[1],
            )
        ]

    for sorted_class_idx, class_dir in enumerate(class_dirs):
        class_idx = (
            class_to_idx[class_dir.name] if class_to_idx is not None else sorted_class_idx
        )
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


def load_class_to_idx(class_index: str, project_root: Path) -> dict[str, int]:
    """Load a configured ImageNet class index mapping."""

    if class_index == TIMM_IMAGENET_1K_CLASS_INDEX:
        return _load_timm_imagenet_1k_class_to_idx()

    path = Path(class_index)
    if not path.is_absolute():
        path = Path(project_root) / path
    if path.suffix.lower() == ".json":
        return _load_class_to_idx_json(path)
    return _load_class_to_idx_lines(path)


def write_class_mapping(class_to_idx: dict[str, int], output_path: Path) -> Path:
    """Write a stable class-to-index mapping artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {
        class_name: class_idx
        for class_name, class_idx in sorted(class_to_idx.items(), key=lambda item: item[1])
    }
    output_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    return output_path


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


def _load_timm_imagenet_1k_class_to_idx() -> dict[str, int]:
    with importlib.resources.files("timm.data").joinpath(
        "_info/imagenet_synsets.txt"
    ).open(encoding="utf-8") as handle:
        return _class_to_idx_from_lines(handle)


def _load_class_to_idx_lines(path: Path) -> dict[str, int]:
    with Path(path).open(encoding="utf-8") as handle:
        return _class_to_idx_from_lines(handle)


def _load_class_to_idx_json(path: Path) -> dict[str, int]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    class_to_idx = {str(class_name): int(class_idx) for class_name, class_idx in data.items()}
    _validate_class_to_idx(class_to_idx)
    return class_to_idx


def _class_to_idx_from_lines(lines: Iterable[str]) -> dict[str, int]:
    class_names = [
        line.strip()
        for line in lines
        if isinstance(line, str) and line.strip() and not line.lstrip().startswith("#")
    ]
    class_to_idx = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
    _validate_class_to_idx(class_to_idx)
    return class_to_idx


def _validate_class_to_idx(class_to_idx: dict[str, int]) -> None:
    if not class_to_idx:
        raise ValueError("class index must not be empty")
    if len(class_to_idx) != len(set(class_to_idx)):
        raise ValueError("class index names must be unique")
    indexes = sorted(class_to_idx.values())
    expected = list(range(len(class_to_idx)))
    if indexes != expected:
        raise ValueError("class index values must be sequential from 0")


def _validate_class_dirs_match_index(
    class_dirs: list[Path],
    class_to_idx: dict[str, int],
) -> None:
    discovered = {path.name for path in class_dirs}
    expected = set(class_to_idx)
    missing = sorted(expected - discovered)
    extra = sorted(discovered - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if extra:
            details.append(f"extra={extra[:5]}")
        raise ValueError(
            "ImageNet class directories must match configured class index: "
            + ", ".join(details)
        )
