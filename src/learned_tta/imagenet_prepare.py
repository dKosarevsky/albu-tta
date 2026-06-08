"""CPU-only ImageNet validation archive preparation."""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpeg", ".jpg"}
GROUND_TRUTH_FILENAME = "ILSVRC2012_validation_ground_truth.txt"
LABEL_MAPPING_NAME = "one_based_configured_class_index"


@dataclass(frozen=True, slots=True)
class ImageNetValPreparationSummary:
    """Summary of a prepared ImageNet validation directory."""

    output_dir: Path
    audit_output_path: Path
    image_count: int
    class_count: int
    images_per_class: dict[str, int]


def prepare_imagenet_val(
    *,
    val_tar_path: Path,
    output_dir: Path,
    class_to_idx: dict[str, int],
    devkit_path: Path | None = None,
    ground_truth_path: Path | None = None,
    audit_output_path: Path | None = None,
    overwrite: bool = False,
) -> ImageNetValPreparationSummary:
    """Prepare official ImageNet validation images as `val/WNID/*.JPEG`."""

    val_tar_path = Path(val_tar_path)
    output_dir = Path(output_dir)
    audit_output_path = Path(audit_output_path or output_dir / "_preparation_audit.json")
    _validate_inputs(
        val_tar_path=val_tar_path,
        class_to_idx=class_to_idx,
        devkit_path=devkit_path,
        ground_truth_path=ground_truth_path,
    )
    labels = _read_labels(devkit_path=devkit_path, ground_truth_path=ground_truth_path)
    class_names_by_label = _class_names_by_one_based_label(class_to_idx)
    image_members = _list_validation_images(val_tar_path)
    if len(labels) != len(image_members):
        raise ValueError(
            f"label count {len(labels)} does not match image count {len(image_members)}"
        )

    _prepare_output_dir(output_dir=output_dir, overwrite=overwrite)
    images_per_class = {class_name: 0 for class_name in class_to_idx}
    with tarfile.open(val_tar_path, "r:*") as archive:
        for member, label in zip(image_members, labels, strict=True):
            class_name = _class_name_for_label(label, class_names_by_label)
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            target_path = class_dir / Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read image from validation tar: {member.name}")
            with source, target_path.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            images_per_class[class_name] += 1

    images_per_class = {
        class_name: count
        for class_name, count in sorted(
            images_per_class.items(),
            key=lambda item: class_to_idx[item[0]],
        )
        if count > 0
    }
    summary = ImageNetValPreparationSummary(
        output_dir=output_dir,
        audit_output_path=audit_output_path,
        image_count=len(image_members),
        class_count=len(images_per_class),
        images_per_class=images_per_class,
    )
    _write_audit(
        summary=summary,
        val_tar_path=val_tar_path,
        label_source_path=ground_truth_path or devkit_path,
        audit_output_path=audit_output_path,
    )
    return summary


def _validate_inputs(
    *,
    val_tar_path: Path,
    class_to_idx: dict[str, int],
    devkit_path: Path | None,
    ground_truth_path: Path | None,
) -> None:
    if not val_tar_path.exists() or not val_tar_path.is_file():
        raise ValueError(f"validation tar does not exist: {val_tar_path}")
    if not tarfile.is_tarfile(val_tar_path):
        raise ValueError(f"validation input is not a tar archive: {val_tar_path}")
    if not class_to_idx:
        raise ValueError("class_to_idx must not be empty")
    if devkit_path is None and ground_truth_path is None:
        raise ValueError("prepare-imagenet-val requires either devkit_path or ground_truth_path")
    if devkit_path is not None and not Path(devkit_path).exists():
        raise ValueError(f"devkit path does not exist: {devkit_path}")
    if ground_truth_path is not None and not Path(ground_truth_path).exists():
        raise ValueError(f"ground truth path does not exist: {ground_truth_path}")


def _read_labels(
    *,
    devkit_path: Path | None,
    ground_truth_path: Path | None,
) -> list[int]:
    if ground_truth_path is not None:
        return _parse_ground_truth_text(Path(ground_truth_path).read_text(encoding="utf-8"))
    if devkit_path is None:
        raise ValueError("prepare-imagenet-val requires either devkit_path or ground_truth_path")

    devkit_path = Path(devkit_path)
    if devkit_path.is_dir():
        matches = sorted(devkit_path.rglob(GROUND_TRUTH_FILENAME))
        if not matches:
            raise ValueError(f"devkit directory does not contain {GROUND_TRUTH_FILENAME}")
        return _parse_ground_truth_text(matches[0].read_text(encoding="utf-8"))
    if devkit_path.name == GROUND_TRUTH_FILENAME:
        return _parse_ground_truth_text(devkit_path.read_text(encoding="utf-8"))
    if not tarfile.is_tarfile(devkit_path):
        raise ValueError(f"devkit input is not a tar archive: {devkit_path}")

    with tarfile.open(devkit_path, "r:*") as archive:
        for member in archive.getmembers():
            if Path(member.name).name != GROUND_TRUTH_FILENAME or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read {GROUND_TRUTH_FILENAME} from devkit")
            with source:
                return _parse_ground_truth_text(source.read().decode("utf-8"))
    raise ValueError(f"devkit archive does not contain {GROUND_TRUTH_FILENAME}")


def _parse_ground_truth_text(text: str) -> list[int]:
    labels: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            labels.append(int(stripped))
        except ValueError as error:
            raise ValueError(f"invalid label on line {line_number}: {stripped!r}") from error
    if not labels:
        raise ValueError("validation ground truth contains no labels")
    return labels


def _class_names_by_one_based_label(class_to_idx: dict[str, int]) -> list[str]:
    ordered = sorted(class_to_idx.items(), key=lambda item: item[1])
    indexes = [class_idx for _class_name, class_idx in ordered]
    expected = list(range(len(ordered)))
    if indexes != expected:
        raise ValueError("class_to_idx values must be sequential from 0")
    return [class_name for class_name, _class_idx in ordered]


def _list_validation_images(val_tar_path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(val_tar_path, "r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES
        ]
    if not members:
        raise ValueError(f"validation tar contains no JPEG files: {val_tar_path}")
    basenames = [Path(member.name).name for member in members]
    if len(basenames) != len(set(basenames)):
        raise ValueError("validation tar contains duplicate image filenames")
    return sorted(members, key=lambda member: Path(member.name).name)


def _class_name_for_label(label: int, class_names_by_label: list[str]) -> str:
    if label < 1 or label > len(class_names_by_label):
        raise ValueError(
            f"label {label} is outside configured class index with "
            f"{len(class_names_by_label)} classes"
        )
    return class_names_by_label[label - 1]


def _prepare_output_dir(*, output_dir: Path, overwrite: bool) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise ValueError(f"output directory is not empty: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_audit(
    *,
    summary: ImageNetValPreparationSummary,
    val_tar_path: Path,
    label_source_path: Path | None,
    audit_output_path: Path,
) -> None:
    audit_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": 1,
        "source": {
            "val_tar": str(val_tar_path),
            "label_source": str(label_source_path) if label_source_path is not None else None,
        },
        "output_dir": str(summary.output_dir),
        "label_mapping": LABEL_MAPPING_NAME,
        "image_count": summary.image_count,
        "class_count": summary.class_count,
        "images_per_class": summary.images_per_class,
    }
    audit_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
