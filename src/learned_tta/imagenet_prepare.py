"""CPU-only ImageNet validation archive preparation."""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SUFFIXES = {".jpeg", ".jpg"}
GROUND_TRUTH_FILENAME = "ILSVRC2012_validation_ground_truth.txt"
META_FILENAME = "meta.mat"
LABEL_MAPPING_ONE_BASED_CLASS_INDEX = "one_based_configured_class_index"
LABEL_MAPPING_OFFICIAL_ILSVRC2012 = "official_ilsvrc2012_id_to_wnid"


@dataclass(frozen=True, slots=True)
class ImageNetValPreparationSummary:
    """Summary of a prepared ImageNet validation directory."""

    output_dir: Path
    audit_output_path: Path
    image_count: int
    class_count: int
    images_per_class: dict[str, int]


@dataclass(frozen=True, slots=True)
class ValidationLabels:
    """Validation labels plus the mapping used to resolve labels into WNIDs."""

    labels: list[int]
    label_to_class_name: dict[int, str] | None
    mapping_name: str


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
    validation_labels = _read_validation_labels(
        devkit_path=devkit_path,
        ground_truth_path=ground_truth_path,
        class_to_idx=class_to_idx,
    )
    labels = validation_labels.labels
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
            class_name = _class_name_for_label(
                label,
                class_names_by_label=class_names_by_label,
                label_to_class_name=validation_labels.label_to_class_name,
            )
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
        label_mapping=validation_labels.mapping_name,
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
    return _read_validation_labels(
        devkit_path=devkit_path,
        ground_truth_path=ground_truth_path,
        class_to_idx=None,
    ).labels


def _read_validation_labels(
    *,
    devkit_path: Path | None,
    ground_truth_path: Path | None,
    class_to_idx: dict[str, int] | None,
) -> ValidationLabels:
    if ground_truth_path is not None:
        return ValidationLabels(
            labels=_parse_ground_truth_text(Path(ground_truth_path).read_text(encoding="utf-8")),
            label_to_class_name=None,
            mapping_name=LABEL_MAPPING_ONE_BASED_CLASS_INDEX,
        )
    if devkit_path is None:
        raise ValueError("prepare-imagenet-val requires either devkit_path or ground_truth_path")

    devkit_path = Path(devkit_path)
    if devkit_path.is_dir():
        matches = sorted(devkit_path.rglob(GROUND_TRUTH_FILENAME))
        if not matches:
            raise ValueError(f"devkit directory does not contain {GROUND_TRUTH_FILENAME}")
        meta_matches = sorted(devkit_path.rglob(META_FILENAME))
        return _validation_labels_from_text_and_meta(
            text=matches[0].read_text(encoding="utf-8"),
            meta_bytes=meta_matches[0].read_bytes() if meta_matches else None,
            class_to_idx=class_to_idx,
        )
    if devkit_path.name == GROUND_TRUTH_FILENAME:
        return ValidationLabels(
            labels=_parse_ground_truth_text(devkit_path.read_text(encoding="utf-8")),
            label_to_class_name=None,
            mapping_name=LABEL_MAPPING_ONE_BASED_CLASS_INDEX,
        )
    if not tarfile.is_tarfile(devkit_path):
        raise ValueError(f"devkit input is not a tar archive: {devkit_path}")

    ground_truth_text: str | None = None
    meta_bytes: bytes | None = None
    with tarfile.open(devkit_path, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_name = Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"could not read {member_name} from devkit")
            with source:
                payload = source.read()
            if member_name == GROUND_TRUTH_FILENAME:
                ground_truth_text = payload.decode("utf-8")
            elif member_name == META_FILENAME:
                meta_bytes = payload
    if ground_truth_text is None:
        raise ValueError(f"devkit archive does not contain {GROUND_TRUTH_FILENAME}")
    return _validation_labels_from_text_and_meta(
        text=ground_truth_text,
        meta_bytes=meta_bytes,
        class_to_idx=class_to_idx,
    )


def _validation_labels_from_text_and_meta(
    *,
    text: str,
    meta_bytes: bytes | None,
    class_to_idx: dict[str, int] | None,
) -> ValidationLabels:
    labels = _parse_ground_truth_text(text)
    if meta_bytes is None or class_to_idx is None:
        return ValidationLabels(
            labels=labels,
            label_to_class_name=None,
            mapping_name=LABEL_MAPPING_ONE_BASED_CLASS_INDEX,
        )
    return ValidationLabels(
        labels=labels,
        label_to_class_name=_parse_devkit_meta_label_mapping(
            meta_bytes=meta_bytes,
            class_to_idx=class_to_idx,
        ),
        mapping_name=LABEL_MAPPING_OFFICIAL_ILSVRC2012,
    )


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


def _parse_devkit_meta_label_mapping(
    *,
    meta_bytes: bytes,
    class_to_idx: dict[str, int],
) -> dict[int, str]:
    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - scipy is a project dependency.
        raise ValueError("reading ImageNet devkit meta.mat requires scipy") from error

    meta = loadmat(BytesIO(meta_bytes), squeeze_me=True, struct_as_record=False)
    synsets = meta.get("synsets")
    if synsets is None:
        raise ValueError("ImageNet devkit meta.mat does not contain synsets")

    label_to_class_name: dict[int, str] = {}
    for row in np.atleast_1d(synsets):
        label = int(_mat_scalar(_mat_field(row, "ILSVRC2012_ID")))
        class_name = str(_mat_scalar(_mat_field(row, "WNID")))
        if class_name in class_to_idx:
            label_to_class_name[label] = class_name
    if not label_to_class_name:
        raise ValueError("ImageNet devkit meta.mat does not match configured class index")
    return label_to_class_name


def _mat_field(row: Any, field_name: str) -> Any:
    if hasattr(row, field_name):
        return getattr(row, field_name)
    try:
        return row[field_name]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ImageNet devkit meta.mat row is missing {field_name}") from error


def _mat_scalar(value: Any) -> Any:
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("ImageNet devkit meta.mat field must be scalar")
        value = value.reshape((-1,))[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _class_name_for_label(
    label: int,
    *,
    class_names_by_label: list[str],
    label_to_class_name: dict[int, str] | None,
) -> str:
    if label_to_class_name is not None:
        try:
            return label_to_class_name[label]
        except KeyError as error:
            raise ValueError(
                f"label {label} is missing from ImageNet devkit meta mapping"
            ) from error
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
    label_mapping: str,
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
        "label_mapping": label_mapping,
        "image_count": summary.image_count,
        "class_count": summary.class_count,
        "images_per_class": summary.images_per_class,
    }
    audit_output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
