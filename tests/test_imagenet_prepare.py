from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from learned_tta.imagenet_prepare import _read_labels, prepare_imagenet_val


def test_prepare_imagenet_val_from_tar_and_devkit(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    devkit_tar = tmp_path / "ILSVRC2012_devkit_t12.tar.gz"
    output_dir = tmp_path / "val"
    audit_path = tmp_path / "prepare_audit.json"
    _write_val_tar(
        val_tar,
        [
            "ILSVRC2012_val_00000001.JPEG",
            "ILSVRC2012_val_00000002.JPEG",
            "ILSVRC2012_val_00000003.JPEG",
            "ILSVRC2012_val_00000004.JPEG",
        ],
    )
    _write_devkit_tar(devkit_tar, labels=[1, 2, 1, 2])

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=output_dir,
        class_to_idx={"n00000001": 0, "n00000002": 1},
        devkit_path=devkit_tar,
        audit_output_path=audit_path,
    )

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert summary.image_count == 4
    assert summary.class_count == 2
    assert summary.images_per_class == {"n00000001": 2, "n00000002": 2}
    assert payload["source"]["val_tar"] == str(val_tar)
    assert payload["source"]["label_source"] == str(devkit_tar)
    assert payload["label_mapping"] == "one_based_configured_class_index"
    assert sorted(path.name for path in (output_dir / "n00000001").iterdir()) == [
        "ILSVRC2012_val_00000001.JPEG",
        "ILSVRC2012_val_00000003.JPEG",
    ]
    assert sorted(path.name for path in (output_dir / "n00000002").iterdir()) == [
        "ILSVRC2012_val_00000002.JPEG",
        "ILSVRC2012_val_00000004.JPEG",
    ]


def test_prepare_imagenet_val_accepts_direct_ground_truth_file(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    output_dir = tmp_path / "val"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=output_dir,
        class_to_idx={"n00000001": 0},
        ground_truth_path=ground_truth,
    )

    assert summary.audit_output_path == output_dir / "_preparation_audit.json"
    assert (output_dir / "n00000001" / "ILSVRC2012_val_00000001.JPEG").exists()


def test_prepare_imagenet_val_rejects_mismatched_label_count(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(
        val_tar,
        ["ILSVRC2012_val_00000001.JPEG", "ILSVRC2012_val_00000002.JPEG"],
    )
    ground_truth.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="label count .* does not match image count"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
            ground_truth_path=ground_truth,
        )


@pytest.mark.parametrize("label", [0, 3])
def test_prepare_imagenet_val_rejects_labels_outside_class_index(
    tmp_path: Path,
    label: int,
) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text(f"{label}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside configured class index"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0, "n00000002": 1},
            ground_truth_path=ground_truth,
        )


def test_prepare_imagenet_val_rejects_non_empty_output_without_overwrite(
    tmp_path: Path,
) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    output_dir = tmp_path / "val"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("stale", encoding="utf-8")
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory is not empty"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=output_dir,
            class_to_idx={"n00000001": 0},
            ground_truth_path=ground_truth,
        )

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=output_dir,
        class_to_idx={"n00000001": 0},
        ground_truth_path=ground_truth,
        overwrite=True,
    )

    assert summary.image_count == 1
    assert not (output_dir / "old.txt").exists()


def test_prepare_imagenet_val_overwrite_removes_stale_directories(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    output_dir = tmp_path / "val"
    (output_dir / "stale_dir").mkdir(parents=True)
    (output_dir / "stale_dir" / "old.txt").write_text("stale", encoding="utf-8")
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=output_dir,
        class_to_idx={"n00000001": 0},
        ground_truth_path=ground_truth,
        overwrite=True,
    )

    assert summary.image_count == 1
    assert not (output_dir / "stale_dir").exists()


def test_prepare_imagenet_val_requires_one_label_source(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])

    with pytest.raises(ValueError, match="requires either devkit_path or ground_truth_path"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
        )


def test_prepare_imagenet_val_accepts_extracted_devkit_directory(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    devkit_dir = tmp_path / "ILSVRC2012_devkit_t12"
    ground_truth = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.parent.mkdir(parents=True)
    ground_truth.write_text("1\n", encoding="utf-8")

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=tmp_path / "val",
        class_to_idx={"n00000001": 0},
        devkit_path=devkit_dir,
    )

    assert summary.image_count == 1


def test_prepare_imagenet_val_accepts_ground_truth_as_devkit_path(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")

    summary = prepare_imagenet_val(
        val_tar_path=val_tar,
        output_dir=tmp_path / "val",
        class_to_idx={"n00000001": 0},
        devkit_path=ground_truth,
    )

    assert summary.image_count == 1


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        ("missing_val_tar", "validation tar does not exist"),
        ("not_tar", "validation input is not a tar archive"),
        ("empty_class_index", "class_to_idx must not be empty"),
        ("missing_devkit", "devkit path does not exist"),
        ("missing_ground_truth", "ground truth path does not exist"),
    ],
)
def test_prepare_imagenet_val_rejects_invalid_inputs(
    tmp_path: Path,
    setup: str,
    match: str,
) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    class_to_idx = {"n00000001": 0}
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")
    devkit_path: Path | None = None
    ground_truth_path: Path | None = ground_truth
    if setup == "missing_val_tar":
        val_tar = tmp_path / "missing.tar"
    elif setup == "not_tar":
        val_tar.write_text("not a tar", encoding="utf-8")
    elif setup == "empty_class_index":
        class_to_idx = {}
    elif setup == "missing_devkit":
        devkit_path = tmp_path / "missing_devkit.tar.gz"
        ground_truth_path = None
    elif setup == "missing_ground_truth":
        ground_truth_path = tmp_path / "missing_ground_truth.txt"

    with pytest.raises(ValueError, match=match):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx=class_to_idx,
            devkit_path=devkit_path,
            ground_truth_path=ground_truth_path,
        )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("not-an-int\n", "invalid label"),
        ("\n", "contains no labels"),
    ],
)
def test_prepare_imagenet_val_rejects_invalid_ground_truth_text(
    tmp_path: Path,
    text: str,
    match: str,
) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
            ground_truth_path=ground_truth,
        )


def test_prepare_imagenet_val_rejects_non_sequential_class_index(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    ground_truth.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sequential from 0"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 1},
            ground_truth_path=ground_truth,
        )


def test_prepare_imagenet_val_rejects_tar_with_no_jpegs(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(val_tar, ["README.txt"])
    ground_truth.write_text("1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no JPEG files"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
            ground_truth_path=ground_truth,
        )


def test_prepare_imagenet_val_rejects_duplicate_image_basenames(tmp_path: Path) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    ground_truth = tmp_path / "ILSVRC2012_validation_ground_truth.txt"
    _write_val_tar(
        val_tar,
        [
            "a/ILSVRC2012_val_00000001.JPEG",
            "b/ILSVRC2012_val_00000001.JPEG",
        ],
    )
    ground_truth.write_text("1\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate image filenames"):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
            ground_truth_path=ground_truth,
        )


@pytest.mark.parametrize(
    ("devkit_kind", "match"),
    [
        ("dir_missing_ground_truth", "does not contain"),
        ("not_tar", "devkit input is not a tar archive"),
        ("tar_missing_ground_truth", "archive does not contain"),
    ],
)
def test_prepare_imagenet_val_rejects_invalid_devkit_inputs(
    tmp_path: Path,
    devkit_kind: str,
    match: str,
) -> None:
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    _write_val_tar(val_tar, ["ILSVRC2012_val_00000001.JPEG"])
    if devkit_kind == "dir_missing_ground_truth":
        devkit_path = tmp_path / "devkit"
        devkit_path.mkdir()
    elif devkit_kind == "not_tar":
        devkit_path = tmp_path / "devkit.txt"
        devkit_path.write_text("not a tar", encoding="utf-8")
    else:
        devkit_path = tmp_path / "devkit.tar.gz"
        with tarfile.open(devkit_path, "w:gz") as archive:
            payload = b"irrelevant"
            info = tarfile.TarInfo(name="devkit/README.txt")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))

    with pytest.raises(ValueError, match=match):
        prepare_imagenet_val(
            val_tar_path=val_tar,
            output_dir=tmp_path / "val",
            class_to_idx={"n00000001": 0},
            devkit_path=devkit_path,
        )


def test_read_labels_rejects_missing_sources() -> None:
    with pytest.raises(ValueError, match="requires either devkit_path or ground_truth_path"):
        _read_labels(devkit_path=None, ground_truth_path=None)


def _write_val_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name in names:
            payload = f"jpeg bytes for {name}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _write_devkit_tar(path: Path, labels: list[int]) -> None:
    payload = ("\n".join(str(label) for label in labels) + "\n").encode()
    info = tarfile.TarInfo(
        name="ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    )
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))
