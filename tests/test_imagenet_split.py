from __future__ import annotations

import csv
from pathlib import Path

import pytest

from learned_tta.imagenet_split import (
    SplitConfig,
    build_stratified_splits,
    discover_imagenet_val,
    write_class_mapping,
    write_split_manifests,
)


def _make_fake_imagenet_val(root: Path, classes: int = 3, images_per_class: int = 50) -> Path:
    val_root = root / "val"
    for class_idx in range(classes):
        class_dir = val_root / f"n{class_idx:08d}"
        class_dir.mkdir(parents=True)
        for image_idx in range(images_per_class):
            (class_dir / f"ILSVRC2012_val_{class_idx:04d}_{image_idx:04d}.JPEG").write_bytes(b"")
    return val_root


def test_discover_imagenet_val_records_sorted_classes(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=2, images_per_class=3)

    records = discover_imagenet_val(val_root)

    assert len(records) == 6
    assert [record.class_name for record in records[:3]] == ["n00000000"] * 3
    assert [record.class_name for record in records[3:]] == ["n00000001"] * 3
    assert {record.class_idx for record in records} == {0, 1}
    assert records[0].image_id == "ILSVRC2012_val_0000_0000"


def test_discover_imagenet_val_uses_explicit_class_mapping(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=2, images_per_class=3)
    class_to_idx = {"n00000001": 0, "n00000000": 1}

    records = discover_imagenet_val(val_root, class_to_idx=class_to_idx)

    assert [record.class_name for record in records[:3]] == ["n00000001"] * 3
    assert [record.class_name for record in records[3:]] == ["n00000000"] * 3
    assert {record.class_name: record.class_idx for record in records} == {
        "n00000001": 0,
        "n00000000": 1,
    }


def test_discover_imagenet_val_rejects_class_dirs_that_do_not_match_mapping(
    tmp_path: Path,
) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=2, images_per_class=3)

    with pytest.raises(ValueError, match="class directories must match configured class index"):
        discover_imagenet_val(
            val_root,
            class_to_idx={"n00000000": 0, "n00000002": 1},
        )


def test_build_stratified_splits_counts_are_disjoint_and_stable(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=3, images_per_class=50)
    records = discover_imagenet_val(val_root)
    config = SplitConfig(seed=20260522)

    splits = build_stratified_splits(records, config)
    repeated = build_stratified_splits(records, config)

    assert {name: [record.image_id for record in split] for name, split in splits.items()} == {
        name: [record.image_id for record in split] for name, split in repeated.items()
    }

    assert len(splits["public_train"]) == 60
    assert len(splits["public_val"]) == 15
    assert len(splits["public"]) == 75
    assert len(splits["private"]) == 75

    for class_idx in range(3):
        assert sum(record.class_idx == class_idx for record in splits["public_train"]) == 20
        assert sum(record.class_idx == class_idx for record in splits["public_val"]) == 5
        assert sum(record.class_idx == class_idx for record in splits["public"]) == 25
        assert sum(record.class_idx == class_idx for record in splits["private"]) == 25

    public_ids = {record.image_id for record in splits["public"]}
    private_ids = {record.image_id for record in splits["private"]}
    assert public_ids.isdisjoint(private_ids)

    public_train_ids = {record.image_id for record in splits["public_train"]}
    public_val_ids = {record.image_id for record in splits["public_val"]}
    assert public_train_ids.isdisjoint(public_val_ids)
    assert public_ids == public_train_ids | public_val_ids


def test_build_stratified_splits_rejects_incomplete_classes(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=1, images_per_class=49)
    records = discover_imagenet_val(val_root)

    with pytest.raises(ValueError, match="expected at least 50 images"):
        build_stratified_splits(records, SplitConfig(seed=20260522))


def test_write_split_manifests(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=1, images_per_class=50)
    splits = build_stratified_splits(discover_imagenet_val(val_root), SplitConfig(seed=20260522))

    written = write_split_manifests(splits, tmp_path / "manifests")

    assert set(written) == {"public_train", "public_val", "public", "private"}
    with written["public_train"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 20
    assert set(rows[0]) == {"split", "image_id", "class_idx", "class_name", "path"}
    assert rows[0]["split"] == "public_train"


def test_write_class_mapping(tmp_path: Path) -> None:
    written = write_class_mapping(
        class_to_idx={"n00000001": 0, "n00000000": 1},
        output_path=tmp_path / "class_to_idx.json",
    )

    assert written.name == "class_to_idx.json"
    assert written.read_text(encoding="utf-8").splitlines() == [
        "{",
        '  "n00000001": 0,',
        '  "n00000000": 1',
        "}",
    ]
