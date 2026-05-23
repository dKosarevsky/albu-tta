from __future__ import annotations

from pathlib import Path

import pytest

from learned_tta.preflight import run_full_run_preflight

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"


def test_run_full_run_preflight_validates_registry_config_and_imagenet_shape(
    tmp_path: Path,
) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=2, images_per_class=50)

    summary = run_full_run_preflight(
        config_path=CONFIG_PATH,
        imagenet_val_dir=val_root,
    )

    assert summary.project_name == "albu-tta"
    assert summary.teacher_model_name == "resnet50.a1_in1k"
    assert summary.class_count == 2
    assert summary.image_count == 100
    assert summary.images_per_class == 50
    assert summary.candidate_count == 100
    assert summary.identity_id == "aug_000"
    assert summary.split_counts == {
        "public_train": 40,
        "public_val": 10,
        "public": 50,
        "private": 50,
    }
    assert summary.artifact_dirs["teacher_cache_dir"].name == "teacher_cache"


def test_run_full_run_preflight_rejects_missing_imagenet_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ImageNet validation directory does not exist"):
        run_full_run_preflight(
            config_path=CONFIG_PATH,
            imagenet_val_dir=tmp_path / "missing" / "val",
        )


def test_run_full_run_preflight_rejects_wrong_images_per_class(tmp_path: Path) -> None:
    val_root = _make_fake_imagenet_val(tmp_path, classes=2, images_per_class=49)

    with pytest.raises(ValueError, match="expected 50 images"):
        run_full_run_preflight(
            config_path=CONFIG_PATH,
            imagenet_val_dir=val_root,
        )


def _make_fake_imagenet_val(
    root: Path,
    classes: int,
    images_per_class: int,
) -> Path:
    val_root = root / "val"
    for class_idx in range(classes):
        class_dir = val_root / f"n{class_idx:08d}"
        class_dir.mkdir(parents=True)
        for image_idx in range(images_per_class):
            (class_dir / f"ILSVRC2012_val_{class_idx:04d}_{image_idx:04d}.JPEG").write_bytes(
                b""
            )
    return val_root
