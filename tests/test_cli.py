from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_tta.cli import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment" / "resnet50_a1_in1k.yaml"


def test_cli_validate_augmentations_reports_candidate_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["validate-augmentations", "--config", str(CONFIG_PATH)])

    captured = capsys.readouterr()

    assert "validated 100 augmentation candidates" in captured.out


def test_cli_validate_augmentations_writes_audit_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audit_path = tmp_path / "augmentation_registry_audit.json"

    main(
        [
            "validate-augmentations",
            "--config",
            str(CONFIG_PATH),
            "--audit-output",
            str(audit_path),
        ]
    )
    captured = capsys.readouterr()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert f"wrote audit {audit_path}" in captured.out
    assert audit["seed"] == 20260522
    assert audit["candidate_count"] == 100
    assert audit["identity_id"] == "aug_000"
    assert audit["candidates"][0]["class_name"] is None
    assert audit["candidates"][0]["serialized_transform"] is None
    assert audit["candidates"][1]["serialized_transform"]["transform"]["seed"] == 20260522


def test_cli_make_splits_writes_manifests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    val_root = _make_fake_imagenet_val(tmp_path)
    output_dir = tmp_path / "manifests"

    main(
        [
            "make-splits",
            "--config",
            str(CONFIG_PATH),
            "--imagenet-val-dir",
            str(val_root),
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert "wrote 4 split manifests" in captured.out
    assert (output_dir / "public_train.csv").exists()
    assert (output_dir / "public_val.csv").exists()
    assert (output_dir / "public.csv").exists()
    assert (output_dir / "private.csv").exists()


def test_cli_check_full_run_reports_preflight_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    val_root = _make_fake_imagenet_val(tmp_path)

    main(
        [
            "check-full-run",
            "--config",
            str(CONFIG_PATH),
            "--imagenet-val-dir",
            str(val_root),
        ]
    )
    captured = capsys.readouterr()

    assert "full run preflight ok" in captured.out
    assert "classes=2" in captured.out
    assert "images=100" in captured.out
    assert "candidates=100" in captured.out


def test_cli_full_run_status_reports_next_step(capsys: pytest.CaptureFixture[str]) -> None:
    main(["full-run-status", "--config", str(CONFIG_PATH)])

    captured = capsys.readouterr()

    assert "full run status:" in captured.out
    assert "required steps complete" in captured.out
    assert "optional:" in captured.out
    assert "next:" in captured.out
    assert "validate_augmentations" in captured.out


def test_cli_full_run_status_can_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    main(["full-run-status", "--config", str(CONFIG_PATH), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["completed_required_steps"] == 0
    assert payload["total_required_steps"] == 12
    assert payload["next_step"]["name"] == "validate_augmentations"
    assert payload["steps"][0]["required"] is True
    assert payload["steps"][0]["outputs"][0].endswith("augmentation_registry_audit.json")


def _make_fake_imagenet_val(root: Path, classes: int = 2, images_per_class: int = 50) -> Path:
    val_root = root / "val"
    for class_idx in range(classes):
        class_dir = val_root / f"n{class_idx:08d}"
        class_dir.mkdir(parents=True)
        for image_idx in range(images_per_class):
            (class_dir / f"ILSVRC2012_val_{class_idx:04d}_{image_idx:04d}.JPEG").write_bytes(b"")
    return val_root
