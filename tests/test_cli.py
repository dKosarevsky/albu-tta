from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)
    output_dir = tmp_path / "manifests"

    main(
        [
            "make-splits",
            "--config",
            str(config_path),
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
    assert (output_dir / "class_to_idx.json").exists()


def test_cli_check_full_run_reports_preflight_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    val_root = _make_fake_imagenet_val(tmp_path)
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(
        [
            "check-full-run",
            "--config",
            str(config_path),
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
    assert "missing=1" in captured.out
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
    assert payload["steps"][0]["missing_output_count"] == 1
    assert payload["steps"][0]["extra_output_count"] == 0
    assert payload["steps"][0]["outputs"][0].endswith("augmentation_registry_audit.json")
    assert payload["steps"][0]["missing_outputs"][0].endswith(
        "augmentation_registry_audit.json"
    )


def test_cli_full_run_status_can_fail_on_incomplete_required_steps(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "full-run-status",
                "--config",
                str(CONFIG_PATH),
                "--fail-on-incomplete",
            ]
        )

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "next: validate_augmentations" in captured.out


def test_cli_full_run_status_can_print_only_next_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["full-run-status", "--config", str(CONFIG_PATH), "--next-command"])

    captured = capsys.readouterr()

    assert captured.err == ""
    assert captured.out.startswith("uv run python -m learned_tta.cli validate-augmentations")
    assert "--audit-output" in captured.out
    assert "full run status:" not in captured.out
    assert captured.out.count("\n") == 1


def test_cli_resume_full_run_reports_background_cache_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_next_full_run_step(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            status="started",
            step_name="cache_public_train",
            command="uv run python -m learned_tta.cli cache-teacher --split public_train",
            log_path=tmp_path / "logs" / "cache_public_train.log",
            pid=12345,
            active_processes=(),
        )

    monkeypatch.setattr(
        "learned_tta.cli.run_next_full_run_step",
        fake_run_next_full_run_step,
    )

    main(
        [
            "resume-full-run",
            "--config",
            str(CONFIG_PATH),
            "--imagenet-val-dir",
            "/content/imagenet_val_prepare/val",
            "--cache-log-dir",
            str(tmp_path / "logs"),
        ]
    )

    captured = capsys.readouterr()

    assert "started background step: cache_public_train" in captured.out
    assert "pid: 12345" in captured.out
    assert calls == [
        {
            "config_path": CONFIG_PATH,
            "imagenet_val_dir": Path("/content/imagenet_val_prepare/val"),
            "cache_log_dir": tmp_path / "logs",
            "dry_run": False,
            "background_cache": True,
            "allow_duplicate_cache": False,
        }
    ]


def test_cli_module_entrypoint_runs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "learned_tta.cli",
            "full-run-status",
            "--config",
            str(CONFIG_PATH),
            "--next-command",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout.startswith("uv run python -m learned_tta.cli")


def _make_fake_imagenet_val(root: Path, classes: int = 2, images_per_class: int = 50) -> Path:
    val_root = root / "val"
    for class_idx in range(classes):
        class_dir = val_root / f"n{class_idx:08d}"
        class_dir.mkdir(parents=True)
        for image_idx in range(images_per_class):
            (class_dir / f"ILSVRC2012_val_{class_idx:04d}_{image_idx:04d}.JPEG").write_bytes(b"")
    return val_root


def _write_test_config(tmp_path: Path, class_count: int, images_per_class: int) -> Path:
    class_index_path = tmp_path / "class_index.txt"
    class_index_path.write_text(
        "\n".join(f"n{class_idx:08d}" for class_idx in range(class_count)) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        f"""
project_name: albu-tta
seed: 20260522

teacher:
  model_name: resnet50.a1_in1k
  pretrained: true

dataset:
  name: imagenet-val
  class_count: {class_count}
  class_index: {class_index_path}
  images_per_class: {images_per_class}
  public_per_class: 25
  private_per_class: 25
  public_train_per_class: 20
  public_val_per_class: 5

augmentations:
  registry_path: {ROOT / "configs" / "augmentations" / "imagenet100.yaml"}
  candidate_count: 100
  identity_id: aug_000

selector:
  output_dim: 100
  max_parameters: 1500000
  top_k_grid:
    - 1
    - 2
    - 4

artifacts:
  root: artifacts
  manifests_dir: artifacts/manifests
  teacher_cache_dir: artifacts/teacher_cache
  selector_dir: artifacts/selector
  reports_dir: reports/resnet50_a1_in1k
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path
