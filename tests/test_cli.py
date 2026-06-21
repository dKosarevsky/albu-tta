from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.cli import main
from learned_tta.selector_features import load_selector_features

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment" / "resnet50_a1_in1k.yaml"


def test_cli_import_does_not_import_albumentations() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import learned_tta.cli; "
                "import learned_tta.teacher_cache; "
                "import learned_tta.data; "
                "print(int('albumentations' in sys.modules))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "0"


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


def test_cli_prepare_imagenet_val_writes_wnid_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=2)
    val_tar = tmp_path / "ILSVRC2012_img_val.tar"
    devkit_tar = tmp_path / "ILSVRC2012_devkit_t12.tar.gz"
    output_dir = tmp_path / "imagenet" / "val"
    audit_path = tmp_path / "prepare_audit.json"
    _write_test_val_tar(
        val_tar,
        [
            "ILSVRC2012_val_00000001.JPEG",
            "ILSVRC2012_val_00000002.JPEG",
        ],
    )
    _write_test_devkit_tar(devkit_tar, labels=[1, 2])

    main(
        [
            "prepare-imagenet-val",
            "--config",
            str(config_path),
            "--val-tar",
            str(val_tar),
            "--devkit",
            str(devkit_tar),
            "--output-dir",
            str(output_dir),
            "--audit-output",
            str(audit_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    assert "prepared ImageNet validation: images=2, classes=2" in captured.out
    assert (output_dir / "n00000000" / "ILSVRC2012_val_00000001.JPEG").exists()
    assert (output_dir / "n00000001" / "ILSVRC2012_val_00000002.JPEG").exists()
    assert payload["label_mapping"] == "one_based_configured_class_index"


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


def test_cli_check_clean_baseline_writes_default_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)
    cache_dir = tmp_path / "artifacts" / "teacher_cache"
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
            aug_id="aug_000",
            image_ids=["img_0", "img_1"],
            class_idxs=np.array([0, 1], dtype=np.int64),
            logits=np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        ),
    )

    main(["check-clean-baseline", "--config", str(config_path)])
    captured = capsys.readouterr()

    output_path = cache_dir / "public_val__aug_000.clean_baseline.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "clean baseline ok" in captured.out
    assert "top1=1.0000" in captured.out
    assert payload["passed"] is True
    assert payload["split"] == "public_val"


def test_cli_summarize_clean_baseline_writes_full_val_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)
    cache_dir = tmp_path / "artifacts" / "teacher_cache"
    for split, image_ids, class_idxs, logits in [
        (
            "public_train",
            ["public_train_0", "public_train_1"],
            np.array([0, 1], dtype=np.int64),
            np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        ),
        (
            "public_val",
            ["public_val_0"],
            np.array([0], dtype=np.int64),
            np.array([[4.0, 0.0]], dtype=np.float32),
        ),
        (
            "private",
            ["private_0", "private_1"],
            np.array([0, 1], dtype=np.int64),
            np.array([[4.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        ),
    ]:
        write_teacher_shard(
            cache_dir,
            TeacherShard(
                split=split,
                aug_id="aug_000",
                image_ids=image_ids,
                class_idxs=class_idxs,
                logits=logits,
            ),
        )

    main(["summarize-clean-baseline", "--config", str(config_path)])
    captured = capsys.readouterr()

    output_path = (
        tmp_path / "reports" / "resnet50_a1_in1k" / "tables" / "clean_center_crop_baseline.json"
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "clean center-crop baseline:" in captured.out
    assert "splits=public_train,public_val,private" in captured.out
    assert "images=5" in captured.out
    assert "top1=0.8000" in captured.out
    assert payload["splits"] == ["public_train", "public_val", "private"]
    assert payload["overall"]["image_count"] == 5.0
    assert payload["overall"]["top1"] == pytest.approx(0.8)


def test_cli_build_selector_features_writes_default_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)
    manifest_path = tmp_path / "artifacts" / "manifests" / "public_train.csv"
    _write_feature_manifest(manifest_path)

    class FakeFeatureModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            means = inputs.mean(dim=(1, 2, 3))
            return torch.stack([means, means + 1.0], dim=1)

    fake_bundle = SimpleNamespace(
        model=FakeFeatureModel(),
        preprocess=lambda image: torch.full((3, 4, 4), float(image.getpixel((0, 0))[0])),
        model_name="resnet50.a1_in1k",
        pretrained=True,
        data_config={"input_size": (3, 4, 4)},
    )
    monkeypatch.setattr(
        "learned_tta.selector_feature_cache.build_timm_feature_extractor",
        lambda model_name, pretrained=True: fake_bundle,
    )

    main(
        [
            "build-selector-features",
            "--config",
            str(config_path),
            "--split",
            "public_train",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ]
    )
    captured = capsys.readouterr()
    output_path = (
        tmp_path
        / "artifacts"
        / "selector"
        / "features"
        / "public_train__resnet50.a1_in1k.features.npz"
    )
    loaded = load_selector_features(output_path)

    assert f"selector features: wrote {output_path}" in captured.out
    assert loaded.image_ids == ["public_train-0", "public_train-1"]
    assert loaded.features.shape == (2, 2)


def test_cli_full_run_status_reports_next_step(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(["full-run-status", "--config", str(config_path)])

    captured = capsys.readouterr()

    assert "full run status:" in captured.out
    assert "required steps complete" in captured.out
    assert "optional:" in captured.out
    assert "missing=1" in captured.out
    assert "next:" in captured.out
    assert "validate_augmentations" in captured.out


def test_cli_full_run_status_can_emit_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(["full-run-status", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["completed_required_steps"] == 0
    assert payload["total_required_steps"] == 14
    assert payload["next_step"]["name"] == "validate_augmentations"
    assert payload["steps"][0]["required"] is True
    assert payload["steps"][0]["missing_output_count"] == 1
    assert payload["steps"][0]["extra_output_count"] == 0
    assert payload["steps"][0]["outputs"][0].endswith("augmentation_registry_audit.json")
    assert payload["steps"][0]["missing_outputs"][0].endswith("augmentation_registry_audit.json")


def test_cli_teacher_cache_plan_can_emit_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)
    cache_dir = tmp_path / "teacher-cache"

    main(
        [
            "teacher-cache-plan",
            "--config",
            str(config_path),
            "--cache-dir",
            str(cache_dir),
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["cache_dir"] == str(cache_dir)
    assert payload["total_predictions"] == 10_000
    assert payload["expected_shards"] == 300
    assert payload["complete_shards"] == 0
    assert payload["logits_bytes_estimate"] == 40_000
    assert payload["splits"][0]["split"] == "public_train"
    assert payload["splits"][0]["expected_images"] == 40
    assert payload["splits"][0]["missing_files"] == 300
    assert "cache-teacher --split public_train" in payload["splits"][0]["next_command"]


def test_cli_teacher_cache_plan_text_reports_next_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(["teacher-cache-plan", "--config", str(config_path), "--split", "public_val"])
    captured = capsys.readouterr()

    assert "teacher cache plan:" in captured.out
    assert "[ ] public_val:" in captured.out
    assert "0 B/" in captured.out
    assert "next public_val:" in captured.out


def test_cli_teacher_backend_plan_text_reports_planned_accelerator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["teacher-backend-plan", "--device", "cpu"])
    captured = capsys.readouterr()

    assert "teacher backend plan:" in captured.out
    assert "recommended_accelerator=openvino" in captured.out
    assert "- pytorch: status=implemented" in captured.out


def test_cli_full_run_status_json_matches_stable_golden(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(["full-run-status", "--config", str(config_path), "--format", "json"])

    captured = capsys.readouterr()
    payload = _normalize_json_paths(json.loads(captured.out), extra_root=tmp_path)

    assert {
        "completed_required_steps": payload["completed_required_steps"],
        "completed_steps": payload["completed_steps"],
        "config_path": payload["config_path"],
        "total_required_steps": payload["total_required_steps"],
        "total_steps": payload["total_steps"],
    } == {
        "completed_required_steps": 0,
        "completed_steps": 0,
        "config_path": "<tmp>/experiment.yaml",
        "total_required_steps": 14,
        "total_steps": 15,
    }
    assert payload["next_step"] == payload["steps"][0]
    assert payload["steps"][:3] == [
        {
            "command": (
                "uv run python -m learned_tta.cli validate-augmentations "
                "--config <tmp>/experiment.yaml "
                "--audit-output <tmp>/artifacts/augmentation_registry_audit.json"
            ),
            "complete": False,
            "extra_output_count": 0,
            "extra_outputs": [],
            "missing_output_count": 1,
            "missing_outputs": ["<tmp>/artifacts/augmentation_registry_audit.json"],
            "name": "validate_augmentations",
            "outputs": ["<tmp>/artifacts/augmentation_registry_audit.json"],
            "required": True,
        },
        {
            "command": (
                "uv run python -m learned_tta.cli make-splits "
                "--config <tmp>/experiment.yaml "
                "--imagenet-val-dir /path/to/imagenet/val"
            ),
            "complete": False,
            "extra_output_count": 0,
            "extra_outputs": [],
            "missing_output_count": 5,
            "missing_outputs": [
                "<tmp>/artifacts/manifests/public_train.csv",
                "<tmp>/artifacts/manifests/public_val.csv",
                "<tmp>/artifacts/manifests/public.csv",
                "<tmp>/artifacts/manifests/private.csv",
                "<tmp>/artifacts/manifests/class_to_idx.json",
            ],
            "name": "make_splits",
            "outputs": [
                "<tmp>/artifacts/manifests/public_train.csv",
                "<tmp>/artifacts/manifests/public_val.csv",
                "<tmp>/artifacts/manifests/public.csv",
                "<tmp>/artifacts/manifests/private.csv",
                "<tmp>/artifacts/manifests/class_to_idx.json",
            ],
            "required": True,
        },
        {
            "command": (
                "uv run python -m learned_tta.cli cache-teacher "
                "--split public_val "
                "--config <tmp>/experiment.yaml "
                "--device cuda --num-workers 2 --candidate-id aug_000"
            ),
            "complete": False,
            "extra_output_count": 0,
            "extra_outputs": [],
            "missing_output_count": 3,
            "missing_outputs": [
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.parquet",
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.logits.npy",
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.run.json",
            ],
            "name": "cache_public_val_identity",
            "outputs": [
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.parquet",
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.logits.npy",
                "<tmp>/artifacts/teacher_cache/public_val__aug_000.run.json",
            ],
            "required": True,
        },
    ]


def test_cli_full_run_status_can_fail_on_incomplete_required_steps(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "full-run-status",
                "--config",
                str(config_path),
                "--fail-on-incomplete",
            ]
        )

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "next: validate_augmentations" in captured.out


def test_cli_full_run_status_can_print_only_next_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    main(["full-run-status", "--config", str(config_path), "--next-command"])

    captured = capsys.readouterr()

    assert captured.err == ""
    assert captured.out.startswith("uv run python -m learned_tta.cli validate-augmentations")
    assert "--audit-output" in captured.out
    assert "full run status:" not in captured.out
    assert captured.out.count("\n") == 1


def test_cli_full_run_status_next_command_can_fail_on_incomplete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "full-run-status",
                "--config",
                str(config_path),
                "--next-command",
                "--fail-on-incomplete",
            ]
        )

    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert captured.out.startswith("uv run python -m learned_tta.cli validate-augmentations")


def test_cli_full_run_status_json_can_fail_on_incomplete(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "full-run-status",
                "--config",
                str(config_path),
                "--format",
                "json",
                "--fail-on-incomplete",
            ]
        )

    payload = json.loads(capsys.readouterr().out)

    assert exc_info.value.code == 1
    assert payload["next_step"]["name"] == "validate_augmentations"


def test_cli_full_run_status_reports_no_next_step_for_complete_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "learned_tta.run_status.inspect_full_run_status",
        lambda _config_path: SimpleNamespace(
            completed_required_steps=14,
            total_required_steps=14,
            completed_steps=15,
            total_steps=15,
            steps=(),
            next_step=None,
        ),
    )

    main(["full-run-status", "--config", str(CONFIG_PATH)])
    captured = capsys.readouterr()

    assert "15 total" in captured.out
    assert "next: none" in captured.out


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
        "learned_tta.run_supervisor.run_next_full_run_step",
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


@pytest.mark.parametrize(
    ("result", "expected_lines"),
    [
        (
            SimpleNamespace(
                status="complete",
                step_name=None,
                command=None,
                log_path=None,
                pid=None,
                active_processes=(),
            ),
            ["full run complete: no required steps left"],
        ),
        (
            SimpleNamespace(
                status="dry-run",
                step_name="validate_augmentations",
                command="uv run python -m learned_tta.cli validate-augmentations",
                log_path=None,
                pid=None,
                active_processes=(),
            ),
            [
                "dry-run: validate_augmentations",
                "uv run python -m learned_tta.cli validate-augmentations",
            ],
        ),
        (
            SimpleNamespace(
                status="active",
                step_name="cache_public_train",
                command="uv run python -m learned_tta.cli cache-teacher --split public_train",
                log_path=None,
                pid=None,
                active_processes=("123 python -m learned_tta.cli cache-teacher",),
            ),
            [
                "cache already active: cache_public_train",
                "123 python -m learned_tta.cli cache-teacher",
                "not starting a duplicate process",
            ],
        ),
        (
            SimpleNamespace(
                status="completed",
                step_name="validate_augmentations",
                command="uv run python -m learned_tta.cli validate-augmentations",
                log_path=None,
                pid=None,
                active_processes=(),
            ),
            [
                "completed step: validate_augmentations",
                "uv run python -m learned_tta.cli validate-augmentations",
            ],
        ),
    ],
)
def test_cli_resume_full_run_reports_non_started_statuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result: SimpleNamespace,
    expected_lines: list[str],
) -> None:
    monkeypatch.setattr(
        "learned_tta.run_supervisor.run_next_full_run_step",
        lambda **_kwargs: result,
    )

    main(["resume-full-run", "--config", str(CONFIG_PATH)])
    captured = capsys.readouterr()

    for expected_line in expected_lines:
        assert expected_line in captured.out


def test_cli_resume_full_run_rejects_unknown_supervisor_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "learned_tta.run_supervisor.run_next_full_run_step",
        lambda **_kwargs: SimpleNamespace(
            status="unknown",
            step_name="mystery",
            command=None,
            log_path=None,
            pid=None,
            active_processes=(),
        ),
    )

    with pytest.raises(ValueError, match="unknown resume result status: unknown"):
        main(["resume-full-run", "--config", str(CONFIG_PATH)])


def test_cli_module_entrypoint_runs(tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path, class_count=2, images_per_class=50)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "learned_tta.cli",
            "full-run-status",
            "--config",
            str(config_path),
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


def _write_feature_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(2):
        image_path = path.parent / f"feature_image_{index}.png"
        image = np.full((8, 8, 3), fill_value=index + 1, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(image_path)
        rows.append(
            {
                "split": "public_train",
                "image_id": f"public_train-{index}",
                "class_idx": index,
                "class_name": f"class-{index}",
                "path": str(image_path),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_test_val_tar(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w") as archive:
        for name in names:
            payload = f"payload for {name}".encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _write_test_devkit_tar(path: Path, labels: list[int]) -> None:
    payload = ("\n".join(str(label) for label in labels) + "\n").encode()
    info = tarfile.TarInfo(name="ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, BytesIO(payload))


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

clean_baseline:
  split: public_val
  min_top1: 0.70
  min_top5: 0.90
  max_nll: 1.60

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


def _normalize_json_paths(value: Any, extra_root: Path | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_json_paths(item, extra_root=extra_root) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_json_paths(item, extra_root=extra_root) for item in value]
    if isinstance(value, str):
        normalized = value.replace(str(ROOT), "<repo>")
        if extra_root is not None:
            normalized = normalized.replace(str(extra_root), "<tmp>")
        return normalized
    return value
