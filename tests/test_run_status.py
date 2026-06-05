from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from learned_tta.augmentations import AugmentationCandidate
from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.config import load_experiment_config
from learned_tta.run_status import (
    _contains_subset,
    _expected_teacher_cache_metadata,
    _extra_teacher_cache_outputs,
    _has_complete_teacher_cache,
    _json_ready,
    _teacher_cache_metadata_matches,
    inspect_full_run_status,
)


def test_inspect_full_run_status_reports_first_missing_step(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path)

    summary = inspect_full_run_status(config_path)

    assert summary.completed_steps == 0
    assert summary.total_steps == 15
    assert summary.completed_required_steps == 0
    assert summary.total_required_steps == 14
    assert summary.next_step is not None
    assert summary.next_step.name == "validate_augmentations"
    assert summary.steps[0].required is True
    assert summary.steps[0].outputs == (
        tmp_path / "project" / "artifacts" / "augmentation_registry_audit.json",
    )
    assert summary.steps[0].missing_outputs == summary.steps[0].outputs
    assert summary.steps[0].extra_outputs == ()
    assert summary.steps[1].name == "make_splits"
    assert len(summary.steps[1].outputs) == 5
    assert summary.steps[1].outputs[-1].name == "class_to_idx.json"


def test_inspect_full_run_status_advances_past_completed_manifests(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    (project_root / "artifacts").mkdir()
    (project_root / "artifacts" / "augmentation_registry_audit.json").write_text(
        "{}",
        encoding="utf-8",
    )
    manifests_dir = project_root / "artifacts" / "manifests"
    manifests_dir.mkdir()
    for split in ("public_train", "public_val", "public", "private"):
        (manifests_dir / f"{split}.csv").write_text("path,label\n", encoding="utf-8")
    (manifests_dir / "class_to_idx.json").write_text("{}", encoding="utf-8")

    summary = inspect_full_run_status(config_path)

    assert summary.completed_steps == 2
    assert summary.completed_required_steps == 2
    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_val_identity"
    assert "cache-teacher --split public_val" in summary.next_step.command
    assert "--candidate-id aug_000" in summary.next_step.command
    assert "--num-workers 2" in summary.next_step.command


def test_inspect_full_run_status_rejects_partial_teacher_cache(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    manifests_dir = artifacts_dir / "manifests"
    cache_dir = artifacts_dir / "teacher_cache"
    _touch(artifacts_dir / "augmentation_registry_audit.json")
    for split in ("public_train", "public_val", "public", "private"):
        _touch(manifests_dir / f"{split}.csv")
    _touch(manifests_dir / "class_to_idx.json")
    _touch_teacher_cache(cache_dir, split="public_val", candidate_count=1)
    _touch(cache_dir / "public_val__aug_000.clean_baseline.json")
    _touch(
        cache_dir / "public_train__aug_000.parquet",
        cache_dir / "public_train__aug_000.logits.npy",
        cache_dir / "public_train__aug_999.parquet",
        cache_dir / "public_train__aug_999.run.json",
    )

    summary = inspect_full_run_status(config_path)
    cache_step = next(step for step in summary.steps if step.name == "cache_public_train")

    assert summary.completed_required_steps == 4
    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert len(cache_step.missing_outputs) == 298
    assert cache_step.missing_outputs[0].name == "public_train__aug_000.run.json"
    assert cache_step.extra_outputs == (
        cache_dir / "public_train__aug_999.parquet",
        cache_dir / "public_train__aug_999.run.json",
    )


def test_inspect_full_run_status_rejects_stale_teacher_cache_metadata(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    manifests_dir = artifacts_dir / "manifests"
    cache_dir = artifacts_dir / "teacher_cache"
    _touch(artifacts_dir / "augmentation_registry_audit.json")
    for split in ("public_train", "public_val", "public", "private"):
        _touch(manifests_dir / f"{split}.csv")
    _touch(manifests_dir / "class_to_idx.json")
    _touch_teacher_cache(cache_dir, split="public_val", candidate_count=1)
    _touch(cache_dir / "public_val__aug_000.clean_baseline.json")
    _touch_teacher_cache(cache_dir, split="public_train", candidate_count=100)
    (cache_dir / "public_train__aug_042.run.json").write_text(
        json.dumps({"version": 1, "split": "public_train", "aug_id": "aug_042", "seed": 999}),
        encoding="utf-8",
    )

    summary = inspect_full_run_status(config_path)
    cache_step = next(step for step in summary.steps if step.name == "cache_public_train")

    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert cache_step.complete is False
    assert cache_step.missing_outputs == (cache_dir / "public_train__aug_042.run.json",)


def test_inspect_full_run_status_rejects_wrong_shape_teacher_cache(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    manifests_dir = artifacts_dir / "manifests"
    cache_dir = artifacts_dir / "teacher_cache"
    _touch(artifacts_dir / "augmentation_registry_audit.json")
    for split in ("public_train", "public_val", "public", "private"):
        _touch(manifests_dir / f"{split}.csv")
    _touch(manifests_dir / "class_to_idx.json")
    _touch_teacher_cache(cache_dir, split="public_val", candidate_count=1)
    _touch(cache_dir / "public_val__aug_000.clean_baseline.json")
    _touch_teacher_cache(cache_dir, split="public_train", candidate_count=100)
    np.save(cache_dir / "public_train__aug_007.logits.npy", np.zeros((1, 2), dtype=np.float16))

    summary = inspect_full_run_status(config_path)
    cache_step = next(step for step in summary.steps if step.name == "cache_public_train")

    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert cache_step.complete is False
    assert cache_dir / "public_train__aug_007.logits.npy" in cache_step.missing_outputs


def test_inspect_full_run_status_rejects_teacher_data_config_drift(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    manifests_dir = artifacts_dir / "manifests"
    cache_dir = artifacts_dir / "teacher_cache"
    _touch(artifacts_dir / "augmentation_registry_audit.json")
    for split in ("public_train", "public_val", "public", "private"):
        _touch(manifests_dir / f"{split}.csv")
    _touch(manifests_dir / "class_to_idx.json")
    _touch_teacher_cache(cache_dir, split="public_val", candidate_count=1)
    _touch(cache_dir / "public_val__aug_000.clean_baseline.json")
    _touch_teacher_cache(cache_dir, split="public_train", candidate_count=100)
    stale_metadata = _teacher_cache_run_metadata(split="public_train", aug_id="aug_013")
    teacher_metadata = cast(dict[str, object], stale_metadata["teacher"])
    teacher_metadata["data_config"] = {"input_size": [3, 256, 256]}
    (cache_dir / "public_train__aug_013.run.json").write_text(
        json.dumps(stale_metadata),
        encoding="utf-8",
    )

    summary = inspect_full_run_status(config_path)
    cache_step = next(step for step in summary.steps if step.name == "cache_public_train")

    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert cache_step.complete is False
    assert cache_step.missing_outputs == (cache_dir / "public_train__aug_013.run.json",)


def test_inspect_full_run_status_does_not_block_on_optional_xgboost(
    tmp_path: Path,
) -> None:
    config_path = _write_experiment_config(tmp_path)
    project_root = tmp_path / "project"
    artifacts_dir = project_root / "artifacts"
    manifests_dir = artifacts_dir / "manifests"
    cache_dir = artifacts_dir / "teacher_cache"
    selector_dir = artifacts_dir / "selector"
    reports_dir = project_root / "reports" / "resnet50_a1_in1k"
    tables_dir = reports_dir / "tables"
    for directory in (artifacts_dir, manifests_dir, cache_dir, selector_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_paths = tuple(
        manifests_dir / f"{split}.csv"
        for split in ("public_train", "public_val", "public", "private")
    )
    _touch(
        artifacts_dir / "augmentation_registry_audit.json",
        *manifest_paths,
        manifests_dir / "class_to_idx.json",
        cache_dir / "public_val__aug_000.clean_baseline.json",
        selector_dir / "public_train_targets.npz",
        selector_dir / "public_val_targets.npz",
        selector_dir / "selector_best.pt",
        selector_dir / "selector_history.csv",
        selector_dir / "public_val_tta_tuning.json",
        selector_dir / "public_val_global_nonnegative_aggregator.json",
        selector_dir / "public_val_class_nonnegative_aggregator.json",
        tables_dir / "private_metrics.csv",
        tables_dir / "corrections.csv",
        reports_dir / "results.md",
        tables_dir / "augmentation_impact.csv",
        tables_dir / "private_metric_deltas.csv",
    )
    for split in ("public_train", "public_val", "private"):
        _touch_teacher_cache(cache_dir, split=split, candidate_count=100)

    summary = inspect_full_run_status(config_path)

    assert summary.completed_steps == 14
    assert summary.total_steps == 15
    assert summary.completed_required_steps == summary.total_required_steps
    assert summary.next_step is None
    assert all(step.missing_outputs == () for step in summary.steps if step.required)
    optional_steps = [step for step in summary.steps if not step.required]
    assert [step.name for step in optional_steps] == ["train_xgboost_aggregator"]
    assert optional_steps[0].complete is False
    assert "xgboost-multiclass" in optional_steps[0].command


def test_run_status_teacher_cache_helpers_cover_metadata_edge_cases(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path)
    config = load_experiment_config(config_path)
    cache_dir = tmp_path / "teacher_cache"

    assert not _has_complete_teacher_cache(
        config,
        cache_dir,
        split="public_train",
        expected_aug_ids=(),
    )

    candidate = AugmentationCandidate(
        id="aug_001",
        name="custom",
        class_name="HorizontalFlip",
        params={"p": 1.0, "limits": (1, 1), "path": Path("local")},
    )
    expected = _expected_teacher_cache_metadata(
        config,
        split="public_train",
        aug_id="aug_001",
        candidate=candidate,
    )
    assert expected["augmentation"] == {
        "id": "aug_001",
        "name": "custom",
        "determinism": "fixed",
        "class_name": "HorizontalFlip",
        "params": {"p": 1.0, "limits": [1, 1], "path": "PosixPath('local')"},
    }
    assert _json_ready({"items": (Path("a"),)}) == {"items": ["PosixPath('a')"]}

    bad_json = tmp_path / "bad.run.json"
    bad_json.write_text("", encoding="utf-8")
    assert not _teacher_cache_metadata_matches(bad_json, expected_metadata=expected)
    not_object = tmp_path / "list.run.json"
    not_object.write_text("[]", encoding="utf-8")
    assert not _teacher_cache_metadata_matches(not_object, expected_metadata=expected)
    assert not _contains_subset({"teacher": "resnet50"}, {"teacher": {"model_name": "x"}})

    _touch(
        cache_dir / "public_train__aug_999.logits.npy",
        cache_dir / "public_train__aug_999.parquet",
        cache_dir / "public_train__aug_999.run.json",
    )
    assert cache_dir / "public_train__aug_999.logits.npy" in _extra_teacher_cache_outputs(
        cache_dir,
        split="public_train",
        expected_aug_ids=("aug_000",),
    )


def _write_experiment_config(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs" / "experiment"
    config_dir.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    config_path = config_dir / "experiment.yaml"
    config_path.write_text(
        """
project_name: albu-tta
seed: 20260522
teacher:
  model_name: resnet50.a1_in1k
  pretrained: true
  data_config:
    input_size: [3, 224, 224]
dataset:
  name: imagenet-val
  class_count: 2
  class_index: timm-imagenet-1k
  images_per_class: 4
  public_per_class: 2
  private_per_class: 2
  public_train_per_class: 1
  public_val_per_class: 1
clean_baseline:
  split: public_val
  min_top1: 0.70
  min_top5: 0.90
  max_nll: 1.60
augmentations:
  registry_path: configs/augmentations/imagenet100.yaml
  candidate_count: 100
  identity_id: aug_000
selector:
  output_dim: 100
  max_parameters: 1500000
  top_k_grid: [1, 2, 4, 8, 16]
artifacts:
  root: artifacts
  manifests_dir: artifacts/manifests
  teacher_cache_dir: artifacts/teacher_cache
  selector_dir: artifacts/selector
  reports_dir: reports/resnet50_a1_in1k
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _touch(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _touch_teacher_cache(cache_dir: Path, split: str, candidate_count: int) -> None:
    for candidate_idx in range(candidate_count):
        aug_id = f"aug_{candidate_idx:03d}"
        rows = _expected_test_rows(split)
        write_teacher_shard(
            cache_dir,
            TeacherShard(
                split=split,
                aug_id=aug_id,
                image_ids=[f"{split}-{row_idx}" for row_idx in range(rows)],
                class_idxs=np.arange(rows, dtype=np.int64) % 2,
                logits=np.zeros((rows, 2), dtype=np.float32),
                run_metadata=_teacher_cache_run_metadata(split=split, aug_id=aug_id),
            ),
        )


def _teacher_cache_run_metadata(split: str, aug_id: str) -> dict[str, object]:
    return {
        "version": 1,
        "split": split,
        "aug_id": aug_id,
        "seed": 20260522,
        "teacher": {
            "model_name": "resnet50.a1_in1k",
            "pretrained": True,
            "num_classes": 2,
            "data_config": {"input_size": [3, 224, 224]},
        },
        "storage": {
            "logits_dtype": "float16",
            "metadata_format": "parquet",
        },
    }


def _expected_test_rows(split: str) -> int:
    if split in {"public_train", "public_val"}:
        return 2
    if split in {"public", "private"}:
        return 4
    raise ValueError(f"unknown test split {split!r}")
