from __future__ import annotations

from pathlib import Path

from learned_tta.run_status import inspect_full_run_status


def test_inspect_full_run_status_reports_first_missing_step(tmp_path: Path) -> None:
    config_path = _write_experiment_config(tmp_path)

    summary = inspect_full_run_status(config_path)

    assert summary.completed_steps == 0
    assert summary.total_steps == 13
    assert summary.completed_required_steps == 0
    assert summary.total_required_steps == 12
    assert summary.next_step is not None
    assert summary.next_step.name == "validate_augmentations"
    assert summary.steps[0].required is True
    assert summary.steps[0].outputs == (
        tmp_path / "project" / "artifacts" / "augmentation_registry_audit.json",
    )
    assert summary.steps[0].missing_outputs == summary.steps[0].outputs
    assert summary.steps[0].extra_outputs == ()


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

    summary = inspect_full_run_status(config_path)

    assert summary.completed_steps == 2
    assert summary.completed_required_steps == 2
    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert "cache-teacher --split public_train" in summary.next_step.command


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
    _touch(
        cache_dir / "public_train__aug_000.parquet",
        cache_dir / "public_train__aug_000.logits.npy",
        cache_dir / "public_train__aug_999.parquet",
    )

    summary = inspect_full_run_status(config_path)
    cache_step = next(step for step in summary.steps if step.name == "cache_public_train")

    assert summary.completed_required_steps == 2
    assert summary.next_step is not None
    assert summary.next_step.name == "cache_public_train"
    assert len(cache_step.missing_outputs) == 198
    assert cache_step.missing_outputs[0].name == "public_train__aug_001.parquet"
    assert cache_step.extra_outputs == (cache_dir / "public_train__aug_999.parquet",)


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

    assert summary.completed_steps == 12
    assert summary.total_steps == 13
    assert summary.completed_required_steps == summary.total_required_steps
    assert summary.next_step is None
    assert all(step.missing_outputs == () for step in summary.steps if step.required)
    optional_steps = [step for step in summary.steps if not step.required]
    assert [step.name for step in optional_steps] == ["train_xgboost_aggregator"]
    assert optional_steps[0].complete is False
    assert "xgboost-multiclass" in optional_steps[0].command


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
dataset:
  name: imagenet-val
  images_per_class: 50
  public_per_class: 25
  private_per_class: 25
  public_train_per_class: 20
  public_val_per_class: 5
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
        _touch(
            cache_dir / f"{split}__{aug_id}.parquet",
            cache_dir / f"{split}__{aug_id}.logits.npy",
        )
