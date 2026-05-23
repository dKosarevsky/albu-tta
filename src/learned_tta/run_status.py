"""Filesystem status checks for the full ImageNet experiment run."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_tta.config import ExperimentConfig, load_experiment_config


@dataclass(frozen=True, slots=True)
class FullRunStepStatus:
    """Completion status for one full-run pipeline step."""

    name: str
    complete: bool
    outputs: tuple[Path, ...]
    command: str
    required: bool


@dataclass(frozen=True, slots=True)
class FullRunStatusSummary:
    """Ordered status summary for the full ImageNet experiment run."""

    config_path: Path
    steps: tuple[FullRunStepStatus, ...]
    completed_steps: int
    total_steps: int
    completed_required_steps: int
    total_required_steps: int
    next_step: FullRunStepStatus | None


@dataclass(frozen=True, slots=True)
class _StepSpec:
    name: str
    outputs: tuple[Path, ...]
    command: str
    complete: Callable[[], bool]
    required: bool = True


def inspect_full_run_status(config_path: Path) -> FullRunStatusSummary:
    """Inspect configured full-run artifacts and report the next missing step."""

    config = load_experiment_config(config_path)
    steps = tuple(_build_step_statuses(config))
    completed_steps = sum(step.complete for step in steps)
    required_steps = tuple(step for step in steps if step.required)
    completed_required_steps = sum(step.complete for step in required_steps)
    next_step = next((step for step in required_steps if not step.complete), None)
    return FullRunStatusSummary(
        config_path=config.path,
        steps=steps,
        completed_steps=completed_steps,
        total_steps=len(steps),
        completed_required_steps=completed_required_steps,
        total_required_steps=len(required_steps),
        next_step=next_step,
    )


def full_run_status_to_dict(summary: FullRunStatusSummary) -> dict[str, Any]:
    """Return a JSON-serializable representation of a full-run status summary."""

    return {
        "config_path": str(summary.config_path),
        "completed_steps": summary.completed_steps,
        "total_steps": summary.total_steps,
        "completed_required_steps": summary.completed_required_steps,
        "total_required_steps": summary.total_required_steps,
        "next_step": _step_to_dict(summary.next_step) if summary.next_step is not None else None,
        "steps": [_step_to_dict(step) for step in summary.steps],
    }


def _build_step_statuses(config: ExperimentConfig) -> list[FullRunStepStatus]:
    audit_path = config.artifacts.root / "augmentation_registry_audit.json"
    manifests = tuple(
        config.artifacts.manifests_dir / f"{split}.csv"
        for split in ("public_train", "public_val", "public", "private")
    )
    train_targets = config.artifacts.selector_dir / "public_train_targets.npz"
    val_targets = config.artifacts.selector_dir / "public_val_targets.npz"
    selector_checkpoint = config.artifacts.selector_dir / "selector_best.pt"
    selector_history = config.artifacts.selector_dir / "selector_history.csv"
    tuning_path = config.artifacts.selector_dir / "public_val_tta_tuning.json"
    global_aggregator = _aggregator_path(
        output_dir=config.artifacts.selector_dir,
        split="public_val",
        method="global-nonnegative",
    )
    class_aggregator = _aggregator_path(
        output_dir=config.artifacts.selector_dir,
        split="public_val",
        method="class-nonnegative",
    )
    xgboost_aggregator = _aggregator_path(
        output_dir=config.artifacts.selector_dir,
        split="public_val",
        method="xgboost-multiclass",
    )
    xgboost_model = xgboost_aggregator.with_suffix(".model.json")
    private_metrics = config.artifacts.reports_dir / "tables" / "private_metrics.csv"
    corrections = config.artifacts.reports_dir / "tables" / "corrections.csv"
    results_md = config.artifacts.reports_dir / "results.md"
    augmentation_impact = config.artifacts.reports_dir / "tables" / "augmentation_impact.csv"
    private_deltas = config.artifacts.reports_dir / "tables" / "private_metric_deltas.csv"
    specs = (
        _StepSpec(
            name="validate_augmentations",
            outputs=(audit_path,),
            command=(
                "uv run python -m learned_tta.cli validate-augmentations "
                f"--config {config.path} --audit-output {audit_path}"
            ),
            complete=lambda: audit_path.exists(),
        ),
        _StepSpec(
            name="make_splits",
            outputs=manifests,
            command=(
                "uv run python -m learned_tta.cli make-splits "
                f"--config {config.path} --imagenet-val-dir /path/to/imagenet/val"
            ),
            complete=lambda: _all_exist(manifests),
        ),
        _cache_step_spec(config, "public_train"),
        _cache_step_spec(config, "public_val"),
        _StepSpec(
            name="build_targets",
            outputs=(train_targets, val_targets),
            command=f"uv run python -m learned_tta.cli build-targets --config {config.path}",
            complete=lambda: train_targets.exists() and val_targets.exists(),
        ),
        _StepSpec(
            name="train_selector",
            outputs=(selector_checkpoint, selector_history),
            command=(
                "uv run python -m learned_tta.cli train-selector "
                f"--config {config.path} --device cuda"
            ),
            complete=lambda: selector_checkpoint.exists() and selector_history.exists(),
        ),
        _StepSpec(
            name="tune_tta",
            outputs=(tuning_path,),
            command=(
                "uv run python -m learned_tta.cli tune-tta --split public_val "
                f"--config {config.path} --device cuda"
            ),
            complete=lambda: tuning_path.exists(),
        ),
        _StepSpec(
            name="train_global_aggregator",
            outputs=(global_aggregator,),
            command=(
                "uv run python -m learned_tta.cli train-aggregator "
                "--method global-nonnegative "
                f"--config {config.path} --split public_val --device cuda"
            ),
            complete=lambda: global_aggregator.exists(),
        ),
        _StepSpec(
            name="train_class_aggregator",
            outputs=(class_aggregator,),
            command=(
                "uv run python -m learned_tta.cli train-aggregator "
                "--method class-nonnegative "
                f"--config {config.path} --split public_val --device cuda"
            ),
            complete=lambda: class_aggregator.exists(),
        ),
        _StepSpec(
            name="train_xgboost_aggregator",
            outputs=(xgboost_aggregator, xgboost_model),
            command=(
                "uv run python -m learned_tta.cli train-aggregator "
                "--method xgboost-multiclass "
                f"--config {config.path} --split public_val"
            ),
            complete=lambda: xgboost_aggregator.exists() and xgboost_model.exists(),
            required=False,
        ),
        _cache_step_spec(config, "private"),
        _StepSpec(
            name="evaluate_private",
            outputs=(private_metrics, corrections),
            command=(
                "uv run python -m learned_tta.cli evaluate-private "
                f"--config {config.path} --device cuda"
            ),
            complete=lambda: private_metrics.exists() and corrections.exists(),
        ),
        _StepSpec(
            name="build_report",
            outputs=(results_md, augmentation_impact, private_deltas),
            command=(
                "uv run python -m learned_tta.cli build-report "
                f"--config {config.path} --device cuda"
            ),
            complete=lambda: results_md.exists()
            and augmentation_impact.exists()
            and private_deltas.exists(),
        ),
    )
    return [
        FullRunStepStatus(
            name=spec.name,
            complete=spec.complete(),
            outputs=spec.outputs,
            command=spec.command,
            required=spec.required,
        )
        for spec in specs
    ]


def _cache_step_spec(config: ExperimentConfig, split: str) -> _StepSpec:
    return _StepSpec(
        name=f"cache_{split}",
        outputs=(config.artifacts.teacher_cache_dir,),
        command=(
            f"uv run python -m learned_tta.cli cache-teacher --split {split} "
            f"--config {config.path} --device cuda"
        ),
        complete=lambda: _has_complete_teacher_cache(config.artifacts.teacher_cache_dir, split),
    )


def _all_exist(paths: tuple[Path, ...]) -> bool:
    return all(path.exists() for path in paths)


def _aggregator_path(output_dir: Path, split: str, method: str) -> Path:
    method_slug = method.replace("-", "_")
    return output_dir / f"{split}_{method_slug}_aggregator.json"


def _has_complete_teacher_cache(cache_dir: Path, split: str) -> bool:
    parquet_shards = sorted(cache_dir.glob(f"{split}__*.parquet"))
    logits_shards = sorted(cache_dir.glob(f"{split}__*.logits.npy"))
    if not parquet_shards:
        return False
    parquet_stems = {path.name.removesuffix(".parquet") for path in parquet_shards}
    logits_stems = {path.name.removesuffix(".logits.npy") for path in logits_shards}
    return parquet_stems == logits_stems


def _step_to_dict(step: FullRunStepStatus) -> dict[str, Any]:
    return {
        "name": step.name,
        "complete": step.complete,
        "required": step.required,
        "outputs": [str(path) for path in step.outputs],
        "command": step.command,
    }
