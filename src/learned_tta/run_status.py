"""Filesystem status checks for the full ImageNet experiment run."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from learned_tta.augmentations import AugmentationCandidate, load_augmentation_registry
from learned_tta.cache import shard_is_complete, teacher_shard_paths
from learned_tta.clean_baseline import clean_baseline_artifact_path
from learned_tta.config import ExperimentConfig, load_experiment_config

CACHE_TEACHER_NUM_WORKERS = 2


@dataclass(frozen=True, slots=True)
class FullRunStepStatus:
    """Completion status for one full-run pipeline step."""

    name: str
    complete: bool
    outputs: tuple[Path, ...]
    command: str
    required: bool
    missing_outputs: tuple[Path, ...]
    extra_outputs: tuple[Path, ...]


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


def _empty_outputs() -> tuple[Path, ...]:
    return ()


@dataclass(frozen=True, slots=True)
class _StepSpec:
    name: str
    outputs: tuple[Path, ...]
    command: str
    complete: Callable[[], bool]
    required: bool = True
    extra_outputs: Callable[[], tuple[Path, ...]] = _empty_outputs
    missing_outputs: Callable[[], tuple[Path, ...]] | None = None


def inspect_full_run_status(config_path: Path) -> FullRunStatusSummary:
    """Inspect configured full-run artifacts and report the next missing step."""

    config = load_experiment_config(config_path)
    expected_aug_ids = _expected_augmentation_ids(config)
    steps = tuple(_build_step_statuses(config, expected_aug_ids=expected_aug_ids))
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


def _build_step_statuses(
    config: ExperimentConfig,
    expected_aug_ids: tuple[str, ...],
) -> list[FullRunStepStatus]:
    audit_path = config.artifacts.root / "augmentation_registry_audit.json"
    manifests = tuple(
        config.artifacts.manifests_dir / f"{split}.csv"
        for split in ("public_train", "public_val", "public", "private")
    )
    class_mapping = config.artifacts.manifests_dir / "class_to_idx.json"
    clean_baseline = clean_baseline_artifact_path(
        cache_dir=config.artifacts.teacher_cache_dir,
        split=config.clean_baseline.split,
        identity_aug_id=config.augmentations.identity_id,
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
            outputs=(*manifests, class_mapping),
            command=(
                "uv run python -m learned_tta.cli make-splits "
                f"--config {config.path} --imagenet-val-dir /path/to/imagenet/val"
            ),
            complete=lambda: _all_exist((*manifests, class_mapping)),
        ),
        _cache_step_spec(
            config,
            config.clean_baseline.split,
            expected_aug_ids=(config.augmentations.identity_id,),
            name=f"cache_{config.clean_baseline.split}_identity",
            candidate_args=f"--candidate-id {config.augmentations.identity_id}",
            allow_extra_outputs=True,
        ),
        _StepSpec(
            name="check_clean_baseline",
            outputs=(clean_baseline,),
            command=f"uv run python -m learned_tta.cli check-clean-baseline --config {config.path}",
            complete=lambda: clean_baseline.exists(),
        ),
        _cache_step_spec(config, "public_train", expected_aug_ids=expected_aug_ids),
        _cache_step_spec(config, "public_val", expected_aug_ids=expected_aug_ids),
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
        _cache_step_spec(config, "private", expected_aug_ids=expected_aug_ids),
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
        _build_step_status(spec)
        for spec in specs
    ]


def _cache_step_spec(
    config: ExperimentConfig,
    split: str,
    expected_aug_ids: tuple[str, ...],
    name: str | None = None,
    candidate_args: str = "",
    allow_extra_outputs: bool = False,
) -> _StepSpec:
    outputs = tuple(
        path
        for aug_id in expected_aug_ids
        for path in (
            config.artifacts.teacher_cache_dir / f"{split}__{aug_id}.parquet",
            config.artifacts.teacher_cache_dir / f"{split}__{aug_id}.logits.npy",
            config.artifacts.teacher_cache_dir / f"{split}__{aug_id}.run.json",
        )
    )
    return _StepSpec(
        name=name or f"cache_{split}",
        outputs=outputs,
        command=(
            f"uv run python -m learned_tta.cli cache-teacher --split {split} "
            f"--config {config.path} --device cuda "
            f"--num-workers {CACHE_TEACHER_NUM_WORKERS}"
            f"{' ' + candidate_args if candidate_args else ''}"
        ),
        complete=lambda: _has_complete_teacher_cache(
            config,
            config.artifacts.teacher_cache_dir,
            split,
            expected_aug_ids=expected_aug_ids,
            allow_extra_outputs=allow_extra_outputs,
        ),
        missing_outputs=lambda: _missing_teacher_cache_outputs(
            config,
            config.artifacts.teacher_cache_dir,
            split,
            expected_aug_ids=expected_aug_ids,
        ),
        extra_outputs=(
            _empty_outputs
            if allow_extra_outputs
            else lambda: _extra_teacher_cache_outputs(
                config.artifacts.teacher_cache_dir,
                split,
                expected_aug_ids=expected_aug_ids,
            )
        ),
    )


def _all_exist(paths: tuple[Path, ...]) -> bool:
    return all(path.exists() for path in paths)


def _build_step_status(spec: _StepSpec) -> FullRunStepStatus:
    missing_outputs = (
        spec.missing_outputs()
        if spec.missing_outputs is not None
        else tuple(path for path in spec.outputs if not path.exists())
    )
    return FullRunStepStatus(
        name=spec.name,
        complete=spec.complete(),
        outputs=spec.outputs,
        command=spec.command,
        required=spec.required,
        missing_outputs=missing_outputs,
        extra_outputs=spec.extra_outputs(),
    )


def _aggregator_path(output_dir: Path, split: str, method: str) -> Path:
    method_slug = method.replace("-", "_")
    return output_dir / f"{split}_{method_slug}_aggregator.json"


def _expected_augmentation_ids(config: ExperimentConfig) -> tuple[str, ...]:
    if config.augmentations.registry_path.exists():
        return tuple(
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        )
    return tuple(
        f"aug_{candidate_idx:03d}"
        for candidate_idx in range(config.augmentations.candidate_count)
    )


def _has_complete_teacher_cache(
    config: ExperimentConfig,
    cache_dir: Path,
    split: str,
    expected_aug_ids: tuple[str, ...],
    allow_extra_outputs: bool = False,
) -> bool:
    if not expected_aug_ids:
        return False
    if _missing_teacher_cache_outputs(
        config,
        cache_dir,
        split,
        expected_aug_ids=expected_aug_ids,
    ):
        return False
    parquet_shards = sorted(cache_dir.glob(f"{split}__*.parquet"))
    logits_shards = sorted(cache_dir.glob(f"{split}__*.logits.npy"))
    run_metadata_shards = sorted(cache_dir.glob(f"{split}__*.run.json"))
    parquet_stems = {path.name.removesuffix(".parquet") for path in parquet_shards}
    logits_stems = {path.name.removesuffix(".logits.npy") for path in logits_shards}
    run_metadata_stems = {
        path.name.removesuffix(".run.json") for path in run_metadata_shards
    }
    expected_stems = {f"{split}__{aug_id}" for aug_id in expected_aug_ids}
    if allow_extra_outputs:
        return (
            expected_stems <= parquet_stems
            and expected_stems <= logits_stems
            and expected_stems <= run_metadata_stems
        )
    return (
        parquet_stems == expected_stems
        and logits_stems == expected_stems
        and run_metadata_stems == expected_stems
    )


def _missing_teacher_cache_outputs(
    config: ExperimentConfig,
    cache_dir: Path,
    split: str,
    expected_aug_ids: tuple[str, ...],
) -> tuple[Path, ...]:
    candidates_by_id = _candidate_metadata_by_id(config)
    missing_outputs = []
    for aug_id in expected_aug_ids:
        paths = teacher_shard_paths(cache_dir, split=split, aug_id=aug_id)
        expected_paths = (paths.metadata_path, paths.logits_path, paths.run_metadata_path)
        missing_outputs.extend(path for path in expected_paths if not path.exists())
        if any(not path.exists() for path in expected_paths):
            continue
        expected_metadata = _expected_teacher_cache_metadata(
            config,
            split=split,
            aug_id=aug_id,
            candidate=candidates_by_id.get(aug_id),
        )
        if not _teacher_cache_metadata_matches(
            paths.run_metadata_path,
            expected_metadata=expected_metadata,
        ):
            missing_outputs.append(paths.run_metadata_path)
            continue
        if not shard_is_complete(
            metadata_path=paths.metadata_path,
            logits_path=paths.logits_path,
            expected_rows=_expected_teacher_cache_rows(config, split),
            expected_classes=config.dataset.class_count,
            run_metadata_path=paths.run_metadata_path,
        ):
            missing_outputs.extend((paths.metadata_path, paths.logits_path))
    return tuple(missing_outputs)


def _candidate_metadata_by_id(
    config: ExperimentConfig,
) -> dict[str, AugmentationCandidate]:
    if not config.augmentations.registry_path.exists():
        return {}
    return {
        candidate.id: candidate
        for candidate in load_augmentation_registry(config.augmentations.registry_path)
    }


def _expected_teacher_cache_metadata(
    config: ExperimentConfig,
    split: str,
    aug_id: str,
    candidate: AugmentationCandidate | None,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "version": 1,
        "split": split,
        "aug_id": aug_id,
        "seed": config.seed,
        "teacher": {
            "model_name": config.teacher.model_name,
            "pretrained": config.teacher.pretrained,
            "num_classes": config.dataset.class_count,
        },
        "storage": {
            "logits_dtype": "float16",
            "metadata_format": "parquet",
        },
    }
    if config.teacher.data_config is not None:
        expected["teacher"]["data_config"] = _json_ready(config.teacher.data_config)
    if candidate is not None:
        expected["augmentation"] = {
            "id": candidate.id,
            "name": candidate.name,
            "determinism": candidate.determinism,
            "class_name": candidate.class_name,
            "params": _json_ready(candidate.params),
        }
    return expected


def _expected_teacher_cache_rows(config: ExperimentConfig, split: str) -> int:
    per_class_by_split = {
        "public_train": config.split.public_train_per_class,
        "public_val": config.split.public_val_per_class,
        "public": config.split.public_per_class,
        "private": config.split.private_per_class,
    }
    try:
        per_class = per_class_by_split[split]
    except KeyError as error:
        raise ValueError(f"unknown split {split!r}") from error
    return config.dataset.class_count * per_class


def _teacher_cache_metadata_matches(
    path: Path,
    expected_metadata: Mapping[str, Any],
) -> bool:
    try:
        with path.open(encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(actual, Mapping):
        return False
    return _contains_subset(actual, _json_ready(dict(expected_metadata)))


def _contains_subset(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(
            key in actual and _contains_subset(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


def _extra_teacher_cache_outputs(
    cache_dir: Path,
    split: str,
    expected_aug_ids: tuple[str, ...],
) -> tuple[Path, ...]:
    expected_stems = {f"{split}__{aug_id}" for aug_id in expected_aug_ids}
    extra_outputs = []
    for path in sorted(cache_dir.glob(f"{split}__*.parquet")):
        if path.name.removesuffix(".parquet") not in expected_stems:
            extra_outputs.append(path)
    for path in sorted(cache_dir.glob(f"{split}__*.logits.npy")):
        if path.name.removesuffix(".logits.npy") not in expected_stems:
            extra_outputs.append(path)
    for path in sorted(cache_dir.glob(f"{split}__*.run.json")):
        if path.name.removesuffix(".run.json") not in expected_stems:
            extra_outputs.append(path)
    return tuple(extra_outputs)


def _step_to_dict(step: FullRunStepStatus) -> dict[str, Any]:
    return {
        "name": step.name,
        "complete": step.complete,
        "required": step.required,
        "outputs": [str(path) for path in step.outputs],
        "missing_output_count": len(step.missing_outputs),
        "missing_outputs": [str(path) for path in step.missing_outputs],
        "extra_output_count": len(step.extra_outputs),
        "extra_outputs": [str(path) for path in step.extra_outputs],
        "command": step.command,
    }
