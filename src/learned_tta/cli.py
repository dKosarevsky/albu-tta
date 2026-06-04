"""Command-line entry points for albu-tta."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from learned_tta.augmentations import (
    load_augmentation_registry,
    validate_augmentation_registry,
    write_augmentation_audit,
)
from learned_tta.config import load_experiment_config
from learned_tta.imagenet_split import (
    build_stratified_splits,
    discover_imagenet_val,
    load_class_to_idx,
    write_class_mapping,
    write_split_manifests,
)
from learned_tta.preflight import run_full_run_preflight
from learned_tta.private_eval import evaluate_private_from_config
from learned_tta.report_builder import build_report_from_config
from learned_tta.run_status import full_run_status_to_dict, inspect_full_run_status
from learned_tta.run_supervisor import run_next_full_run_step
from learned_tta.selector_training import train_selector_from_config
from learned_tta.smoke import run_smoke_e2e
from learned_tta.stacking import train_aggregator_from_config
from learned_tta.target_builder import build_selector_targets_from_config
from learned_tta.teacher_cache import cache_teacher_from_config
from learned_tta.tta_tuning import tune_tta_from_config


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)

    if command == "validate-augmentations":
        _cmd_validate_augmentations(
            config_path=Path(args.config),
            audit_output=_optional_path(args.audit_output),
        )
    elif command == "run-smoke":
        _cmd_run_smoke(
            config_path=Path(args.config),
            output_dir=Path(args.output_dir),
            candidate_count=int(args.candidate_count),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            epochs=int(args.epochs),
            device=str(args.device),
        )
    elif command == "make-splits":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_make_splits(
            config_path=Path(args.config),
            imagenet_val_dir=Path(args.imagenet_val_dir),
            output_dir=output_dir,
        )
    elif command == "check-full-run":
        _cmd_check_full_run(
            config_path=Path(args.config),
            imagenet_val_dir=Path(args.imagenet_val_dir),
        )
    elif command == "full-run-status":
        _cmd_full_run_status(
            config_path=Path(args.config),
            output_format=str(args.format),
            fail_on_incomplete=bool(args.fail_on_incomplete),
            next_command=bool(args.next_command),
        )
    elif command == "resume-full-run":
        _cmd_resume_full_run(
            config_path=Path(args.config),
            imagenet_val_dir=_optional_path(args.imagenet_val_dir),
            cache_log_dir=_optional_path(args.cache_log_dir),
            dry_run=bool(args.dry_run),
            background_cache=not bool(args.foreground_cache),
            allow_duplicate_cache=bool(args.allow_duplicate_cache),
        )
    elif command == "cache-teacher":
        manifest_path = Path(args.manifest) if args.manifest is not None else None
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_cache_teacher(
            config_path=Path(args.config),
            split=str(args.split),
            manifest_path=manifest_path,
            output_dir=output_dir,
            candidate_ids=args.candidate_id,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            resume=not bool(args.no_resume),
            device=str(args.device),
        )
    elif command == "build-targets":
        cache_dir = Path(args.cache_dir) if args.cache_dir is not None else None
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_build_targets(
            config_path=Path(args.config),
            cache_dir=cache_dir,
            output_dir=output_dir,
            train_split=str(args.train_split),
            val_split=str(args.val_split),
            candidate_ids=args.candidate_id,
        )
    elif command == "train-selector":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_train_selector(
            config_path=Path(args.config),
            train_manifest_path=_optional_path(args.train_manifest),
            val_manifest_path=_optional_path(args.val_manifest),
            train_targets_path=_optional_path(args.train_targets),
            val_targets_path=_optional_path(args.val_targets),
            cache_dir=_optional_path(args.cache_dir),
            output_dir=output_dir,
            val_split=str(args.val_split),
            candidate_ids=args.candidate_id,
            top_k_grid=args.top_k,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            rank_weight=float(args.rank_weight),
            device=str(args.device),
        )
    elif command == "tune-tta":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_tune_tta(
            config_path=Path(args.config),
            split=str(args.split),
            manifest_path=_optional_path(args.manifest),
            cache_dir=_optional_path(args.cache_dir),
            checkpoint_path=_optional_path(args.checkpoint),
            output_dir=output_dir,
            candidate_ids=args.candidate_id,
            top_k_grid=args.top_k,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
        )
    elif command == "train-aggregator":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_train_aggregator(
            config_path=Path(args.config),
            split=str(args.split),
            cache_dir=_optional_path(args.cache_dir),
            output_dir=output_dir,
            output_path=_optional_path(args.output_path),
            candidate_ids=args.candidate_id,
            method=str(args.method),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            l1_penalty=float(args.l1_penalty),
            active_threshold=float(args.active_threshold),
            device=str(args.device),
        )
    elif command == "evaluate-private":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_evaluate_private(
            config_path=Path(args.config),
            split=str(args.split),
            manifest_path=_optional_path(args.manifest),
            cache_dir=_optional_path(args.cache_dir),
            checkpoint_path=_optional_path(args.checkpoint),
            tuning_path=_optional_path(args.tuning),
            output_dir=output_dir,
            candidate_ids=args.candidate_id,
            global_aggregator_path=_optional_path(args.global_aggregator),
            class_aggregator_path=_optional_path(args.class_aggregator),
            xgboost_aggregator_path=_optional_path(args.xgboost_aggregator),
            random_seeds=args.random_seed,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
        )
    elif command == "build-report":
        report_dir = Path(args.report_dir) if args.report_dir is not None else None
        _cmd_build_report(
            config_path=Path(args.config),
            report_dir=report_dir,
            private_metrics_path=_optional_path(args.private_metrics),
            corrections_path=_optional_path(args.corrections),
            selector_history_path=_optional_path(args.selector_history),
            tuning_path=_optional_path(args.tuning),
            impact_targets_path=_optional_path(args.impact_targets),
            impact_manifest_path=_optional_path(args.impact_manifest),
            checkpoint_path=_optional_path(args.checkpoint),
            global_aggregator_path=_optional_path(args.global_aggregator),
            class_aggregator_path=_optional_path(args.class_aggregator),
            xgboost_aggregator_path=_optional_path(args.xgboost_aggregator),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
        )
    else:
        parser.error(f"unknown command {command!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="learned-tta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-augmentations",
        help="Validate the configured AlbumentationsX candidate registry.",
    )
    validate.add_argument("--config", required=True, help="Path to experiment YAML config.")
    validate.add_argument(
        "--audit-output",
        help="Optional path for a stable JSON audit of the loaded augmentation registry.",
    )

    run_smoke = subparsers.add_parser(
        "run-smoke",
        help="Run a tiny synthetic end-to-end pipeline without loading timm or ImageNet.",
    )
    run_smoke.add_argument("--config", required=True, help="Path to experiment YAML config.")
    run_smoke.add_argument(
        "--output-dir",
        required=True,
        help="Directory for synthetic smoke artifacts.",
    )
    run_smoke.add_argument("--candidate-count", type=int, default=3)
    run_smoke.add_argument("--image-size", type=int, default=16)
    run_smoke.add_argument("--batch-size", type=int, default=2)
    run_smoke.add_argument("--num-workers", type=int, default=0)
    run_smoke.add_argument("--epochs", type=int, default=1)
    run_smoke.add_argument("--device", default="cpu")

    make_splits = subparsers.add_parser(
        "make-splits",
        help="Create stratified ImageNet validation split manifests.",
    )
    make_splits.add_argument("--config", required=True, help="Path to experiment YAML config.")
    make_splits.add_argument(
        "--imagenet-val-dir",
        required=True,
        help="Path to ImageNet validation directory laid out as val/class_name/image.JPEG.",
    )
    make_splits.add_argument(
        "--output-dir",
        help="Manifest output directory. Defaults to artifacts.manifests_dir from config.",
    )

    check_full_run = subparsers.add_parser(
        "check-full-run",
        help="Validate full ImageNet-run prerequisites without launching inference.",
    )
    check_full_run.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    check_full_run.add_argument(
        "--imagenet-val-dir",
        required=True,
        help="Path to ImageNet validation directory laid out as val/class_name/image.JPEG.",
    )

    full_run_status = subparsers.add_parser(
        "full-run-status",
        help="Inspect full ImageNet-run artifacts and print the next missing step.",
    )
    full_run_status.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    full_run_status.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the artifact status summary.",
    )
    full_run_status.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit with code 1 when any required full-run step is incomplete.",
    )
    full_run_status.add_argument(
        "--next-command",
        action="store_true",
        help="Print only the next required full-run command, if any.",
    )

    resume_full_run = subparsers.add_parser(
        "resume-full-run",
        help="Run the next missing full-run step with Colab-safe cache supervision.",
    )
    resume_full_run.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    resume_full_run.add_argument(
        "--imagenet-val-dir",
        help=(
            "ImageNet validation directory used when the next command is make-splits. "
            "Required only while split manifests are missing."
        ),
    )
    resume_full_run.add_argument(
        "--cache-log-dir",
        help=(
            "Directory for background cache-teacher logs. Defaults to artifacts/logs "
            "under the project root."
        ),
    )
    resume_full_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the supervised action without launching it.",
    )
    resume_full_run.add_argument(
        "--foreground-cache",
        action="store_true",
        help="Run cache-teacher in the foreground instead of detaching it.",
    )
    resume_full_run.add_argument(
        "--allow-duplicate-cache",
        action="store_true",
        help="Start cache-teacher even if another matching cache process is active.",
    )

    cache_teacher = subparsers.add_parser(
        "cache-teacher",
        help="Run teacher inference and write per-augmentation cache shards.",
    )
    cache_teacher.add_argument("--config", required=True, help="Path to experiment YAML config.")
    cache_teacher.add_argument("--split", required=True, help="Split name to cache.")
    cache_teacher.add_argument(
        "--manifest",
        help="Split manifest CSV. Defaults to artifacts.manifests_dir/{split}.csv.",
    )
    cache_teacher.add_argument(
        "--output-dir",
        help="Teacher cache output directory. Defaults to artifacts.teacher_cache_dir.",
    )
    cache_teacher.add_argument(
        "--candidate-id",
        action="append",
        help="Augmentation candidate id to cache. May be passed more than once.",
    )
    cache_teacher.add_argument("--batch-size", type=int, default=64)
    cache_teacher.add_argument("--num-workers", type=int, default=4)
    cache_teacher.add_argument("--device", default="cpu")
    cache_teacher.add_argument("--no-resume", action="store_true")

    build_targets = subparsers.add_parser(
        "build-targets",
        help="Build selector target artifacts from teacher cache shards.",
    )
    build_targets.add_argument("--config", required=True, help="Path to experiment YAML config.")
    build_targets.add_argument(
        "--cache-dir",
        help="Teacher cache directory. Defaults to artifacts.teacher_cache_dir.",
    )
    build_targets.add_argument(
        "--output-dir",
        help="Selector output directory. Defaults to artifacts.selector_dir.",
    )
    build_targets.add_argument("--train-split", default="public_train")
    build_targets.add_argument("--val-split", default="public_val")
    build_targets.add_argument(
        "--candidate-id",
        action="append",
        help="Augmentation candidate id to include. May be passed more than once.",
    )

    train_selector = subparsers.add_parser(
        "train-selector",
        help="Train the small selector CNN from clean images and selector targets.",
    )
    train_selector.add_argument("--config", required=True, help="Path to experiment YAML config.")
    train_selector.add_argument("--train-manifest")
    train_selector.add_argument("--val-manifest")
    train_selector.add_argument("--train-targets")
    train_selector.add_argument("--val-targets")
    train_selector.add_argument("--cache-dir")
    train_selector.add_argument("--output-dir")
    train_selector.add_argument("--val-split", default="public_val")
    train_selector.add_argument("--candidate-id", action="append")
    train_selector.add_argument("--top-k", type=int, action="append")
    train_selector.add_argument("--image-size", type=int, default=224)
    train_selector.add_argument("--batch-size", type=int, default=64)
    train_selector.add_argument("--num-workers", type=int, default=4)
    train_selector.add_argument("--epochs", type=int, default=20)
    train_selector.add_argument("--learning-rate", type=float, default=1e-3)
    train_selector.add_argument("--rank-weight", type=float, default=0.2)
    train_selector.add_argument("--device", default="cpu")

    tune_tta = subparsers.add_parser(
        "tune-tta",
        help="Tune learned TTA top-k on a validation split.",
    )
    tune_tta.add_argument("--config", required=True, help="Path to experiment YAML config.")
    tune_tta.add_argument("--split", default="public_val")
    tune_tta.add_argument("--manifest")
    tune_tta.add_argument("--cache-dir")
    tune_tta.add_argument("--checkpoint")
    tune_tta.add_argument("--output-dir")
    tune_tta.add_argument("--candidate-id", action="append")
    tune_tta.add_argument("--top-k", type=int, action="append")
    tune_tta.add_argument("--image-size", type=int, default=224)
    tune_tta.add_argument("--batch-size", type=int, default=64)
    tune_tta.add_argument("--num-workers", type=int, default=4)
    tune_tta.add_argument("--device", default="cpu")

    train_aggregator = subparsers.add_parser(
        "train-aggregator",
        help="Train learned non-negative TTA aggregation weights from cached logits.",
    )
    train_aggregator.add_argument("--config", required=True, help="Path to experiment YAML config.")
    train_aggregator.add_argument("--split", default="public_val")
    train_aggregator.add_argument("--cache-dir")
    train_aggregator.add_argument("--output-dir")
    train_aggregator.add_argument("--output-path")
    train_aggregator.add_argument("--candidate-id", action="append")
    train_aggregator.add_argument(
        "--method",
        choices=["global-nonnegative", "class-nonnegative", "xgboost-multiclass"],
        default="global-nonnegative",
    )
    train_aggregator.add_argument("--epochs", type=int, default=200)
    train_aggregator.add_argument("--learning-rate", type=float, default=0.05)
    train_aggregator.add_argument(
        "--l1-penalty",
        type=float,
        default=0.0,
        help=(
            "Sparsity regularization strength for normalized non-negative weights. "
            "Kept under the original name for CLI compatibility."
        ),
    )
    train_aggregator.add_argument(
        "--active-threshold",
        type=float,
        default=1e-6,
        help="Weights at or below this threshold are pruned after aggregation training.",
    )
    train_aggregator.add_argument("--device", default="cpu")

    evaluate_private = subparsers.add_parser(
        "evaluate-private",
        help="Evaluate frozen learned TTA and baselines on the private split.",
    )
    evaluate_private.add_argument("--config", required=True, help="Path to experiment YAML config.")
    evaluate_private.add_argument("--split", default="private")
    evaluate_private.add_argument("--manifest")
    evaluate_private.add_argument("--cache-dir")
    evaluate_private.add_argument("--checkpoint")
    evaluate_private.add_argument("--tuning")
    evaluate_private.add_argument("--global-aggregator")
    evaluate_private.add_argument("--class-aggregator")
    evaluate_private.add_argument("--xgboost-aggregator")
    evaluate_private.add_argument("--output-dir")
    evaluate_private.add_argument("--candidate-id", action="append")
    evaluate_private.add_argument("--random-seed", type=int, action="append")
    evaluate_private.add_argument("--image-size", type=int, default=224)
    evaluate_private.add_argument("--batch-size", type=int, default=64)
    evaluate_private.add_argument("--num-workers", type=int, default=4)
    evaluate_private.add_argument("--device", default="cpu")

    build_report = subparsers.add_parser(
        "build-report",
        help="Build final markdown, tables, and SVG figures from experiment artifacts.",
    )
    build_report.add_argument("--config", required=True, help="Path to experiment YAML config.")
    build_report.add_argument("--report-dir")
    build_report.add_argument("--private-metrics")
    build_report.add_argument("--corrections")
    build_report.add_argument("--selector-history")
    build_report.add_argument("--tuning")
    build_report.add_argument("--impact-targets")
    build_report.add_argument("--impact-manifest")
    build_report.add_argument("--checkpoint")
    build_report.add_argument("--global-aggregator")
    build_report.add_argument("--class-aggregator")
    build_report.add_argument("--xgboost-aggregator")
    build_report.add_argument("--image-size", type=int, default=224)
    build_report.add_argument("--batch-size", type=int, default=64)
    build_report.add_argument("--num-workers", type=int, default=4)
    build_report.add_argument("--device", default="cpu")
    return parser


def _cmd_validate_augmentations(config_path: Path, audit_output: Path | None) -> None:
    config = load_experiment_config(config_path)
    candidates = load_augmentation_registry(config.augmentations.registry_path)
    validate_augmentation_registry(
        candidates=candidates,
        expected_count=config.augmentations.candidate_count,
    )
    message = f"validated {len(candidates)} augmentation candidates"
    if audit_output is not None:
        written = write_augmentation_audit(candidates, audit_output, seed=config.seed)
        message = f"{message}; wrote audit {written}"
    print(message)


def _cmd_run_smoke(
    config_path: Path,
    output_dir: Path,
    candidate_count: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    device: str,
) -> None:
    summary = run_smoke_e2e(
        config_path=config_path,
        output_dir=output_dir,
        candidate_count=candidate_count,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        device=device,
    )
    print(f"smoke run: wrote {summary.results_md}")


def _cmd_make_splits(
    config_path: Path,
    imagenet_val_dir: Path,
    output_dir: Path | None,
) -> None:
    config = load_experiment_config(config_path)
    class_to_idx = load_class_to_idx(config.dataset.class_index, config.project_root)
    records = discover_imagenet_val(imagenet_val_dir, class_to_idx=class_to_idx)
    splits = build_stratified_splits(records, config.split)
    target_dir = output_dir if output_dir is not None else config.artifacts.manifests_dir
    written = write_split_manifests(splits, target_dir)
    mapping_path = write_class_mapping(class_to_idx, Path(target_dir) / "class_to_idx.json")
    print(f"wrote {len(written)} split manifests to {target_dir}; wrote {mapping_path}")


def _cmd_check_full_run(config_path: Path, imagenet_val_dir: Path) -> None:
    summary = run_full_run_preflight(
        config_path=config_path,
        imagenet_val_dir=imagenet_val_dir,
    )
    print(
        "full run preflight ok: "
        f"classes={summary.class_count}, "
        f"images={summary.image_count}, "
        f"candidates={summary.candidate_count}, "
        f"teacher={summary.teacher_model_name}"
    )


def _cmd_full_run_status(
    config_path: Path,
    output_format: str,
    fail_on_incomplete: bool,
    next_command: bool,
) -> None:
    summary = inspect_full_run_status(config_path)
    if next_command:
        if summary.next_step is not None:
            print(summary.next_step.command)
            if fail_on_incomplete:
                raise SystemExit(1)
        return

    if output_format == "json":
        print(json.dumps(full_run_status_to_dict(summary), indent=2, sort_keys=True))
        if fail_on_incomplete and summary.next_step is not None:
            raise SystemExit(1)
        return

    print(
        "full run status: "
        f"{summary.completed_required_steps}/{summary.total_required_steps} "
        "required steps complete "
        f"({summary.completed_steps}/{summary.total_steps} total)"
    )
    for step in summary.steps:
        marker = "x" if step.complete else " "
        label = "optional: " if not step.required else ""
        diagnostics = (
            f" missing={len(step.missing_outputs)} extra={len(step.extra_outputs)}"
            if not step.complete
            else ""
        )
        print(f"[{marker}] {label}{step.name}{diagnostics}")
    if summary.next_step is None:
        print("next: none")
    else:
        print(f"next: {summary.next_step.name}")
        print(f"command: {summary.next_step.command}")
    if fail_on_incomplete and summary.next_step is not None:
        raise SystemExit(1)


def _cmd_resume_full_run(
    config_path: Path,
    imagenet_val_dir: Path | None,
    cache_log_dir: Path | None,
    dry_run: bool,
    background_cache: bool,
    allow_duplicate_cache: bool,
) -> None:
    result = run_next_full_run_step(
        config_path=config_path,
        imagenet_val_dir=imagenet_val_dir,
        cache_log_dir=cache_log_dir,
        dry_run=dry_run,
        background_cache=background_cache,
        allow_duplicate_cache=allow_duplicate_cache,
    )

    if result.status == "complete":
        print("full run complete: no required steps left")
    elif result.status == "dry-run":
        print(f"dry-run: {result.step_name}")
        print(result.command)
    elif result.status == "active":
        print(f"cache already active: {result.step_name}")
        for process in result.active_processes:
            print(process)
        print("not starting a duplicate process")
    elif result.status == "started":
        print(f"started background step: {result.step_name}")
        print(f"pid: {result.pid}")
        print(f"log: {result.log_path}")
        print(result.command)
    elif result.status == "completed":
        print(f"completed step: {result.step_name}")
        print(result.command)
    else:
        raise ValueError(f"unknown resume result status: {result.status}")


def _cmd_cache_teacher(
    config_path: Path,
    split: str,
    manifest_path: Path | None,
    output_dir: Path | None,
    candidate_ids: list[str] | None,
    batch_size: int,
    num_workers: int,
    resume: bool,
    device: str,
) -> None:
    summary = cache_teacher_from_config(
        config_path=config_path,
        split=split,
        manifest_path=manifest_path,
        output_dir=output_dir,
        candidate_ids=candidate_ids,
        batch_size=batch_size,
        num_workers=num_workers,
        resume=resume,
        device=device,
    )
    print(
        f"teacher cache {summary.split}: wrote {_plural(len(summary.written), 'shard')}, "
        f"skipped {_plural(len(summary.skipped), 'shard')}"
    )


def _plural(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def _cmd_build_targets(
    config_path: Path,
    cache_dir: Path | None,
    output_dir: Path | None,
    train_split: str,
    val_split: str,
    candidate_ids: list[str] | None,
) -> None:
    summary = build_selector_targets_from_config(
        config_path=config_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        candidate_ids=candidate_ids,
    )
    print(
        f"selector targets: wrote {summary.train_path.name} and {summary.val_path.name} "
        f"for {_plural(len(summary.aug_ids), 'augmentation')}"
    )


def _cmd_train_selector(
    config_path: Path,
    train_manifest_path: Path | None,
    val_manifest_path: Path | None,
    train_targets_path: Path | None,
    val_targets_path: Path | None,
    cache_dir: Path | None,
    output_dir: Path | None,
    val_split: str,
    candidate_ids: list[str] | None,
    top_k_grid: list[int] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    epochs: int,
    learning_rate: float,
    rank_weight: float,
    device: str,
) -> None:
    summary = train_selector_from_config(
        config_path=config_path,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        train_targets_path=train_targets_path,
        val_targets_path=val_targets_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        val_split=val_split,
        candidate_ids=candidate_ids,
        top_k_grid=top_k_grid,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=learning_rate,
        rank_weight=rank_weight,
        device=device,
    )
    print(
        f"selector training: best epoch {summary.best_epoch}, "
        f"best val nll {summary.best_val_nll:.6g}, "
        f"best val loss {summary.best_val_loss:.6g}, checkpoint {summary.checkpoint_path}"
    )


def _optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _cmd_tune_tta(
    config_path: Path,
    split: str,
    manifest_path: Path | None,
    cache_dir: Path | None,
    checkpoint_path: Path | None,
    output_dir: Path | None,
    candidate_ids: list[str] | None,
    top_k_grid: list[int] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    summary = tune_tta_from_config(
        config_path=config_path,
        split=split,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        candidate_ids=candidate_ids,
        top_k_grid=top_k_grid,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(
        f"tta tuning {summary.split}: best k {summary.best_k}, "
        f"wrote {summary.result_path}"
    )


def _cmd_train_aggregator(
    config_path: Path,
    split: str,
    cache_dir: Path | None,
    output_dir: Path | None,
    output_path: Path | None,
    candidate_ids: list[str] | None,
    method: str,
    epochs: int,
    learning_rate: float,
    l1_penalty: float,
    active_threshold: float,
    device: str,
) -> None:
    summary = train_aggregator_from_config(
        config_path=config_path,
        split=split,
        cache_dir=cache_dir,
        output_dir=output_dir,
        output_path=output_path,
        candidate_ids=candidate_ids,
        method=method,
        epochs=epochs,
        learning_rate=learning_rate,
        l1_penalty=l1_penalty,
        active_threshold=active_threshold,
        device=device,
    )
    print(f"aggregator {summary.method}: wrote {summary.path}")


def _cmd_evaluate_private(
    config_path: Path,
    split: str,
    manifest_path: Path | None,
    cache_dir: Path | None,
    checkpoint_path: Path | None,
    tuning_path: Path | None,
    output_dir: Path | None,
    candidate_ids: list[str] | None,
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
    xgboost_aggregator_path: Path | None,
    random_seeds: list[int] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    summary = evaluate_private_from_config(
        config_path=config_path,
        split=split,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        tuning_path=tuning_path,
        output_dir=output_dir,
        candidate_ids=candidate_ids,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
        random_seeds=random_seeds,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(
        f"private evaluation: best k {summary.best_k}, "
        f"wrote {summary.private_metrics_csv}"
    )


def _cmd_build_report(
    config_path: Path,
    report_dir: Path | None,
    private_metrics_path: Path | None,
    corrections_path: Path | None,
    selector_history_path: Path | None,
    tuning_path: Path | None,
    impact_targets_path: Path | None,
    impact_manifest_path: Path | None,
    checkpoint_path: Path | None,
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
    xgboost_aggregator_path: Path | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    summary = build_report_from_config(
        config_path=config_path,
        report_dir=report_dir,
        private_metrics_path=private_metrics_path,
        corrections_path=corrections_path,
        selector_history_path=selector_history_path,
        tuning_path=tuning_path,
        impact_targets_path=impact_targets_path,
        impact_manifest_path=impact_manifest_path,
        checkpoint_path=checkpoint_path,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(f"report: wrote {summary.results_md}")


if __name__ == "__main__":
    main()
