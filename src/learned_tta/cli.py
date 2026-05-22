"""Command-line entry points for albu-tta."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from learned_tta.augmentations import load_augmentation_registry, validate_augmentation_registry
from learned_tta.config import load_experiment_config
from learned_tta.imagenet_split import (
    build_stratified_splits,
    discover_imagenet_val,
    write_split_manifests,
)
from learned_tta.private_eval import evaluate_private_from_config
from learned_tta.report_builder import build_report_from_config
from learned_tta.selector_training import train_selector_from_config
from learned_tta.target_builder import build_selector_targets_from_config
from learned_tta.teacher_cache import cache_teacher_from_config
from learned_tta.tta_tuning import tune_tta_from_config


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    command = str(args.command)

    if command == "validate-augmentations":
        _cmd_validate_augmentations(config_path=Path(args.config))
    elif command == "make-splits":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_make_splits(
            config_path=Path(args.config),
            imagenet_val_dir=Path(args.imagenet_val_dir),
            output_dir=output_dir,
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
            output_dir=output_dir,
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
            tuning_path=_optional_path(args.tuning),
            impact_targets_path=_optional_path(args.impact_targets),
            impact_manifest_path=_optional_path(args.impact_manifest),
            checkpoint_path=_optional_path(args.checkpoint),
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
    train_selector.add_argument("--output-dir")
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
    build_report.add_argument("--tuning")
    build_report.add_argument("--impact-targets")
    build_report.add_argument("--impact-manifest")
    build_report.add_argument("--checkpoint")
    build_report.add_argument("--image-size", type=int, default=224)
    build_report.add_argument("--batch-size", type=int, default=64)
    build_report.add_argument("--num-workers", type=int, default=4)
    build_report.add_argument("--device", default="cpu")
    return parser


def _cmd_validate_augmentations(config_path: Path) -> None:
    config = load_experiment_config(config_path)
    candidates = load_augmentation_registry(config.augmentations.registry_path)
    validate_augmentation_registry(
        candidates=candidates,
        expected_count=config.augmentations.candidate_count,
    )
    print(f"validated {len(candidates)} augmentation candidates")


def _cmd_make_splits(
    config_path: Path,
    imagenet_val_dir: Path,
    output_dir: Path | None,
) -> None:
    config = load_experiment_config(config_path)
    records = discover_imagenet_val(imagenet_val_dir)
    splits = build_stratified_splits(records, config.split)
    target_dir = output_dir if output_dir is not None else config.artifacts.manifests_dir
    written = write_split_manifests(splits, target_dir)
    print(f"wrote {len(written)} split manifests to {target_dir}")


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
    output_dir: Path | None,
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
        output_dir=output_dir,
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


def _cmd_evaluate_private(
    config_path: Path,
    split: str,
    manifest_path: Path | None,
    cache_dir: Path | None,
    checkpoint_path: Path | None,
    tuning_path: Path | None,
    output_dir: Path | None,
    candidate_ids: list[str] | None,
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
    tuning_path: Path | None,
    impact_targets_path: Path | None,
    impact_manifest_path: Path | None,
    checkpoint_path: Path | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    summary = build_report_from_config(
        config_path=config_path,
        report_dir=report_dir,
        private_metrics_path=private_metrics_path,
        tuning_path=tuning_path,
        impact_targets_path=impact_targets_path,
        impact_manifest_path=impact_manifest_path,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(f"report: wrote {summary.results_md}")
