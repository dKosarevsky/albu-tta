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
from learned_tta.target_builder import build_selector_targets_from_config
from learned_tta.teacher_cache import cache_teacher_from_config


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
