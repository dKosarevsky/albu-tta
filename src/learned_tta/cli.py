"""Command-line entry points for albu-tta."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from learned_tta.clean_baseline import (
    check_clean_baseline_from_config,
    summarize_clean_center_crop_baseline_from_config,
)
from learned_tta.config import load_experiment_config
from learned_tta.imagenet_prepare import prepare_imagenet_val
from learned_tta.imagenet_split import (
    build_stratified_splits,
    discover_imagenet_val,
    load_class_to_idx,
    write_class_mapping,
    write_split_manifests,
)
from learned_tta.inference_backends import (
    build_teacher_backend_plan,
    teacher_backend_plan_to_dict,
)
from learned_tta.targets import TRAINABLE_SELECTOR_TARGET_KINDS


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
    elif command == "prepare-imagenet-val":
        _cmd_prepare_imagenet_val(
            config_path=Path(args.config),
            val_tar_path=Path(args.val_tar),
            output_dir=Path(args.output_dir),
            devkit_path=_optional_path(args.devkit),
            ground_truth_path=_optional_path(args.ground_truth),
            audit_output_path=_optional_path(args.audit_output),
            overwrite=bool(args.overwrite),
        )
    elif command == "check-full-run":
        _cmd_check_full_run(
            config_path=Path(args.config),
            imagenet_val_dir=Path(args.imagenet_val_dir),
        )
    elif command == "check-clean-baseline":
        _cmd_check_clean_baseline(
            config_path=Path(args.config),
            split=args.split,
            cache_dir=_optional_path(args.cache_dir),
            output_path=_optional_path(args.output_path),
            min_top1=args.min_top1,
            min_top5=args.min_top5,
            max_nll=args.max_nll,
        )
    elif command == "summarize-clean-baseline":
        _cmd_summarize_clean_baseline(
            config_path=Path(args.config),
            splits=args.split,
            cache_dir=_optional_path(args.cache_dir),
            output_path=_optional_path(args.output_path),
        )
    elif command == "full-run-status":
        _cmd_full_run_status(
            config_path=Path(args.config),
            output_format=str(args.format),
            fail_on_incomplete=bool(args.fail_on_incomplete),
            next_command=bool(args.next_command),
        )
    elif command == "teacher-cache-plan":
        _cmd_teacher_cache_plan(
            config_path=Path(args.config),
            cache_dir=_optional_path(args.cache_dir),
            splits=tuple(args.split) if args.split is not None else None,
            output_format=str(args.format),
        )
    elif command == "teacher-backend-plan":
        _cmd_teacher_backend_plan(
            device=str(args.device),
            output_format=str(args.format),
        )
    elif command == "teacher-cache-diagnostics":
        _cmd_teacher_cache_diagnostics(
            config_path=Path(args.config),
            split=str(args.split),
            cache_dir=_optional_path(args.cache_dir),
            candidate_ids=args.candidate_id,
            top_n=int(args.top_n),
            output_path=_optional_path(args.output),
            output_format=str(args.format),
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
            backend=str(args.backend),
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
            target_kind=str(args.target_kind),
        )
    elif command == "build-selector-features":
        _cmd_build_selector_features(
            config_path=Path(args.config),
            split=str(args.split),
            manifest_path=_optional_path(args.manifest),
            output_path=_optional_path(args.output),
            model_name=str(args.model_name) if args.model_name is not None else None,
            pretrained=bool(args.pretrained),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=str(args.device),
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
            usefulness_head=args.usefulness_head,
            usefulness_tau=args.usefulness_tau,
            usefulness_weight=args.usefulness_weight,
            device=str(args.device),
        )
    elif command == "train-selector-ablation":
        output_dir = Path(args.output_dir) if args.output_dir is not None else None
        _cmd_train_selector_ablation(
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
            device=str(args.device),
            variant_names=tuple(args.ablation_variant) if args.ablation_variant else None,
            train_features_path=_optional_path(args.train_features),
            val_features_path=_optional_path(args.val_features),
            force=bool(args.force),
        )
    elif command == "train-pairwise-selector":
        _cmd_train_pairwise_selector(
            train_manifest_path=Path(args.train_manifest),
            val_manifest_path=Path(args.val_manifest),
            train_targets_path=Path(args.train_targets),
            val_targets_path=Path(args.val_targets),
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            identity_aug_id=str(args.identity_aug_id),
            train_features_path=_optional_path(args.train_features),
            val_features_path=_optional_path(args.val_features),
            top_k_grid=args.top_k,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            hidden_dim=int(args.hidden_dim),
            usefulness_tau=float(args.usefulness_tau),
            usefulness_weight=float(args.usefulness_weight),
            positive_gain_weight=float(args.positive_gain_weight),
            target_mode=str(args.target_mode),
            selection_metric=str(args.selection_metric),
            device=str(args.device),
        )
    elif command == "train-pairwise-selector-comparison":
        _cmd_train_pairwise_selector_comparison(
            train_manifest_path=Path(args.train_manifest),
            val_manifest_path=Path(args.val_manifest),
            train_targets_path=Path(args.train_targets),
            val_targets_path=Path(args.val_targets),
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            identity_aug_id=str(args.identity_aug_id),
            train_features_path=_optional_path(args.train_features),
            val_features_path=_optional_path(args.val_features),
            top_k_grid=args.top_k,
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            hidden_dim=int(args.hidden_dim),
            usefulness_tau=float(args.usefulness_tau),
            usefulness_weight=float(args.usefulness_weight),
            positive_gain_weight=float(args.positive_gain_weight),
            device=str(args.device),
        )
    elif command == "evaluate-pairwise-selector":
        _cmd_evaluate_pairwise_selector(
            manifest_path=Path(args.manifest),
            cache_dir=Path(args.cache_dir),
            checkpoint_path=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            identity_aug_id=str(args.identity_aug_id),
            features_path=_optional_path(args.features),
            top_k=int(args.top_k),
            batch_size=int(args.batch_size),
            strategy_name=str(args.strategy_name),
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
            adaptive_threshold_grid=args.adaptive_threshold,
            adaptive_max_k_grid=args.adaptive_max_k,
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
            selector_ablation_path=_optional_path(args.selector_ablation),
            selector_diagnostics_path=_optional_path(args.selector_diagnostics),
            adaptive_selection_counts_path=_optional_path(args.adaptive_selection_counts),
            compute_policy_frontier_path=_optional_path(args.compute_policy_frontier),
            pairwise_selector_comparison_path=_optional_path(args.pairwise_selector_comparison),
            selector_error_analysis_path=_optional_path(args.selector_error_analysis),
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

    prepare_imagenet_val = subparsers.add_parser(
        "prepare-imagenet-val",
        help="Prepare official ImageNet validation tar into val/WNID/*.JPEG layout.",
    )
    prepare_imagenet_val.add_argument("--config", required=True, help="Path to experiment YAML.")
    prepare_imagenet_val.add_argument(
        "--val-tar",
        required=True,
        help="Path to local ILSVRC2012_img_val.tar.",
    )
    prepare_imagenet_val.add_argument(
        "--output-dir",
        required=True,
        help="Output validation directory to create, e.g. /datasets/imagenet/val.",
    )
    prepare_imagenet_val.add_argument(
        "--devkit",
        help="Path to local ILSVRC2012_devkit_t12.tar.gz or extracted devkit directory.",
    )
    prepare_imagenet_val.add_argument(
        "--ground-truth",
        help="Path to ILSVRC2012_validation_ground_truth.txt. Overrides --devkit.",
    )
    prepare_imagenet_val.add_argument(
        "--audit-output",
        help="Optional JSON audit path. Defaults to OUTPUT_DIR/_preparation_audit.json.",
    )
    prepare_imagenet_val.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
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

    check_clean_baseline = subparsers.add_parser(
        "check-clean-baseline",
        help="Validate clean identity-cache metrics before full teacher caching.",
    )
    check_clean_baseline.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    check_clean_baseline.add_argument("--split")
    check_clean_baseline.add_argument("--cache-dir")
    check_clean_baseline.add_argument("--output-path")
    check_clean_baseline.add_argument("--min-top1", type=float)
    check_clean_baseline.add_argument("--min-top5", type=float)
    check_clean_baseline.add_argument("--max-nll", type=float)

    summarize_clean_baseline = subparsers.add_parser(
        "summarize-clean-baseline",
        help="Summarize clean CenterCrop metrics over identity-cache validation shards.",
    )
    summarize_clean_baseline.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    summarize_clean_baseline.add_argument(
        "--split",
        action="append",
        help=(
            "Split to include. Repeatable. Defaults to public_train, public_val, "
            "and private, which covers ImageNet-val once."
        ),
    )
    summarize_clean_baseline.add_argument("--cache-dir")
    summarize_clean_baseline.add_argument("--output-path")

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

    teacher_cache_plan = subparsers.add_parser(
        "teacher-cache-plan",
        help="Summarize expected teacher-cache logits work and current shard progress.",
    )
    teacher_cache_plan.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    teacher_cache_plan.add_argument(
        "--cache-dir",
        help="Teacher cache directory. Defaults to artifacts.teacher_cache_dir.",
    )
    teacher_cache_plan.add_argument(
        "--split",
        action="append",
        help=(
            "Split to include. Repeatable. Defaults to public_train, public_val, "
            "and private, which covers the 5M logits run once."
        ),
    )
    teacher_cache_plan.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the teacher-cache plan.",
    )

    teacher_backend_plan = subparsers.add_parser(
        "teacher-backend-plan",
        help="Show implemented and planned teacher inference backends.",
    )
    teacher_backend_plan.add_argument(
        "--device",
        default="cuda",
        help="Target device family for recommendations, e.g. cuda or cpu.",
    )
    teacher_backend_plan.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for the teacher backend plan.",
    )

    teacher_cache_diagnostics = subparsers.add_parser(
        "teacher-cache-diagnostics",
        help="Summarize completed teacher-cache metadata for a split without loading logits.",
    )
    teacher_cache_diagnostics.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    teacher_cache_diagnostics.add_argument("--split", default="public_val")
    teacher_cache_diagnostics.add_argument(
        "--cache-dir",
        help="Teacher cache directory. Defaults to artifacts.teacher_cache_dir.",
    )
    teacher_cache_diagnostics.add_argument(
        "--candidate-id",
        action="append",
        help="Augmentation candidate id to include. May be passed more than once.",
    )
    teacher_cache_diagnostics.add_argument("--top-n", type=int, default=10)
    teacher_cache_diagnostics.add_argument("--output", help="Optional JSON output path.")
    teacher_cache_diagnostics.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for diagnostics.",
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
    cache_teacher.add_argument(
        "--backend",
        default="pytorch",
        help=(
            "Teacher inference backend. Only pytorch is implemented; TensorRT, "
            "ONNXRuntime, and OpenVINO are documented planned accelerators."
        ),
    )
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
        "--target-kind",
        choices=TRAINABLE_SELECTOR_TARGET_KINDS,
        default="gain",
        help=(
            "High-is-better selector target to train against. Raw nll stays diagnostic-only; "
            "use negative_nll for a trainable loss-based target."
        ),
    )
    build_targets.add_argument(
        "--candidate-id",
        action="append",
        help="Augmentation candidate id to include. May be passed more than once.",
    )

    build_selector_features = subparsers.add_parser(
        "build-selector-features",
        help="Build cached pretrained image features for selector MLP training.",
    )
    build_selector_features.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    build_selector_features.add_argument("--split", default="public_train")
    build_selector_features.add_argument("--manifest")
    build_selector_features.add_argument("--output")
    build_selector_features.add_argument("--model-name")
    build_selector_features.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained weights for smoke/debug feature extraction.",
    )
    build_selector_features.set_defaults(pretrained=True)
    build_selector_features.add_argument("--batch-size", type=int, default=64)
    build_selector_features.add_argument("--num-workers", type=int, default=4)
    build_selector_features.add_argument("--device", default="cpu")

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
    usefulness_group = train_selector.add_mutually_exclusive_group()
    usefulness_group.add_argument("--usefulness-head", dest="usefulness_head", action="store_true")
    usefulness_group.add_argument(
        "--no-usefulness-head",
        dest="usefulness_head",
        action="store_false",
    )
    train_selector.set_defaults(usefulness_head=None)
    train_selector.add_argument("--usefulness-tau", type=float)
    train_selector.add_argument("--usefulness-weight", type=float)
    train_selector.add_argument("--device", default="cpu")

    train_selector_ablation = subparsers.add_parser(
        "train-selector-ablation",
        help="Train selector loss ablation variants.",
    )
    train_selector_ablation.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )
    train_selector_ablation.add_argument("--train-manifest")
    train_selector_ablation.add_argument("--val-manifest")
    train_selector_ablation.add_argument("--train-targets")
    train_selector_ablation.add_argument("--val-targets")
    train_selector_ablation.add_argument("--train-features")
    train_selector_ablation.add_argument("--val-features")
    train_selector_ablation.add_argument("--cache-dir")
    train_selector_ablation.add_argument("--output-dir")
    train_selector_ablation.add_argument("--val-split", default="public_val")
    train_selector_ablation.add_argument("--candidate-id", action="append")
    train_selector_ablation.add_argument("--top-k", type=int, action="append")
    train_selector_ablation.add_argument("--image-size", type=int, default=224)
    train_selector_ablation.add_argument("--batch-size", type=int, default=64)
    train_selector_ablation.add_argument("--num-workers", type=int, default=4)
    train_selector_ablation.add_argument("--epochs", type=int, default=5)
    train_selector_ablation.add_argument("--learning-rate", type=float, default=1e-3)
    train_selector_ablation.add_argument("--device", default="cpu")
    train_selector_ablation.add_argument(
        "--ablation-variant",
        action="append",
        help="Run only this selector ablation variant. May be repeated.",
    )
    train_selector_ablation.add_argument(
        "--force",
        action="store_true",
        help="Retrain variants even when selector_best.pt and selector_history.csv already exist.",
    )

    train_pairwise_selector = subparsers.add_parser(
        "train-pairwise-selector",
        help="Train a pairwise image/augmentation selector MLP.",
    )
    train_pairwise_selector.add_argument("--train-manifest", required=True)
    train_pairwise_selector.add_argument("--val-manifest", required=True)
    train_pairwise_selector.add_argument("--train-targets", required=True)
    train_pairwise_selector.add_argument("--val-targets", required=True)
    train_pairwise_selector.add_argument("--train-features")
    train_pairwise_selector.add_argument("--val-features")
    train_pairwise_selector.add_argument("--cache-dir", required=True)
    train_pairwise_selector.add_argument("--output-dir", required=True)
    train_pairwise_selector.add_argument("--identity-aug-id", default="aug_000")
    train_pairwise_selector.add_argument("--candidate-id", action="append")
    train_pairwise_selector.add_argument("--top-k", type=int, action="append")
    train_pairwise_selector.add_argument("--batch-size", type=int, default=1024)
    train_pairwise_selector.add_argument("--epochs", type=int, default=5)
    train_pairwise_selector.add_argument("--learning-rate", type=float, default=1e-3)
    train_pairwise_selector.add_argument("--hidden-dim", type=int, default=128)
    train_pairwise_selector.add_argument("--usefulness-tau", type=float, default=0.01)
    train_pairwise_selector.add_argument("--usefulness-weight", type=float, default=0.0)
    train_pairwise_selector.add_argument("--positive-gain-weight", type=float, default=0.0)
    train_pairwise_selector.add_argument(
        "--target-mode",
        choices=["nll_gain", "top1_delta"],
        default="nll_gain",
    )
    train_pairwise_selector.add_argument(
        "--selection-metric",
        choices=["val_tta_nll", "val_tta_top1"],
        default="val_tta_nll",
    )
    train_pairwise_selector.add_argument("--device", default="cpu")

    train_pairwise_comparison = subparsers.add_parser(
        "train-pairwise-selector-comparison",
        help="Train NLL-gain and top-1-delta pairwise selector variants.",
    )
    train_pairwise_comparison.add_argument("--train-manifest", required=True)
    train_pairwise_comparison.add_argument("--val-manifest", required=True)
    train_pairwise_comparison.add_argument("--train-targets", required=True)
    train_pairwise_comparison.add_argument("--val-targets", required=True)
    train_pairwise_comparison.add_argument("--train-features")
    train_pairwise_comparison.add_argument("--val-features")
    train_pairwise_comparison.add_argument("--cache-dir", required=True)
    train_pairwise_comparison.add_argument("--output-dir", required=True)
    train_pairwise_comparison.add_argument("--identity-aug-id", default="aug_000")
    train_pairwise_comparison.add_argument("--candidate-id", action="append")
    train_pairwise_comparison.add_argument("--top-k", type=int, action="append")
    train_pairwise_comparison.add_argument("--batch-size", type=int, default=1024)
    train_pairwise_comparison.add_argument("--epochs", type=int, default=5)
    train_pairwise_comparison.add_argument("--learning-rate", type=float, default=1e-3)
    train_pairwise_comparison.add_argument("--hidden-dim", type=int, default=128)
    train_pairwise_comparison.add_argument("--usefulness-tau", type=float, default=0.01)
    train_pairwise_comparison.add_argument("--usefulness-weight", type=float, default=0.0)
    train_pairwise_comparison.add_argument("--positive-gain-weight", type=float, default=0.0)
    train_pairwise_comparison.add_argument("--device", default="cpu")

    evaluate_pairwise = subparsers.add_parser(
        "evaluate-pairwise-selector",
        help="Evaluate a pairwise selector checkpoint on a cached split.",
    )
    evaluate_pairwise.add_argument("--manifest", required=True)
    evaluate_pairwise.add_argument("--cache-dir", required=True)
    evaluate_pairwise.add_argument("--checkpoint", required=True)
    evaluate_pairwise.add_argument("--output-dir", required=True)
    evaluate_pairwise.add_argument("--identity-aug-id", default="aug_000")
    evaluate_pairwise.add_argument("--features")
    evaluate_pairwise.add_argument("--top-k", type=int, default=16)
    evaluate_pairwise.add_argument("--batch-size", type=int, default=8192)
    evaluate_pairwise.add_argument("--strategy-name", default="pairwise_topk_uniform")
    evaluate_pairwise.add_argument("--device", default="cpu")

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
    tune_tta.add_argument("--adaptive-threshold", type=float, action="append")
    tune_tta.add_argument("--adaptive-max-k", type=int, action="append")
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
    build_report.add_argument("--selector-ablation")
    build_report.add_argument("--selector-diagnostics")
    build_report.add_argument("--adaptive-selection-counts")
    build_report.add_argument("--compute-policy-frontier")
    build_report.add_argument("--pairwise-selector-comparison")
    build_report.add_argument("--selector-error-analysis")
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
    from learned_tta.augmentations import (
        load_augmentation_registry,
        validate_augmentation_registry,
        write_augmentation_audit,
    )

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
    from learned_tta.smoke import run_smoke_e2e

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


def _cmd_prepare_imagenet_val(
    *,
    config_path: Path,
    val_tar_path: Path,
    output_dir: Path,
    devkit_path: Path | None,
    ground_truth_path: Path | None,
    audit_output_path: Path | None,
    overwrite: bool,
) -> None:
    config = load_experiment_config(config_path)
    class_to_idx = load_class_to_idx(config.dataset.class_index, config.project_root)
    summary = prepare_imagenet_val(
        val_tar_path=val_tar_path,
        output_dir=output_dir,
        class_to_idx=class_to_idx,
        devkit_path=devkit_path,
        ground_truth_path=ground_truth_path,
        audit_output_path=audit_output_path,
        overwrite=overwrite,
    )
    print(
        "prepared ImageNet validation: "
        f"images={summary.image_count}, "
        f"classes={summary.class_count}, "
        f"output_dir={summary.output_dir}, "
        f"audit={summary.audit_output_path}"
    )


def _cmd_check_full_run(config_path: Path, imagenet_val_dir: Path) -> None:
    from learned_tta.preflight import run_full_run_preflight

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


def _cmd_check_clean_baseline(
    *,
    config_path: Path,
    split: str | None,
    cache_dir: Path | None,
    output_path: Path | None,
    min_top1: float | None,
    min_top5: float | None,
    max_nll: float | None,
) -> None:
    report = check_clean_baseline_from_config(
        config_path,
        split=split,
        cache_dir=cache_dir,
        output_path=output_path,
        min_top1=min_top1,
        min_top5=min_top5,
        max_nll=max_nll,
    )
    print(
        "clean baseline ok: "
        f"split={report.split}, "
        f"images={int(report.metrics['image_count'])}, "
        f"top1={report.metrics['top1']:.4f}, "
        f"top5={report.metrics['top5']:.4f}, "
        f"nll={report.metrics['nll']:.4f}"
    )


def _cmd_summarize_clean_baseline(
    *,
    config_path: Path,
    splits: list[str] | None,
    cache_dir: Path | None,
    output_path: Path | None,
) -> None:
    summary = summarize_clean_center_crop_baseline_from_config(
        config_path,
        splits=splits,
        cache_dir=cache_dir,
        output_path=output_path,
    )
    print(
        "clean center-crop baseline: "
        f"splits={','.join(summary.splits)}, "
        f"images={int(summary.overall['image_count'])}, "
        f"top1={summary.overall['top1']:.4f}, "
        f"top5={summary.overall['top5']:.4f}, "
        f"nll={summary.overall['nll']:.4f}, "
        f"wrote {summary.output_path}"
    )


def _cmd_full_run_status(
    config_path: Path,
    output_format: str,
    fail_on_incomplete: bool,
    next_command: bool,
) -> None:
    from learned_tta.run_status import full_run_status_to_dict, inspect_full_run_status

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


def _cmd_teacher_cache_plan(
    *,
    config_path: Path,
    cache_dir: Path | None,
    splits: tuple[str, ...] | None,
    output_format: str,
) -> None:
    from learned_tta.teacher_cache_plan import (
        build_teacher_cache_plan,
        teacher_cache_plan_to_dict,
    )

    plan = build_teacher_cache_plan(
        config_path=config_path,
        cache_dir=cache_dir,
        splits=splits if splits is not None else ("public_train", "public_val", "private"),
    )
    if output_format == "json":
        print(json.dumps(teacher_cache_plan_to_dict(plan), indent=2, sort_keys=True))
        return

    print(
        "teacher cache plan: "
        f"predictions={plan.total_predictions:,}, "
        f"shards={plan.complete_shards}/{plan.expected_shards}, "
        f"missing_files={plan.missing_files}, "
        f"stale_or_malformed={plan.stale_or_malformed_shards}, "
        f"logits={_format_bytes(plan.completed_logits_bytes)}/"
        f"{_format_bytes(plan.logits_bytes_estimate)}"
    )
    for split in plan.splits:
        marker = "x" if split.complete else " "
        print(
            f"[{marker}] {split.split}: "
            f"images={split.expected_images:,}, "
            f"predictions={split.expected_predictions:,}, "
            f"shards={split.complete_shards}/{split.expected_shards}, "
            f"missing_files={split.missing_files}, "
            f"stale_or_malformed={split.stale_or_malformed_shards}, "
            f"logits={_format_bytes(split.completed_logits_bytes)}/"
            f"{_format_bytes(split.logits_bytes_estimate)}"
        )
        if split.next_command is not None:
            print(f"next {split.split}: {split.next_command}")


def _cmd_teacher_backend_plan(*, device: str, output_format: str) -> None:
    plan = build_teacher_backend_plan(device=device)
    if output_format == "json":
        print(json.dumps(teacher_backend_plan_to_dict(plan), indent=2, sort_keys=True))
        return

    print(
        "teacher backend plan: "
        f"active={plan.active_backend}, "
        f"device={plan.device}, "
        f"recommended_accelerator={plan.recommended_accelerator}"
    )
    for backend in plan.backends:
        print(
            f"- {backend.name}: status={backend.status}, "
            f"device={backend.device}, role={backend.role}"
        )


def _cmd_teacher_cache_diagnostics(
    *,
    config_path: Path,
    split: str,
    cache_dir: Path | None,
    candidate_ids: list[str] | None,
    top_n: int,
    output_path: Path | None,
    output_format: str,
) -> None:
    from learned_tta.augmentations import load_augmentation_registry
    from learned_tta.teacher_cache_diagnostics import (
        summarize_teacher_cache_diagnostics,
        teacher_cache_diagnostics_to_dict,
    )

    config = load_experiment_config(config_path)
    aug_ids = candidate_ids
    if aug_ids is None:
        aug_ids = [
            candidate.id
            for candidate in load_augmentation_registry(config.augmentations.registry_path)
        ]
    summary = summarize_teacher_cache_diagnostics(
        cache_dir=cache_dir or config.artifacts.teacher_cache_dir,
        split=split,
        aug_ids=aug_ids,
        identity_aug_id=config.augmentations.identity_id,
        top_n=top_n,
    )
    payload = teacher_cache_diagnostics_to_dict(summary)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(
        "teacher cache diagnostics: "
        f"split={summary.split}, "
        f"images={summary.image_count:,}, "
        f"candidates={summary.candidate_count}, "
        f"clean_top1={summary.clean_top1:.4f}, "
        f"clean_top5={summary.clean_top5:.4f}, "
        f"clean_nll={summary.clean_nll:.4f}, "
        f"best_single={summary.best_single_aug_id} "
        f"gain={summary.best_single_aug_mean_gain:.6g}, "
        f"oracle_gain={summary.oracle_best_mean_gain:.6g}"
    )
    if output_path is not None:
        print(f"wrote {output_path}")


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    unit = units[0]
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.2f} {unit}"


def _cmd_resume_full_run(
    config_path: Path,
    imagenet_val_dir: Path | None,
    cache_log_dir: Path | None,
    dry_run: bool,
    background_cache: bool,
    allow_duplicate_cache: bool,
) -> None:
    from learned_tta.run_supervisor import run_next_full_run_step

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
    backend: str,
) -> None:
    from learned_tta.teacher_cache import cache_teacher_from_config

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
        backend=backend,
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
    target_kind: str,
) -> None:
    from learned_tta.target_builder import build_selector_targets_from_config

    summary = build_selector_targets_from_config(
        config_path=config_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        train_split=train_split,
        val_split=val_split,
        candidate_ids=candidate_ids,
        target_kind=target_kind,
    )
    print(
        f"selector targets: wrote {summary.train_path.name} and {summary.val_path.name} "
        f"for {_plural(len(summary.aug_ids), 'augmentation')} "
        f"target_kind={summary.target_kind}"
    )


def _cmd_build_selector_features(
    config_path: Path,
    split: str,
    manifest_path: Path | None,
    output_path: Path | None,
    model_name: str | None,
    pretrained: bool,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    from learned_tta.selector_feature_cache import (
        build_timm_feature_extractor,
        extract_selector_features_from_manifest,
    )

    config = load_experiment_config(config_path)
    resolved_model_name = model_name or config.teacher.model_name
    resolved_manifest_path = manifest_path or config.artifacts.manifests_dir / f"{split}.csv"
    resolved_output_path = (
        output_path
        or config.artifacts.selector_dir
        / "features"
        / f"{split}__{resolved_model_name}.features.npz"
    )
    bundle = build_timm_feature_extractor(
        model_name=resolved_model_name,
        pretrained=pretrained,
    )
    written = extract_selector_features_from_manifest(
        manifest_path=resolved_manifest_path,
        output_path=resolved_output_path,
        bundle=bundle,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(f"selector features: wrote {written}")


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
    usefulness_head: bool | None,
    usefulness_tau: float | None,
    usefulness_weight: float | None,
    device: str,
) -> None:
    from learned_tta.selector_training import train_selector_from_config

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
        usefulness_head=usefulness_head,
        usefulness_tau=usefulness_tau,
        usefulness_weight=usefulness_weight,
        device=device,
    )
    print(
        f"selector training: best epoch {summary.best_epoch}, "
        f"best val nll {summary.best_val_nll:.6g}, "
        f"best val loss {summary.best_val_loss:.6g}, checkpoint {summary.checkpoint_path}"
    )


def _cmd_train_selector_ablation(
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
    device: str,
    variant_names: tuple[str, ...] | None,
    train_features_path: Path | None,
    val_features_path: Path | None,
    force: bool,
) -> None:
    from learned_tta.selector_training import train_selector_loss_ablation_from_config

    summary = train_selector_loss_ablation_from_config(
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
        device=device,
        variant_names=variant_names,
        train_features_path=train_features_path,
        val_features_path=val_features_path,
        skip_completed=not force,
    )
    print(f"selector ablation: wrote {summary.results_csv}")


def _cmd_train_pairwise_selector(
    train_manifest_path: Path,
    val_manifest_path: Path,
    train_targets_path: Path,
    val_targets_path: Path,
    cache_dir: Path,
    output_dir: Path,
    identity_aug_id: str,
    train_features_path: Path | None,
    val_features_path: Path | None,
    top_k_grid: list[int] | None,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    usefulness_tau: float,
    usefulness_weight: float,
    positive_gain_weight: float,
    target_mode: str,
    selection_metric: str,
    device: str,
) -> None:
    from learned_tta.pairwise_selector import train_pairwise_selector_from_artifacts

    summary = train_pairwise_selector_from_artifacts(
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        train_targets_path=train_targets_path,
        val_targets_path=val_targets_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        identity_aug_id=identity_aug_id,
        train_features_path=train_features_path,
        val_features_path=val_features_path,
        top_k_grid=top_k_grid,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        usefulness_tau=usefulness_tau,
        usefulness_weight=usefulness_weight,
        positive_gain_weight=positive_gain_weight,
        target_mode=target_mode,
        selection_metric=selection_metric,
        device=device,
    )
    print(f"pairwise selector: wrote {summary.summary_csv}")


def _cmd_train_pairwise_selector_comparison(
    train_manifest_path: Path,
    val_manifest_path: Path,
    train_targets_path: Path,
    val_targets_path: Path,
    cache_dir: Path,
    output_dir: Path,
    identity_aug_id: str,
    train_features_path: Path | None,
    val_features_path: Path | None,
    top_k_grid: list[int] | None,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    usefulness_tau: float,
    usefulness_weight: float,
    positive_gain_weight: float,
    device: str,
) -> None:
    from learned_tta.pairwise_selector import train_pairwise_selector_comparison_from_artifacts

    summary = train_pairwise_selector_comparison_from_artifacts(
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
        train_targets_path=train_targets_path,
        val_targets_path=val_targets_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
        identity_aug_id=identity_aug_id,
        train_features_path=train_features_path,
        val_features_path=val_features_path,
        top_k_grid=top_k_grid,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        usefulness_tau=usefulness_tau,
        usefulness_weight=usefulness_weight,
        positive_gain_weight=positive_gain_weight,
        device=device,
    )
    print(f"pairwise selector comparison: wrote {summary.results_csv}")


def _cmd_evaluate_pairwise_selector(
    manifest_path: Path,
    cache_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    identity_aug_id: str,
    features_path: Path | None,
    top_k: int,
    batch_size: int,
    strategy_name: str,
    device: str,
) -> None:
    from learned_tta.pairwise_selector import evaluate_pairwise_selector_from_artifacts

    summary = evaluate_pairwise_selector_from_artifacts(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        identity_aug_id=identity_aug_id,
        features_path=features_path,
        top_k=top_k,
        batch_size=batch_size,
        strategy_name=strategy_name,
        device=device,
    )
    print(f"pairwise selector evaluation: wrote {summary.metrics_csv}")


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
    adaptive_threshold_grid: list[float] | None,
    adaptive_max_k_grid: list[int] | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    from learned_tta.tta_tuning import tune_tta_from_config

    summary = tune_tta_from_config(
        config_path=config_path,
        split=split,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        candidate_ids=candidate_ids,
        top_k_grid=top_k_grid,
        adaptive_threshold_grid=adaptive_threshold_grid,
        adaptive_max_k_grid=adaptive_max_k_grid,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    print(f"tta tuning {summary.split}: best k {summary.best_k}, wrote {summary.result_path}")


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
    from learned_tta.stacking import train_aggregator_from_config

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
    from learned_tta.private_eval import evaluate_private_from_config

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
    print(f"private evaluation: best k {summary.best_k}, wrote {summary.private_metrics_csv}")


def _cmd_build_report(
    config_path: Path,
    report_dir: Path | None,
    private_metrics_path: Path | None,
    corrections_path: Path | None,
    selector_history_path: Path | None,
    selector_ablation_path: Path | None,
    selector_diagnostics_path: Path | None,
    adaptive_selection_counts_path: Path | None,
    compute_policy_frontier_path: Path | None,
    pairwise_selector_comparison_path: Path | None,
    selector_error_analysis_path: Path | None,
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
    from learned_tta.report_builder import build_report_from_config

    summary = build_report_from_config(
        config_path=config_path,
        report_dir=report_dir,
        private_metrics_path=private_metrics_path,
        corrections_path=corrections_path,
        selector_history_path=selector_history_path,
        selector_ablation_path=selector_ablation_path,
        selector_diagnostics_path=selector_diagnostics_path,
        adaptive_selection_counts_path=adaptive_selection_counts_path,
        compute_policy_frontier_path=compute_policy_frontier_path,
        pairwise_selector_comparison_path=pairwise_selector_comparison_path,
        selector_error_analysis_path=selector_error_analysis_path,
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
