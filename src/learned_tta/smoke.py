"""Synthetic end-to-end smoke runner for the learned TTA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.imagenet_split import (
    SplitConfig,
    build_stratified_splits,
    discover_imagenet_val,
    write_split_manifests,
)
from learned_tta.private_eval import evaluate_private_from_artifacts
from learned_tta.report_builder import build_report_from_artifacts
from learned_tta.selector_training import train_selector_from_artifacts
from learned_tta.stacking import default_aggregator_path, train_aggregator_from_artifacts
from learned_tta.target_builder import build_selector_targets_from_cache
from learned_tta.teacher import TeacherBundle
from learned_tta.teacher_cache import run_teacher_cache
from learned_tta.tta_tuning import tune_tta_from_artifacts


@dataclass(frozen=True, slots=True)
class SmokeRunSummary:
    """Important artifacts produced by `run_smoke_e2e`."""

    output_dir: Path
    manifests_dir: Path
    teacher_cache_dir: Path
    selector_dir: Path
    reports_dir: Path
    selector_checkpoint: Path
    tuning_json: Path
    private_metrics_csv: Path
    results_md: Path
    candidate_ids: list[str]


def run_smoke_e2e(
    config_path: Path,
    output_dir: Path,
    candidate_count: int = 2,
    image_size: int = 16,
    batch_size: int = 2,
    num_workers: int = 0,
    epochs: int = 1,
    device: str | torch.device = "cpu",
) -> SmokeRunSummary:
    """Run a tiny synthetic pipeline from manifests through final report artifacts."""

    if candidate_count < 2:
        raise ValueError("candidate_count must include identity and at least one transform")

    config = load_experiment_config(config_path)
    output_dir = Path(output_dir)
    manifests_dir = output_dir / "manifests"
    teacher_cache_dir = output_dir / "teacher_cache"
    selector_dir = output_dir / "selector"
    reports_dir = output_dir / "reports"
    synthetic_val_dir = output_dir / "synthetic_imagenet_val"

    candidates = load_augmentation_registry(config.augmentations.registry_path)[
        :candidate_count
    ]
    candidate_ids = [candidate.id for candidate in candidates]
    if candidate_ids[0] != config.augmentations.identity_id:
        raise ValueError("the first smoke candidate must be the configured identity")

    _write_synthetic_imagenet_val(
        synthetic_val_dir,
        class_count=2,
        images_per_class=4,
        image_size=max(image_size, 8),
    )
    splits = build_stratified_splits(
        discover_imagenet_val(synthetic_val_dir),
        SplitConfig(
            seed=config.seed,
            public_per_class=2,
            private_per_class=2,
            public_train_per_class=1,
            public_val_per_class=1,
        ),
    )
    manifests = write_split_manifests(splits, manifests_dir)

    teacher = _smoke_teacher_bundle()
    for split in ("public_train", "public_val", "private"):
        run_teacher_cache(
            split=split,
            records=_records_from_manifest(manifests[split]),
            candidates=candidates,
            teacher=teacher,
            output_dir=teacher_cache_dir,
            seed=config.seed,
            batch_size=batch_size,
            num_workers=num_workers,
            resume=False,
            device=device,
        )

    target_summary = build_selector_targets_from_cache(
        cache_dir=teacher_cache_dir,
        output_dir=selector_dir,
        train_split="public_train",
        val_split="public_val",
        aug_ids=candidate_ids,
        identity_aug_id=config.augmentations.identity_id,
    )
    train_summary = train_selector_from_artifacts(
        train_manifest_path=manifests["public_train"],
        val_manifest_path=manifests["public_val"],
        train_targets_path=target_summary.train_path,
        val_targets_path=target_summary.val_path,
        output_dir=selector_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        epochs=epochs,
        learning_rate=1e-3,
        rank_weight=0.2,
        val_cache_dir=teacher_cache_dir,
        val_split="public_val",
        aug_ids=candidate_ids,
        top_k_grid=list(range(candidate_count)),
        identity_aug_id=config.augmentations.identity_id,
        device=device,
    )
    tuning_summary = tune_tta_from_artifacts(
        split="public_val",
        manifest_path=manifests["public_val"],
        cache_dir=teacher_cache_dir,
        checkpoint_path=train_summary.checkpoint_path,
        output_dir=selector_dir,
        aug_ids=candidate_ids,
        top_k_grid=list(range(candidate_count)),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )
    global_aggregator_path = default_aggregator_path(
        selector_dir,
        split="public_val",
        method="global-nonnegative",
    )
    class_aggregator_path = default_aggregator_path(
        selector_dir,
        split="public_val",
        method="class-nonnegative",
    )
    for method, output_path in (
        ("global-nonnegative", global_aggregator_path),
        ("class-nonnegative", class_aggregator_path),
    ):
        train_aggregator_from_artifacts(
            split="public_val",
            cache_dir=teacher_cache_dir,
            output_path=output_path,
            aug_ids=candidate_ids,
            method=method,
            epochs=max(epochs, 20),
            learning_rate=0.1,
            l1_penalty=0.0,
            active_threshold=1e-6,
            device=device,
        )
    private_summary = evaluate_private_from_artifacts(
        split="private",
        manifest_path=manifests["private"],
        cache_dir=teacher_cache_dir,
        checkpoint_path=train_summary.checkpoint_path,
        tuning_path=tuning_summary.result_path,
        output_dir=reports_dir,
        aug_ids=candidate_ids,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seeds=[1, 2, 3],
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )
    report_summary = build_report_from_artifacts(
        report_dir=reports_dir,
        private_metrics_path=private_summary.private_metrics_csv,
        tuning_path=tuning_summary.result_path,
        impact_targets_path=target_summary.val_path,
        impact_manifest_path=manifests["public_val"],
        checkpoint_path=train_summary.checkpoint_path,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        corrections_path=private_summary.corrections_csv,
        selector_history_path=train_summary.history_csv,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )
    return SmokeRunSummary(
        output_dir=output_dir,
        manifests_dir=manifests_dir,
        teacher_cache_dir=teacher_cache_dir,
        selector_dir=selector_dir,
        reports_dir=reports_dir,
        selector_checkpoint=train_summary.checkpoint_path,
        tuning_json=tuning_summary.result_path,
        private_metrics_csv=private_summary.private_metrics_csv,
        results_md=report_summary.results_md,
        candidate_ids=candidate_ids,
    )


def _write_synthetic_imagenet_val(
    val_dir: Path,
    class_count: int,
    images_per_class: int,
    image_size: int,
) -> None:
    val_dir.mkdir(parents=True, exist_ok=True)
    for class_idx in range(class_count):
        class_dir = val_dir / f"n{class_idx:08d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        base_value = 48 if class_idx == 0 else 208
        for image_idx in range(images_per_class):
            image = np.full(
                (image_size, image_size, 3),
                fill_value=base_value + image_idx,
                dtype=np.uint8,
            )
            Image.fromarray(image, mode="RGB").save(
                class_dir / f"smoke_{class_idx:02d}_{image_idx:02d}.png"
            )


def _records_from_manifest(path: Path) -> list[ManifestRecord]:
    return load_manifest(path)


def _smoke_teacher_bundle() -> TeacherBundle:
    def preprocess(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    return TeacherBundle(
        model=_SmokeTeacher(),
        data_config={"input_size": (3, 16, 16)},
        preprocess=preprocess,
    )


class _SmokeTeacher(torch.nn.Module):
    num_classes = 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        means = images.mean(dim=(1, 2, 3))
        class_one_score = (means - 0.5) * 8.0
        return torch.stack([-class_one_score, class_one_score], dim=1)
