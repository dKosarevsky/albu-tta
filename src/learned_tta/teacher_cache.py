"""Teacher cache inference runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learned_tta.augmentations import AugmentationCandidate, load_augmentation_registry
from learned_tta.cache import (
    TeacherShard,
    shard_is_complete,
    teacher_shard_paths,
    write_teacher_shard,
)
from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest, make_teacher_dataloader
from learned_tta.teacher import TeacherBundle, load_teacher


@dataclass(frozen=True, slots=True)
class TeacherCacheRunSummary:
    """Summary of one teacher cache run."""

    split: str
    output_dir: Path
    written: list[str]
    skipped: list[str]


def run_teacher_cache(
    split: str,
    records: list[ManifestRecord],
    candidates: list[AugmentationCandidate],
    teacher: TeacherBundle,
    output_dir: Path,
    seed: int,
    batch_size: int,
    num_workers: int,
    resume: bool = True,
    device: str | torch.device = "cpu",
) -> TeacherCacheRunSummary:
    """Run teacher inference for one split over a list of augmentation candidates."""

    output_dir = Path(output_dir)
    torch_device = torch.device(device)
    model = teacher.model
    if hasattr(model, "to"):
        model.to(torch_device)
    if hasattr(model, "eval"):
        model.eval()

    expected_classes = _model_num_classes(model)
    written: list[str] = []
    skipped: list[str] = []

    for candidate in candidates:
        paths = teacher_shard_paths(output_dir, split=split, aug_id=candidate.id)
        if (
            resume
            and expected_classes is not None
            and shard_is_complete(
                metadata_path=paths.metadata_path,
                logits_path=paths.logits_path,
                expected_rows=len(records),
                expected_classes=expected_classes,
            )
        ):
            skipped.append(candidate.id)
            continue

        shard = _infer_candidate(
            split=split,
            records=records,
            candidate=candidate,
            teacher=teacher,
            model=model,
            device=torch_device,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        write_teacher_shard(output_dir, shard)
        written.append(candidate.id)

    return TeacherCacheRunSummary(
        split=split,
        output_dir=output_dir,
        written=written,
        skipped=skipped,
    )


def cache_teacher_from_config(
    config_path: Path,
    split: str,
    manifest_path: Path | None = None,
    output_dir: Path | None = None,
    candidate_ids: list[str] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    resume: bool = True,
    device: str | torch.device = "cpu",
) -> TeacherCacheRunSummary:
    """Load config, manifest, registry, and teacher, then run teacher caching."""

    config = load_experiment_config(config_path)
    resolved_manifest_path = manifest_path or config.artifacts.manifests_dir / f"{split}.csv"
    resolved_output_dir = output_dir or config.artifacts.teacher_cache_dir
    candidates = load_augmentation_registry(config.augmentations.registry_path)
    if candidate_ids is not None:
        candidates = _filter_candidates(candidates, candidate_ids)
    teacher = load_teacher(
        model_name=config.teacher.model_name,
        pretrained=config.teacher.pretrained,
    )
    return run_teacher_cache(
        split=split,
        records=load_manifest(resolved_manifest_path),
        candidates=candidates,
        teacher=teacher,
        output_dir=resolved_output_dir,
        seed=config.seed,
        batch_size=batch_size,
        num_workers=num_workers,
        resume=resume,
        device=device,
    )


@torch.inference_mode()
def _infer_candidate(
    split: str,
    records: list[ManifestRecord],
    candidate: AugmentationCandidate,
    teacher: TeacherBundle,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> TeacherShard:
    dataloader = make_teacher_dataloader(
        records=records,
        candidate=candidate,
        preprocess=teacher.preprocess,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    image_ids: list[str] = []
    class_idxs = []
    logits = []
    for batch in dataloader:
        batch_logits = model(batch.images.to(device))
        logits.append(batch_logits.detach().cpu().numpy().astype(np.float32))
        image_ids.extend(batch.image_ids)
        class_idxs.append(batch.class_idxs.cpu().numpy().astype(np.int64))

    return TeacherShard(
        split=split,
        aug_id=candidate.id,
        image_ids=image_ids,
        class_idxs=np.concatenate(class_idxs),
        logits=np.concatenate(logits, axis=0),
    )


def _filter_candidates(
    candidates: list[AugmentationCandidate],
    candidate_ids: list[str],
) -> list[AugmentationCandidate]:
    by_id = {candidate.id: candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"unknown augmentation candidate ids: {', '.join(missing)}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def _model_num_classes(model: object) -> int | None:
    value = getattr(model, "num_classes", None)
    if isinstance(value, int):
        return value
    return None
