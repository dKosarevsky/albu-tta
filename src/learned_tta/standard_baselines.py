"""Standard ImageNet evaluation baselines for report comparisons."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from learned_tta.config import load_experiment_config
from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.metrics import classification_metrics, expected_calibration_error
from learned_tta.teacher import load_teacher
from learned_tta.tta_eval import evaluate_selected_tta


@dataclass(frozen=True, slots=True)
class TenCropLogits:
    """Saved 10-crop logits and row metadata."""

    crop_logits: np.ndarray
    class_idxs: np.ndarray
    image_ids: list[str]


@dataclass(frozen=True, slots=True)
class TenCropRunSummary:
    """Summary of a generated 10-crop baseline artifact."""

    logits_path: Path
    metrics_path: Path
    metrics: dict[str, float]


def evaluate_cached_standard_baselines(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    identity_aug_id: str = "aug_000",
    hflip_aug_id: str = "aug_005",
    reference_aug_count: int = 100,
) -> dict[str, dict[str, float]]:
    """Evaluate standard cached baselines from existing teacher-cache logits.

    `clean_center_crop` maps to the identity augmentation. `center_crop_hflip`
    maps to identity plus the configured horizontal-flip candidate when present.
    """

    _validate_reference_aug_count(reference_aug_count)
    if identity_aug_id not in logits_by_aug:
        raise ValueError(f"identity augmentation {identity_aug_id!r} is missing")

    baselines = {
        "clean_center_crop": _with_reference_compute(
            evaluate_selected_tta(logits_by_aug, [identity_aug_id], class_idxs),
            forwards_per_image=1.0,
            reference_aug_count=reference_aug_count,
        )
    }
    if hflip_aug_id in logits_by_aug:
        baselines["center_crop_hflip"] = _with_reference_compute(
            evaluate_selected_tta(
                logits_by_aug,
                [identity_aug_id, hflip_aug_id],
                class_idxs,
            ),
            forwards_per_image=2.0,
            reference_aug_count=reference_aug_count,
        )
    return baselines


def evaluate_ten_crop_logits(
    crop_logits: np.ndarray,
    class_idxs: np.ndarray,
    reference_aug_count: int = 100,
) -> dict[str, float]:
    """Evaluate a standard 10-crop baseline from logits shaped `[N, 10, C]`."""

    _validate_reference_aug_count(reference_aug_count)
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    probabilities = ten_crop_probabilities(crop_logits)
    if class_idxs.shape != (probabilities.shape[0],):
        raise ValueError("class_idxs must have shape [num_images]")

    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 5))
    metrics["ece"] = expected_calibration_error(probabilities, class_idxs)
    return _with_reference_compute(
        metrics,
        forwards_per_image=10.0,
        reference_aug_count=reference_aug_count,
    )


def ten_crop_probabilities(crop_logits: np.ndarray) -> np.ndarray:
    """Average softmax probabilities from logits shaped `[N, 10, C]`."""

    crop_logits = _validate_ten_crop_logits(crop_logits)
    return _softmax_3d(crop_logits).mean(axis=1).astype(np.float32)


def write_ten_crop_logits(
    path: Path,
    crop_logits: np.ndarray,
    class_idxs: np.ndarray,
    image_ids: list[str],
) -> Path:
    """Write 10-crop logits as a compact `.npz` artifact."""

    crop_logits = _validate_ten_crop_logits(crop_logits)
    class_idxs = np.asarray(class_idxs, dtype=np.int64)
    if class_idxs.shape != (crop_logits.shape[0],):
        raise ValueError("class_idxs must have shape [num_images]")
    if len(image_ids) != crop_logits.shape[0]:
        raise ValueError("image_ids length must match num_images")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        crop_logits=crop_logits,
        class_idxs=class_idxs,
        image_ids=np.asarray(image_ids, dtype=str),
    )
    return path


def load_ten_crop_logits(path: Path) -> TenCropLogits:
    """Load 10-crop logits written by `write_ten_crop_logits`."""

    with np.load(Path(path), allow_pickle=False) as data:
        crop_logits = _validate_ten_crop_logits(data["crop_logits"])
        class_idxs = np.asarray(data["class_idxs"], dtype=np.int64)
        image_ids = [str(image_id) for image_id in data["image_ids"].tolist()]
    if class_idxs.shape != (crop_logits.shape[0],):
        raise ValueError("class_idxs must have shape [num_images]")
    if len(image_ids) != crop_logits.shape[0]:
        raise ValueError("image_ids length must match num_images")
    return TenCropLogits(crop_logits=crop_logits, class_idxs=class_idxs, image_ids=image_ids)


def evaluate_ten_crop_artifact(
    path: Path,
    expected_class_idxs: np.ndarray | None = None,
    reference_aug_count: int = 100,
) -> dict[str, float]:
    """Evaluate a saved 10-crop logits artifact."""

    artifact = load_ten_crop_logits(path)
    if expected_class_idxs is not None and not np.array_equal(
        artifact.class_idxs,
        np.asarray(expected_class_idxs, dtype=np.int64),
    ):
        raise ValueError("10-crop class_idxs do not match private cache order")
    return evaluate_ten_crop_logits(
        artifact.crop_logits,
        artifact.class_idxs,
        reference_aug_count=reference_aug_count,
    )


def run_ten_crop_baseline_from_config(
    config_path: Path,
    split: str = "private",
    manifest_path: Path | None = None,
    output_logits_path: Path | None = None,
    metrics_output_path: Path | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cpu",
) -> TenCropRunSummary:
    """Run standard 10-crop inference and write logits plus metrics artifacts."""

    config = load_experiment_config(config_path)
    resolved_manifest_path = manifest_path or config.artifacts.manifests_dir / f"{split}.csv"
    resolved_logits_path = (
        output_logits_path
        or config.artifacts.reports_dir / "tables" / f"{split}_ten_crop_logits.npz"
    )
    resolved_metrics_path = (
        metrics_output_path
        or config.artifacts.reports_dir / "tables" / f"{split}_ten_crop_metrics.csv"
    )
    records = load_manifest(resolved_manifest_path)
    bundle = load_teacher(
        model_name=config.teacher.model_name,
        pretrained=config.teacher.pretrained,
    )
    data_config = dict(bundle.data_config)
    if config.teacher.data_config is not None:
        data_config.update(config.teacher.data_config)
    crop_logits, class_idxs, image_ids = predict_ten_crop_logits(
        records=records,
        model=bundle.model,
        data_config=data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    write_ten_crop_logits(
        path=resolved_logits_path,
        crop_logits=crop_logits,
        class_idxs=class_idxs,
        image_ids=image_ids,
    )
    metrics = evaluate_ten_crop_logits(
        crop_logits,
        class_idxs,
        reference_aug_count=config.augmentations.candidate_count,
    )
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"strategy": "ten_crop", **metrics}]).to_csv(
        resolved_metrics_path,
        index=False,
    )
    return TenCropRunSummary(
        logits_path=resolved_logits_path,
        metrics_path=resolved_metrics_path,
        metrics=metrics,
    )


def predict_ten_crop_logits(
    records: list[ManifestRecord],
    model: Any,
    data_config: dict[str, Any],
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run model inference over all 10 crops for each manifest record."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    transform = make_ten_crop_transform(data_config)
    dataset = _TenCropDataset(records=records, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_collate_ten_crop_batch,
    )
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    all_logits: list[np.ndarray] = []
    all_class_idxs: list[np.ndarray] = []
    all_image_ids: list[str] = []
    with torch.inference_mode():
        for image_ids, class_idxs, crop_images in dataloader:
            batch_size_actual, crop_count = crop_images.shape[:2]
            flat_images = crop_images.reshape(
                batch_size_actual * crop_count,
                *crop_images.shape[2:],
            ).to(torch_device)
            logits = model(flat_images).detach().cpu().numpy()  # type: ignore[operator]
            all_logits.append(logits.reshape(batch_size_actual, crop_count, -1))
            all_class_idxs.append(class_idxs.numpy())
            all_image_ids.extend(image_ids)
    if not all_logits:
        raise ValueError("manifest contains no records")
    return (
        np.concatenate(all_logits, axis=0).astype(np.float32),
        np.concatenate(all_class_idxs, axis=0).astype(np.int64),
        all_image_ids,
    )


def make_ten_crop_transform(data_config: dict[str, Any]) -> Callable[[Any], torch.Tensor]:
    """Create a standard ImageNet 10-crop preprocessing transform."""

    import math

    from timm.data.transforms import str_to_interp_mode
    from torchvision import transforms
    from torchvision.transforms import functional as functional_transform

    input_size_obj = data_config.get("input_size", (3, 224, 224))
    if not isinstance(input_size_obj, list | tuple) or len(input_size_obj) < 2:
        raise ValueError("data_config input_size must be a sequence ending with height, width")
    input_size = [int(value) for value in input_size_obj]
    crop_size = (int(input_size[-2]), int(input_size[-1]))
    crop_pct = float(data_config.get("crop_pct", 0.875))
    if crop_pct <= 0.0:
        raise ValueError("data_config crop_pct must be positive")
    scale_size = tuple(math.floor(size / crop_pct) for size in crop_size)
    interpolation = str(data_config.get("interpolation", "bilinear"))
    mean = _float_sequence(data_config.get("mean", (0.485, 0.456, 0.406)), "mean")
    std = _float_sequence(data_config.get("std", (0.229, 0.224, 0.225)), "std")
    resize_size = scale_size[0] if scale_size[0] == scale_size[1] else scale_size
    resize = transforms.Resize(resize_size, interpolation=str_to_interp_mode(interpolation))
    ten_crop = transforms.TenCrop(crop_size[0] if crop_size[0] == crop_size[1] else crop_size)

    def transform(image):
        resized = resize(image)
        crops = ten_crop(resized)
        tensors = [
            functional_transform.normalize(
                functional_transform.to_tensor(crop),
                mean=mean,
                std=std,
            )
            for crop in crops
        ]
        return torch.stack(tensors, dim=0)

    return transform


def _with_reference_compute(
    metrics: dict[str, float],
    forwards_per_image: float,
    reference_aug_count: int,
) -> dict[str, float]:
    row = dict(metrics)
    row["forwards_per_image"] = float(forwards_per_image)
    row["relative_compute_vs_all"] = float(forwards_per_image) / float(reference_aug_count)
    return row


def _validate_reference_aug_count(reference_aug_count: int) -> None:
    if reference_aug_count <= 0:
        raise ValueError("reference_aug_count must be positive")


def _validate_ten_crop_logits(crop_logits: np.ndarray) -> np.ndarray:
    crop_logits = np.asarray(crop_logits, dtype=np.float32)
    if crop_logits.ndim != 3:
        raise ValueError("crop_logits must have shape [num_images, 10, num_classes]")
    if crop_logits.shape[1] != 10:
        raise ValueError("crop_logits must contain exactly 10 crops per image")
    return crop_logits


def _softmax_3d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=2, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=2, keepdims=True)


def _float_sequence(value: object, name: str) -> list[float]:
    if not isinstance(value, Sequence):
        raise ValueError(f"data_config {name} must be a sequence")
    return [float(item) for item in cast(Sequence[Any], value)]


class _TenCropDataset(torch.utils.data.Dataset[tuple[str, int, torch.Tensor]]):
    def __init__(
        self,
        records: list[ManifestRecord],
        transform: Callable[[Any], torch.Tensor],
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        from PIL import Image

        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return record.image_id, record.class_idx, tensor


def _collate_ten_crop_batch(
    samples: list[tuple[str, int, torch.Tensor]],
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    image_ids = [sample[0] for sample in samples]
    class_idxs = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    crop_images = torch.stack([sample[2] for sample in samples], dim=0)
    return image_ids, class_idxs, crop_images
