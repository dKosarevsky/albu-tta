"""Pretrained image feature cache builder for selector MLP baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from learned_tta.data import ManifestRecord, load_manifest
from learned_tta.selector_features import save_selector_features


@dataclass(frozen=True, slots=True)
class FeatureExtractorBundle:
    """Loaded pretrained feature extractor plus preprocessing."""

    model: torch.nn.Module
    preprocess: Any
    model_name: str
    pretrained: bool
    data_config: dict[str, Any]


SelectorFeatureBatch = tuple[list[str], torch.Tensor]


class SelectorFeatureImageDataset(torch.utils.data.Dataset[tuple[str, torch.Tensor]]):
    """Manifest-backed image dataset for pretrained feature extraction."""

    def __init__(self, records: list[ManifestRecord], preprocess: Any) -> None:
        self.records = records
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return record.image_id, tensor


def build_timm_feature_extractor(
    model_name: str,
    *,
    pretrained: bool = True,
) -> FeatureExtractorBundle:
    """Load a timm model configured to return pooled pretrained embeddings."""

    import timm
    import timm.data

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    return FeatureExtractorBundle(
        model=model,
        preprocess=preprocess,
        model_name=model_name,
        pretrained=pretrained,
        data_config=dict(data_config),
    )


def extract_selector_features_from_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    bundle: FeatureExtractorBundle,
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
) -> Path:
    """Extract pretrained image embeddings for one manifest and write a selector feature cache."""

    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest must contain at least one row")
    split = records[0].split
    if any(record.split != split for record in records):
        raise ValueError("manifest must contain a single split")

    torch_device = torch.device(device)
    model = bundle.model.to(torch_device)
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        SelectorFeatureImageDataset(records=records, preprocess=bundle.preprocess),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    image_ids: list[str] = []
    feature_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_image_ids, images in dataloader:
            outputs = model(images.to(torch_device))
            features = _flatten_model_outputs(outputs)
            image_ids.extend(str(image_id) for image_id in batch_image_ids)
            feature_chunks.append(features.cpu().numpy().astype(np.float32))

    features_array = np.concatenate(feature_chunks, axis=0)
    feature_names = [f"feature_{index:04d}" for index in range(features_array.shape[1])]
    return save_selector_features(
        path=output_path,
        split=split,
        model_name=bundle.model_name,
        image_ids=image_ids,
        features=features_array,
        feature_names=feature_names,
        metadata={
            "pretrained": bundle.pretrained,
            "data_config": _json_ready(bundle.data_config),
        },
    )


def _flatten_model_outputs(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, tuple | list):
        if not outputs:
            raise ValueError("feature extractor returned an empty output sequence")
        outputs = outputs[0]
    if not isinstance(outputs, torch.Tensor):
        raise ValueError("feature extractor must return a tensor or a non-empty tensor sequence")
    if outputs.ndim == 4:
        outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, 1).flatten(1)
    elif outputs.ndim > 2:
        outputs = outputs.flatten(1)
    if outputs.ndim != 2:
        raise ValueError("feature extractor output must flatten to shape [images, features]")
    return outputs.float()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)
