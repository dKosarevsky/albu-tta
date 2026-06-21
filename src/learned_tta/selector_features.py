"""Feature extraction helpers for lightweight selector baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SavedSelectorFeatures:
    """Cached per-image selector features."""

    split: str
    model_name: str
    image_ids: list[str]
    features: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_logit_features(
    logits: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Build per-image clean-pass features from model logits."""

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [images, classes]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")

    probs = _softmax(logits)
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    capped_top_k = min(top_k, logits.shape[1])
    confidence = sorted_probs[:, 0]
    second_prob = sorted_probs[:, 1] if logits.shape[1] > 1 else np.zeros_like(confidence)
    margin = confidence - second_prob
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    pred_class = np.argmax(logits, axis=1).astype(np.float32)

    columns = [
        confidence.astype(np.float32),
        margin.astype(np.float32),
        entropy.astype(np.float32),
        pred_class,
    ]
    names = [
        "clean_confidence",
        "clean_margin",
        "clean_entropy",
        "clean_pred_class",
    ]
    for index in range(capped_top_k):
        columns.append(sorted_probs[:, index].astype(np.float32))
        names.append(f"clean_top{index + 1}_prob")

    return np.stack(columns, axis=1).astype(np.float32), names


def save_selector_features(
    path: Path,
    *,
    split: str,
    model_name: str,
    image_ids: list[str],
    features: np.ndarray,
    feature_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write cached selector features as a compact NPZ artifact."""

    path = Path(path)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must have shape [images, features]")
    if len(image_ids) != features.shape[0]:
        raise ValueError("image_ids length must match feature rows")
    if len(feature_names) != features.shape[1]:
        raise ValueError("feature_names length must match feature columns")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array([1], dtype=np.int64),
        split=np.array([split]),
        model_name=np.array([model_name]),
        image_ids=np.asarray(image_ids, dtype=str),
        features=features,
        feature_names=np.asarray(feature_names, dtype=str),
        metadata_json=np.array([json.dumps(metadata or {}, sort_keys=True)]),
    )
    return path


def load_selector_features(path: Path) -> SavedSelectorFeatures:
    """Load cached selector features from an NPZ artifact."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        image_ids = [str(value) for value in data["image_ids"].tolist()]
        feature_names = [str(value) for value in data["feature_names"].tolist()]
        split = str(data["split"][0])
        model_name = str(data["model_name"][0])
        metadata_json = str(data["metadata_json"][0])
    if features.ndim != 2:
        raise ValueError("cached selector features must have shape [images, features]")
    if len(image_ids) != features.shape[0]:
        raise ValueError("cached selector feature image_ids must match feature rows")
    if len(feature_names) != features.shape[1]:
        raise ValueError("cached selector feature_names must match feature columns")
    metadata = json.loads(metadata_json)
    if not isinstance(metadata, dict):
        raise ValueError("cached selector feature metadata must be a JSON object")
    return SavedSelectorFeatures(
        split=split,
        model_name=model_name,
        image_ids=image_ids,
        features=features,
        feature_names=feature_names,
        metadata=metadata,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
