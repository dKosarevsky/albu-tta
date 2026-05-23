"""AlbumentationsX candidate registry helpers."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


@dataclass(frozen=True, slots=True)
class AugmentationCandidate:
    """One deterministic TTA augmentation candidate."""

    id: str
    name: str
    class_name: str | None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_identity(self) -> bool:
        return self.class_name is None

    @property
    def transform_spec(self) -> tuple[str | None, tuple[tuple[str, str], ...]]:
        return (
            self.class_name,
            tuple(sorted((key, repr(value)) for key, value in self.params.items())),
        )


def load_augmentation_registry(path: Path) -> list[AugmentationCandidate]:
    """Load augmentation candidates from a YAML registry."""

    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    candidates = []
    for raw_candidate in data["candidates"]:
        transform = raw_candidate.get("transform")
        if transform is None:
            class_name = None
            params: dict[str, Any] = {}
        else:
            class_name = transform["class_name"]
            params = dict(transform.get("params", {}))

        candidates.append(
            AugmentationCandidate(
                id=raw_candidate["id"],
                name=raw_candidate["name"],
                class_name=class_name,
                params=params,
            )
        )

    return candidates


def validate_augmentation_registry(
    candidates: list[AugmentationCandidate],
    expected_count: int,
) -> None:
    """Validate candidate count, ids, class names, and deterministic parameters."""

    ids = [candidate.id for candidate in candidates]
    if len(candidates) != expected_count:
        raise ValueError(f"expected {expected_count} candidates, found {len(candidates)}")
    if len(set(ids)) != len(ids):
        raise ValueError("augmentation candidate ids must be unique")

    expected_ids = [f"aug_{idx:03d}" for idx in range(expected_count)]
    if ids != expected_ids:
        raise ValueError("augmentation candidate ids must be sequential from aug_000")

    transform_specs = [
        candidate.transform_spec for candidate in candidates if not candidate.is_identity
    ]
    if len(set(transform_specs)) != len(transform_specs):
        raise ValueError("augmentation transform specs must be unique")

    for candidate in candidates:
        if candidate.is_identity:
            continue
        if not hasattr(A, candidate.class_name or ""):
            raise ValueError(f"unknown Albumentations transform {candidate.class_name}")
        if candidate.params.get("p") != 1.0:
            raise ValueError(f"{candidate.id} must set p=1.0")


def build_augmentation_audit(candidates: list[AugmentationCandidate], seed: int) -> dict[str, Any]:
    """Build a stable JSON-ready audit payload for the augmentation registry."""

    identity_id = next((candidate.id for candidate in candidates if candidate.is_identity), None)
    return {
        "version": 1,
        "seed": int(seed),
        "candidate_count": len(candidates),
        "identity_id": identity_id,
        "runtime": _runtime_audit(),
        "candidates": [
            {
                "id": candidate.id,
                "name": candidate.name,
                "class_name": candidate.class_name,
                "params": _json_ready(candidate.params),
                "serialized_transform": _serialized_transform(candidate, seed=seed),
            }
            for candidate in candidates
        ],
    }


def write_augmentation_audit(
    candidates: list[AugmentationCandidate],
    output_path: Path,
    seed: int,
) -> Path:
    """Write a stable registry audit JSON artifact."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_augmentation_audit(candidates, seed=seed), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def instantiate_candidate(candidate: AugmentationCandidate, seed: int) -> A.Compose | None:
    """Instantiate one candidate as an AlbumentationsX Compose pipeline."""

    if candidate.is_identity:
        return None

    transform_cls = getattr(A, candidate.class_name or "")
    transform = transform_cls(**candidate.params)
    return A.Compose([transform], seed=seed, telemetry=False)


def apply_candidate(
    candidate: AugmentationCandidate,
    image: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply one candidate to an RGB uint8 image."""

    pipeline = instantiate_candidate(candidate, seed=seed)
    if pipeline is None:
        return image.copy()
    return pipeline(image=image)["image"]


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _serialized_transform(candidate: AugmentationCandidate, seed: int) -> dict[str, Any] | None:
    pipeline = instantiate_candidate(candidate, seed=seed)
    if pipeline is None:
        return None
    return _json_ready(pipeline.to_dict())


def _runtime_audit() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "packages": {
            "albumentationsx": _package_version("albumentationsx"),
            "opencv-python-headless": _package_version("opencv-python-headless"),
            "numpy": np.__version__,
            "pyyaml": _package_version("pyyaml"),
        },
        "opencv": {
            "version": cv2.__version__,
            "threads": int(cv2.getNumThreads()),
            "opencl": bool(cv2.ocl.useOpenCL()),
        },
    }


def _package_version(distribution_name: str) -> str:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"
