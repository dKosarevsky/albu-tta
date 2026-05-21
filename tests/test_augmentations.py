from __future__ import annotations

from pathlib import Path

import numpy as np

from learned_tta.augmentations import (
    apply_candidate,
    load_augmentation_registry,
    validate_augmentation_registry,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "augmentations" / "imagenet100.yaml"


def test_augmentation_registry_has_exactly_100_single_transform_candidates() -> None:
    candidates = load_augmentation_registry(REGISTRY_PATH)

    validate_augmentation_registry(candidates, expected_count=100)

    assert len(candidates) == 100
    assert candidates[0].id == "aug_000"
    assert candidates[0].class_name is None
    assert [candidate.id for candidate in candidates] == [f"aug_{idx:03d}" for idx in range(100)]
    assert all(
        candidate.params.get("p") == 1.0 for candidate in candidates if not candidate.is_identity
    )


def test_augmentation_registry_outputs_are_deterministic_with_fixed_seed() -> None:
    candidates = load_augmentation_registry(REGISTRY_PATH)
    image = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)

    for candidate in candidates:
        first = apply_candidate(candidate, image, seed=20260522)
        second = apply_candidate(candidate, image, seed=20260522)
        assert np.array_equal(first, second), candidate.id
        assert first.shape == image.shape, candidate.id
        assert first.dtype == image.dtype, candidate.id
