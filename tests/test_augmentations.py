from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learned_tta.augmentations import (
    AugmentationCandidate,
    apply_candidate,
    build_augmentation_audit,
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
    assert candidates[0].determinism == "fixed"
    assert [candidate.id for candidate in candidates] == [f"aug_{idx:03d}" for idx in range(100)]
    assert all(
        candidate.params.get("p") == 1.0 for candidate in candidates if not candidate.is_identity
    )
    assert {
        candidate.name: candidate.determinism
        for candidate in candidates
        if candidate.class_name == "PlanckianJitter"
    } == {
        "planckian_5000_6000": "seeded_stochastic",
        "planckian_5500_6500": "seeded_stochastic",
        "planckian_6000_7500": "seeded_stochastic",
    }


def test_fixed_augmentation_candidates_must_collapse_range_params() -> None:
    candidates = [
        AugmentationCandidate(
            id="aug_000",
            name="brightness_range_not_fixed",
            class_name="RandomBrightnessContrast",
            params={"brightness_range": [0.1, 0.2], "contrast_range": [0.0, 0.0], "p": 1.0},
            determinism="fixed",
        )
    ]

    with pytest.raises(ValueError, match="aug_000 fixed candidate has non-fixed range"):
        validate_augmentation_registry(candidates, expected_count=1)


def test_seeded_stochastic_candidates_may_have_non_collapsed_ranges() -> None:
    candidates = [
        AugmentationCandidate(
            id="aug_000",
            name="planckian_seeded",
            class_name="PlanckianJitter",
            params={"mode": "blackbody", "temperature_range": [5000, 6000], "p": 1.0},
            determinism="seeded_stochastic",
        )
    ]

    validate_augmentation_registry(candidates, expected_count=1)


@pytest.mark.parametrize(
    ("candidates", "expected_count", "match"),
    [
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="identity",
                    class_name=None,
                ),
                AugmentationCandidate(
                    id="aug_001",
                    name="duplicate",
                    class_name=None,
                ),
            ],
            1,
            "expected 1 candidates, found 2",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="identity",
                    class_name=None,
                ),
                AugmentationCandidate(
                    id="aug_000",
                    name="duplicate",
                    class_name=None,
                ),
            ],
            2,
            "augmentation candidate ids must be unique",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_001",
                    name="identity",
                    class_name=None,
                ),
            ],
            1,
            "augmentation candidate ids must be sequential from aug_000",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="flip_a",
                    class_name="HorizontalFlip",
                    params={"p": 1.0},
                ),
                AugmentationCandidate(
                    id="aug_001",
                    name="flip_b",
                    class_name="HorizontalFlip",
                    params={"p": 1.0},
                ),
            ],
            2,
            "augmentation transform specs must be unique",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="bad_determinism",
                    class_name=None,
                    determinism="random",
                ),
            ],
            1,
            "determinism must be one of",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="bad_transform",
                    class_name="NoSuchTransform",
                    params={"p": 1.0},
                ),
            ],
            1,
            "unknown Albumentations transform NoSuchTransform",
        ),
        (
            [
                AugmentationCandidate(
                    id="aug_000",
                    name="missing_p",
                    class_name="HorizontalFlip",
                    params={},
                ),
            ],
            1,
            "aug_000 must set p=1.0",
        ),
    ],
)
def test_validate_augmentation_registry_rejects_invalid_candidates(
    candidates: list[AugmentationCandidate],
    expected_count: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_augmentation_registry(candidates, expected_count=expected_count)


def test_augmentation_registry_outputs_are_deterministic_with_fixed_seed() -> None:
    candidates = load_augmentation_registry(REGISTRY_PATH)
    image = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)

    for candidate in candidates:
        first = apply_candidate(candidate, image, seed=20260522)
        second = apply_candidate(candidate, image, seed=20260522)
        assert np.array_equal(first, second), candidate.id
        assert first.shape == image.shape, candidate.id
        assert first.dtype == image.dtype, candidate.id


def test_build_augmentation_audit_is_stable_json_payload() -> None:
    candidates = load_augmentation_registry(REGISTRY_PATH)

    audit = build_augmentation_audit(candidates, seed=20260522)
    encoded = json.dumps(audit, sort_keys=True)

    assert "aug_000" in encoded
    assert audit["version"] == 1
    assert audit["seed"] == 20260522
    assert audit["candidate_count"] == 100
    assert audit["identity_id"] == "aug_000"
    assert audit["runtime"]["python"]
    assert audit["runtime"]["packages"]["albumentationsx"]
    assert audit["runtime"]["packages"]["opencv-python-headless"]
    assert audit["runtime"]["packages"]["numpy"]
    assert audit["runtime"]["opencv"]["version"]
    assert isinstance(audit["runtime"]["opencv"]["threads"], int)
    assert audit["candidates"][0] == {
        "id": "aug_000",
        "name": "identity",
        "determinism": "fixed",
        "class_name": None,
        "params": {},
        "serialized_transform": None,
    }
    assert audit["candidates"][19]["id"] == "aug_019"
    assert audit["candidates"][19]["params"] == {
        "brightness_range": [0.1, 0.1],
        "contrast_range": [0.0, 0.0],
        "p": 1.0,
    }
    serialized = audit["candidates"][19]["serialized_transform"]
    assert serialized["transform"]["seed"] == 20260522
    assert serialized["transform"]["transforms"][0]["__class_fullname__"] == (
        "RandomBrightnessContrast"
    )
