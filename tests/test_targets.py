from __future__ import annotations

import numpy as np
import pytest

from learned_tta.targets import (
    TargetStats,
    compute_gain_targets,
    compute_target_stats,
    compute_true_class_nll,
    load_selector_targets,
    save_selector_targets,
    standardize_gain_targets,
)


@pytest.fixture
def logits_by_aug() -> dict[str, np.ndarray]:
    return {
        "aug_000": np.array(
            [
                [3.0, 1.0, -1.0],
                [0.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "aug_001": np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 3.0],
            ],
            dtype=np.float32,
        ),
        "aug_002": np.array(
            [
                [4.0, 0.0, -1.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }


@pytest.fixture
def class_idxs() -> np.ndarray:
    return np.array([0, 2], dtype=np.int64)


@pytest.mark.parametrize(
    ("logits", "class_idx", "expected_prob"),
    [
        (np.array([[3.0, 1.0, -1.0]], dtype=np.float32), np.array([0]), 0.8668133),
        (np.array([[0.0, 2.0, 1.0]], dtype=np.float32), np.array([2]), 0.2447285),
    ],
)
def test_compute_true_class_nll(
    logits: np.ndarray,
    class_idx: np.ndarray,
    expected_prob: float,
) -> None:
    result = compute_true_class_nll(logits, class_idx)

    np.testing.assert_allclose(
        result.prob_true,
        np.array([expected_prob], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(result.nll_true, -np.log([expected_prob]), rtol=1e-5)


@pytest.mark.parametrize(
    ("logits", "class_idxs", "match"),
    [
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0], dtype=np.int64),
            "logits must have shape",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.array([0], dtype=np.int64),
            "class_idxs must have shape",
        ),
    ],
)
def test_compute_true_class_nll_rejects_invalid_shapes(
    logits: np.ndarray,
    class_idxs: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        compute_true_class_nll(logits, class_idxs)


def test_compute_gain_targets_uses_clean_nll_baseline(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
) -> None:
    targets = compute_gain_targets(logits_by_aug, class_idxs, identity_aug_id="aug_000")

    assert targets.aug_ids == ["aug_000", "aug_001", "aug_002"]
    assert targets.gain.shape == (2, 3)
    np.testing.assert_allclose(targets.gain[:, 0], np.zeros(2), atol=1e-7)
    assert targets.gain[0, 2] > 0.0
    assert targets.gain[1, 1] > 0.0
    assert targets.gain[1, 2] < 0.0


def test_compute_gain_targets_requires_identity_logits(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="identity augmentation 'aug_999' is missing"):
        compute_gain_targets(logits_by_aug, class_idxs, identity_aug_id="aug_999")


def test_compute_target_stats_and_standardize_round_trip() -> None:
    gain = np.array(
        [
            [0.0, 1.0, -1.0],
            [0.0, 3.0, -3.0],
            [0.0, 5.0, -5.0],
        ],
        dtype=np.float32,
    )

    stats = compute_target_stats(gain)
    standardized = standardize_gain_targets(gain, stats)

    assert isinstance(stats, TargetStats)
    np.testing.assert_allclose(stats.mean, np.array([0.0, 3.0, -3.0], dtype=np.float32))
    assert stats.std[0] == pytest.approx(1.0)
    np.testing.assert_allclose(standardized.mean(axis=0), np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(standardized[:, 0], np.zeros(3), atol=1e-6)


def test_compute_target_stats_rejects_non_matrix_gain() -> None:
    with pytest.raises(ValueError, match="gain must have shape"):
        compute_target_stats(np.array([0.0, 1.0], dtype=np.float32))


def test_standardize_gain_targets_rejects_non_matrix_gain() -> None:
    with pytest.raises(ValueError, match="gain must have shape"):
        standardize_gain_targets(
            np.array([0.0, 1.0], dtype=np.float32),
            TargetStats(mean=np.zeros(2, dtype=np.float32), std=np.ones(2, dtype=np.float32)),
        )


def test_standardize_gain_targets_rejects_mismatched_stats() -> None:
    gain = np.zeros((2, 3), dtype=np.float32)
    stats = TargetStats(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="stats shape"):
        standardize_gain_targets(gain, stats)


def test_save_and_load_selector_targets(tmp_path) -> None:
    aug_ids = ["aug_000", "aug_001"]
    gain = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    stats = compute_target_stats(gain)
    target_z = standardize_gain_targets(gain, stats)

    path = tmp_path / "targets.npz"
    save_selector_targets(path, aug_ids=aug_ids, gain=gain, target_z=target_z, stats=stats)
    loaded = load_selector_targets(path)

    assert loaded.aug_ids == aug_ids
    np.testing.assert_array_equal(loaded.gain, gain)
    np.testing.assert_array_equal(loaded.target_z, target_z)
    np.testing.assert_array_equal(loaded.stats.mean, stats.mean)
    np.testing.assert_array_equal(loaded.stats.std, stats.std)
