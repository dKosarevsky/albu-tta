from __future__ import annotations

import numpy as np
import pytest

from learned_tta.targets import (
    TargetStats,
    compute_gain_targets,
    compute_selector_target_matrices,
    compute_target_stats,
    compute_true_class_nll,
    load_selector_targets,
    save_selector_targets,
    select_selector_target_matrix,
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


def test_compute_selector_target_matrices_exposes_trainable_ablation_targets(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
) -> None:
    matrices = compute_selector_target_matrices(
        logits_by_aug,
        class_idxs,
        identity_aug_id="aug_000",
        softmax_temperature=0.5,
    )

    assert matrices.aug_ids == ["aug_000", "aug_001", "aug_002"]
    assert matrices.gain.shape == (2, 3)
    assert matrices.nll.shape == (2, 3)
    np.testing.assert_allclose(matrices.negative_nll, -matrices.nll)
    np.testing.assert_allclose(matrices.helpfulness, (matrices.gain > 0.0).astype(np.float32))
    assert matrices.rank[0, 2] == pytest.approx(1.0)
    assert matrices.rank[0, 1] == pytest.approx(0.0)
    np.testing.assert_allclose(matrices.softmax_weight.sum(axis=1), np.ones(2), atol=1e-6)
    assert np.all(matrices.softmax_weight >= 0.0)
    np.testing.assert_allclose(
        matrices.true_logit[:, 0],
        np.array([3.0, 1.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "target_kind",
    ["gain", "negative_nll", "helpfulness", "rank", "softmax_weight", "true_logit"],
)
def test_select_selector_target_matrix_returns_high_is_better_target(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
    target_kind: str,
) -> None:
    matrices = compute_selector_target_matrices(
        logits_by_aug,
        class_idxs,
        identity_aug_id="aug_000",
    )

    selected = select_selector_target_matrix(matrices, target_kind)

    assert selected.shape == matrices.gain.shape


def test_select_selector_target_matrix_rejects_raw_nll_for_training(
    logits_by_aug: dict[str, np.ndarray],
    class_idxs: np.ndarray,
) -> None:
    matrices = compute_selector_target_matrices(
        logits_by_aug,
        class_idxs,
        identity_aug_id="aug_000",
    )

    with pytest.raises(ValueError, match="raw nll is diagnostic-only"):
        select_selector_target_matrix(matrices, "nll")


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
    image_ids = ["image-0", "image-1"]
    gain = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    stats = compute_target_stats(gain)
    target_z = standardize_gain_targets(gain, stats)

    path = tmp_path / "targets.npz"
    save_selector_targets(
        path,
        aug_ids=aug_ids,
        image_ids=image_ids,
        gain=gain,
        target_z=target_z,
        stats=stats,
    )
    loaded = load_selector_targets(path)

    assert loaded.aug_ids == aug_ids
    assert loaded.image_ids == image_ids
    assert loaded.target_kind == "gain"
    assert loaded.higher_is_better is True
    np.testing.assert_array_equal(loaded.gain, gain)
    np.testing.assert_array_equal(loaded.target_z, target_z)
    np.testing.assert_array_equal(loaded.stats.mean, stats.mean)
    np.testing.assert_array_equal(loaded.stats.std, stats.std)


def test_save_and_load_selector_targets_preserves_target_kind(tmp_path) -> None:
    path = tmp_path / "targets.npz"
    save_selector_targets(
        path,
        aug_ids=["aug_000", "aug_001"],
        image_ids=["image-0", "image-1"],
        gain=np.zeros((2, 2), dtype=np.float32),
        target_z=np.ones((2, 2), dtype=np.float32),
        stats=TargetStats(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
        ),
        target_kind="softmax_weight",
        higher_is_better=True,
    )

    loaded = load_selector_targets(path)

    assert loaded.target_kind == "softmax_weight"
    assert loaded.higher_is_better is True
