from __future__ import annotations

import numpy as np
import pytest

from learned_tta.tta_eval import (
    _mean_selection_size,
    average_probabilities,
    class_weighted_probabilities,
    evaluate_all_100_uniform,
    evaluate_class_weighted_tta,
    evaluate_clean,
    evaluate_fixed_light_tta,
    evaluate_global_weighted_tta,
    evaluate_learned_topk_softmax_weighted,
    evaluate_learned_topk_uniform,
    evaluate_oracle_topk_uniform,
    evaluate_random_topk,
    evaluate_selected_tta,
    fixed_light_tta_selection,
    global_weighted_probabilities,
    learned_topk_selection,
    oracle_selection_recall,
    oracle_topk_selection,
    random_topk_selection,
    select_best_k,
    weighted_average_probabilities,
)


@pytest.fixture
def logits_by_aug() -> dict[str, np.ndarray]:
    return {
        "aug_000": np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        "aug_001": np.array([[1.0, 3.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float32),
        "aug_002": np.array([[4.0, 0.0, 0.0], [3.0, 0.0, 1.0]], dtype=np.float32),
        "aug_003": np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]], dtype=np.float32),
    }


@pytest.fixture
def aug_ids() -> list[str]:
    return ["aug_000", "aug_001", "aug_002", "aug_003"]


@pytest.fixture
def predicted_gain() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.2, 0.8, -0.1],
            [0.0, 0.2, -0.3, 0.9],
        ],
        dtype=np.float32,
    )


def test_average_probabilities_averages_softmax_outputs(
    logits_by_aug: dict[str, np.ndarray],
) -> None:
    probabilities = average_probabilities(logits_by_aug, selected_aug_ids=["aug_000", "aug_002"])

    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2), rtol=1e-6)


def test_learned_topk_selection_always_includes_identity() -> None:
    predicted_gain = np.array(
        [
            [0.0, 0.2, 0.8, -1.0],
            [0.0, 0.7, -0.1, 0.3],
        ],
        dtype=np.float32,
    )

    selected = learned_topk_selection(
        aug_ids=["aug_000", "aug_001", "aug_002", "aug_003"],
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=2,
    )

    assert selected == [
        ["aug_000", "aug_002", "aug_001"],
        ["aug_000", "aug_001", "aug_003"],
    ]


def test_learned_topk_zero_selects_identity_only(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    predicted_gain: np.ndarray,
) -> None:
    class_idxs = np.array([0, 2], dtype=np.int64)

    selected = learned_topk_selection(
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=0,
    )
    uniform = evaluate_learned_topk_uniform(
        logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=0,
    )
    weighted = evaluate_learned_topk_softmax_weighted(
        logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=0,
    )
    clean = evaluate_clean(logits_by_aug, class_idxs=class_idxs, identity_aug_id="aug_000")

    assert selected == [["aug_000"], ["aug_000"]]
    assert uniform["forwards_per_image"] == pytest.approx(1.0)
    assert weighted["forwards_per_image"] == pytest.approx(1.0)
    for key, value in clean.items():
        assert uniform[key] == pytest.approx(value)
        assert weighted[key] == pytest.approx(value)


def test_fixed_and_random_selection_are_reproducible(aug_ids: list[str]) -> None:
    fixed = fixed_light_tta_selection(aug_ids, identity_aug_id="aug_000", k=2)
    first = random_topk_selection(
        aug_ids,
        num_images=3,
        identity_aug_id="aug_000",
        k=2,
        seed=20260522,
    )
    second = random_topk_selection(
        aug_ids,
        num_images=3,
        identity_aug_id="aug_000",
        k=2,
        seed=20260522,
    )

    assert fixed == ["aug_000", "aug_001", "aug_002"]
    assert first == second
    assert all(selection[0] == "aug_000" for selection in first)


def test_oracle_topk_selection_uses_true_gain(logits_by_aug: dict[str, np.ndarray]) -> None:
    selected = oracle_topk_selection(
        logits_by_aug,
        class_idxs=np.array([0, 2], dtype=np.int64),
        identity_aug_id="aug_000",
        k=1,
    )

    assert selected == [["aug_000", "aug_002"], ["aug_000", "aug_003"]]


def test_evaluate_selected_tta_supports_per_image_selection(
    logits_by_aug: dict[str, np.ndarray],
) -> None:
    selected = [["aug_000", "aug_002"], ["aug_000", "aug_001"]]

    metrics = evaluate_selected_tta(
        logits_by_aug,
        selected_aug_ids=selected,
        class_idxs=np.array([0, 2], dtype=np.int64),
    )

    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["nll"] > 0.0


@pytest.mark.parametrize(
    "strategy",
    [
        "clean",
        "fixed_light_tta",
        "random_topk",
        "all_100_uniform",
        "learned_topk_uniform",
        "oracle_topk_uniform",
    ],
)
def test_tta_strategies_report_shared_metric_schema(
    strategy: str,
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    predicted_gain: np.ndarray,
) -> None:
    class_idxs = np.array([0, 2], dtype=np.int64)

    if strategy == "clean":
        metrics = evaluate_clean(logits_by_aug, class_idxs, identity_aug_id="aug_000")
    elif strategy == "fixed_light_tta":
        metrics = evaluate_fixed_light_tta(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            identity_aug_id="aug_000",
            k=1,
        )
    elif strategy == "random_topk":
        metrics = evaluate_random_topk(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            identity_aug_id="aug_000",
            k=1,
            seed=20260522,
        )
    elif strategy == "all_100_uniform":
        metrics = evaluate_all_100_uniform(logits_by_aug, class_idxs, aug_ids=aug_ids)
    elif strategy == "learned_topk_uniform":
        metrics = evaluate_learned_topk_uniform(
            logits_by_aug,
            class_idxs,
            aug_ids=aug_ids,
            predicted_gain=predicted_gain,
            identity_aug_id="aug_000",
            k=1,
        )
    else:
        metrics = evaluate_oracle_topk_uniform(
            logits_by_aug,
            class_idxs,
            identity_aug_id="aug_000",
            k=1,
        )

    assert set(metrics) == {
        "top1",
        "top5",
        "nll",
        "ece",
        "forwards_per_image",
        "relative_compute_vs_all",
    }
    assert metrics["forwards_per_image"] > 0.0
    assert 0.0 < metrics["relative_compute_vs_all"] <= 1.0


def test_learned_weighted_tta_uses_selector_scores(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    predicted_gain: np.ndarray,
) -> None:
    class_idxs = np.array([0, 2], dtype=np.int64)

    uniform = evaluate_learned_topk_uniform(
        logits_by_aug,
        class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=1,
    )
    weighted = evaluate_learned_topk_softmax_weighted(
        logits_by_aug,
        class_idxs,
        aug_ids=aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=1,
    )

    assert weighted["forwards_per_image"] == pytest.approx(uniform["forwards_per_image"])
    assert weighted["nll"] != pytest.approx(uniform["nll"])


def test_global_weighted_tta_uses_nonzero_weights_as_compute(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
) -> None:
    class_idxs = np.array([0, 2], dtype=np.int64)

    metrics = evaluate_global_weighted_tta(
        logits_by_aug,
        class_idxs,
        aug_ids=aug_ids,
        weights=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        active_threshold=1e-6,
    )

    assert metrics["top1"] == pytest.approx(0.5)
    assert metrics["forwards_per_image"] == pytest.approx(1.0)
    assert metrics["relative_compute_vs_all"] == pytest.approx(0.25)


def test_class_weighted_tta_supports_per_class_augmentation_weights(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
) -> None:
    class_idxs = np.array([0, 2], dtype=np.int64)
    class_weights = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    metrics = evaluate_class_weighted_tta(
        logits_by_aug,
        class_idxs,
        aug_ids=aug_ids,
        class_weights=class_weights,
        active_threshold=1e-6,
    )

    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["forwards_per_image"] == pytest.approx(3.0)
    assert metrics["relative_compute_vs_all"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("selected", "oracle", "expected_recall"),
    [
        ([["aug_000", "aug_001"]], [["aug_000", "aug_001"]], 1.0),
        ([["aug_000", "aug_001"]], [["aug_000", "aug_002"]], 0.0),
        ([["aug_000", "aug_001", "aug_002"]], [["aug_000", "aug_002", "aug_003"]], 0.5),
    ],
)
def test_oracle_selection_recall_ignores_identity(
    selected: list[list[str]],
    oracle: list[list[str]],
    expected_recall: float,
) -> None:
    recall = oracle_selection_recall(
        selected_aug_ids=selected,
        oracle_aug_ids=oracle,
        identity_aug_id="aug_000",
    )

    assert recall == pytest.approx(expected_recall)


def test_select_best_k_prefers_best_metric_then_lower_compute() -> None:
    results_by_k = {
        1: {"nll": 0.7},
        2: {"nll": 0.5},
        4: {"nll": 0.5},
    }

    assert select_best_k(results_by_k, metric="nll", higher_is_better=False) == 2


def test_select_best_k_can_maximize_metric_and_rejects_empty_results() -> None:
    assert select_best_k(
        {
            1: {"top1": 0.7},
            2: {"top1": 0.8},
            4: {"top1": 0.8},
        },
        metric="top1",
        higher_is_better=True,
    ) == 2
    with pytest.raises(ValueError, match="results_by_k must not be empty"):
        select_best_k({}, metric="nll", higher_is_better=False)


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        (np.array([0.0, 1.0], dtype=np.float32), "predicted_gain must have shape"),
        (np.zeros((3, 4), dtype=np.float32), "predicted_gain shape must match"),
    ],
)
def test_weighted_average_probabilities_rejects_invalid_score_shapes(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    scores: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        weighted_average_probabilities(
            logits_by_aug=logits_by_aug,
            selected_aug_ids=[["aug_000"], ["aug_001"]],
            aug_ids=aug_ids,
            predicted_gain=scores,
        )


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        (np.array([0.0, 1.0], dtype=np.float32), "predicted_gain must have shape"),
        (np.zeros((2, 3), dtype=np.float32), "predicted_gain width must match"),
    ],
)
def test_learned_topk_selection_rejects_invalid_score_shapes(
    aug_ids: list[str],
    scores: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        learned_topk_selection(
            aug_ids=aug_ids,
            predicted_gain=scores,
            identity_aug_id="aug_000",
            k=1,
        )


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        (np.array([1.0, -0.1, 0.0, 0.0], dtype=np.float32), "weights must be non-negative"),
        (
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "weights must contain at least one positive value",
        ),
        (np.array([1.0, 0.0], dtype=np.float32), "weights must have shape"),
    ],
)
def test_global_weighted_probabilities_rejects_invalid_weights(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    weights: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        global_weighted_probabilities(logits_by_aug, aug_ids=aug_ids, weights=weights)


@pytest.mark.parametrize(
    ("class_weights", "match"),
    [
        (np.ones((3,), dtype=np.float32), "class_weights must have shape"),
        (np.ones((3, 2), dtype=np.float32), "class_weights must have shape"),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "class_weights must be non-negative",
        ),
        (np.zeros((3, 4), dtype=np.float32), "each class must have at least one positive"),
        (np.ones((2, 4), dtype=np.float32), "class_weights class count must match"),
    ],
)
def test_class_weighted_probabilities_rejects_invalid_weights(
    logits_by_aug: dict[str, np.ndarray],
    aug_ids: list[str],
    class_weights: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        class_weighted_probabilities(
            logits_by_aug=logits_by_aug,
            aug_ids=aug_ids,
            class_weights=class_weights,
        )


def test_mean_selection_size_reports_zero_for_empty_selection() -> None:
    assert _mean_selection_size([]) == pytest.approx(0.0)
