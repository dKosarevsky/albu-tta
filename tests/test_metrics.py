from __future__ import annotations

import numpy as np
import pytest

from learned_tta.metrics import classification_metrics, expected_calibration_error


@pytest.fixture
def probabilities() -> np.ndarray:
    return np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ],
        dtype=np.float32,
    )


def test_classification_metrics_reports_topk_and_nll(probabilities: np.ndarray) -> None:
    class_idxs = np.array([0, 1, 2], dtype=np.int64)

    metrics = classification_metrics(probabilities, class_idxs, topk=(1, 2))

    assert metrics["top1"] == pytest.approx(1 / 3)
    assert metrics["top2"] == pytest.approx(2 / 3)
    assert metrics["nll"] == pytest.approx(float(-np.log([0.8, 0.3, 0.1]).mean()))


@pytest.mark.parametrize(
    ("probabilities", "class_idxs", "expected_message"),
    [
        (np.ones(3, dtype=np.float32), np.array([0], dtype=np.int64), "probabilities"),
        (np.ones((2, 3), dtype=np.float32), np.array([0], dtype=np.int64), "class_idxs"),
    ],
)
def test_classification_metrics_validates_shapes(
    probabilities: np.ndarray,
    class_idxs: np.ndarray,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        classification_metrics(probabilities, class_idxs)


def test_expected_calibration_error_bins_confidence(probabilities: np.ndarray) -> None:
    class_idxs = np.array([0, 1, 2], dtype=np.int64)

    ece = expected_calibration_error(probabilities, class_idxs, bins=2)

    assert ece == pytest.approx(0.3)
