from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from learned_tta.reporting import (
    _selection_frequency,
    build_augmentation_impact_table,
    build_compute_table,
    build_correction_table,
    build_metrics_table,
    build_results_markdown,
    write_report_artifacts,
)


@pytest.fixture
def metrics_by_strategy() -> dict[str, dict[str, float]]:
    return {
        "clean": {
            "top1": 0.76,
            "top5": 0.93,
            "nll": 0.95,
            "ece": 0.04,
            "forwards_per_image": 1.0,
            "relative_compute_vs_all": 0.01,
        },
        "learned_topk_uniform": {
            "top1": 0.77,
            "top5": 0.94,
            "nll": 0.91,
            "ece": 0.03,
            "forwards_per_image": 5.0,
            "relative_compute_vs_all": 0.05,
        },
    }


def test_build_metrics_table_preserves_strategy_order(
    metrics_by_strategy: dict[str, dict[str, float]],
) -> None:
    table = build_metrics_table(metrics_by_strategy)

    assert table["strategy"].tolist() == ["clean", "learned_topk_uniform"]
    assert table["nll"].tolist() == pytest.approx([0.95, 0.91])


def test_build_compute_table_keeps_only_compute_columns(
    metrics_by_strategy: dict[str, dict[str, float]],
) -> None:
    table = build_compute_table(metrics_by_strategy)

    assert table.columns.tolist() == [
        "strategy",
        "forwards_per_image",
        "relative_compute_vs_all",
    ]
    assert table["relative_compute_vs_all"].tolist() == pytest.approx([0.01, 0.05])


def test_build_correction_table_counts_clean_tta_transitions() -> None:
    table = build_correction_table(
        clean_correct=np.array([True, True, False, False]),
        predictions_by_strategy={
            "clean": np.array([0, 1, 0, 1]),
            "tta": np.array([0, 0, 1, 1]),
        },
        class_idxs=np.array([0, 1, 1, 0]),
    )

    clean_row = table[table["strategy"] == "clean"].iloc[0]
    tta_row = table[table["strategy"] == "tta"].iloc[0]

    assert clean_row["clean_wrong_tta_right"] == 0
    assert clean_row["clean_right_tta_wrong"] == 0
    assert tta_row["clean_wrong_tta_right"] == 1
    assert tta_row["clean_right_tta_wrong"] == 1
    assert tta_row["both_right"] == 1
    assert tta_row["both_wrong"] == 1


def test_build_correction_table_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="clean_correct and class_idxs"):
        build_correction_table(
            clean_correct=np.array([True, False]),
            predictions_by_strategy={},
            class_idxs=np.array([0]),
        )

    with pytest.raises(ValueError, match="predictions for tta must match"):
        build_correction_table(
            clean_correct=np.array([True, False]),
            predictions_by_strategy={"tta": np.array([0])},
            class_idxs=np.array([0, 1]),
        )


@pytest.mark.parametrize(
    ("selected_aug_ids", "oracle_aug_ids", "expected_selection", "expected_oracle"),
    [
        (
            [["aug_000", "aug_001"], ["aug_000", "aug_002"]],
            [["aug_000", "aug_001"], ["aug_000", "aug_001"]],
            [1.0, 0.5, 0.5],
            [1.0, 1.0, 0.0],
        ),
    ],
)
def test_build_augmentation_impact_table_counts_selection_frequency(
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
    expected_selection: list[float],
    expected_oracle: list[float],
) -> None:
    table = build_augmentation_impact_table(
        aug_ids=["aug_000", "aug_001", "aug_002"],
        gain=np.array([[0.0, 1.0, -1.0], [0.0, 3.0, 2.0]], dtype=np.float32),
        selected_aug_ids=selected_aug_ids,
        oracle_aug_ids=oracle_aug_ids,
    )

    assert table["mean_gain"].tolist() == pytest.approx([0.0, 2.0, 0.5])
    assert table["selection_frequency"].tolist() == pytest.approx(expected_selection)
    assert table["oracle_frequency"].tolist() == pytest.approx(expected_oracle)


@pytest.mark.parametrize(
    ("gain", "selected_aug_ids", "oracle_aug_ids", "match"),
    [
        (
            np.array([0.0, 1.0], dtype=np.float32),
            [["aug_000"]],
            [["aug_000"]],
            "gain must have shape",
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            [["aug_000"]],
            [["aug_000"], ["aug_000"]],
            "gain shape must match",
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            [["aug_000"], ["aug_000"]],
            [["aug_000"]],
            "oracle_aug_ids length",
        ),
    ],
)
def test_build_augmentation_impact_table_rejects_shape_mismatches(
    gain: np.ndarray,
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_augmentation_impact_table(
            aug_ids=["aug_000", "aug_001"],
            gain=gain,
            selected_aug_ids=selected_aug_ids,
            oracle_aug_ids=oracle_aug_ids,
        )


def test_selection_frequency_handles_empty_selections() -> None:
    frequency = _selection_frequency(["aug_000", "aug_001"], selected_aug_ids=[])

    assert frequency.tolist() == pytest.approx([0.0, 0.0])


def test_build_results_markdown_avoids_state_of_the_art_claims(
    metrics_by_strategy: dict[str, dict[str, float]],
) -> None:
    markdown = build_results_markdown(
        public_metrics=metrics_by_strategy,
        private_metrics=metrics_by_strategy,
        tuned_k=4,
    )

    assert "state-of-the-art" not in markdown.lower()
    assert "tuned k: 4" in markdown.lower()
    assert "learned_topk_uniform" in markdown


def test_write_report_artifacts_creates_expected_files(
    tmp_path: Path,
    metrics_by_strategy: dict[str, dict[str, float]],
) -> None:
    written = write_report_artifacts(
        report_dir=tmp_path,
        public_metrics=metrics_by_strategy,
        private_metrics=metrics_by_strategy,
        aug_ids=["aug_000", "aug_001"],
        gain=np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32),
        selected_aug_ids=[["aug_000", "aug_001"], ["aug_000", "aug_001"]],
        oracle_aug_ids=[["aug_000", "aug_001"], ["aug_000", "aug_001"]],
        tuned_k=1,
    )

    assert written.results_md == tmp_path / "results.md"
    assert written.public_metrics_csv == tmp_path / "tables" / "public_metrics.csv"
    assert written.private_metrics_csv == tmp_path / "tables" / "private_metrics.csv"
    assert written.compute_csv == tmp_path / "tables" / "compute.csv"
    assert written.augmentation_impact_csv == tmp_path / "tables" / "augmentation_impact.csv"
    assert all(path.exists() for path in written)
    assert pd.read_csv(written.augmentation_impact_csv)["aug_id"].tolist() == [
        "aug_000",
        "aug_001",
    ]
