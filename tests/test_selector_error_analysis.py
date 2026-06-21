from __future__ import annotations

import numpy as np
import pandas as pd

from learned_tta.selector_error_analysis import build_selector_error_analysis_table


def test_build_selector_error_analysis_table_summarizes_policy_failures() -> None:
    logits_by_aug = {
        "aug_000": np.asarray([[5.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        "aug_001": np.asarray([[5.0, 0.0], [0.0, 4.0]], dtype=np.float32),
    }
    class_idxs = np.asarray([0, 1], dtype=np.int64)
    predicted_gain = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    table = build_selector_error_analysis_table(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        aug_ids=["aug_000", "aug_001"],
        predicted_gain=predicted_gain,
        identity_aug_id="aug_000",
        k=1,
        confidence_bins=[0.0, 0.75, 1.01],
    )

    assert table["images"].sum() == 2
    assert table["clean_wrong_tta_right"].sum() == 1
    assert table["clean_right_tta_wrong"].sum() == 0
    assert table["mean_oracle_recall"].mean() == 1.0


def test_build_selector_error_analysis_table_writes_csv(tmp_path) -> None:
    logits_by_aug = {
        "aug_000": np.asarray([[5.0, 0.0]], dtype=np.float32),
        "aug_001": np.asarray([[0.0, 5.0]], dtype=np.float32),
    }
    output_path = tmp_path / "selector_error_analysis.csv"

    table = build_selector_error_analysis_table(
        logits_by_aug=logits_by_aug,
        class_idxs=np.asarray([0], dtype=np.int64),
        aug_ids=["aug_000", "aug_001"],
        predicted_gain=np.asarray([[0.0, 1.0]], dtype=np.float32),
        identity_aug_id="aug_000",
        k=1,
        output_path=output_path,
    )

    loaded = pd.read_csv(output_path)
    assert loaded["images"].tolist() == table["images"].tolist()
