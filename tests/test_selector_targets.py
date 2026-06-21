from __future__ import annotations

import numpy as np
import pytest


def test_selector_targets_build_gain_logit_margin_and_top1_fix() -> None:
    from learned_tta.selector_targets import build_selector_targets

    clean_logits = np.array([[2.0, 0.0], [0.1, 1.0]], dtype=np.float32)
    aug_logits = np.array(
        [
            [[0.0, 3.0], [3.0, 0.0]],
            [[0.2, 1.2], [1.5, 0.5]],
        ],
        dtype=np.float32,
    )
    labels = np.array([1, 0], dtype=np.int64)

    targets = build_selector_targets(clean_logits, aug_logits, labels)

    assert set(targets) >= {"nll_gain", "true_logit_gain", "margin_gain", "top1_fix"}
    assert targets["nll_gain"].shape == (2, 2)
    assert targets["true_logit_gain"][0].tolist() == pytest.approx([3.0, 0.0])
    assert targets["margin_gain"][0, 0] > targets["margin_gain"][0, 1]
    assert targets["top1_fix"][0, 0] == 1.0
    assert targets["top1_fix"][0, 1] == 0.0
    assert targets["top1_fix"][1, 0] == 0.0
    assert targets["top1_fix"][1, 1] == 1.0


def test_selector_targets_reject_shape_mismatch() -> None:
    from learned_tta.selector_targets import build_selector_targets

    with pytest.raises(ValueError, match="aug_logits"):
        build_selector_targets(
            clean_logits=np.zeros((2, 3), dtype=np.float32),
            aug_logits=np.zeros((2, 3), dtype=np.float32),
            labels=np.zeros(2, dtype=np.int64),
        )
