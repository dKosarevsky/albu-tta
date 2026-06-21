from __future__ import annotations

import numpy as np
import pytest


def test_clean_logit_features_include_margin_entropy_and_topk() -> None:
    from learned_tta.selector_features import clean_logit_features

    logits = np.array([[4.0, 2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    features, names = clean_logit_features(logits, top_k=2)

    assert names[:4] == [
        "clean_confidence",
        "clean_margin",
        "clean_entropy",
        "clean_pred_class",
    ]
    assert "clean_top1_prob" in names
    assert "clean_top2_prob" in names
    assert features.shape == (2, len(names))
    assert (
        features[0, names.index("clean_confidence")] > features[1, names.index("clean_confidence")]
    )
    assert features[0, names.index("clean_margin")] > features[1, names.index("clean_margin")]
    assert features[1, names.index("clean_entropy")] == pytest.approx(np.log(3.0))


def test_clean_logit_features_reject_bad_inputs() -> None:
    from learned_tta.selector_features import clean_logit_features

    with pytest.raises(ValueError, match="shape"):
        clean_logit_features(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="top_k"):
        clean_logit_features(np.zeros((2, 3), dtype=np.float32), top_k=0)
