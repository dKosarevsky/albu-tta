from __future__ import annotations

import pytest


def test_oracle_gap_summary_quantifies_clean_oracle_and_learned_capture() -> None:
    from learned_tta.selector_analysis import summarize_oracle_gap

    summary = summarize_oracle_gap(
        clean_top1=0.80,
        oracle_top1=0.87,
        learned_top1=0.815,
        clean_nll=0.90,
        oracle_nll=0.50,
        learned_nll=0.75,
        forwards_per_image=17.0,
    )

    assert summary["top1_oracle_delta_pp"] == pytest.approx(7.0)
    assert summary["top1_learned_delta_pp"] == pytest.approx(1.5)
    assert summary["top1_oracle_capture"] == pytest.approx(0.2142857)
    assert summary["nll_oracle_delta"] == pytest.approx(-0.40)
    assert summary["nll_learned_delta"] == pytest.approx(-0.15)
    assert summary["nll_oracle_capture"] == pytest.approx(0.375)
    assert summary["forwards_per_image"] == 17.0
