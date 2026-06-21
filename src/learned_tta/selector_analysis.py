"""Analysis helpers for learned-selector oracle gap reporting."""

from __future__ import annotations


def summarize_oracle_gap(
    *,
    clean_top1: float,
    oracle_top1: float,
    learned_top1: float,
    clean_nll: float,
    oracle_nll: float,
    learned_nll: float,
    forwards_per_image: float,
) -> dict[str, float]:
    """Summarize how much oracle headroom the learned selector captures."""

    top1_oracle_delta = oracle_top1 - clean_top1
    top1_learned_delta = learned_top1 - clean_top1
    nll_oracle_delta = oracle_nll - clean_nll
    nll_learned_delta = learned_nll - clean_nll
    return {
        "clean_top1": float(clean_top1),
        "oracle_top1": float(oracle_top1),
        "learned_top1": float(learned_top1),
        "top1_oracle_delta_pp": float(top1_oracle_delta * 100.0),
        "top1_learned_delta_pp": float(top1_learned_delta * 100.0),
        "top1_oracle_capture": _capture_ratio(top1_learned_delta, top1_oracle_delta),
        "clean_nll": float(clean_nll),
        "oracle_nll": float(oracle_nll),
        "learned_nll": float(learned_nll),
        "nll_oracle_delta": float(nll_oracle_delta),
        "nll_learned_delta": float(nll_learned_delta),
        "nll_oracle_capture": _capture_ratio(-nll_learned_delta, -nll_oracle_delta),
        "forwards_per_image": float(forwards_per_image),
    }


def _capture_ratio(learned_delta: float, oracle_delta: float) -> float:
    if oracle_delta == 0.0:
        return 0.0
    return float(learned_delta / oracle_delta)
