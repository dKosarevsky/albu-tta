"""Final report builder for learned TTA experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from learned_tta.config import load_experiment_config
from learned_tta.reporting import (
    build_augmentation_impact_table,
    build_compute_table,
    build_metrics_table,
)
from learned_tta.targets import load_selector_targets
from learned_tta.tta_eval import learned_topk_selection, oracle_selection_recall
from learned_tta.tta_tuning import predict_selector_scores


@dataclass(frozen=True, slots=True)
class ReportBuildSummary:
    """Paths written by `build_report_from_artifacts`."""

    results_md: Path
    public_metrics_csv: Path
    private_metrics_csv: Path
    compute_csv: Path
    augmentation_impact_csv: Path
    gain_distribution_svg: Path
    oracle_overlap_svg: Path
    best_k: int


def build_report_from_artifacts(
    report_dir: Path,
    private_metrics_path: Path,
    tuning_path: Path,
    impact_targets_path: Path,
    impact_manifest_path: Path,
    checkpoint_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str | torch.device = "cpu",
    identity_aug_id: str = "aug_000",
) -> ReportBuildSummary:
    """Build final markdown, tables, and SVG plots from generated experiment artifacts."""

    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tuning = _load_tuning(tuning_path)
    best_k = _json_int(tuning["best_k"])
    targets = load_selector_targets(impact_targets_path)
    predicted_gain = predict_selector_scores(
        checkpoint_path=checkpoint_path,
        records=_load_manifest_records(impact_manifest_path),
        output_dim=len(targets.aug_ids),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    selected_aug_ids = learned_topk_selection(
        aug_ids=targets.aug_ids,
        predicted_gain=predicted_gain,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )
    oracle_aug_ids = learned_topk_selection(
        aug_ids=targets.aug_ids,
        predicted_gain=targets.gain,
        identity_aug_id=identity_aug_id,
        k=best_k,
    )

    private_metrics = _read_metrics_csv(private_metrics_path)
    public_metrics = _public_metrics_from_tuning(tuning)
    impact_table = build_augmentation_impact_table(
        aug_ids=targets.aug_ids,
        gain=targets.gain,
        selected_aug_ids=selected_aug_ids,
        oracle_aug_ids=oracle_aug_ids,
    )

    paths = ReportBuildSummary(
        results_md=report_dir / "results.md",
        public_metrics_csv=tables_dir / "public_metrics.csv",
        private_metrics_csv=tables_dir / "private_metrics.csv",
        compute_csv=tables_dir / "compute.csv",
        augmentation_impact_csv=tables_dir / "augmentation_impact.csv",
        gain_distribution_svg=figures_dir / "gain_distribution.svg",
        oracle_overlap_svg=figures_dir / "oracle_overlap.svg",
        best_k=best_k,
    )

    build_metrics_table(public_metrics).to_csv(paths.public_metrics_csv, index=False)
    build_metrics_table(private_metrics).to_csv(paths.private_metrics_csv, index=False)
    build_compute_table(private_metrics).to_csv(paths.compute_csv, index=False)
    impact_table.to_csv(paths.augmentation_impact_csv, index=False)
    paths.gain_distribution_svg.write_text(
        _gain_distribution_svg(targets.gain),
        encoding="utf-8",
    )
    paths.oracle_overlap_svg.write_text(
        _oracle_overlap_svg(selected_aug_ids, oracle_aug_ids, identity_aug_id),
        encoding="utf-8",
    )
    paths.results_md.write_text(
        _results_markdown(
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            best_k=best_k,
            recall=oracle_selection_recall(selected_aug_ids, oracle_aug_ids, identity_aug_id),
        ),
        encoding="utf-8",
    )
    return paths


def build_report_from_config(
    config_path: Path,
    report_dir: Path | None = None,
    private_metrics_path: Path | None = None,
    tuning_path: Path | None = None,
    impact_targets_path: Path | None = None,
    impact_manifest_path: Path | None = None,
    checkpoint_path: Path | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str | torch.device = "cpu",
) -> ReportBuildSummary:
    """Load experiment config and build the final report from default artifact locations."""

    config = load_experiment_config(config_path)
    resolved_report_dir = report_dir or config.artifacts.reports_dir
    return build_report_from_artifacts(
        report_dir=resolved_report_dir,
        private_metrics_path=private_metrics_path
        or resolved_report_dir / "tables" / "private_metrics.csv",
        tuning_path=tuning_path or config.artifacts.selector_dir / "public_val_tta_tuning.json",
        impact_targets_path=impact_targets_path
        or config.artifacts.selector_dir / "public_val_targets.npz",
        impact_manifest_path=impact_manifest_path
        or config.artifacts.manifests_dir / "public_val.csv",
        checkpoint_path=checkpoint_path or config.artifacts.selector_dir / "selector_best.pt",
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        identity_aug_id=config.augmentations.identity_id,
    )


def _load_manifest_records(path: Path):
    from learned_tta.data import load_manifest

    return load_manifest(path)


def _load_tuning(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return dict(json.load(handle))


def _read_metrics_csv(path: Path) -> dict[str, dict[str, float]]:
    table = pd.read_csv(path)
    metrics: dict[str, dict[str, float]] = {}
    for row in table.to_dict(orient="records"):
        strategy = str(row.pop("strategy"))
        metrics[strategy] = {str(key): float(value) for key, value in row.items()}
    return metrics


def _public_metrics_from_tuning(tuning: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    best_k = str(_json_int(tuning["best_k"]))
    results_by_k = tuning.get("results_by_k")
    if not isinstance(results_by_k, dict) or best_k not in results_by_k:
        raise ValueError("tuning JSON must contain results_by_k for best_k")
    best_metrics = results_by_k[best_k]
    if not isinstance(best_metrics, dict):
        raise ValueError("best_k metrics must be a JSON object")
    return {
        "learned_topk_uniform": {
            str(key): float(value) for key, value in best_metrics.items()
        }
    }


def _json_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected integer-compatible JSON value, got {type(value).__name__}")


def _results_markdown(
    public_metrics: dict[str, dict[str, float]],
    private_metrics: dict[str, dict[str, float]],
    best_k: int,
    recall: float,
) -> str:
    public_table = build_metrics_table(public_metrics)
    private_table = build_metrics_table(private_metrics)
    compute_table = build_compute_table(private_metrics)
    return "\n".join(
        [
            "# albu-tta ResNet50 Case Study",
            "",
            f"Tuned k: {best_k}",
            f"Public-val oracle top-k recall: {recall:.6g}",
            "",
            "This report is a single-architecture ImageNet validation case study. "
            "Run additional architectures before making broad leaderboard claims.",
            "",
            "## Public Validation Metrics",
            "",
            _markdown_table(public_table),
            "",
            "## Private Metrics",
            "",
            _markdown_table(private_table),
            "",
            "## Compute",
            "",
            _markdown_table(compute_table),
            "",
            "## Augmentation Impact",
            "",
            "- Table: `tables/augmentation_impact.csv`",
            "",
            "![Gain distribution](figures/gain_distribution.svg)",
            "",
            "![Learned versus oracle overlap](figures/oracle_overlap.svg)",
            "",
        ]
    )


def _gain_distribution_svg(gain: np.ndarray) -> str:
    values = np.asarray(gain, dtype=np.float32).ravel()
    counts, edges = np.histogram(values, bins=min(12, max(1, values.size)))
    return _bar_svg(
        title="Gain Distribution",
        labels=[f"{edges[index]:.2g}" for index in range(len(counts))],
        values=counts.astype(float).tolist(),
        y_label="count",
    )


def _oracle_overlap_svg(
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
    identity_aug_id: str,
) -> str:
    recalls = [
        _single_recall(selected, oracle, identity_aug_id)
        for selected, oracle in zip(selected_aug_ids, oracle_aug_ids, strict=True)
    ]
    counts, edges = np.histogram(recalls, bins=np.linspace(0.0, 1.0, 6))
    return _bar_svg(
        title="Learned vs Oracle Top-k Recall",
        labels=[f"{edges[index]:.1f}" for index in range(len(counts))],
        values=counts.astype(float).tolist(),
        y_label="images",
    )


def _single_recall(selected: list[str], oracle: list[str], identity_aug_id: str) -> float:
    selected_set = set(selected) - {identity_aug_id}
    oracle_set = set(oracle) - {identity_aug_id}
    if not oracle_set:
        return 1.0
    return len(selected_set & oracle_set) / len(oracle_set)


def _bar_svg(title: str, labels: list[str], values: list[float], y_label: str) -> str:
    width = 720
    height = 360
    margin_left = 56
    margin_bottom = 54
    plot_width = width - margin_left - 28
    plot_height = height - 74 - margin_bottom
    max_value = max(values) if values else 1.0
    if max_value <= 0.0:
        max_value = 1.0
    bar_gap = 6
    bar_width = max(1.0, (plot_width - bar_gap * max(0, len(values) - 1)) / max(1, len(values)))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" '
        'font-family="Arial" font-size="18" font-weight="700">'
        f"{_escape_xml(title)}</text>",
        f'<text x="18" y="{height / 2:.0f}" transform="rotate(-90 18 {height / 2:.0f})" '
        'text-anchor="middle" font-family="Arial" font-size="12">'
        f"{_escape_xml(y_label)}</text>",
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
        f'x2="{width - 28}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="58" '
        f'x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
    ]
    for index, value in enumerate(values):
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = plot_height * (value / max_value)
        y = height - margin_bottom - bar_height
        label_x = x + bar_width / 2
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="#2f6f9f"/>',
                f'<text x="{label_x:.2f}" y="{height - margin_bottom + 16}" '
                'text-anchor="middle" font-family="Arial" font-size="10">'
                f"{_escape_xml(labels[index])}</text>",
                f'<text x="{label_x:.2f}" y="{y - 4:.2f}" text-anchor="middle" '
                'font-family="Arial" font-size="10">'
                f"{value:.0f}</text>",
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _markdown_table(table: pd.DataFrame) -> str:
    columns = [str(column) for column in table.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [_format_markdown_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
