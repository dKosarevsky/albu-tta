"""Report table and markdown artifact builders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "strategy",
    "top1",
    "top5",
    "nll",
    "ece",
    "forwards_per_image",
    "relative_compute_vs_all",
]


@dataclass(frozen=True, slots=True)
class ReportArtifactPaths:
    """Paths written by `write_report_artifacts`."""

    results_md: Path
    public_metrics_csv: Path
    private_metrics_csv: Path
    compute_csv: Path
    augmentation_impact_csv: Path

    def __iter__(self) -> Iterator[Path]:
        yield self.results_md
        yield self.public_metrics_csv
        yield self.private_metrics_csv
        yield self.compute_csv
        yield self.augmentation_impact_csv


def build_metrics_table(
    metrics_by_strategy: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Build a stable strategy metrics table."""

    rows = []
    for strategy, metrics in metrics_by_strategy.items():
        row: dict[str, str | float] = {"strategy": strategy}
        for column in METRIC_COLUMNS[1:]:
            row[column] = float(metrics[column])
        rows.append(row)
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_compute_table(
    metrics_by_strategy: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Build the compute summary table from strategy metrics."""

    metrics_table = build_metrics_table(metrics_by_strategy)
    return metrics_table[
        [
            "strategy",
            "forwards_per_image",
            "relative_compute_vs_all",
        ]
    ].copy()


def build_augmentation_impact_table(
    aug_ids: list[str],
    gain: np.ndarray,
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
) -> pd.DataFrame:
    """Build per-augmentation mean gain and selection frequency table."""

    gain = np.asarray(gain, dtype=np.float32)
    if gain.ndim != 2:
        raise ValueError("gain must have shape [num_images, augmentations]")
    if gain.shape != (len(selected_aug_ids), len(aug_ids)):
        raise ValueError("gain shape must match selected_aug_ids and aug_ids")
    if len(oracle_aug_ids) != gain.shape[0]:
        raise ValueError("oracle_aug_ids length must match gain rows")

    return pd.DataFrame(
        {
            "aug_id": aug_ids,
            "mean_gain": gain.mean(axis=0),
            "selection_frequency": _selection_frequency(aug_ids, selected_aug_ids),
            "oracle_frequency": _selection_frequency(aug_ids, oracle_aug_ids),
        }
    )


def build_results_markdown(
    public_metrics: Mapping[str, Mapping[str, float]],
    private_metrics: Mapping[str, Mapping[str, float]],
    tuned_k: int,
) -> str:
    """Build a short markdown result summary for the article artifact folder."""

    public_table = build_metrics_table(public_metrics)
    private_table = build_metrics_table(private_metrics)
    compute_table = build_compute_table(private_metrics)
    return "\n".join(
        [
            "# albu-tta ResNet50 Case Study",
            "",
            f"Tuned k: {tuned_k}",
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
        ]
    )


def write_report_artifacts(
    report_dir: Path,
    public_metrics: Mapping[str, Mapping[str, float]],
    private_metrics: Mapping[str, Mapping[str, float]],
    aug_ids: list[str],
    gain: np.ndarray,
    selected_aug_ids: list[list[str]],
    oracle_aug_ids: list[list[str]],
    tuned_k: int,
) -> ReportArtifactPaths:
    """Write markdown and CSV report artifacts."""

    report_dir = Path(report_dir)
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    paths = ReportArtifactPaths(
        results_md=report_dir / "results.md",
        public_metrics_csv=tables_dir / "public_metrics.csv",
        private_metrics_csv=tables_dir / "private_metrics.csv",
        compute_csv=tables_dir / "compute.csv",
        augmentation_impact_csv=tables_dir / "augmentation_impact.csv",
    )

    public_table = build_metrics_table(public_metrics)
    private_table = build_metrics_table(private_metrics)
    compute_table = build_compute_table(private_metrics)
    impact_table = build_augmentation_impact_table(
        aug_ids=aug_ids,
        gain=gain,
        selected_aug_ids=selected_aug_ids,
        oracle_aug_ids=oracle_aug_ids,
    )
    paths.results_md.write_text(
        build_results_markdown(
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            tuned_k=tuned_k,
        ),
        encoding="utf-8",
    )
    public_table.to_csv(paths.public_metrics_csv, index=False)
    private_table.to_csv(paths.private_metrics_csv, index=False)
    compute_table.to_csv(paths.compute_csv, index=False)
    impact_table.to_csv(paths.augmentation_impact_csv, index=False)
    return paths


def _selection_frequency(aug_ids: list[str], selected_aug_ids: list[list[str]]) -> np.ndarray:
    if not selected_aug_ids:
        return np.zeros(len(aug_ids), dtype=np.float32)

    counts = {aug_id: 0 for aug_id in aug_ids}
    for image_aug_ids in selected_aug_ids:
        for aug_id in set(image_aug_ids):
            counts[aug_id] += 1
    return np.asarray(
        [counts[aug_id] / len(selected_aug_ids) for aug_id in aug_ids],
        dtype=np.float32,
    )


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
