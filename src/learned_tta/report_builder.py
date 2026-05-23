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

from learned_tta.augmentations import load_augmentation_registry
from learned_tta.config import load_experiment_config
from learned_tta.reporting import (
    METRIC_COLUMNS,
    build_augmentation_impact_table,
    build_compute_table,
    build_metrics_table,
)
from learned_tta.stacking import (
    default_aggregator_path,
    load_aggregation_artifact,
    load_xgboost_aggregation_artifact,
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
    transform_class_impact_csv: Path | None
    aggregation_weights_csv: Path | None
    class_augmentation_weights_csv: Path | None
    xgboost_feature_importance_csv: Path | None
    corrections_csv: Path | None
    selector_history_csv: Path | None
    gain_distribution_svg: Path
    oracle_overlap_svg: Path
    aggregation_weights_svg: Path | None
    xgboost_feature_importance_svg: Path | None
    corrections_svg: Path | None
    selector_history_svg: Path | None
    transform_class_impact_svg: Path | None
    best_k: int


@dataclass(frozen=True, slots=True)
class _AggregationTables:
    weights: pd.DataFrame | None
    class_weights: pd.DataFrame | None


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
    augmentation_registry_path: Path | None = None,
    global_aggregator_path: Path | None = None,
    class_aggregator_path: Path | None = None,
    xgboost_aggregator_path: Path | None = None,
    corrections_path: Path | None = None,
    selector_history_path: Path | None = None,
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
    public_split = str(tuning.get("split", "public_val"))
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
    public_metrics = _public_metrics_from_tuning(
        tuning=tuning,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
        xgboost_aggregator_path=xgboost_aggregator_path,
    )
    augmentation_metadata = _augmentation_metadata_table(
        aug_ids=targets.aug_ids,
        registry_path=augmentation_registry_path,
    )
    impact_table = build_augmentation_impact_table(
        aug_ids=targets.aug_ids,
        gain=targets.gain,
        selected_aug_ids=selected_aug_ids,
        oracle_aug_ids=oracle_aug_ids,
    )
    impact_table = _attach_augmentation_metadata(impact_table, augmentation_metadata)
    transform_class_impact = _build_transform_class_impact_table(impact_table)
    aggregation_tables = _build_aggregation_tables(
        aug_ids=targets.aug_ids,
        global_aggregator_path=global_aggregator_path,
        class_aggregator_path=class_aggregator_path,
    )
    aggregation_tables = _attach_aggregation_metadata(
        aggregation_tables,
        augmentation_metadata,
    )
    xgboost_importance = _build_xgboost_feature_importance_table(
        aug_ids=targets.aug_ids,
        xgboost_aggregator_path=xgboost_aggregator_path,
    )
    if xgboost_importance is not None:
        xgboost_importance = _attach_augmentation_metadata(
            xgboost_importance,
            augmentation_metadata,
        )
    corrections_table = _read_corrections_csv(corrections_path) if corrections_path else None
    selector_history = (
        _read_selector_history_csv(selector_history_path) if selector_history_path else None
    )

    paths = ReportBuildSummary(
        results_md=report_dir / "results.md",
        public_metrics_csv=tables_dir / "public_metrics.csv",
        private_metrics_csv=tables_dir / "private_metrics.csv",
        compute_csv=tables_dir / "compute.csv",
        augmentation_impact_csv=tables_dir / "augmentation_impact.csv",
        transform_class_impact_csv=(
            tables_dir / "transform_class_impact.csv"
            if transform_class_impact is not None
            else None
        ),
        aggregation_weights_csv=(
            tables_dir / "aggregation_weights.csv"
            if aggregation_tables.weights is not None
            else None
        ),
        class_augmentation_weights_csv=(
            tables_dir / "class_augmentation_weights.csv"
            if aggregation_tables.class_weights is not None
            else None
        ),
        xgboost_feature_importance_csv=(
            tables_dir / "xgboost_feature_importance.csv"
            if xgboost_importance is not None
            else None
        ),
        corrections_csv=tables_dir / "corrections.csv" if corrections_table is not None else None,
        selector_history_csv=(
            tables_dir / "selector_history.csv" if selector_history is not None else None
        ),
        gain_distribution_svg=figures_dir / "gain_distribution.svg",
        oracle_overlap_svg=figures_dir / "oracle_overlap.svg",
        aggregation_weights_svg=(
            figures_dir / "aggregation_weights.svg"
            if aggregation_tables.weights is not None
            else None
        ),
        xgboost_feature_importance_svg=(
            figures_dir / "xgboost_feature_importance.svg"
            if xgboost_importance is not None
            else None
        ),
        corrections_svg=figures_dir / "corrections.svg" if corrections_table is not None else None,
        selector_history_svg=(
            figures_dir / "selector_history.svg" if selector_history is not None else None
        ),
        transform_class_impact_svg=(
            figures_dir / "transform_class_impact.svg"
            if transform_class_impact is not None
            else None
        ),
        best_k=best_k,
    )

    build_metrics_table(public_metrics).to_csv(paths.public_metrics_csv, index=False)
    build_metrics_table(private_metrics).to_csv(paths.private_metrics_csv, index=False)
    _build_split_compute_table(
        public_split=public_split,
        public_metrics=public_metrics,
        private_metrics=private_metrics,
    ).to_csv(paths.compute_csv, index=False)
    impact_table.to_csv(paths.augmentation_impact_csv, index=False)
    if paths.transform_class_impact_csv is not None and transform_class_impact is not None:
        transform_class_impact.to_csv(paths.transform_class_impact_csv, index=False)
    if paths.aggregation_weights_csv is not None and aggregation_tables.weights is not None:
        aggregation_tables.weights.to_csv(paths.aggregation_weights_csv, index=False)
    if (
        paths.class_augmentation_weights_csv is not None
        and aggregation_tables.class_weights is not None
    ):
        aggregation_tables.class_weights.to_csv(
            paths.class_augmentation_weights_csv,
            index=False,
        )
    if paths.xgboost_feature_importance_csv is not None and xgboost_importance is not None:
        xgboost_importance.to_csv(paths.xgboost_feature_importance_csv, index=False)
    if paths.corrections_csv is not None and corrections_table is not None:
        corrections_table.to_csv(paths.corrections_csv, index=False)
    if paths.selector_history_csv is not None and selector_history is not None:
        selector_history.to_csv(paths.selector_history_csv, index=False)
    paths.gain_distribution_svg.write_text(
        _gain_distribution_svg(targets.gain),
        encoding="utf-8",
    )
    paths.oracle_overlap_svg.write_text(
        _oracle_overlap_svg(selected_aug_ids, oracle_aug_ids, identity_aug_id),
        encoding="utf-8",
    )
    if paths.aggregation_weights_svg is not None and aggregation_tables.weights is not None:
        paths.aggregation_weights_svg.write_text(
            _aggregation_weights_svg(aggregation_tables.weights),
            encoding="utf-8",
        )
    if (
        paths.xgboost_feature_importance_svg is not None
        and xgboost_importance is not None
    ):
        paths.xgboost_feature_importance_svg.write_text(
            _xgboost_feature_importance_svg(xgboost_importance),
            encoding="utf-8",
        )
    if paths.corrections_svg is not None and corrections_table is not None:
        paths.corrections_svg.write_text(
            _corrections_svg(corrections_table),
            encoding="utf-8",
        )
    if paths.selector_history_svg is not None and selector_history is not None:
        paths.selector_history_svg.write_text(
            _selector_history_svg(selector_history),
            encoding="utf-8",
        )
    if paths.transform_class_impact_svg is not None and transform_class_impact is not None:
        paths.transform_class_impact_svg.write_text(
            _transform_class_impact_svg(transform_class_impact),
            encoding="utf-8",
        )
    paths.results_md.write_text(
        _results_markdown(
            public_metrics=public_metrics,
            private_metrics=private_metrics,
            public_split=public_split,
            best_k=best_k,
            recall=oracle_selection_recall(selected_aug_ids, oracle_aug_ids, identity_aug_id),
            has_aggregation_weights=aggregation_tables.weights is not None,
            has_class_weights=aggregation_tables.class_weights is not None,
            has_xgboost_importance=xgboost_importance is not None,
            has_corrections=corrections_table is not None,
            has_selector_history=selector_history is not None,
            has_transform_class_impact=transform_class_impact is not None,
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
    global_aggregator_path: Path | None = None,
    class_aggregator_path: Path | None = None,
    xgboost_aggregator_path: Path | None = None,
    corrections_path: Path | None = None,
    selector_history_path: Path | None = None,
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
        global_aggregator_path=global_aggregator_path
        or _existing_path(
            default_aggregator_path(
                config.artifacts.selector_dir,
                split="public_val",
                method="global-nonnegative",
            )
        ),
        class_aggregator_path=class_aggregator_path
        or _existing_path(
            default_aggregator_path(
                config.artifacts.selector_dir,
                split="public_val",
                method="class-nonnegative",
            )
        ),
        xgboost_aggregator_path=xgboost_aggregator_path
        or _existing_path(
            default_aggregator_path(
                config.artifacts.selector_dir,
                split="public_val",
                method="xgboost-multiclass",
            )
        ),
        corrections_path=corrections_path
        or _existing_path(resolved_report_dir / "tables" / "corrections.csv"),
        selector_history_path=selector_history_path
        or _existing_path(config.artifacts.selector_dir / "selector_history.csv"),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_registry_path=config.augmentations.registry_path,
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


def _read_corrections_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required_columns = {
        "strategy",
        "clean_correct",
        "tta_correct",
        "both_right",
        "clean_wrong_tta_right",
        "clean_right_tta_wrong",
        "both_wrong",
        "num_images",
    }
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"corrections CSV is missing columns: {sorted(missing)}")
    return table[
        [
            "strategy",
            "clean_correct",
            "tta_correct",
            "both_right",
            "clean_wrong_tta_right",
            "clean_right_tta_wrong",
            "both_wrong",
            "num_images",
        ]
    ].copy()


def _read_selector_history_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required_columns = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_spearman",
        "val_tta_best_k",
        "val_tta_top1",
        "val_tta_top5",
        "val_tta_nll",
        "val_tta_ece",
        "val_tta_oracle_recall",
    }
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"selector history CSV is missing columns: {sorted(missing)}")
    return table[
        [
            "epoch",
            "train_loss",
            "val_loss",
            "val_spearman",
            "val_tta_best_k",
            "val_tta_top1",
            "val_tta_top5",
            "val_tta_nll",
            "val_tta_ece",
            "val_tta_oracle_recall",
        ]
    ].copy()


def _build_aggregation_tables(
    aug_ids: list[str],
    global_aggregator_path: Path | None,
    class_aggregator_path: Path | None,
) -> _AggregationTables:
    if global_aggregator_path is None and class_aggregator_path is None:
        return _AggregationTables(weights=None, class_weights=None)

    global_weights: np.ndarray | None = None
    global_active: np.ndarray | None = None
    class_weights: np.ndarray | None = None
    class_active_threshold: float | None = None

    if global_aggregator_path is not None:
        artifact = load_aggregation_artifact(global_aggregator_path)
        _validate_aggregator_aug_ids(artifact.aug_ids, aug_ids, global_aggregator_path)
        global_weights = np.asarray(artifact.weights, dtype=np.float32)
        global_active = global_weights > artifact.active_threshold

    if class_aggregator_path is not None:
        artifact = load_aggregation_artifact(class_aggregator_path)
        _validate_aggregator_aug_ids(artifact.aug_ids, aug_ids, class_aggregator_path)
        class_weights = np.asarray(artifact.weights, dtype=np.float32)
        if class_weights.ndim != 2 or class_weights.shape[1] != len(aug_ids):
            raise ValueError("class aggregation weights must have shape [classes, augmentations]")
        class_active_threshold = artifact.active_threshold

    weights_table = pd.DataFrame({"aug_id": aug_ids})
    if global_weights is not None and global_active is not None:
        weights_table["global_weight"] = global_weights
        weights_table["global_active"] = global_active
    if class_weights is not None and class_active_threshold is not None:
        weights_table["class_mean_weight"] = class_weights.mean(axis=0)
        weights_table["class_max_weight"] = class_weights.max(axis=0)
        weights_table["class_active_frequency"] = (
            class_weights > class_active_threshold
        ).mean(axis=0)

    class_table = None
    if class_weights is not None:
        class_table = pd.DataFrame(
            [
                {
                    "class_idx": class_idx,
                    "aug_id": aug_id,
                    "weight": float(class_weights[class_idx, aug_index]),
                }
                for class_idx in range(class_weights.shape[0])
                for aug_index, aug_id in enumerate(aug_ids)
            ]
        )

    return _AggregationTables(weights=weights_table, class_weights=class_table)


def _attach_aggregation_metadata(
    tables: _AggregationTables,
    metadata: pd.DataFrame | None,
) -> _AggregationTables:
    return _AggregationTables(
        weights=(
            _attach_augmentation_metadata(tables.weights, metadata)
            if tables.weights is not None
            else None
        ),
        class_weights=(
            _attach_augmentation_metadata(tables.class_weights, metadata)
            if tables.class_weights is not None
            else None
        ),
    )


def _augmentation_metadata_table(
    aug_ids: list[str],
    registry_path: Path | None,
) -> pd.DataFrame | None:
    if registry_path is None:
        return None

    candidates = load_augmentation_registry(registry_path)
    metadata_by_id = {
        candidate.id: {
            "aug_id": candidate.id,
            "augmentation_name": candidate.name,
            "transform_class": candidate.class_name or "identity",
        }
        for candidate in candidates
    }
    missing = [aug_id for aug_id in aug_ids if aug_id not in metadata_by_id]
    if missing:
        raise ValueError(f"augmentation registry is missing ids: {missing}")
    return pd.DataFrame([metadata_by_id[aug_id] for aug_id in aug_ids])


def _attach_augmentation_metadata(
    table: pd.DataFrame,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    if metadata is None:
        return table
    return table.merge(metadata, on="aug_id", how="left", validate="many_to_one")


def _build_transform_class_impact_table(impact_table: pd.DataFrame) -> pd.DataFrame | None:
    if "transform_class" not in impact_table.columns:
        return None

    return (
        impact_table.groupby("transform_class", as_index=False, sort=True)
        .agg(
            candidate_count=("aug_id", "count"),
            mean_gain=("mean_gain", "mean"),
            selection_frequency=("selection_frequency", "mean"),
            oracle_frequency=("oracle_frequency", "mean"),
        )
        .loc[
            :,
            [
                "transform_class",
                "candidate_count",
                "mean_gain",
                "selection_frequency",
                "oracle_frequency",
            ],
        ]
    )


def _build_xgboost_feature_importance_table(
    aug_ids: list[str],
    xgboost_aggregator_path: Path | None,
) -> pd.DataFrame | None:
    if xgboost_aggregator_path is None:
        return None

    artifact = load_xgboost_aggregation_artifact(xgboost_aggregator_path)
    _validate_aggregator_aug_ids(artifact.aug_ids, aug_ids, xgboost_aggregator_path)
    feature_importance = np.asarray(artifact.feature_importance, dtype=np.float32)
    if feature_importance.shape != (len(aug_ids),):
        raise ValueError("xgboost feature importance must have shape [augmentations]")
    return pd.DataFrame(
        {
            "aug_id": aug_ids,
            "feature_importance": feature_importance,
        }
    )


def _validate_aggregator_aug_ids(
    aggregator_aug_ids: list[str],
    expected_aug_ids: list[str],
    path: Path,
) -> None:
    if aggregator_aug_ids != expected_aug_ids:
        raise ValueError(f"aggregator aug_ids in {path} must match selector target aug_ids")


def _public_metrics_from_tuning(
    tuning: Mapping[str, Any],
    global_aggregator_path: Path | None = None,
    class_aggregator_path: Path | None = None,
    xgboost_aggregator_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    best_k = str(_json_int(tuning["best_k"]))
    results_by_k = tuning.get("results_by_k")
    if not isinstance(results_by_k, dict) or best_k not in results_by_k:
        raise ValueError("tuning JSON must contain results_by_k for best_k")
    best_metrics = results_by_k[best_k]
    if not isinstance(best_metrics, dict):
        raise ValueError("best_k metrics must be a JSON object")
    metrics = {
        "learned_topk_uniform": {
            str(key): float(value) for key, value in best_metrics.items()
        }
    }
    if global_aggregator_path is not None:
        artifact = load_aggregation_artifact(global_aggregator_path)
        metrics["global_weighted_tta"] = _complete_metrics(
            artifact.metrics,
            strategy="global_weighted_tta",
        )
    if class_aggregator_path is not None:
        artifact = load_aggregation_artifact(class_aggregator_path)
        metrics["class_weighted_tta"] = _complete_metrics(
            artifact.metrics,
            strategy="class_weighted_tta",
        )
    if xgboost_aggregator_path is not None:
        artifact = load_xgboost_aggregation_artifact(xgboost_aggregator_path)
        metrics["xgboost_multiclass"] = _complete_metrics(
            artifact.metrics,
            strategy="xgboost_multiclass",
        )
    return metrics


def _complete_metrics(metrics: Mapping[str, float], strategy: str) -> dict[str, float]:
    missing = [column for column in METRIC_COLUMNS[1:] if column not in metrics]
    if missing:
        raise ValueError(f"{strategy} metrics are missing columns: {missing}")
    return {column: float(metrics[column]) for column in METRIC_COLUMNS[1:]}


def _build_split_compute_table(
    public_split: str,
    public_metrics: Mapping[str, Mapping[str, float]],
    private_metrics: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    public_compute = build_compute_table(public_metrics)
    public_compute.insert(0, "split", public_split)
    private_compute = build_compute_table(private_metrics)
    private_compute.insert(0, "split", "private")
    return pd.concat([public_compute, private_compute], ignore_index=True)


def _json_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected integer-compatible JSON value, got {type(value).__name__}")


def _results_markdown(
    public_metrics: dict[str, dict[str, float]],
    private_metrics: dict[str, dict[str, float]],
    public_split: str,
    best_k: int,
    recall: float,
    has_aggregation_weights: bool,
    has_class_weights: bool,
    has_xgboost_importance: bool,
    has_corrections: bool,
    has_selector_history: bool,
    has_transform_class_impact: bool,
) -> str:
    public_table = build_metrics_table(public_metrics)
    private_table = build_metrics_table(private_metrics)
    compute_table = _build_split_compute_table(
        public_split=public_split,
        public_metrics=public_metrics,
        private_metrics=private_metrics,
    )
    lines = [
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
    if has_transform_class_impact:
        lines.extend(
            [
                "- Transform-class table: `tables/transform_class_impact.csv`",
                "",
                "![Transform-class impact](figures/transform_class_impact.svg)",
                "",
            ]
        )
    if has_aggregation_weights:
        lines.extend(
            [
                "## Learned Aggregation Weights",
                "",
                "- Table: `tables/aggregation_weights.csv`",
                "",
                "![Aggregation weights](figures/aggregation_weights.svg)",
                "",
            ]
        )
        if has_class_weights:
            lines.extend(
                [
                    "- Class-specific table: `tables/class_augmentation_weights.csv`",
                    "",
                ]
            )
    if has_xgboost_importance:
        lines.extend(
            [
                "## XGBoost Stacker Diagnostics",
                "",
                "- Table: `tables/xgboost_feature_importance.csv`",
                "",
                "![XGBoost feature importance](figures/xgboost_feature_importance.svg)",
                "",
            ]
        )
    if has_corrections:
        lines.extend(
            [
                "## Corrections and Corruptions",
                "",
                "- Table: `tables/corrections.csv`",
                "",
                "![TTA corrections and corruptions](figures/corrections.svg)",
                "",
            ]
        )
    if has_selector_history:
        lines.extend(
            [
                "## Selector Training Diagnostics",
                "",
                "- Table: `tables/selector_history.csv`",
                "",
                "![Selector training diagnostics](figures/selector_history.svg)",
                "",
            ]
        )
    return "\n".join(lines)


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


def _aggregation_weights_svg(table: pd.DataFrame) -> str:
    labels = table["aug_id"].astype(str).tolist()
    if "global_weight" in table.columns:
        values = table["global_weight"].astype(float).tolist()
        title = "Global Aggregation Weights"
    elif "class_mean_weight" in table.columns:
        values = table["class_mean_weight"].astype(float).tolist()
        title = "Mean Class Aggregation Weights"
    else:
        values = []
        title = "Aggregation Weights"
    return _bar_svg(title=title, labels=labels, values=values, y_label="weight")


def _xgboost_feature_importance_svg(table: pd.DataFrame) -> str:
    return _bar_svg(
        title="XGBoost Feature Importance",
        labels=table["aug_id"].astype(str).tolist(),
        values=table["feature_importance"].astype(float).tolist(),
        y_label="importance",
    )


def _transform_class_impact_svg(table: pd.DataFrame) -> str:
    return _grouped_bar_svg(
        title="Transform-class Selection Frequencies",
        labels=table["transform_class"].astype(str).tolist(),
        series=[
            (
                "learned selection",
                table["selection_frequency"].astype(float).tolist(),
                "#2f6f9f",
            ),
            (
                "oracle selection",
                table["oracle_frequency"].astype(float).tolist(),
                "#8a5a9f",
            ),
        ],
        y_label="frequency",
    )


def _corrections_svg(table: pd.DataFrame) -> str:
    labels = table["strategy"].astype(str).tolist()
    return _grouped_bar_svg(
        title="TTA Corrections and Corruptions",
        labels=labels,
        series=[
            (
                "clean wrong -> TTA right",
                table["clean_wrong_tta_right"].astype(float).tolist(),
                "#247a4d",
            ),
            (
                "clean right -> TTA wrong",
                table["clean_right_tta_wrong"].astype(float).tolist(),
                "#b24a3b",
            ),
        ],
        y_label="images",
    )


def _selector_history_svg(table: pd.DataFrame) -> str:
    labels = [str(int(epoch)) for epoch in table["epoch"].astype(float).tolist()]
    return _grouped_bar_svg(
        title="Selector Validation Diagnostics",
        labels=labels,
        series=[
            (
                "TTA NLL",
                table["val_tta_nll"].astype(float).tolist(),
                "#2f6f9f",
            ),
            (
                "Oracle recall",
                table["val_tta_oracle_recall"].astype(float).tolist(),
                "#8a5a9f",
            ),
        ],
        y_label="value",
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


def _grouped_bar_svg(
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
) -> str:
    width = 760
    height = 390
    margin_left = 64
    margin_bottom = 74
    plot_width = width - margin_left - 28
    plot_height = height - 98 - margin_bottom
    max_value = max((max(values) for _, values, _ in series if values), default=1.0)
    if max_value <= 0.0:
        max_value = 1.0
    group_gap = 12
    bar_gap = 3
    group_width = max(
        1.0,
        (plot_width - group_gap * max(0, len(labels) - 1)) / max(1, len(labels)),
    )
    bar_width = max(1.0, (group_width - bar_gap * max(0, len(series) - 1)) / len(series))
    baseline_y = height - margin_bottom
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
        f'<line x1="{margin_left}" y1="{baseline_y}" '
        f'x2="{width - 28}" y2="{baseline_y}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="70" '
        f'x2="{margin_left}" y2="{baseline_y}" stroke="#222"/>',
    ]
    legend_x = margin_left
    for legend_index, (name, _, color) in enumerate(series):
        x = legend_x + legend_index * 190
        elements.extend(
            [
                f'<rect x="{x}" y="46" width="12" height="12" fill="{color}"/>',
                f'<text x="{x + 18}" y="56" font-family="Arial" font-size="11">'
                f"{_escape_xml(name)}</text>",
            ]
        )
    for label_index, label in enumerate(labels):
        group_x = margin_left + label_index * (group_width + group_gap)
        for series_index, (_, values, color) in enumerate(series):
            value = values[label_index]
            x = group_x + series_index * (bar_width + bar_gap)
            bar_height = plot_height * (value / max_value)
            y = baseline_y - bar_height
            label_x = x + bar_width / 2
            elements.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{bar_height:.2f}" fill="{color}"/>',
                    f'<text x="{label_x:.2f}" y="{y - 4:.2f}" text-anchor="middle" '
                    'font-family="Arial" font-size="9">'
                    f"{value:.0f}</text>",
                ]
            )
        elements.append(
            f'<text x="{group_x + group_width / 2:.2f}" y="{baseline_y + 16}" '
            'text-anchor="end" font-family="Arial" font-size="10" '
            f'transform="rotate(-30 {group_x + group_width / 2:.2f} {baseline_y + 16})">'
            f"{_escape_xml(label)}</text>"
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


def _existing_path(path: Path) -> Path | None:
    if path.exists():
        return path
    return None
