from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.report_builder import build_report_from_artifacts
from learned_tta.selector_model import SelectorCNN
from learned_tta.stacking import AggregationArtifact, XGBoostAggregationArtifact
from learned_tta.targets import TargetStats, save_selector_targets


@pytest.fixture
def report_artifacts(tmp_path: Path) -> dict[str, Path]:
    private_metrics_csv = tmp_path / "private_metrics.csv"
    pd.DataFrame(
        [
            {
                "strategy": "clean",
                "top1": 0.5,
                "top5": 1.0,
                "nll": 0.8,
                "ece": 0.1,
                "forwards_per_image": 1.0,
                "relative_compute_vs_all": 0.5,
            },
            {
                "strategy": "learned_topk_uniform",
                "top1": 1.0,
                "top5": 1.0,
                "nll": 0.4,
                "ece": 0.05,
                "forwards_per_image": 2.0,
                "relative_compute_vs_all": 1.0,
            },
        ]
    ).to_csv(private_metrics_csv, index=False)

    corrections_csv = tmp_path / "corrections.csv"
    pd.DataFrame(
        [
            {
                "strategy": "clean",
                "clean_correct": 1,
                "tta_correct": 1,
                "both_right": 1,
                "clean_wrong_tta_right": 0,
                "clean_right_tta_wrong": 0,
                "both_wrong": 1,
                "num_images": 2,
            },
            {
                "strategy": "learned_topk_uniform",
                "clean_correct": 1,
                "tta_correct": 2,
                "both_right": 1,
                "clean_wrong_tta_right": 1,
                "clean_right_tta_wrong": 0,
                "both_wrong": 0,
                "num_images": 2,
            },
        ]
    ).to_csv(corrections_csv, index=False)

    selector_history_csv = tmp_path / "selector_history.csv"
    pd.DataFrame(
        [
            {
                "epoch": 1.0,
                "train_loss": 0.8,
                "val_loss": 0.7,
                "val_spearman": 0.1,
                "val_tta_best_k": 1.0,
                "val_tta_top1": 0.5,
                "val_tta_top5": 1.0,
                "val_tta_nll": 0.6,
                "val_tta_ece": 0.2,
                "val_tta_oracle_recall": 0.25,
            },
            {
                "epoch": 2.0,
                "train_loss": 0.4,
                "val_loss": 0.5,
                "val_spearman": 0.3,
                "val_tta_best_k": 1.0,
                "val_tta_top1": 1.0,
                "val_tta_top5": 1.0,
                "val_tta_nll": 0.3,
                "val_tta_ece": 0.1,
                "val_tta_oracle_recall": 0.75,
            },
        ]
    ).to_csv(selector_history_csv, index=False)

    tuning_json = tmp_path / "public_val_tta_tuning.json"
    tuning_json.write_text(
        json.dumps(
            {
                "best_k": 1,
                "results_by_k": {
                    "1": {
                        "top1": 1.0,
                        "top5": 1.0,
                        "nll": 0.3,
                        "ece": 0.02,
                        "forwards_per_image": 2.0,
                        "relative_compute_vs_all": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = _write_manifest(tmp_path, count=3)
    targets_path = _write_targets(tmp_path / "public_val_targets.npz")
    checkpoint_path = _write_selector_checkpoint(tmp_path / "selector_best.pt", output_dim=3)
    global_aggregator_path = tmp_path / "global_aggregator.json"
    class_aggregator_path = tmp_path / "class_aggregator.json"
    xgboost_aggregator_path = tmp_path / "xgboost_aggregator.json"
    AggregationArtifact(
        method="global-nonnegative",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        weights=np.array([0.1, 0.7, 0.2], dtype=np.float32),
        active_threshold=1e-6,
        metrics={
            "top1": 1.0,
            "top5": 1.0,
            "nll": 0.1,
            "ece": 0.02,
            "forwards_per_image": 3.0,
            "relative_compute_vs_all": 1.0,
        },
    ).save(global_aggregator_path)
    AggregationArtifact(
        method="class-nonnegative",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        weights=np.array(
            [
                [0.5, 0.4, 0.1],
                [0.1, 0.3, 0.6],
            ],
            dtype=np.float32,
        ),
        active_threshold=1e-6,
        metrics={
            "top1": 1.0,
            "top5": 1.0,
            "nll": 0.2,
            "ece": 0.03,
            "forwards_per_image": 3.0,
            "relative_compute_vs_all": 1.0,
        },
    ).save(class_aggregator_path)
    xgboost_model_path = tmp_path / "xgboost.model.json"
    xgboost_model_path.write_text("fake xgboost model", encoding="utf-8")
    XGBoostAggregationArtifact(
        method="xgboost-multiclass",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        model_path=xgboost_model_path,
        num_classes=2,
        feature_count=6,
        feature_importance=np.array([0.1, 0.7, 0.2], dtype=np.float32),
        metrics={
            "top1": 1.0,
            "top5": 1.0,
            "nll": 0.15,
            "ece": 0.025,
            "forwards_per_image": 3.0,
            "relative_compute_vs_all": 1.0,
        },
    ).save(xgboost_aggregator_path)
    return {
        "private_metrics": private_metrics_csv,
        "corrections": corrections_csv,
        "selector_history": selector_history_csv,
        "tuning": tuning_json,
        "manifest": manifest_path,
        "targets": targets_path,
        "checkpoint": checkpoint_path,
        "global_aggregator": global_aggregator_path,
        "class_aggregator": class_aggregator_path,
        "xgboost_aggregator": xgboost_aggregator_path,
    }


def test_build_report_from_artifacts_writes_tables_markdown_and_plots(
    tmp_path: Path,
    report_artifacts: dict[str, Path],
) -> None:
    summary = build_report_from_artifacts(
        report_dir=tmp_path / "report",
        private_metrics_path=report_artifacts["private_metrics"],
        tuning_path=report_artifacts["tuning"],
        impact_targets_path=report_artifacts["targets"],
        impact_manifest_path=report_artifacts["manifest"],
        checkpoint_path=report_artifacts["checkpoint"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        augmentation_registry_path=Path(__file__).resolve().parents[1]
        / "configs/augmentations/imagenet100.yaml",
        global_aggregator_path=report_artifacts["global_aggregator"],
        class_aggregator_path=report_artifacts["class_aggregator"],
        xgboost_aggregator_path=report_artifacts["xgboost_aggregator"],
        corrections_path=report_artifacts["corrections"],
        selector_history_path=report_artifacts["selector_history"],
        device="cpu",
    )

    markdown = summary.results_md.read_text(encoding="utf-8")
    impact = pd.read_csv(summary.augmentation_impact_csv)
    public_metrics = pd.read_csv(summary.public_metrics_csv)
    compute = pd.read_csv(summary.compute_csv)
    assert summary.aggregation_weights_csv is not None
    assert summary.class_augmentation_weights_csv is not None
    assert summary.aggregation_weights_svg is not None
    assert summary.xgboost_feature_importance_csv is not None
    assert summary.xgboost_feature_importance_svg is not None
    assert summary.corrections_csv is not None
    assert summary.corrections_svg is not None
    assert summary.selector_history_csv is not None
    assert summary.selector_history_svg is not None
    assert summary.transform_class_impact_csv is not None
    assert summary.transform_class_impact_svg is not None
    aggregation_weights = pd.read_csv(summary.aggregation_weights_csv)
    class_weights = pd.read_csv(summary.class_augmentation_weights_csv)
    xgboost_importance = pd.read_csv(summary.xgboost_feature_importance_csv)
    corrections = pd.read_csv(summary.corrections_csv)
    selector_history = pd.read_csv(summary.selector_history_csv)
    transform_class_impact = pd.read_csv(summary.transform_class_impact_csv)

    assert summary.best_k == 1
    assert summary.public_metrics_csv.exists()
    assert summary.private_metrics_csv.exists()
    assert summary.compute_csv.exists()
    assert summary.gain_distribution_svg.exists()
    assert summary.oracle_overlap_svg.exists()
    assert summary.aggregation_weights_csv.exists()
    assert summary.class_augmentation_weights_csv.exists()
    assert summary.aggregation_weights_svg.exists()
    assert summary.xgboost_feature_importance_csv.exists()
    assert summary.xgboost_feature_importance_svg.exists()
    assert summary.corrections_csv.exists()
    assert summary.corrections_svg.exists()
    assert summary.selector_history_csv.exists()
    assert summary.selector_history_svg.exists()
    assert summary.transform_class_impact_csv.exists()
    assert summary.transform_class_impact_svg.exists()
    assert impact["aug_id"].tolist() == ["aug_000", "aug_001", "aug_002"]
    assert impact["augmentation_name"].tolist() == [
        "identity",
        "horizontal_flip",
        "vertical_flip",
    ]
    assert impact["transform_class"].tolist() == [
        "identity",
        "HorizontalFlip",
        "VerticalFlip",
    ]
    assert public_metrics["strategy"].tolist() == [
        "learned_topk_uniform",
        "global_weighted_tta",
        "class_weighted_tta",
        "xgboost_multiclass",
    ]
    assert public_metrics["nll"].tolist() == pytest.approx([0.3, 0.1, 0.2, 0.15])
    assert compute.columns.tolist() == [
        "split",
        "strategy",
        "forwards_per_image",
        "relative_compute_vs_all",
    ]
    assert compute["split"].tolist() == [
        "public_val",
        "public_val",
        "public_val",
        "public_val",
        "private",
        "private",
    ]
    assert compute["strategy"].tolist() == [
        "learned_topk_uniform",
        "global_weighted_tta",
        "class_weighted_tta",
        "xgboost_multiclass",
        "clean",
        "learned_topk_uniform",
    ]
    assert aggregation_weights["global_weight"].tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert aggregation_weights["augmentation_name"].tolist() == [
        "identity",
        "horizontal_flip",
        "vertical_flip",
    ]
    assert set(class_weights.columns) == {
        "class_idx",
        "aug_id",
        "augmentation_name",
        "transform_class",
        "weight",
    }
    assert xgboost_importance["feature_importance"].tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert xgboost_importance["augmentation_name"].tolist() == [
        "identity",
        "horizontal_flip",
        "vertical_flip",
    ]
    assert transform_class_impact.columns.tolist() == [
        "transform_class",
        "candidate_count",
        "mean_gain",
        "selection_frequency",
        "oracle_frequency",
    ]
    assert transform_class_impact["transform_class"].tolist() == [
        "HorizontalFlip",
        "VerticalFlip",
        "identity",
    ]
    assert transform_class_impact["candidate_count"].tolist() == [1, 1, 1]
    assert corrections["strategy"].tolist() == ["clean", "learned_topk_uniform"]
    assert corrections["clean_wrong_tta_right"].tolist() == [0, 1]
    assert selector_history["val_tta_oracle_recall"].tolist() == pytest.approx([0.25, 0.75])
    assert "state-of-the-art" not in markdown.lower()
    assert "figures/gain_distribution.svg" in markdown
    assert "figures/aggregation_weights.svg" in markdown
    assert "figures/xgboost_feature_importance.svg" in markdown
    assert "figures/corrections.svg" in markdown
    assert "figures/selector_history.svg" in markdown
    assert "figures/transform_class_impact.svg" in markdown
    assert "Top mean-gain augmentations" in markdown
    assert "Top learned-selection augmentations" in markdown
    assert "Top oracle-selection augmentations" in markdown
    assert "Top transform classes by mean gain" in markdown
    assert "| aug_002 | vertical_flip | VerticalFlip |" in markdown
    assert "| VerticalFlip | 1 |" in markdown
    assert "aggregation_weights.csv" in markdown
    assert "public_val" in markdown
    assert "private" in markdown
    assert "global_weighted_tta" in markdown
    assert "class_weighted_tta" in markdown
    assert "xgboost_multiclass" in markdown
    assert "xgboost_feature_importance.csv" in markdown
    assert "corrections.csv" in markdown
    assert "selector_history.csv" in markdown
    assert "augmentation_impact.csv" in markdown
    assert "transform_class_impact.csv" in markdown


def test_build_report_cli_writes_final_results(
    tmp_path: Path,
    report_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    report_dir = tmp_path / "report"

    main(
        [
            "build-report",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--report-dir",
            str(report_dir),
            "--private-metrics",
            str(report_artifacts["private_metrics"]),
            "--corrections",
            str(report_artifacts["corrections"]),
            "--selector-history",
            str(report_artifacts["selector_history"]),
            "--tuning",
            str(report_artifacts["tuning"]),
            "--impact-targets",
            str(report_artifacts["targets"]),
            "--impact-manifest",
            str(report_artifacts["manifest"]),
            "--checkpoint",
            str(report_artifacts["checkpoint"]),
            "--global-aggregator",
            str(report_artifacts["global_aggregator"]),
            "--class-aggregator",
            str(report_artifacts["class_aggregator"]),
            "--xgboost-aggregator",
            str(report_artifacts["xgboost_aggregator"]),
            "--image-size",
            "16",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert "report: wrote" in captured.out
    assert (report_dir / "results.md").exists()
    assert (report_dir / "tables" / "augmentation_impact.csv").exists()
    assert (report_dir / "tables" / "aggregation_weights.csv").exists()
    assert (report_dir / "tables" / "corrections.csv").exists()
    assert (report_dir / "tables" / "selector_history.csv").exists()
    assert (report_dir / "figures" / "oracle_overlap.svg").exists()
    assert (report_dir / "figures" / "aggregation_weights.svg").exists()
    assert (report_dir / "tables" / "xgboost_feature_importance.csv").exists()
    assert (report_dir / "figures" / "xgboost_feature_importance.svg").exists()
    assert (report_dir / "figures" / "corrections.svg").exists()
    assert (report_dir / "figures" / "selector_history.svg").exists()


def _write_manifest(root: Path, count: int) -> Path:
    rows = []
    for index in range(count):
        path = root / f"public_val_{index}.png"
        image = np.full((12, 12, 3), fill_value=30 + index, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(path)
        rows.append(
            {
                "split": "public_val",
                "image_id": f"public_val-{index}",
                "class_idx": index % 2,
                "class_name": f"class-{index % 2}",
                "path": str(path),
            }
        )
    manifest_path = root / "public_val.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _write_targets(path: Path) -> Path:
    gain = np.array(
        [
            [0.0, 0.5, -0.1],
            [0.0, 0.2, 0.8],
            [0.0, -0.3, 0.4],
        ],
        dtype=np.float32,
    )
    stats = TargetStats(
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    save_selector_targets(
        path=path,
        aug_ids=["aug_000", "aug_001", "aug_002"],
        gain=gain,
        target_z=gain,
        stats=stats,
    )
    return path


def _write_selector_checkpoint(path: Path, output_dim: int) -> Path:
    model = SelectorCNN(output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    torch.save(
        {
            "epoch": 1,
            "val_nll": 0.0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
        },
        path,
    )
    return path
