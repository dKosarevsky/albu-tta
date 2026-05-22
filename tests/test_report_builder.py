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
    return {
        "private_metrics": private_metrics_csv,
        "tuning": tuning_json,
        "manifest": manifest_path,
        "targets": targets_path,
        "checkpoint": checkpoint_path,
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
        device="cpu",
    )

    markdown = summary.results_md.read_text(encoding="utf-8")
    impact = pd.read_csv(summary.augmentation_impact_csv)

    assert summary.best_k == 1
    assert summary.public_metrics_csv.exists()
    assert summary.private_metrics_csv.exists()
    assert summary.compute_csv.exists()
    assert summary.gain_distribution_svg.exists()
    assert summary.oracle_overlap_svg.exists()
    assert impact["aug_id"].tolist() == ["aug_000", "aug_001", "aug_002"]
    assert "state-of-the-art" not in markdown.lower()
    assert "figures/gain_distribution.svg" in markdown
    assert "augmentation_impact.csv" in markdown


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
            "--tuning",
            str(report_artifacts["tuning"]),
            "--impact-targets",
            str(report_artifacts["targets"]),
            "--impact-manifest",
            str(report_artifacts["manifest"]),
            "--checkpoint",
            str(report_artifacts["checkpoint"]),
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
    assert (report_dir / "figures" / "oracle_overlap.svg").exists()


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
