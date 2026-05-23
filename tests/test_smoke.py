from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from learned_tta.smoke import run_smoke_e2e

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"


def test_run_smoke_e2e_writes_end_to_end_artifacts(tmp_path: Path) -> None:
    summary = run_smoke_e2e(
        config_path=CONFIG_PATH,
        output_dir=tmp_path / "smoke",
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        device="cpu",
    )

    private_metrics = pd.read_csv(summary.private_metrics_csv)
    impact = pd.read_csv(summary.reports_dir / "tables" / "augmentation_impact.csv")

    assert summary.results_md.exists()
    assert summary.selector_checkpoint.exists()
    assert summary.tuning_json.exists()
    assert summary.private_metrics_csv.exists()
    assert (summary.selector_dir / "selector_history.csv").exists()
    assert (summary.reports_dir / "tables" / "corrections.csv").exists()
    assert (summary.reports_dir / "tables" / "selector_history.csv").exists()
    assert (summary.reports_dir / "tables" / "aggregation_weights.csv").exists()
    assert (summary.reports_dir / "tables" / "transform_class_impact.csv").exists()
    assert (summary.reports_dir / "figures" / "corrections.svg").exists()
    assert (summary.reports_dir / "figures" / "selector_history.svg").exists()
    assert (summary.reports_dir / "figures" / "transform_class_impact.svg").exists()
    assert "corrections.csv" in summary.results_md.read_text(encoding="utf-8")
    assert "selector_history.csv" in summary.results_md.read_text(encoding="utf-8")
    assert "transform_class_impact.csv" in summary.results_md.read_text(encoding="utf-8")
    assert "transform_class_impact.svg" in summary.results_md.read_text(encoding="utf-8")
    assert summary.candidate_ids == ["aug_000", "aug_001", "aug_002"]
    assert impact["augmentation_name"].tolist() == [
        "identity",
        "horizontal_flip",
        "vertical_flip",
    ]
    assert {"global_weighted_tta", "class_weighted_tta"} <= set(private_metrics["strategy"])
    assert set(private_metrics["strategy"]) == {
        "clean",
        "fixed_light_tta",
        "random_topk",
        "all_100_uniform",
        "global_weighted_tta",
        "class_weighted_tta",
        "learned_topk_uniform",
        "learned_topk_softmax_weighted",
        "oracle_topk_uniform",
    }


def test_run_smoke_cli_writes_final_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "smoke"
    main(
        [
            "run-smoke",
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--image-size",
            "16",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--epochs",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert "smoke run: wrote" in captured.out
    assert (output_dir / "reports" / "results.md").exists()
    saved_tuning = json.loads(
        (output_dir / "selector" / "public_val_tta_tuning.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_tuning["predicted_gain_shape"] == [2, 3]
