from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.private_eval import evaluate_private_from_artifacts
from learned_tta.selector_model import SelectorCNN
from learned_tta.stacking import AggregationArtifact, XGBoostAggregationArtifact


@pytest.fixture
def private_eval_artifacts(tmp_path: Path) -> dict[str, Path]:
    manifest_path = _write_manifest(tmp_path, split="private", count=2)
    cache_dir = _write_cache(tmp_path / "teacher_cache")
    checkpoint_path = _write_selector_checkpoint(tmp_path / "selector_best.pt", output_dim=3)
    global_aggregator_path = tmp_path / "global_aggregator.json"
    class_aggregator_path = tmp_path / "class_aggregator.json"
    xgboost_aggregator_path = tmp_path / "xgboost_aggregator.json"
    AggregationArtifact(
        method="global-nonnegative",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        weights=np.array([0.2, 0.7, 0.1], dtype=np.float32),
        active_threshold=1e-6,
        metrics={"nll": 0.0},
    ).save(global_aggregator_path)
    AggregationArtifact(
        method="class-nonnegative",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        weights=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        active_threshold=1e-6,
        metrics={"nll": 0.0},
    ).save(class_aggregator_path)
    xgboost_model_path = tmp_path / "xgboost.model.json"
    xgboost_model_path.write_text("fake xgboost model", encoding="utf-8")
    XGBoostAggregationArtifact(
        method="xgboost-multiclass",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        model_path=xgboost_model_path,
        num_classes=2,
        feature_count=6,
        feature_importance=np.array([0.2, 0.6, 0.2], dtype=np.float32),
        metrics={"nll": 0.0},
    ).save(xgboost_aggregator_path)
    tuning_path = tmp_path / "public_val_tta_tuning.json"
    tuning_path.write_text(json.dumps({"best_k": 1}), encoding="utf-8")
    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "checkpoint": checkpoint_path,
        "global_aggregator": global_aggregator_path,
        "class_aggregator": class_aggregator_path,
        "xgboost_aggregator": xgboost_aggregator_path,
        "tuning": tuning_path,
    }


def test_evaluate_private_from_artifacts_writes_metric_tables(
    tmp_path: Path,
    private_eval_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_xgboost(monkeypatch)
    summary = evaluate_private_from_artifacts(
        split="private",
        manifest_path=private_eval_artifacts["manifest"],
        cache_dir=private_eval_artifacts["cache_dir"],
        checkpoint_path=private_eval_artifacts["checkpoint"],
        tuning_path=private_eval_artifacts["tuning"],
        output_dir=tmp_path / "reports",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        random_seeds=[1, 5],
        global_aggregator_path=private_eval_artifacts["global_aggregator"],
        class_aggregator_path=private_eval_artifacts["class_aggregator"],
        xgboost_aggregator_path=private_eval_artifacts["xgboost_aggregator"],
        device="cpu",
    )

    table = pd.read_csv(summary.private_metrics_csv)

    assert summary.best_k == 1
    assert summary.private_metrics_csv.exists()
    assert summary.compute_csv.exists()
    assert summary.corrections_csv.exists()
    assert summary.global_topn_metrics_csv is not None
    assert summary.global_topn_metrics_csv.exists()
    assert set(table["strategy"]) == {
        "clean",
        "fixed_light_tta",
        "random_topk",
        "all_100_uniform",
        "learned_topk_uniform",
        "learned_topk_softmax_weighted",
        "global_weighted_tta",
        "class_weighted_tta",
        "xgboost_multiclass",
        "oracle_topk_uniform",
    }
    corrections = pd.read_csv(summary.corrections_csv)
    assert set(corrections.columns) == {
        "strategy",
        "clean_correct",
        "tta_correct",
        "both_right",
        "clean_wrong_tta_right",
        "clean_right_tta_wrong",
        "both_wrong",
        "num_images",
    }
    assert set(table["strategy"]) <= set(corrections["strategy"])
    random_row = corrections[corrections["strategy"] == "random_topk"].iloc[0]
    assert random_row["clean_correct"] == pytest.approx(2.0)
    assert random_row["tta_correct"] == pytest.approx(1.5)
    assert random_row["clean_right_tta_wrong"] == pytest.approx(0.5)
    global_topn = pd.read_csv(summary.global_topn_metrics_csv)
    assert global_topn["top_n"].tolist() == [1, 2, 3]
    assert global_topn["selected_aug_ids"].tolist() == [
        "aug_001",
        "aug_001 aug_000",
        "aug_001 aug_000 aug_002",
    ]
    assert global_topn["forwards_per_image"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert global_topn["relative_compute_vs_all"].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_evaluate_private_from_artifacts_adds_adaptive_strategy_when_tuned(
    tmp_path: Path,
    private_eval_artifacts: dict[str, Path],
) -> None:
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=3,
        usefulness_head=True,
    )
    tuning_path = tmp_path / "public_val_tta_tuning.json"
    tuning_path.write_text(
        json.dumps(
            {
                "split": "public_val",
                "best_k": 1,
                "best_adaptive_threshold": 0.25,
                "best_adaptive_max_k": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_private_from_artifacts(
        split="private",
        manifest_path=private_eval_artifacts["manifest"],
        cache_dir=private_eval_artifacts["cache_dir"],
        checkpoint_path=checkpoint_path,
        tuning_path=tuning_path,
        output_dir=tmp_path / "reports",
        aug_ids=["aug_000", "aug_001", "aug_002"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        random_seeds=[1],
        device="cpu",
    )
    table = pd.read_csv(summary.private_metrics_csv)

    assert "learned_adaptive_uniform" in set(table["strategy"])
    assert summary.metrics_by_strategy["learned_adaptive_uniform"][
        "forwards_per_image"
    ] == pytest.approx(2.0)


def test_evaluate_private_from_artifacts_rejects_public_val_split(
    tmp_path: Path,
    private_eval_artifacts: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="evaluate-private split must be private"):
        evaluate_private_from_artifacts(
            split="public_val",
            manifest_path=private_eval_artifacts["manifest"],
            cache_dir=private_eval_artifacts["cache_dir"],
            checkpoint_path=private_eval_artifacts["checkpoint"],
            tuning_path=private_eval_artifacts["tuning"],
            output_dir=tmp_path / "reports",
            aug_ids=["aug_000", "aug_001", "aug_002"],
            image_size=16,
            batch_size=2,
            num_workers=0,
            random_seeds=[1],
            device="cpu",
        )


def test_evaluate_private_from_artifacts_rejects_private_tuning_artifact(
    tmp_path: Path,
    private_eval_artifacts: dict[str, Path],
) -> None:
    bad_tuning_path = tmp_path / "private_tta_tuning.json"
    bad_tuning_path.write_text(
        json.dumps({"best_k": 1, "split": "private"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tuning artifact split must be public_val"):
        evaluate_private_from_artifacts(
            split="private",
            manifest_path=private_eval_artifacts["manifest"],
            cache_dir=private_eval_artifacts["cache_dir"],
            checkpoint_path=private_eval_artifacts["checkpoint"],
            tuning_path=bad_tuning_path,
            output_dir=tmp_path / "reports",
            aug_ids=["aug_000", "aug_001", "aug_002"],
            image_size=16,
            batch_size=2,
            num_workers=0,
            random_seeds=[1],
            device="cpu",
        )


def test_evaluate_private_cli_writes_private_metrics(
    tmp_path: Path,
    private_eval_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "reports"

    main(
        [
            "evaluate-private",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--manifest",
            str(private_eval_artifacts["manifest"]),
            "--cache-dir",
            str(private_eval_artifacts["cache_dir"]),
            "--checkpoint",
            str(private_eval_artifacts["checkpoint"]),
            "--tuning",
            str(private_eval_artifacts["tuning"]),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--candidate-id",
            "aug_002",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--image-size",
            "16",
        ]
    )
    captured = capsys.readouterr()

    assert "private evaluation: best k 1" in captured.out
    assert (output_dir / "tables" / "private_metrics.csv").exists()
    assert (output_dir / "tables" / "corrections.csv").exists()


def _write_manifest(root: Path, split: str, count: int) -> Path:
    rows = []
    for index in range(count):
        path = root / f"{split}_{index}.png"
        image = np.full((12, 12, 3), fill_value=30 + index, dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(path)
        rows.append(
            {
                "split": split,
                "image_id": f"{split}-{index}",
                "class_idx": index % 2,
                "class_name": f"class-{index % 2}",
                "path": str(path),
            }
        )
    manifest_path = root / f"{split}.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _write_cache(cache_dir: Path) -> Path:
    image_ids = ["private-0", "private-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="private",
            aug_id="aug_000",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="private",
            aug_id="aug_001",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="private",
            aug_id="aug_002",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[4.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        ),
    )
    return cache_dir


def _write_selector_checkpoint(
    path: Path,
    output_dim: int,
    usefulness_head: bool = False,
) -> Path:
    model = SelectorCNN(output_dim=output_dim, usefulness_head=usefulness_head)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    torch.save(
        {
            "epoch": 1,
            "val_nll": 0.0,
            "aug_ids": [f"aug_{index:03d}" for index in range(output_dim)],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "usefulness_head": usefulness_head,
        },
        path,
    )
    return path


def _install_fake_xgboost(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeXGBClassifier:
        def load_model(self, path: str | Path) -> None:
            assert Path(path).exists()

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            assert features.shape == (2, 6)
            return np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "xgboost",
        SimpleNamespace(XGBClassifier=FakeXGBClassifier),
    )
