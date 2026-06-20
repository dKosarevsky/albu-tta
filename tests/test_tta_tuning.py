from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.data import load_manifest
from learned_tta.selector_model import SelectorCNN
from learned_tta.tta_tuning import predict_selector_scores, tune_tta_from_artifacts


@pytest.fixture
def tuning_artifacts(tmp_path: Path) -> dict[str, Path]:
    manifest_path = _write_manifest(tmp_path, split="public_val", count=2)
    cache_dir = _write_cache(tmp_path / "teacher_cache")
    checkpoint_path = _write_selector_checkpoint(tmp_path / "selector_best.pt", output_dim=2)
    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "checkpoint": checkpoint_path,
    }


def test_tune_tta_from_artifacts_selects_and_writes_best_k(
    tmp_path: Path,
    tuning_artifacts: dict[str, Path],
) -> None:
    summary = tune_tta_from_artifacts(
        split="public_val",
        manifest_path=tuning_artifacts["manifest"],
        cache_dir=tuning_artifacts["cache_dir"],
        checkpoint_path=tuning_artifacts["checkpoint"],
        output_dir=tmp_path / "selector",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[0, 1],
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    saved = json.loads(summary.result_path.read_text(encoding="utf-8"))

    assert summary.best_k in {0, 1}
    assert set(summary.results_by_k) == {0, 1}
    assert saved["best_k"] == summary.best_k
    assert saved["split"] == "public_val"
    assert summary.predicted_gain_shape == (2, 2)


def test_tune_tta_from_artifacts_writes_adaptive_public_val_tuning(
    tmp_path: Path,
    tuning_artifacts: dict[str, Path],
) -> None:
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=2,
        usefulness_head=True,
    )

    summary = tune_tta_from_artifacts(
        split="public_val",
        manifest_path=tuning_artifacts["manifest"],
        cache_dir=tuning_artifacts["cache_dir"],
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "selector",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[0, 1],
        adaptive_threshold_grid=[0.25, 0.75],
        adaptive_max_k_grid=[0, 1],
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    saved = json.loads(summary.result_path.read_text(encoding="utf-8"))

    assert summary.best_adaptive_threshold in {0.25, 0.75}
    assert summary.best_adaptive_max_k in {0, 1}
    assert saved["best_adaptive_threshold"] == summary.best_adaptive_threshold
    assert saved["best_adaptive_max_k"] == summary.best_adaptive_max_k
    assert saved["predicted_useful_shape"] == [2, 2]
    assert set(saved["adaptive_results"]) == {
        "threshold=0.25,max_k=0",
        "threshold=0.25,max_k=1",
        "threshold=0.75,max_k=0",
        "threshold=0.75,max_k=1",
    }


def test_tune_tta_from_artifacts_writes_selector_diagnostics(
    tmp_path: Path,
    tuning_artifacts: dict[str, Path],
) -> None:
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=2,
        usefulness_head=True,
    )

    summary = tune_tta_from_artifacts(
        split="public_val",
        manifest_path=tuning_artifacts["manifest"],
        cache_dir=tuning_artifacts["cache_dir"],
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "selector",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[0, 1],
        adaptive_threshold_grid=[0.25, 0.75],
        adaptive_max_k_grid=[0, 1],
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    saved = json.loads(summary.result_path.read_text(encoding="utf-8"))

    assert summary.diagnostics_path is not None
    assert summary.selection_counts_path is not None
    assert summary.diagnostics_path.exists()
    assert summary.selection_counts_path.exists()
    assert saved["selector_diagnostics"]["gain_pearson"] == pytest.approx(
        summary.selector_diagnostics["gain_pearson"]
    )
    assert set(saved["selector_diagnostics"]["topk_hit_rate_by_k"]) == {"1"}
    assert saved["selector_diagnostics"]["usefulness_calibration"]["threshold"] == pytest.approx(
        0.01
    )
    selection_counts = pd.read_csv(summary.selection_counts_path)
    assert selection_counts["threshold"].tolist() == [0.25, 0.75]
    assert "mean_forwards_per_image" in selection_counts.columns


def test_tune_tta_from_artifacts_rejects_private_split(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, split="private", count=2)
    cache_dir = _write_cache(tmp_path / "teacher_cache", split="private")
    checkpoint_path = _write_selector_checkpoint(tmp_path / "selector_best.pt", output_dim=2)

    with pytest.raises(ValueError, match="tune-tta split must be public_val"):
        tune_tta_from_artifacts(
            split="private",
            manifest_path=manifest_path,
            cache_dir=cache_dir,
            checkpoint_path=checkpoint_path,
            output_dir=tmp_path / "selector",
            aug_ids=["aug_000", "aug_001"],
            top_k_grid=[1],
            image_size=16,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )


def test_tune_tta_cli_writes_result_json(
    tmp_path: Path,
    tuning_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "selector"

    main(
        [
            "tune-tta",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--split",
            "public_val",
            "--manifest",
            str(tuning_artifacts["manifest"]),
            "--cache-dir",
            str(tuning_artifacts["cache_dir"]),
            "--checkpoint",
            str(tuning_artifacts["checkpoint"]),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--top-k",
            "0",
            "--top-k",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--image-size",
            "16",
        ]
    )
    captured = capsys.readouterr()

    assert "tta tuning public_val: best k" in captured.out
    assert (output_dir / "public_val_tta_tuning.json").exists()


def test_predict_selector_scores_returns_unstandardized_gain(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, split="public_val", count=2)
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=2,
        target_mean=np.array([0.25, -0.5], dtype=np.float32),
        target_std=np.array([2.0, 4.0], dtype=np.float32),
    )

    scores = predict_selector_scores(
        checkpoint_path=checkpoint_path,
        records=load_manifest(manifest_path),
        output_dim=2,
        aug_ids=["aug_000", "aug_001"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    np.testing.assert_allclose(
        scores,
        np.array([[0.25, -0.5], [0.25, -0.5]], dtype=np.float32),
    )


def test_predict_selector_scores_loads_usefulness_head_checkpoint(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, split="public_val", count=2)
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=2,
        target_mean=np.array([0.25, -0.5], dtype=np.float32),
        target_std=np.array([2.0, 4.0], dtype=np.float32),
        usefulness_head=True,
    )

    scores = predict_selector_scores(
        checkpoint_path=checkpoint_path,
        records=load_manifest(manifest_path),
        output_dim=2,
        aug_ids=["aug_000", "aug_001"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    assert scores.shape == (2, 2)


def test_predict_selector_scores_rejects_checkpoint_aug_id_order_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, split="public_val", count=2)
    checkpoint_path = _write_selector_checkpoint(
        tmp_path / "selector_best.pt",
        output_dim=2,
        target_mean=np.array([0.0, 0.0], dtype=np.float32),
        target_std=np.array([1.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="checkpoint aug_ids must match requested aug_ids"):
        predict_selector_scores(
            checkpoint_path=checkpoint_path,
            records=load_manifest(manifest_path),
            output_dim=2,
            aug_ids=["aug_001", "aug_000"],
            image_size=16,
            batch_size=2,
            num_workers=0,
            device="cpu",
        )


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


def _write_cache(cache_dir: Path, split: str = "public_val") -> Path:
    image_ids = [f"{split}-0", f"{split}-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split=split,
            aug_id="aug_000",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split=split,
            aug_id="aug_001",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    )
    return cache_dir


def _write_selector_checkpoint(
    path: Path,
    output_dim: int,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    usefulness_head: bool = False,
) -> Path:
    model = SelectorCNN(output_dim=output_dim, usefulness_head=usefulness_head)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    checkpoint: dict[str, object] = {
        "epoch": 1,
        "val_nll": 0.0,
        "aug_ids": [f"aug_{index:03d}" for index in range(output_dim)],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "usefulness_head": usefulness_head,
    }
    if target_mean is not None and target_std is not None:
        checkpoint["target_mean"] = target_mean
        checkpoint["target_std"] = target_std
    torch.save(
        checkpoint,
        path,
    )
    return path
