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
        image_size=16,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )

    np.testing.assert_allclose(
        scores,
        np.array([[0.25, -0.5], [0.25, -0.5]], dtype=np.float32),
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


def _write_cache(cache_dir: Path) -> Path:
    image_ids = ["public_val-0", "public_val-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
            aug_id="aug_000",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
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
) -> Path:
    model = SelectorCNN(output_dim=output_dim)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)
    checkpoint: dict[str, object] = {
        "epoch": 1,
        "val_nll": 0.0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
    }
    if target_mean is not None and target_std is not None:
        checkpoint["aug_ids"] = [f"aug_{index:03d}" for index in range(output_dim)]
        checkpoint["target_mean"] = target_mean
        checkpoint["target_std"] = target_std
    torch.save(
        checkpoint,
        path,
    )
    return path
