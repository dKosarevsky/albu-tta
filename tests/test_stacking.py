from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.stacking import (
    AggregationArtifact,
    load_aggregation_artifact,
    train_aggregator_from_artifacts,
    train_class_nonnegative_weights,
    train_global_nonnegative_weights,
)


def test_train_global_nonnegative_weights_prefers_helpful_augmentation() -> None:
    logits_by_aug = {
        "aug_000": np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
        "aug_001": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        "aug_002": np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float32),
    }

    artifact = train_global_nonnegative_weights(
        logits_by_aug=logits_by_aug,
        class_idxs=np.array([0, 1], dtype=np.int64),
        aug_ids=["aug_000", "aug_001", "aug_002"],
        epochs=80,
        learning_rate=0.1,
        l1_penalty=0.0,
        active_threshold=1e-6,
        device="cpu",
    )

    assert artifact.method == "global-nonnegative"
    assert artifact.weights.shape == (3,)
    assert np.all(artifact.weights >= 0.0)
    assert artifact.aug_ids[int(np.argmax(artifact.weights))] == "aug_001"
    assert artifact.metrics["nll"] < 0.1


def test_train_class_nonnegative_weights_learns_per_class_profiles() -> None:
    logits_by_aug = {
        "aug_000": np.array([[5.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "aug_001": np.array([[1.0, 0.0], [0.0, 5.0]], dtype=np.float32),
    }

    artifact = train_class_nonnegative_weights(
        logits_by_aug=logits_by_aug,
        class_idxs=np.array([0, 1], dtype=np.int64),
        aug_ids=["aug_000", "aug_001"],
        epochs=80,
        learning_rate=0.1,
        l1_penalty=0.0,
        active_threshold=1e-6,
        device="cpu",
    )

    assert artifact.method == "class-nonnegative"
    assert artifact.weights.shape == (2, 2)
    assert np.all(artifact.weights >= 0.0)
    np.testing.assert_allclose(artifact.weights.sum(axis=1), np.ones(2), rtol=1e-6)
    assert artifact.metrics["nll"] < 0.4


def test_train_aggregator_from_artifacts_writes_json(tmp_path: Path) -> None:
    cache_dir = _write_public_val_cache(tmp_path / "teacher_cache")
    output_path = tmp_path / "selector" / "global.json"

    summary = train_aggregator_from_artifacts(
        split="public_val",
        cache_dir=cache_dir,
        output_path=output_path,
        aug_ids=["aug_000", "aug_001"],
        method="global-nonnegative",
        epochs=20,
        learning_rate=0.1,
        l1_penalty=0.0,
        active_threshold=1e-6,
        device="cpu",
    )
    loaded = load_aggregation_artifact(summary.path)

    assert summary.path == output_path
    assert loaded.method == "global-nonnegative"
    assert loaded.aug_ids == ["aug_000", "aug_001"]
    assert loaded.weights.shape == (2,)


def test_train_aggregator_cli_writes_default_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    cache_dir = _write_public_val_cache(tmp_path / "teacher_cache")
    output_dir = tmp_path / "selector"

    main(
        [
            "train-aggregator",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--split",
            "public_val",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--method",
            "global-nonnegative",
            "--epochs",
            "20",
            "--learning-rate",
            "0.1",
        ]
    )
    captured = capsys.readouterr()

    assert "aggregator global-nonnegative: wrote" in captured.out
    assert (output_dir / "public_val_global_nonnegative_aggregator.json").exists()


def test_train_aggregator_rejects_unknown_method(tmp_path: Path) -> None:
    cache_dir = _write_public_val_cache(tmp_path / "teacher_cache")

    with pytest.raises(ValueError, match="unknown aggregator method"):
        train_aggregator_from_artifacts(
            split="public_val",
            cache_dir=cache_dir,
            output_path=tmp_path / "bad.json",
            aug_ids=["aug_000", "aug_001"],
            method="xgboost",
            epochs=1,
            learning_rate=0.1,
            l1_penalty=0.0,
            active_threshold=1e-6,
            device="cpu",
        )


def test_aggregation_artifact_roundtrips(tmp_path: Path) -> None:
    artifact = AggregationArtifact(
        method="global-nonnegative",
        aug_ids=["aug_000", "aug_001"],
        weights=np.array([0.2, 0.8], dtype=np.float32),
        active_threshold=1e-6,
        metrics={"nll": 0.1},
    )
    path = tmp_path / "weights.json"

    artifact.save(path)
    loaded = load_aggregation_artifact(path)

    assert loaded.method == artifact.method
    assert loaded.aug_ids == artifact.aug_ids
    np.testing.assert_allclose(loaded.weights, artifact.weights)
    assert loaded.metrics == artifact.metrics


def _write_public_val_cache(cache_dir: Path) -> Path:
    image_ids = ["public_val-0", "public_val-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
            aug_id="aug_000",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[0.0, 2.0], [2.0, 0.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_val",
            aug_id="aug_001",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32),
        ),
    )
    return cache_dir
