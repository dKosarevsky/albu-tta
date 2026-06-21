from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.selector_features import save_selector_features
from learned_tta.selector_training import (
    DEFAULT_SELECTOR_LOSS_ABLATIONS,
    SelectorTrainingSummary,
    make_selector_dataloader,
    train_selector_from_artifacts,
    train_selector_loss_ablation_from_artifacts,
)
from learned_tta.targets import TargetStats, save_selector_targets


@pytest.fixture
def selector_training_artifacts(tmp_path: Path) -> dict[str, Path]:
    train_manifest = _write_manifest(tmp_path, split="public_train", count=4)
    val_manifest = _write_manifest(tmp_path, split="public_val", count=2)
    train_targets = _write_targets(tmp_path / "public_train_targets.npz", rows=4)
    val_targets = _write_targets(tmp_path / "public_val_targets.npz", rows=2)
    cache_dir = _write_cache(tmp_path / "teacher_cache")
    return {
        "train_manifest": train_manifest,
        "val_manifest": val_manifest,
        "train_targets": train_targets,
        "val_targets": val_targets,
        "cache_dir": cache_dir,
    }


def test_make_selector_dataloader_returns_image_target_batches(
    selector_training_artifacts: dict[str, Path],
) -> None:
    dataloader = make_selector_dataloader(
        manifest_path=selector_training_artifacts["train_manifest"],
        targets_path=selector_training_artifacts["train_targets"],
        image_size=16,
        batch_size=2,
        num_workers=0,
        shuffle=False,
    )

    images, targets, gain = next(iter(dataloader))

    assert images.shape == (2, 3, 16, 16)
    assert targets.shape == (2, 2)
    assert gain.shape == (2, 2)
    assert images.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert gain.dtype == torch.float32


def test_make_selector_dataloader_rejects_target_manifest_order_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, split="public_train", count=2)
    targets_path = _write_targets(
        tmp_path / "public_train_targets.npz",
        rows=2,
        image_ids=["public_train-1", "public_train-0"],
    )

    with pytest.raises(ValueError, match="selector target image_ids must match manifest image_ids"):
        make_selector_dataloader(
            manifest_path=manifest_path,
            targets_path=targets_path,
            image_size=16,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )


def test_train_selector_from_artifacts_saves_best_checkpoint(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    summary = train_selector_from_artifacts(
        train_manifest_path=selector_training_artifacts["train_manifest"],
        val_manifest_path=selector_training_artifacts["val_manifest"],
        train_targets_path=selector_training_artifacts["train_targets"],
        val_targets_path=selector_training_artifacts["val_targets"],
        output_dir=tmp_path / "selector",
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        rank_weight=0.2,
        usefulness_head=True,
        usefulness_tau=0.01,
        usefulness_weight=0.05,
        val_cache_dir=selector_training_artifacts["cache_dir"],
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[1],
        device="cpu",
    )

    checkpoint = torch.load(summary.checkpoint_path, weights_only=False)
    history = pd.read_csv(summary.history_csv)

    assert isinstance(summary, SelectorTrainingSummary)
    assert summary.checkpoint_path.exists()
    assert summary.history_csv.exists()
    assert summary.best_epoch == 1
    assert summary.history[0]["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert checkpoint["aug_ids"] == ["aug_000", "aug_001"]
    assert checkpoint["target_mean"].tolist() == pytest.approx([0.0, 0.0])
    assert checkpoint["target_std"].tolist() == pytest.approx([1.0, 1.0])
    assert checkpoint["target_kind"] == "gain"
    assert checkpoint["higher_is_better"] is True
    assert checkpoint["usefulness_head"] is True
    assert checkpoint["usefulness_tau"] == pytest.approx(0.01)
    assert checkpoint["usefulness_weight"] == pytest.approx(0.05)
    assert "train_regression_loss" in summary.history[0]
    assert "train_rank_loss" in summary.history[0]
    assert "train_usefulness_bce" in summary.history[0]
    assert "val_regression_loss" in summary.history[0]
    assert "val_rank_loss" in summary.history[0]
    assert "val_usefulness_bce" in summary.history[0]
    assert "val_tta_nll" in summary.history[0]
    assert summary.history[0]["val_tta_best_k"] == 1
    assert "val_tta_oracle_recall" in summary.history[0]
    assert 0.0 <= summary.history[0]["val_tta_oracle_recall"] <= 1.0
    assert history["val_tta_oracle_recall"].iloc[0] == pytest.approx(
        summary.history[0]["val_tta_oracle_recall"]
    )
    assert checkpoint["val_nll"] == pytest.approx(summary.history[0]["val_tta_nll"])
    assert checkpoint["val_nll"] != pytest.approx(summary.history[0]["val_loss"])
    assert summary.best_val_nll == pytest.approx(checkpoint["val_nll"])


def test_train_selector_from_artifacts_rejects_private_validation_split(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="train-selector split must be public_val"):
        train_selector_from_artifacts(
            train_manifest_path=selector_training_artifacts["train_manifest"],
            val_manifest_path=selector_training_artifacts["val_manifest"],
            train_targets_path=selector_training_artifacts["train_targets"],
            val_targets_path=selector_training_artifacts["val_targets"],
            output_dir=tmp_path / "selector",
            image_size=16,
            batch_size=2,
            num_workers=0,
            epochs=1,
            learning_rate=1e-3,
            rank_weight=0.2,
            val_cache_dir=selector_training_artifacts["cache_dir"],
            val_split="private",
            aug_ids=["aug_000", "aug_001"],
            top_k_grid=[1],
            device="cpu",
        )


def test_train_selector_cli_writes_checkpoint(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "selector"

    main(
        [
            "train-selector",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--train-manifest",
            str(selector_training_artifacts["train_manifest"]),
            "--val-manifest",
            str(selector_training_artifacts["val_manifest"]),
            "--train-targets",
            str(selector_training_artifacts["train_targets"]),
            "--val-targets",
            str(selector_training_artifacts["val_targets"]),
            "--cache-dir",
            str(selector_training_artifacts["cache_dir"]),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--top-k",
            "1",
            "--epochs",
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

    assert "selector training: best epoch 1" in captured.out
    assert "best val nll" in captured.out
    assert (output_dir / "selector_best.pt").exists()
    assert (output_dir / "selector_history.csv").exists()


def test_train_selector_loss_ablation_from_artifacts_writes_variant_table(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    summary = train_selector_loss_ablation_from_artifacts(
        train_manifest_path=selector_training_artifacts["train_manifest"],
        val_manifest_path=selector_training_artifacts["val_manifest"],
        train_targets_path=selector_training_artifacts["train_targets"],
        val_targets_path=selector_training_artifacts["val_targets"],
        output_dir=tmp_path / "selector_ablation",
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        val_cache_dir=selector_training_artifacts["cache_dir"],
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[1],
        identity_aug_id="aug_000",
        device="cpu",
    )
    table = pd.read_csv(summary.results_csv)

    assert summary.results_csv.exists()
    assert table["variant"].tolist() == [
        "gain_only",
        "gain_rank",
        "gain_rank_bce",
        "gain_listwise_topk",
        "clean_logits_mlp_gain_rank",
        "clean_logits_mlp_gain_listwise",
    ]
    assert table["model_family"].tolist() == [
        "image_cnn",
        "image_cnn",
        "image_cnn",
        "image_cnn",
        "mlp",
        "mlp",
    ]
    assert table["feature_mode"].tolist() == [
        "image",
        "image",
        "image",
        "image",
        "clean_logits",
        "clean_logits",
    ]
    assert table["target_mode"].tolist() == ["nll_gain"] * 6
    assert table["usefulness_head"].tolist() == [False, False, True, False, False, False]
    assert table["rank_weight"].tolist() == pytest.approx([0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert table["listwise_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.1, 0.0, 0.1])
    assert set(table.columns) >= {
        "variant",
        "feature_mode",
        "target_mode",
        "listwise_weight",
        "listwise_top_k",
        "best_epoch",
        "best_val_loss",
        "best_val_nll",
        "checkpoint_path",
        "history_csv",
    }
    for checkpoint_path in table["checkpoint_path"]:
        assert Path(checkpoint_path).exists()


def test_train_selector_loss_ablation_from_artifacts_filters_variants(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    summary = train_selector_loss_ablation_from_artifacts(
        train_manifest_path=selector_training_artifacts["train_manifest"],
        val_manifest_path=selector_training_artifacts["val_manifest"],
        train_targets_path=selector_training_artifacts["train_targets"],
        val_targets_path=selector_training_artifacts["val_targets"],
        output_dir=tmp_path / "selector_ablation",
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        val_cache_dir=selector_training_artifacts["cache_dir"],
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[1],
        identity_aug_id="aug_000",
        device="cpu",
        variant_names=("clean_logits_mlp_gain_rank",),
    )
    table = pd.read_csv(summary.results_csv)

    assert table["variant"].tolist() == ["clean_logits_mlp_gain_rank"]
    assert table["feature_mode"].tolist() == ["clean_logits"]
    assert table["model_family"].tolist() == ["mlp"]
    assert (tmp_path / "selector_ablation" / "clean_logits_mlp_gain_rank").exists()
    assert not (tmp_path / "selector_ablation" / "gain_only").exists()


def test_train_selector_loss_ablation_from_artifacts_trains_pretrained_feature_variant(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    train_features = _write_selector_features(
        tmp_path / "public_train_features.npz",
        split="public_train",
        rows=4,
    )
    val_features = _write_selector_features(
        tmp_path / "public_val_features.npz",
        split="public_val",
        rows=2,
    )

    summary = train_selector_loss_ablation_from_artifacts(
        train_manifest_path=selector_training_artifacts["train_manifest"],
        val_manifest_path=selector_training_artifacts["val_manifest"],
        train_targets_path=selector_training_artifacts["train_targets"],
        val_targets_path=selector_training_artifacts["val_targets"],
        output_dir=tmp_path / "selector_ablation",
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        val_cache_dir=selector_training_artifacts["cache_dir"],
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[1],
        identity_aug_id="aug_000",
        device="cpu",
        variant_names=("pretrained_mlp_gain_rank",),
        train_features_path=train_features,
        val_features_path=val_features,
    )
    table = pd.read_csv(summary.results_csv)

    assert table["variant"].tolist() == ["pretrained_mlp_gain_rank"]
    assert table["feature_mode"].tolist() == ["pretrained"]
    assert table["model_family"].tolist() == ["mlp"]
    assert (tmp_path / "selector_ablation" / "pretrained_mlp_gain_rank").exists()


def test_train_selector_loss_ablation_from_artifacts_rejects_unknown_variant(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="unknown selector ablation variant"):
        train_selector_loss_ablation_from_artifacts(
            train_manifest_path=selector_training_artifacts["train_manifest"],
            val_manifest_path=selector_training_artifacts["val_manifest"],
            train_targets_path=selector_training_artifacts["train_targets"],
            val_targets_path=selector_training_artifacts["val_targets"],
            output_dir=tmp_path / "selector_ablation",
            image_size=16,
            batch_size=2,
            num_workers=0,
            epochs=1,
            learning_rate=1e-3,
            val_cache_dir=selector_training_artifacts["cache_dir"],
            val_split="public_val",
            aug_ids=["aug_000", "aug_001"],
            top_k_grid=[1],
            identity_aug_id="aug_000",
            device="cpu",
            variant_names=("does_not_exist",),
        )


def test_train_selector_loss_ablation_from_artifacts_skips_completed_variant(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
) -> None:
    output_dir = tmp_path / "selector_ablation"
    variant_dir = output_dir / "gain_only"
    variant_dir.mkdir(parents=True)
    checkpoint_path = variant_dir / "selector_best.pt"
    checkpoint_path.write_bytes(b"already-trained")
    pd.DataFrame(
        [
            {"epoch": 1, "val_loss": 0.9, "val_tta_nll": 0.7},
            {"epoch": 2, "val_loss": 0.8, "val_tta_nll": 0.6},
        ]
    ).to_csv(variant_dir / "selector_history.csv", index=False)

    summary = train_selector_loss_ablation_from_artifacts(
        train_manifest_path=selector_training_artifacts["train_manifest"],
        val_manifest_path=selector_training_artifacts["val_manifest"],
        train_targets_path=selector_training_artifacts["train_targets"],
        val_targets_path=selector_training_artifacts["val_targets"],
        output_dir=output_dir,
        image_size=16,
        batch_size=2,
        num_workers=0,
        epochs=1,
        learning_rate=1e-3,
        val_cache_dir=selector_training_artifacts["cache_dir"],
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        top_k_grid=[1],
        identity_aug_id="aug_000",
        device="cpu",
        variant_names=("gain_only",),
    )
    table = pd.read_csv(summary.results_csv)

    assert table["variant"].tolist() == ["gain_only"]
    assert table["status"].tolist() == ["skipped"]
    assert table["best_epoch"].tolist() == [2]
    assert table["best_val_loss"].tolist() == pytest.approx([0.8])
    assert table["best_val_nll"].tolist() == pytest.approx([0.6])
    assert checkpoint_path.read_bytes() == b"already-trained"


def test_default_selector_loss_ablation_specs_keep_image_feature_mode() -> None:
    assert {spec.feature_mode for spec in DEFAULT_SELECTOR_LOSS_ABLATIONS} == {
        "image",
        "clean_logits",
    }


def test_default_selector_loss_ablation_specs_keep_nll_gain_target_mode() -> None:
    assert {spec.target_mode for spec in DEFAULT_SELECTOR_LOSS_ABLATIONS} == {"nll_gain"}


def test_selector_loss_ablation_includes_listwise_variant() -> None:
    names = {spec.variant for spec in DEFAULT_SELECTOR_LOSS_ABLATIONS}

    assert "gain_listwise_topk" in names


def test_train_selector_loss_ablation_cli_writes_variant_table(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "selector_ablation"

    main(
        [
            "train-selector-ablation",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--train-manifest",
            str(selector_training_artifacts["train_manifest"]),
            "--val-manifest",
            str(selector_training_artifacts["val_manifest"]),
            "--train-targets",
            str(selector_training_artifacts["train_targets"]),
            "--val-targets",
            str(selector_training_artifacts["val_targets"]),
            "--cache-dir",
            str(selector_training_artifacts["cache_dir"]),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--top-k",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--image-size",
            "16",
            "--ablation-variant",
            "clean_logits_mlp_gain_rank",
            "--force",
        ]
    )
    captured = capsys.readouterr()
    table = pd.read_csv(output_dir / "selector_loss_ablation.csv")

    assert "selector ablation: wrote" in captured.out
    assert (output_dir / "selector_loss_ablation.csv").exists()
    assert table["variant"].tolist() == ["clean_logits_mlp_gain_rank"]


def test_train_selector_loss_ablation_cli_accepts_pretrained_feature_paths(
    tmp_path: Path,
    selector_training_artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "selector_ablation"
    train_features = _write_selector_features(
        tmp_path / "public_train_features.npz",
        split="public_train",
        rows=4,
    )
    val_features = _write_selector_features(
        tmp_path / "public_val_features.npz",
        split="public_val",
        rows=2,
    )

    main(
        [
            "train-selector-ablation",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--train-manifest",
            str(selector_training_artifacts["train_manifest"]),
            "--val-manifest",
            str(selector_training_artifacts["val_manifest"]),
            "--train-targets",
            str(selector_training_artifacts["train_targets"]),
            "--val-targets",
            str(selector_training_artifacts["val_targets"]),
            "--cache-dir",
            str(selector_training_artifacts["cache_dir"]),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--top-k",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--image-size",
            "16",
            "--ablation-variant",
            "pretrained_mlp_gain_rank",
            "--train-features",
            str(train_features),
            "--val-features",
            str(val_features),
            "--force",
        ]
    )
    captured = capsys.readouterr()
    table = pd.read_csv(output_dir / "selector_loss_ablation.csv")

    assert "selector ablation: wrote" in captured.out
    assert table["variant"].tolist() == ["pretrained_mlp_gain_rank"]
    assert table["feature_mode"].tolist() == ["pretrained"]


def _write_manifest(root: Path, split: str, count: int) -> Path:
    rows = []
    for index in range(count):
        path = root / f"{split}_{index}.png"
        image = np.full((12, 12, 3), fill_value=20 + index, dtype=np.uint8)
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


def _write_targets(path: Path, rows: int, image_ids: list[str] | None = None) -> Path:
    gain = np.stack(
        [
            np.linspace(0.0, 0.3, rows, dtype=np.float32),
            np.linspace(0.4, -0.2, rows, dtype=np.float32),
        ],
        axis=1,
    )
    stats = TargetStats(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
    )
    save_selector_targets(
        path=path,
        aug_ids=["aug_000", "aug_001"],
        image_ids=image_ids
        or [f"{path.stem.removesuffix('_targets')}-{index}" for index in range(rows)],
        gain=gain,
        target_z=gain,
        stats=stats,
    )
    return path


def _write_selector_features(path: Path, split: str, rows: int) -> Path:
    feature_count = 6
    features = np.arange(rows * feature_count, dtype=np.float32).reshape(rows, feature_count)
    features = features / max(float(features.max()), 1.0)
    return save_selector_features(
        path=path,
        split=split,
        model_name="fake_pretrained",
        image_ids=[f"{split}-{index}" for index in range(rows)],
        features=features,
        feature_names=[f"feature_{index:04d}" for index in range(feature_count)],
        metadata={"pretrained": True},
    )


def _write_cache(cache_dir: Path) -> Path:
    for split, rows in (("public_train", 4), ("public_val", 2)):
        image_ids = [f"{split}-{index}" for index in range(rows)]
        class_idxs = np.array([index % 2 for index in range(rows)], dtype=np.int64)
        clean_logits = np.stack(
            [
                np.array([3.0, 0.0], dtype=np.float32)
                if class_idx == 0
                else np.array([0.0, 3.0], dtype=np.float32)
                for class_idx in class_idxs
            ],
            axis=0,
        )
        aug_logits = clean_logits.copy()
        aug_logits[:, 0] += 1.0
        write_teacher_shard(
            cache_dir,
            TeacherShard(
                split=split,
                aug_id="aug_000",
                image_ids=image_ids,
                class_idxs=class_idxs,
                logits=clean_logits,
            ),
        )
        write_teacher_shard(
            cache_dir,
            TeacherShard(
                split=split,
                aug_id="aug_001",
                image_ids=image_ids,
                class_idxs=class_idxs,
                logits=aug_logits,
            ),
        )
    return cache_dir
