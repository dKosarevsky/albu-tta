from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.target_builder import build_selector_targets_from_cache
from learned_tta.targets import load_selector_targets


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "teacher_cache"
    _write_split_cache(
        cache,
        split="public_train",
        identity_logits=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        aug_logits=np.array([[4.0, 0.0], [3.0, 0.0]], dtype=np.float32),
    )
    _write_split_cache(
        cache,
        split="public_val",
        identity_logits=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        aug_logits=np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    return cache


def test_build_selector_targets_from_cache_uses_train_stats_for_val(
    tmp_path: Path,
    cache_dir: Path,
) -> None:
    output_dir = tmp_path / "selector"

    summary = build_selector_targets_from_cache(
        cache_dir=cache_dir,
        output_dir=output_dir,
        train_split="public_train",
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )

    train_targets = load_selector_targets(summary.train_path)
    val_targets = load_selector_targets(summary.val_path)

    assert summary.aug_ids == ["aug_000", "aug_001"]
    assert summary.train_rows == 2
    assert summary.val_rows == 2
    assert train_targets.aug_ids == ["aug_000", "aug_001"]
    assert val_targets.aug_ids == ["aug_000", "aug_001"]
    assert train_targets.image_ids == ["image-0", "image-1"]
    assert val_targets.image_ids == ["image-0", "image-1"]
    np.testing.assert_allclose(val_targets.stats.mean, train_targets.stats.mean)
    np.testing.assert_allclose(val_targets.stats.std, train_targets.stats.std)
    assert train_targets.gain.shape == (2, 2)
    assert val_targets.gain.shape == (2, 2)
    assert train_targets.target_kind == "gain"
    assert val_targets.target_kind == "gain"


def test_build_selector_targets_from_cache_can_use_softmax_weight_target(
    tmp_path: Path,
    cache_dir: Path,
) -> None:
    output_dir = tmp_path / "selector"

    summary = build_selector_targets_from_cache(
        cache_dir=cache_dir,
        output_dir=output_dir,
        train_split="public_train",
        val_split="public_val",
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
        target_kind="softmax_weight",
    )

    train_targets = load_selector_targets(summary.train_path)
    train_weights = train_targets.target_z * train_targets.stats.std + train_targets.stats.mean

    assert summary.target_kind == "softmax_weight"
    assert train_targets.target_kind == "softmax_weight"
    assert train_targets.higher_is_better is True
    np.testing.assert_allclose(train_weights.sum(axis=1), np.ones(2), atol=1e-6)
    assert np.all(train_weights >= 0.0)


@pytest.mark.parametrize(
    ("train_split", "val_split", "match"),
    [
        ("private", "public_val", "train_split must be public_train"),
        ("public_train", "private", "val_split must be public_val"),
    ],
)
def test_build_selector_targets_from_cache_rejects_private_leakage(
    tmp_path: Path,
    cache_dir: Path,
    train_split: str,
    val_split: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_selector_targets_from_cache(
            cache_dir=cache_dir,
            output_dir=tmp_path / "selector",
            train_split=train_split,
            val_split=val_split,
            aug_ids=["aug_000", "aug_001"],
            identity_aug_id="aug_000",
        )


def test_build_selector_targets_from_cache_rejects_image_id_order_mismatch(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "teacher_cache"
    _write_split_cache(
        cache,
        split="public_train",
        identity_logits=np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        aug_logits=np.array([[4.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        aug_image_ids=["image-1", "image-0"],
    )
    _write_split_cache(
        cache,
        split="public_val",
        identity_logits=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        aug_logits=np.array([[3.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="image_id order mismatch"):
        build_selector_targets_from_cache(
            cache_dir=cache,
            output_dir=tmp_path / "selector",
            train_split="public_train",
            val_split="public_val",
            aug_ids=["aug_000", "aug_001"],
            identity_aug_id="aug_000",
        )


def test_build_targets_cli_writes_default_artifacts(
    tmp_path: Path,
    cache_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    output_dir = tmp_path / "selector"

    main(
        [
            "build-targets",
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs/experiment/resnet50_a1_in1k.yaml"),
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(output_dir),
            "--candidate-id",
            "aug_000",
            "--candidate-id",
            "aug_001",
            "--target-kind",
            "true_logit",
        ]
    )
    captured = capsys.readouterr()

    assert (
        "selector targets: wrote public_train_targets.npz and public_val_targets.npz"
        in captured.out
    )
    assert "target_kind=true_logit" in captured.out
    assert (output_dir / "public_train_targets.npz").exists()
    assert (output_dir / "public_val_targets.npz").exists()
    assert (
        load_selector_targets(output_dir / "public_train_targets.npz").target_kind
        == "true_logit"
    )


def _write_split_cache(
    cache_dir: Path,
    split: str,
    identity_logits: np.ndarray,
    aug_logits: np.ndarray,
    aug_image_ids: list[str] | None = None,
) -> None:
    image_ids = ["image-0", "image-1"]
    class_idxs = np.array([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split=split,
            aug_id="aug_000",
            image_ids=image_ids,
            class_idxs=class_idxs,
            logits=identity_logits,
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split=split,
            aug_id="aug_001",
            image_ids=aug_image_ids or image_ids,
            class_idxs=class_idxs,
            logits=aug_logits,
        ),
    )
