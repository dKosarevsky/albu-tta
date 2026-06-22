from __future__ import annotations

import numpy as np
import pytest


def test_evaluate_cached_standard_baselines_adds_clean_and_hflip_rows() -> None:
    from learned_tta.standard_baselines import evaluate_cached_standard_baselines

    logits_by_aug = {
        "aug_000": np.array(
            [
                [4.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        ),
        "aug_005": np.array(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }
    class_idxs = np.array([0, 1, 2], dtype=np.int64)

    metrics = evaluate_cached_standard_baselines(
        logits_by_aug=logits_by_aug,
        class_idxs=class_idxs,
        identity_aug_id="aug_000",
        hflip_aug_id="aug_005",
        reference_aug_count=100,
    )

    assert set(metrics) == {"clean_center_crop", "center_crop_hflip"}
    assert metrics["clean_center_crop"]["top1"] == pytest.approx(2 / 3)
    assert metrics["clean_center_crop"]["forwards_per_image"] == pytest.approx(1.0)
    assert metrics["clean_center_crop"]["relative_compute_vs_all"] == pytest.approx(0.01)
    assert metrics["center_crop_hflip"]["top1"] == pytest.approx(1.0)
    assert metrics["center_crop_hflip"]["forwards_per_image"] == pytest.approx(2.0)
    assert metrics["center_crop_hflip"]["relative_compute_vs_all"] == pytest.approx(0.02)


def test_evaluate_cached_standard_baselines_skips_missing_hflip() -> None:
    from learned_tta.standard_baselines import evaluate_cached_standard_baselines

    metrics = evaluate_cached_standard_baselines(
        logits_by_aug={
            "aug_000": np.array([[2.0, 0.0]], dtype=np.float32),
        },
        class_idxs=np.array([0], dtype=np.int64),
        identity_aug_id="aug_000",
        hflip_aug_id="aug_005",
        reference_aug_count=100,
    )

    assert set(metrics) == {"clean_center_crop"}


def test_evaluate_ten_crop_logits_averages_crop_probabilities() -> None:
    from learned_tta.standard_baselines import evaluate_ten_crop_logits, ten_crop_probabilities

    crop_logits = np.array(
        [
            [[0.0, 4.0, 0.0]] * 6 + [[3.0, 0.0, 0.0]] * 4,
            [[0.0, 0.0, 4.0]] * 7 + [[0.0, 3.0, 0.0]] * 3,
        ],
        dtype=np.float32,
    )
    class_idxs = np.array([1, 2], dtype=np.int64)

    metrics = evaluate_ten_crop_logits(
        crop_logits=crop_logits,
        class_idxs=class_idxs,
        reference_aug_count=100,
    )
    probabilities = ten_crop_probabilities(crop_logits)

    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["forwards_per_image"] == pytest.approx(10.0)
    assert metrics["relative_compute_vs_all"] == pytest.approx(0.10)
    assert probabilities.shape == (2, 3)
    assert probabilities.argmax(axis=1).tolist() == [1, 2]


def test_ten_crop_npz_roundtrip(tmp_path) -> None:
    from learned_tta.standard_baselines import load_ten_crop_logits, write_ten_crop_logits

    path = tmp_path / "ten_crop_logits.npz"
    crop_logits = np.ones((2, 10, 3), dtype=np.float32)
    class_idxs = np.array([0, 2], dtype=np.int64)
    image_ids = ["a", "b"]

    write_ten_crop_logits(
        path=path,
        crop_logits=crop_logits,
        class_idxs=class_idxs,
        image_ids=image_ids,
    )
    loaded = load_ten_crop_logits(path)

    assert loaded.image_ids == image_ids
    assert np.array_equal(loaded.class_idxs, class_idxs)
    assert np.array_equal(loaded.crop_logits, crop_logits)


def test_evaluate_ten_crop_logits_rejects_bad_shapes() -> None:
    from learned_tta.standard_baselines import evaluate_ten_crop_logits

    with pytest.raises(ValueError, match="shape \\[num_images, 10, num_classes\\]"):
        evaluate_ten_crop_logits(
            crop_logits=np.zeros((2, 3), dtype=np.float32),
            class_idxs=np.array([0, 1], dtype=np.int64),
        )

    with pytest.raises(ValueError, match="exactly 10 crops"):
        evaluate_ten_crop_logits(
            crop_logits=np.zeros((2, 9, 3), dtype=np.float32),
            class_idxs=np.array([0, 1], dtype=np.int64),
        )
