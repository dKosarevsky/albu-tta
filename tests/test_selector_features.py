from __future__ import annotations

import numpy as np
import pytest


def test_clean_logit_features_include_margin_entropy_and_topk() -> None:
    from learned_tta.selector_features import clean_logit_features

    logits = np.array([[4.0, 2.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    features, names = clean_logit_features(logits, top_k=2)

    assert names[:4] == [
        "clean_confidence",
        "clean_margin",
        "clean_entropy",
        "clean_pred_class",
    ]
    assert "clean_top1_prob" in names
    assert "clean_top2_prob" in names
    assert features.shape == (2, len(names))
    assert (
        features[0, names.index("clean_confidence")] > features[1, names.index("clean_confidence")]
    )
    assert features[0, names.index("clean_margin")] > features[1, names.index("clean_margin")]
    assert features[1, names.index("clean_entropy")] == pytest.approx(np.log(3.0))


def test_clean_logit_features_reject_bad_inputs() -> None:
    from learned_tta.selector_features import clean_logit_features

    with pytest.raises(ValueError, match="shape"):
        clean_logit_features(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError, match="top_k"):
        clean_logit_features(np.zeros((2, 3), dtype=np.float32), top_k=0)
    with pytest.raises(ValueError, match="at least one class"):
        clean_logit_features(np.zeros((2, 0), dtype=np.float32))


def test_clean_logit_uncertainty_features_include_true_class_and_top_logits() -> None:
    from learned_tta.selector_features import clean_logit_uncertainty_features

    logits = np.array([[4.0, 2.0, 1.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    class_idxs = np.array([0, 2], dtype=np.int64)
    features, names = clean_logit_uncertainty_features(logits, class_idxs, top_k=2)

    assert names[:8] == [
        "clean_confidence",
        "clean_true_prob",
        "clean_prob_margin",
        "clean_logit_margin",
        "clean_entropy",
        "clean_true_logit",
        "clean_pred_class",
        "clean_pred_is_true",
    ]
    assert "clean_top1_prob" in names
    assert "clean_top2_prob" in names
    assert "clean_top1_logit" in names
    assert "clean_top2_logit" in names
    assert features.shape == (2, len(names))
    assert features[0, names.index("clean_pred_is_true")] == pytest.approx(1.0)
    assert features[1, names.index("clean_pred_is_true")] == pytest.approx(0.0)
    assert features[0, names.index("clean_true_prob")] > features[1, names.index("clean_true_prob")]


def test_clean_logit_uncertainty_features_reject_bad_class_indices() -> None:
    from learned_tta.selector_features import clean_logit_uncertainty_features

    with pytest.raises(ValueError, match="class_idxs"):
        clean_logit_uncertainty_features(
            np.zeros((2, 3), dtype=np.float32),
            np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="class_idxs"):
        clean_logit_uncertainty_features(
            np.zeros((2, 3), dtype=np.float32),
            np.array([0, 3], dtype=np.int64),
        )


def test_selector_feature_cache_roundtrip(tmp_path) -> None:
    from learned_tta.selector_features import load_selector_features, save_selector_features

    path = tmp_path / "features.npz"
    features = np.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=np.float32)

    save_selector_features(
        path=path,
        split="public_train",
        model_name="resnet50.a1_in1k",
        image_ids=["img-1", "img-2"],
        features=features,
        feature_names=["f0", "f1", "f2"],
        metadata={"pretrained": True, "data_config": {"input_size": [3, 224, 224]}},
    )
    loaded = load_selector_features(path)

    assert loaded.split == "public_train"
    assert loaded.model_name == "resnet50.a1_in1k"
    assert loaded.image_ids == ["img-1", "img-2"]
    assert loaded.feature_names == ["f0", "f1", "f2"]
    assert loaded.metadata["pretrained"] is True
    assert loaded.features.dtype == np.float32
    np.testing.assert_allclose(loaded.features, features)


def test_selector_feature_cache_rejects_shape_mismatch(tmp_path) -> None:
    from learned_tta.selector_features import save_selector_features

    with pytest.raises(ValueError, match="shape"):
        save_selector_features(
            path=tmp_path / "features_1d.npz",
            split="public_train",
            model_name="resnet50.a1_in1k",
            image_ids=["img-1"],
            features=np.zeros(3, dtype=np.float32),
            feature_names=["f0", "f1", "f2"],
        )
    with pytest.raises(ValueError, match="image_ids"):
        save_selector_features(
            path=tmp_path / "features.npz",
            split="public_train",
            model_name="resnet50.a1_in1k",
            image_ids=["img-1"],
            features=np.zeros((2, 3), dtype=np.float32),
            feature_names=["f0", "f1", "f2"],
        )
    with pytest.raises(ValueError, match="feature_names"):
        save_selector_features(
            path=tmp_path / "features_names.npz",
            split="public_train",
            model_name="resnet50.a1_in1k",
            image_ids=["img-1", "img-2"],
            features=np.zeros((2, 3), dtype=np.float32),
            feature_names=["f0", "f1"],
        )


def test_selector_feature_cache_rejects_corrupt_npz(tmp_path) -> None:
    from learned_tta.selector_features import load_selector_features

    def write_npz(path, *, features, image_ids, feature_names, metadata_json="[{}]") -> None:
        np.savez_compressed(
            path,
            version=np.array([1], dtype=np.int64),
            split=np.array(["public_train"]),
            model_name=np.array(["fake"]),
            image_ids=np.asarray(image_ids, dtype=str),
            features=np.asarray(features, dtype=np.float32),
            feature_names=np.asarray(feature_names, dtype=str),
            metadata_json=np.array([metadata_json]),
        )

    path = tmp_path / "bad_features.npz"
    write_npz(path, features=np.zeros(3), image_ids=["img-1"], feature_names=["f0"])
    with pytest.raises(ValueError, match="shape"):
        load_selector_features(path)

    write_npz(
        path, features=np.zeros((2, 3)), image_ids=["img-1"], feature_names=["f0", "f1", "f2"]
    )
    with pytest.raises(ValueError, match="image_ids"):
        load_selector_features(path)

    write_npz(path, features=np.zeros((2, 3)), image_ids=["img-1", "img-2"], feature_names=["f0"])
    with pytest.raises(ValueError, match="feature_names"):
        load_selector_features(path)

    write_npz(
        path,
        features=np.zeros((1, 1)),
        image_ids=["img-1"],
        feature_names=["f0"],
        metadata_json="[]",
    )
    with pytest.raises(ValueError, match="metadata"):
        load_selector_features(path)
