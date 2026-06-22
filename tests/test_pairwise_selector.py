from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from learned_tta.cache import TeacherShard, write_teacher_shard
from learned_tta.pairwise_selector import (
    PairwiseFeatureBundle,
    PairwiseSelectorMLP,
    _fit_optional_pairwise_feature_projection,
    _load_confidence_bucket_policy,
    _prepare_optional_image_features,
    apply_pairwise_feature_projection,
    build_pairwise_feature_bundle,
    build_pairwise_hard_example_weights,
    build_pairwise_inference_bundle,
    build_pairwise_marginal_logit_gain_targets,
    evaluate_pairwise_selector_from_artifacts,
    fit_pairwise_feature_projection,
    pairwise_listwise_topk_loss,
    pairwise_policy_loss,
    pairwise_topk_kl_loss,
    train_pairwise_selector_comparison_from_artifacts,
)
from learned_tta.selector_features import save_selector_features
from learned_tta.targets import TargetStats, save_selector_targets


def test_build_pairwise_feature_bundle_concatenates_image_and_aug_features(
    tmp_path: Path,
) -> None:
    artifacts = _write_pairwise_artifacts(tmp_path)
    features_path = save_selector_features(
        tmp_path / "features.npz",
        split="public_train",
        model_name="fake",
        image_ids=["image-0", "image-1"],
        features=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        feature_names=["pre_f0", "pre_f1", "pre_f2"],
    )

    bundle = build_pairwise_feature_bundle(
        manifest_path=artifacts["manifest"],
        targets_path=artifacts["targets"],
        cache_dir=artifacts["cache_dir"],
        identity_aug_id="aug_000",
        features_path=features_path,
    )

    assert bundle.image_ids == ["image-0", "image-1"]
    assert bundle.aug_ids == ["aug_000", "aug_001"]
    assert bundle.features.shape == (4, 17)
    assert bundle.targets.tolist() == pytest.approx([0.0, 0.4, 0.0, -0.2])
    np.testing.assert_allclose(
        bundle.target_matrix,
        np.asarray([[0.0, 0.4], [0.0, -0.2]], dtype=np.float32),
    )
    assert bundle.row_image_indices.tolist() == [0, 0, 1, 1]
    assert bundle.row_aug_indices.tolist() == [0, 1, 0, 1]
    assert "clean_true_prob" in bundle.feature_names
    assert "clean_top1_logit" in bundle.feature_names
    assert bundle.feature_names[-2:] == ["aug_onehot:aug_000", "aug_onehot:aug_001"]


def test_build_pairwise_feature_bundle_supports_top1_delta_targets(tmp_path: Path) -> None:
    artifacts = _write_pairwise_artifacts(tmp_path)

    bundle = build_pairwise_feature_bundle(
        manifest_path=artifacts["manifest"],
        targets_path=artifacts["targets"],
        cache_dir=artifacts["cache_dir"],
        identity_aug_id="aug_000",
        target_mode="top1_delta",
    )

    np.testing.assert_allclose(
        bundle.target_matrix,
        np.asarray([[0.0, 0.0], [0.0, -1.0]], dtype=np.float32),
    )
    assert bundle.targets.tolist() == pytest.approx([0.0, 0.0, 0.0, -1.0])


def test_build_pairwise_marginal_logit_gain_targets_scores_ensemble_contribution() -> None:
    logits_by_aug = {
        "aug_000": np.asarray([[1.0, 0.0], [0.2, 0.8]], dtype=np.float32),
        "aug_001": np.asarray([[3.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        "aug_002": np.asarray([[0.0, 3.0], [0.0, 1.5]], dtype=np.float32),
    }
    labels = np.asarray([0, 1], dtype=np.int64)

    targets = build_pairwise_marginal_logit_gain_targets(
        logits_by_aug=logits_by_aug,
        aug_ids=["aug_000", "aug_001", "aug_002"],
        class_idxs=labels,
        identity_aug_id="aug_000",
    )

    assert targets.shape == (2, 3)
    np.testing.assert_allclose(targets[:, 0], np.zeros(2), atol=1e-6)
    assert targets[0, 1] > 0.0
    assert targets[0, 2] < 0.0
    assert targets[1, 1] < 0.0
    assert targets[1, 2] > 0.0


def test_build_pairwise_feature_bundle_supports_marginal_logit_gain_targets(
    tmp_path: Path,
) -> None:
    artifacts = _write_pairwise_artifacts(tmp_path)

    bundle = build_pairwise_feature_bundle(
        manifest_path=artifacts["manifest"],
        targets_path=artifacts["targets"],
        cache_dir=artifacts["cache_dir"],
        identity_aug_id="aug_000",
        target_mode="marginal_logit_gain",
    )

    assert bundle.target_matrix.shape == (2, 2)
    np.testing.assert_allclose(bundle.target_matrix[:, 0], np.zeros(2), atol=1e-6)
    assert bundle.target_matrix[0, 1] > 0.0
    assert bundle.target_matrix[1, 1] < 0.0


def test_build_pairwise_inference_bundle_does_not_require_targets(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)

    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )

    assert bundle.image_ids == ["image-0", "image-1"]
    assert bundle.aug_ids == ["aug_000", "aug_001"]
    assert bundle.features.shape == (4, 14)
    assert bundle.target_matrix.shape == (2, 2)
    assert bundle.targets.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_build_pairwise_inference_bundle_accepts_precomputed_features(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    features_path = save_selector_features(
        tmp_path / "features.npz",
        split="private",
        model_name="fake",
        image_ids=["image-0", "image-1"],
        features=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        feature_names=["pre_f0", "pre_f1", "pre_f2"],
    )

    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
        features_path=features_path,
    )

    assert bundle.features.shape == (4, 17)
    assert "pre_f0" in bundle.feature_names


def test_build_pairwise_feature_bundle_projects_precomputed_features(tmp_path: Path) -> None:
    artifacts = _write_pairwise_artifacts(tmp_path)
    features_path = save_selector_features(
        tmp_path / "features.npz",
        split="public_train",
        model_name="fake",
        image_ids=["image-0", "image-1"],
        features=np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            dtype=np.float32,
        ),
        feature_names=["pre_f0", "pre_f1", "pre_f2", "pre_f3"],
    )

    bundle = build_pairwise_feature_bundle(
        manifest_path=artifacts["manifest"],
        targets_path=artifacts["targets"],
        cache_dir=artifacts["cache_dir"],
        identity_aug_id="aug_000",
        features_path=features_path,
        feature_projection_dim=2,
        feature_projection_seed=123,
    )

    assert bundle.features.shape == (4, 16)
    assert bundle.feature_names[-4:-2] == [
        "projected_feature_0000",
        "projected_feature_0001",
    ]


def test_fit_pairwise_feature_projection_pca_whitens_features() -> None:
    features = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 1.0],
            [3.0, 3.0, 2.0],
            [4.0, 4.0, 3.0],
        ],
        dtype=np.float32,
    )

    projection = fit_pairwise_feature_projection(
        features,
        projection_dim=2,
        method="pca_whiten",
    )

    projected = (features - projection["mean"]) @ projection["components"]
    projected = projected / projection["scale"]
    assert projected.shape == (4, 2)
    np.testing.assert_allclose(projected.mean(axis=0), np.zeros(2), atol=1e-6)
    assert np.all(np.isfinite(projected))


def test_pairwise_feature_projection_validates_inputs() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        fit_pairwise_feature_projection(np.asarray([1.0, 2.0], dtype=np.float32), projection_dim=1)
    with pytest.raises(ValueError, match="positive"):
        fit_pairwise_feature_projection(features, projection_dim=0)
    with pytest.raises(ValueError, match="pca_whiten"):
        fit_pairwise_feature_projection(features, projection_dim=1, method="random")
    with pytest.raises(ValueError, match="positive"):
        _prepare_optional_image_features(
            features,
            ["f0", "f1"],
            projection_dim=0,
            projection_seed=0,
            projection_method="random",
            projection_state=None,
        )
    with pytest.raises(ValueError, match="feature_projection_method"):
        _prepare_optional_image_features(
            features,
            ["f0", "f1"],
            projection_dim=1,
            projection_seed=0,
            projection_method="bad",
            projection_state=None,
        )

    passthrough, names = _prepare_optional_image_features(
        features,
        ["f0", "f1"],
        projection_dim=4,
        projection_seed=0,
        projection_method="random",
        projection_state=None,
    )

    np.testing.assert_allclose(passthrough, features)
    assert names == ["f0", "f1"]


def test_apply_pairwise_feature_projection_validates_state() -> None:
    features = np.asarray([[1.0, 2.0]], dtype=np.float32)
    valid_state = {
        "method": "pca_whiten",
        "mean": np.zeros(2, dtype=np.float32),
        "components": np.ones((2, 1), dtype=np.float32),
        "scale": np.ones(1, dtype=np.float32),
        "feature_names": ["pca_whiten_feature_0000"],
    }

    with pytest.raises(ValueError, match="unsupported"):
        apply_pairwise_feature_projection(features, {"method": "random"})
    with pytest.raises(ValueError, match="shape"):
        apply_pairwise_feature_projection(np.asarray([1.0, 2.0], dtype=np.float32), valid_state)
    with pytest.raises(ValueError, match="mean"):
        apply_pairwise_feature_projection(features, {**valid_state, "mean": np.zeros(3)})
    with pytest.raises(ValueError, match="components"):
        apply_pairwise_feature_projection(features, {**valid_state, "components": np.ones((3, 1))})
    with pytest.raises(ValueError, match="scale"):
        apply_pairwise_feature_projection(features, {**valid_state, "scale": np.ones(2)})


def test_optional_pca_projection_requires_train_features() -> None:
    assert (
        _fit_optional_pairwise_feature_projection(
            features_path=None,
            projection_dim=1,
            projection_method="random",
        )
        is None
    )
    assert (
        _fit_optional_pairwise_feature_projection(
            features_path=None,
            projection_dim=None,
            projection_method="pca_whiten",
        )
        is None
    )
    with pytest.raises(ValueError, match="train-features"):
        _fit_optional_pairwise_feature_projection(
            features_path=None,
            projection_dim=1,
            projection_method="pca_whiten",
        )


def test_pairwise_score_matrix_rejects_wrong_row_count(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )

    with pytest.raises(ValueError, match="row_scores shape"):
        bundle.score_matrix(np.zeros(3, dtype=np.float32))


def test_build_pairwise_hard_example_weights_upweights_clean_wrong_images(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )

    weights = build_pairwise_hard_example_weights(
        bundle,
        hard_example_weight=2.0,
        confidence_threshold=0.75,
    )

    np.testing.assert_allclose(weights.reshape(2, 2)[0], np.ones(2))
    np.testing.assert_allclose(weights.reshape(2, 2)[1], np.full(2, 3.0))


def test_pairwise_selector_mlp_scores_one_row_per_image_aug_pair() -> None:
    model = PairwiseSelectorMLP(input_dim=5, hidden_dim=7)
    scores = model(torch.zeros((4, 5), dtype=torch.float32))

    assert scores.shape == (4,)


def test_pairwise_policy_loss_weights_positive_gain_rows() -> None:
    predicted = torch.zeros(2, dtype=torch.float32)
    target = torch.tensor([0.0, 1.0], dtype=torch.float32)

    unweighted = pairwise_policy_loss(
        predicted_gain=predicted,
        target_gain=target,
        positive_gain_weight=0.0,
    )
    weighted = pairwise_policy_loss(
        predicted_gain=predicted,
        target_gain=target,
        positive_gain_weight=3.0,
    )

    assert weighted["regression_loss"] > unweighted["regression_loss"]
    assert weighted["loss"] > unweighted["loss"]


def test_pairwise_policy_loss_adds_usefulness_bce() -> None:
    predicted = torch.zeros(2, dtype=torch.float32)
    target = torch.tensor([-0.1, 0.2], dtype=torch.float32)

    without_bce = pairwise_policy_loss(
        predicted_gain=predicted,
        target_gain=target,
        usefulness_weight=0.0,
    )
    with_bce = pairwise_policy_loss(
        predicted_gain=predicted,
        target_gain=target,
        usefulness_logits=predicted,
        usefulness_tau=0.01,
        usefulness_weight=0.5,
    )

    assert with_bce["usefulness_bce"] > 0.0
    assert with_bce["loss"] > without_bce["loss"]


def test_pairwise_listwise_topk_loss_rewards_matching_topk_membership() -> None:
    targets = torch.tensor([[0.0, 1.0, 0.5]], dtype=torch.float32)
    good_scores = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
    bad_scores = torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32)

    assert pairwise_listwise_topk_loss(good_scores, targets, top_k=2) < pairwise_listwise_topk_loss(
        bad_scores,
        targets,
        top_k=2,
    )


def test_pairwise_topk_kl_loss_prefers_oracle_topk_distribution() -> None:
    targets = torch.tensor([[0.0, 2.0, 1.0, -1.0]], dtype=torch.float32)
    good_scores = torch.tensor([[0.0, 3.0, 1.5, -2.0]], dtype=torch.float32)
    bad_scores = torch.tensor([[3.0, 0.0, -1.0, 1.5]], dtype=torch.float32)

    assert pairwise_topk_kl_loss(
        good_scores,
        targets,
        top_k=2,
        target_temperature=0.5,
    ) < pairwise_topk_kl_loss(
        bad_scores,
        targets,
        top_k=2,
        target_temperature=0.5,
    )


def test_pairwise_listwise_losses_validate_inputs() -> None:
    predicted = torch.zeros((2, 3), dtype=torch.float32)
    target = torch.zeros((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="same shape"):
        pairwise_listwise_topk_loss(predicted, torch.zeros((2, 2)), top_k=1)
    with pytest.raises(ValueError, match="images"):
        pairwise_listwise_topk_loss(torch.zeros(3), torch.zeros(3), top_k=1)
    with pytest.raises(ValueError, match="positive"):
        pairwise_listwise_topk_loss(predicted, target, top_k=0)
    with pytest.raises(ValueError, match="same shape"):
        pairwise_topk_kl_loss(predicted, torch.zeros((2, 2)), top_k=1)
    with pytest.raises(ValueError, match="images"):
        pairwise_topk_kl_loss(torch.zeros(3), torch.zeros(3), top_k=1)
    with pytest.raises(ValueError, match="positive"):
        pairwise_topk_kl_loss(predicted, target, top_k=0)
    with pytest.raises(ValueError, match="temperature"):
        pairwise_topk_kl_loss(predicted, target, top_k=1, target_temperature=0.0)


def test_pairwise_policy_loss_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        pairwise_policy_loss(torch.zeros(2), torch.zeros(1))

    with pytest.raises(ValueError, match="row_weights"):
        pairwise_policy_loss(
            torch.zeros(2),
            torch.zeros(2),
            row_weights=torch.ones(1),
        )

    with pytest.raises(ValueError, match="usefulness_logits"):
        pairwise_policy_loss(
            torch.zeros(2),
            torch.zeros(2),
            usefulness_logits=torch.zeros(1),
            usefulness_weight=0.5,
        )


def test_train_pairwise_selector_cli_writes_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    artifacts = _write_pairwise_artifacts(tmp_path)
    features_path = save_selector_features(
        tmp_path / "features.npz",
        split="public_train",
        model_name="fake",
        image_ids=["image-0", "image-1"],
        features=np.asarray(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            dtype=np.float32,
        ),
        feature_names=["pre_f0", "pre_f1", "pre_f2", "pre_f3"],
    )
    output_dir = tmp_path / "pairwise"

    main(
        [
            "train-pairwise-selector",
            "--train-manifest",
            str(artifacts["manifest"]),
            "--val-manifest",
            str(artifacts["manifest"]),
            "--train-targets",
            str(artifacts["targets"]),
            "--val-targets",
            str(artifacts["targets"]),
            "--cache-dir",
            str(artifacts["cache_dir"]),
            "--output-dir",
            str(output_dir),
            "--train-features",
            str(features_path),
            "--val-features",
            str(features_path),
            "--feature-projection-dim",
            "2",
            "--feature-projection-seed",
            "123",
            "--feature-projection-method",
            "pca_whiten",
            "--listwise-weight",
            "0.1",
            "--listwise-top-k",
            "1",
            "--listwise-loss",
            "topk_kl",
            "--hard-example-weight",
            "2.0",
            "--hard-example-confidence-threshold",
            "0.75",
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
        ]
    )
    captured = capsys.readouterr()

    assert "pairwise selector: wrote" in captured.out
    assert (output_dir / "pairwise_selector_summary.csv").exists()


def test_train_pairwise_selector_comparison_writes_nll_and_top1_variants(tmp_path: Path) -> None:
    artifacts = _write_pairwise_artifacts(tmp_path)
    output_dir = tmp_path / "pairwise_comparison"

    summary = train_pairwise_selector_comparison_from_artifacts(
        train_manifest_path=artifacts["manifest"],
        val_manifest_path=artifacts["manifest"],
        train_targets_path=artifacts["targets"],
        val_targets_path=artifacts["targets"],
        cache_dir=artifacts["cache_dir"],
        output_dir=output_dir,
        top_k_grid=[1],
        batch_size=2,
        epochs=1,
        hidden_dim=8,
    )
    table = pd.read_csv(summary.results_csv)

    assert table["variant"].tolist() == ["pairwise_nll_gain", "pairwise_top1_delta"]
    assert table["target_mode"].tolist() == ["nll_gain", "top1_delta"]
    assert table["selection_metric"].tolist() == ["val_tta_nll", "val_tta_top1"]
    assert table["best_val_top1"].notna().all()
    assert table["best_val_nll"].notna().all()


def test_evaluate_pairwise_selector_from_artifacts_writes_metrics_and_error_analysis(
    tmp_path: Path,
) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )
    checkpoint_path = _write_pairwise_checkpoint(tmp_path, bundle)

    summary = evaluate_pairwise_selector_from_artifacts(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "eval",
        identity_aug_id="aug_000",
        top_k=1,
        batch_size=2,
    )
    metrics = pd.read_csv(summary.metrics_csv)
    scores = np.load(summary.scores_npz)
    error_analysis = pd.read_csv(summary.error_analysis_csv)

    assert metrics["strategy"].tolist() == [
        "clean",
        "pairwise_topk_uniform",
        "pairwise_topk_uniform_softmax_weighted",
        "pairwise_topk_uniform_confidence_adaptive",
        "oracle_topk_uniform",
    ]
    assert summary.metrics["top1"] == pytest.approx(1.0)
    assert scores["predicted_gain"].shape == (2, 2)
    assert error_analysis["clean_wrong_tta_right"].sum() == 1


def test_evaluate_pairwise_selector_from_artifacts_applies_confidence_policy(
    tmp_path: Path,
) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )
    checkpoint_path = _write_pairwise_checkpoint(tmp_path, bundle)
    confidence_policy_path = tmp_path / "confidence_policy.json"
    confidence_policy_path.write_text(
        json.dumps({"confidence_bins": [0.0, 1.01], "bucket_k": [0]}),
        encoding="utf-8",
    )

    summary = evaluate_pairwise_selector_from_artifacts(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "eval_policy",
        identity_aug_id="aug_000",
        top_k=1,
        batch_size=2,
        confidence_policy_path=confidence_policy_path,
    )
    metrics = pd.read_csv(summary.metrics_csv)

    assert "pairwise_topk_uniform_confidence_policy" in metrics["strategy"].tolist()


def test_load_confidence_bucket_policy_validates_json(tmp_path: Path) -> None:
    policy_path = tmp_path / "policy.json"

    policy_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        _load_confidence_bucket_policy(policy_path)

    policy_path.write_text(json.dumps({"confidence_bins": [0.0, 1.01]}), encoding="utf-8")
    with pytest.raises(ValueError, match="confidence_bins and bucket_k"):
        _load_confidence_bucket_policy(policy_path)

    policy_path.write_text(
        json.dumps({"confidence_bins": [0.0, 0.5, 1.01], "bucket_k": [1]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bucket_k"):
        _load_confidence_bucket_policy(policy_path)

    policy_path.write_text(
        json.dumps({"confidence_bins": [0.0, 1.01], "bucket_k": [1]}),
        encoding="utf-8",
    )

    assert _load_confidence_bucket_policy(policy_path) == {
        "confidence_bins": [0.0, 1.01],
        "bucket_k": [1],
    }


def test_evaluate_pairwise_selector_validates_checkpoint_feature_names(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )
    checkpoint_path = tmp_path / "pairwise_selector_bad_features.pt"
    model = PairwiseSelectorMLP(input_dim=bundle.features.shape[1], hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": bundle.features.shape[1],
            "hidden_dim": 8,
            "aug_ids": bundle.aug_ids,
            "feature_names": ["wrong_feature_name"],
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="feature_names"):
        evaluate_pairwise_selector_from_artifacts(
            manifest_path=artifacts["manifest"],
            cache_dir=artifacts["cache_dir"],
            checkpoint_path=checkpoint_path,
            output_dir=tmp_path / "eval",
            identity_aug_id="aug_000",
            top_k=1,
            batch_size=2,
        )


def test_evaluate_pairwise_selector_validates_checkpoint_keys(tmp_path: Path) -> None:
    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    checkpoint_path = tmp_path / "pairwise_selector_missing_keys.pt"
    torch.save({"input_dim": 1}, checkpoint_path)

    with pytest.raises(ValueError, match="missing required keys"):
        evaluate_pairwise_selector_from_artifacts(
            manifest_path=artifacts["manifest"],
            cache_dir=artifacts["cache_dir"],
            checkpoint_path=checkpoint_path,
            output_dir=tmp_path / "eval",
            identity_aug_id="aug_000",
            top_k=1,
            batch_size=2,
        )


def test_evaluate_pairwise_selector_cli_writes_metrics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from learned_tta.cli import main

    artifacts = _write_pairwise_inference_artifacts(tmp_path)
    bundle = build_pairwise_inference_bundle(
        manifest_path=artifacts["manifest"],
        cache_dir=artifacts["cache_dir"],
        aug_ids=["aug_000", "aug_001"],
        identity_aug_id="aug_000",
    )
    checkpoint_path = _write_pairwise_checkpoint(tmp_path, bundle)
    output_dir = tmp_path / "eval_cli"

    main(
        [
            "evaluate-pairwise-selector",
            "--manifest",
            str(artifacts["manifest"]),
            "--cache-dir",
            str(artifacts["cache_dir"]),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--feature-projection-dim",
            "2",
            "--feature-projection-seed",
            "123",
            "--top-k",
            "1",
            "--batch-size",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert "pairwise selector evaluation: wrote" in captured.out
    assert (output_dir / "pairwise_selector_metrics.csv").exists()
    assert (output_dir / "pairwise_selector_scores.npz").exists()


def _write_pairwise_artifacts(root: Path) -> dict[str, Path]:
    manifest_path = _write_manifest(root)
    targets_path = root / "targets.npz"
    save_selector_targets(
        path=targets_path,
        aug_ids=["aug_000", "aug_001"],
        image_ids=["image-0", "image-1"],
        gain=np.asarray([[0.0, 0.4], [0.0, -0.2]], dtype=np.float32),
        target_z=np.asarray([[0.0, 0.4], [0.0, -0.2]], dtype=np.float32),
        stats=TargetStats(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
        ),
    )
    cache_dir = root / "teacher_cache"
    class_idxs = np.asarray([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_train",
            aug_id="aug_000",
            image_ids=["image-0", "image-1"],
            class_idxs=class_idxs,
            logits=np.asarray([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="public_train",
            aug_id="aug_001",
            image_ids=["image-0", "image-1"],
            class_idxs=class_idxs,
            logits=np.asarray([[4.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        ),
    )
    return {"manifest": manifest_path, "targets": targets_path, "cache_dir": cache_dir}


def _write_pairwise_inference_artifacts(root: Path) -> dict[str, Path]:
    manifest_path = _write_manifest(root, split="private")
    cache_dir = root / "teacher_cache"
    class_idxs = np.asarray([0, 1], dtype=np.int64)
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="private",
            aug_id="aug_000",
            image_ids=["image-0", "image-1"],
            class_idxs=class_idxs,
            logits=np.asarray([[3.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        ),
    )
    write_teacher_shard(
        cache_dir,
        TeacherShard(
            split="private",
            aug_id="aug_001",
            image_ids=["image-0", "image-1"],
            class_idxs=class_idxs,
            logits=np.asarray([[4.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        ),
    )
    return {"manifest": manifest_path, "cache_dir": cache_dir}


def _write_pairwise_checkpoint(root: Path, bundle: PairwiseFeatureBundle) -> Path:
    checkpoint_path = root / "pairwise_selector_best.pt"
    model = PairwiseSelectorMLP(input_dim=bundle.features.shape[1], hidden_dim=8)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": bundle.features.shape[1],
            "hidden_dim": 8,
            "aug_ids": bundle.aug_ids,
            "feature_names": bundle.feature_names,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_manifest(root: Path, split: str = "public_train") -> Path:
    rows = []
    for index in range(2):
        path = root / f"image_{index}.png"
        Image.fromarray(np.full((8, 8, 3), 20 + index, dtype=np.uint8), mode="RGB").save(path)
        rows.append(
            {
                "split": split,
                "image_id": f"image-{index}",
                "class_idx": index,
                "class_name": f"class-{index}",
                "path": str(path),
            }
        )
    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path
