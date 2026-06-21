from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from learned_tta.selector_model import SelectorCNN
from learned_tta.targets import TargetStats
from learned_tta.train_selector import (
    CheckpointState,
    _unpack_selector_batch,
    _usefulness_bce_loss,
    evaluate_regression,
    listwise_topk_loss,
    pairwise_rank_loss,
    save_checkpoint_if_best,
    selector_loss,
    selector_loss_components,
    spearman_correlation,
    train_one_epoch,
)


@pytest.fixture
def ranked_targets() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 2.0, -1.0],
            [1.0, -2.0, 0.0],
        ],
        dtype=torch.float32,
    )


def test_pairwise_rank_loss_is_lower_for_correct_order(ranked_targets: torch.Tensor) -> None:
    correct_predictions = ranked_targets * 2.0
    reversed_predictions = -ranked_targets

    assert pairwise_rank_loss(correct_predictions, ranked_targets) < pairwise_rank_loss(
        reversed_predictions,
        ranked_targets,
    )


@pytest.mark.parametrize(
    ("predictions", "targets", "match"),
    [
        (
            torch.zeros(2, 3),
            torch.zeros(2, 2),
            "predictions and targets must have matching shapes",
        ),
        (
            torch.zeros(3),
            torch.zeros(3),
            "predictions and targets must have shape",
        ),
    ],
)
def test_pairwise_rank_loss_rejects_invalid_shapes(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        pairwise_rank_loss(predictions, targets)


def test_pairwise_rank_loss_returns_zero_when_targets_have_no_ordering() -> None:
    loss = pairwise_rank_loss(torch.zeros(2, 3), torch.zeros(2, 3))

    assert loss.item() == pytest.approx(0.0)


def test_selector_loss_combines_smooth_l1_and_rank_loss(ranked_targets: torch.Tensor) -> None:
    predictions = ranked_targets + 0.1

    loss = selector_loss(predictions, ranked_targets, rank_weight=0.2)

    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_selector_loss_components_add_usefulness_bce_with_identity_mask() -> None:
    predictions = torch.zeros(1, 3)
    targets = torch.zeros(1, 3)
    gain = torch.tensor([[100.0, 0.02, -0.03]], dtype=torch.float32)
    good_logits = torch.tensor([[-100.0, 10.0, -10.0]], dtype=torch.float32)
    bad_logits = torch.tensor([[100.0, -10.0, 10.0]], dtype=torch.float32)

    good = selector_loss_components(
        predictions=predictions,
        targets=targets,
        useful_logits=good_logits,
        gain=gain,
        usefulness_tau=0.01,
        usefulness_weight=0.5,
        identity_index=0,
    )
    bad = selector_loss_components(
        predictions=predictions,
        targets=targets,
        useful_logits=bad_logits,
        gain=gain,
        usefulness_tau=0.01,
        usefulness_weight=0.5,
        identity_index=0,
    )

    assert good.usefulness_bce < bad.usefulness_bce
    assert good.total < bad.total


def test_listwise_topk_loss_is_lower_when_topk_membership_matches() -> None:
    targets = torch.tensor([[0.0, 2.0, 1.0, -1.0]], dtype=torch.float32)
    good_predictions = torch.tensor([[0.0, 2.0, 1.0, -1.0]], dtype=torch.float32)
    bad_predictions = torch.tensor([[2.0, -1.0, 0.0, 1.0]], dtype=torch.float32)

    assert listwise_topk_loss(good_predictions, targets, top_k=2) < listwise_topk_loss(
        bad_predictions,
        targets,
        top_k=2,
    )


def test_listwise_topk_loss_validates_inputs() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        listwise_topk_loss(torch.zeros(1, 2), torch.zeros(1, 3), top_k=2)
    with pytest.raises(ValueError, match="shape"):
        listwise_topk_loss(torch.zeros(3), torch.zeros(3), top_k=2)
    assert listwise_topk_loss(
        torch.zeros(1, 2), torch.zeros(1, 2), top_k=0
    ).item() == pytest.approx(0.0)


def test_usefulness_bce_loss_validates_inputs() -> None:
    gain = torch.tensor([[0.0, 0.1]], dtype=torch.float32)

    assert _usefulness_bce_loss(None, gain, 0.01, identity_index=None).item() == pytest.approx(0.0)
    with pytest.raises(ValueError, match="gain is required"):
        _usefulness_bce_loss(torch.zeros(1, 2), None, 0.01, identity_index=None)
    with pytest.raises(ValueError, match="matching shapes"):
        _usefulness_bce_loss(torch.zeros(1, 3), gain, 0.01, identity_index=None)
    with pytest.raises(ValueError, match="shape"):
        _usefulness_bce_loss(torch.zeros(2), torch.zeros(2), 0.01, identity_index=None)
    with pytest.raises(ValueError, match="out of bounds"):
        _usefulness_bce_loss(torch.zeros(1, 2), gain, 0.01, identity_index=3)
    assert _usefulness_bce_loss(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        0.01,
        identity_index=0,
    ).item() == pytest.approx(0.0)


def test_unpack_selector_batch_rejects_bad_tuple_size() -> None:
    with pytest.raises(ValueError, match="selector batch"):
        _unpack_selector_batch((torch.zeros(1, 2),))


@pytest.mark.parametrize(
    ("predictions", "targets", "expected_sign"),
    [
        (torch.tensor([[0.0, 1.0, 2.0]]), torch.tensor([[0.0, 1.0, 2.0]]), 1),
        (torch.tensor([[2.0, 1.0, 0.0]]), torch.tensor([[0.0, 1.0, 2.0]]), -1),
    ],
)
def test_spearman_correlation_reports_rank_direction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    expected_sign: int,
) -> None:
    correlation = spearman_correlation(predictions.float(), targets.float())

    assert correlation * expected_sign > 0


def test_spearman_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="predictions and targets must have matching shapes"):
        spearman_correlation(torch.zeros(2, 3), torch.zeros(2, 2))


def test_save_checkpoint_if_best_only_updates_on_improvement(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    state = CheckpointState(best_val_nll=1.0, path=tmp_path / "selector.pt")
    stats = TargetStats(
        mean=np.array([0.25], dtype=np.float32),
        std=np.array([2.0], dtype=np.float32),
    )

    improved = save_checkpoint_if_best(
        state=state,
        val_nll=0.9,
        epoch=3,
        model=model,
        optimizer=optimizer,
        aug_ids=["aug_000"],
        target_stats=stats,
        usefulness_head=True,
        usefulness_tau=0.01,
        usefulness_weight=0.05,
    )
    not_improved = save_checkpoint_if_best(
        state=improved,
        val_nll=1.1,
        epoch=4,
        model=model,
        optimizer=optimizer,
        aug_ids=["aug_000"],
        target_stats=stats,
    )
    checkpoint = torch.load(improved.path, weights_only=False)

    assert improved.best_val_nll == pytest.approx(0.9)
    assert improved.best_epoch == 3
    assert improved.path.exists()
    assert checkpoint["aug_ids"] == ["aug_000"]
    assert checkpoint["target_mean"].tolist() == pytest.approx([0.25])
    assert checkpoint["target_std"].tolist() == pytest.approx([2.0])
    assert checkpoint["usefulness_head"] is True
    assert checkpoint["usefulness_tau"] == pytest.approx(0.01)
    assert checkpoint["usefulness_weight"] == pytest.approx(0.05)
    assert not_improved == improved


def test_train_one_epoch_updates_model_parameters() -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.ones(4, 4), torch.zeros(4, 3)),
        batch_size=2,
    )
    before = model.weight.detach().clone()

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics["loss"] > 0.0
    assert not torch.equal(model.weight, before)


def test_train_one_epoch_reports_usefulness_components() -> None:
    model = SelectorCNN(output_dim=3, usefulness_head=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.ones(4, 3, 16, 16),
            torch.zeros(4, 3),
            torch.tensor(
                [
                    [0.0, 0.02, -0.01],
                    [0.0, -0.03, 0.04],
                    [0.0, 0.05, -0.02],
                    [0.0, -0.01, 0.03],
                ],
                dtype=torch.float32,
            ),
        ),
        batch_size=2,
    )

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        usefulness_tau=0.01,
        usefulness_weight=0.5,
        identity_index=0,
    )

    assert metrics["loss"] > 0.0
    assert metrics["regression_loss"] >= 0.0
    assert metrics["rank_loss"] >= 0.0
    assert metrics["usefulness_bce"] > 0.0


def test_evaluate_regression_reports_loss_and_spearman() -> None:
    model = torch.nn.Linear(4, 3)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.ones(4, 4), torch.zeros(4, 3)),
        batch_size=2,
    )

    metrics = evaluate_regression(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert set(metrics) == {
        "loss",
        "regression_loss",
        "rank_loss",
        "usefulness_bce",
        "listwise_topk_loss",
        "spearman",
    }


def test_train_and_evaluate_return_zero_metrics_for_empty_dataloader() -> None:
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    empty: list[tuple[torch.Tensor, torch.Tensor]] = []

    train_metrics = train_one_epoch(
        model=model,
        dataloader=empty,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    eval_metrics = evaluate_regression(
        model=model,
        dataloader=empty,
        device=torch.device("cpu"),
    )

    assert train_metrics == {
        "loss": 0.0,
        "regression_loss": 0.0,
        "rank_loss": 0.0,
        "usefulness_bce": 0.0,
        "listwise_topk_loss": 0.0,
    }
    assert eval_metrics == {
        "loss": 0.0,
        "regression_loss": 0.0,
        "rank_loss": 0.0,
        "usefulness_bce": 0.0,
        "listwise_topk_loss": 0.0,
        "spearman": 0.0,
    }
