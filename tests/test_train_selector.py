from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from learned_tta.targets import TargetStats
from learned_tta.train_selector import (
    CheckpointState,
    evaluate_regression,
    pairwise_rank_loss,
    save_checkpoint_if_best,
    selector_loss,
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


def test_selector_loss_combines_smooth_l1_and_rank_loss(ranked_targets: torch.Tensor) -> None:
    predictions = ranked_targets + 0.1

    loss = selector_loss(predictions, ranked_targets, rank_weight=0.2)

    assert loss.ndim == 0
    assert loss.item() > 0.0


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

    assert set(metrics) == {"loss", "spearman"}
