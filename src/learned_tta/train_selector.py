"""Training primitives for the selector CNN."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from learned_tta.targets import TargetStats


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """Best-checkpoint tracking state."""

    best_val_nll: float
    path: Path
    best_epoch: int | None = None


def pairwise_rank_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise logistic rank loss over augmentation scores within each image."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if predictions.ndim != 2:
        raise ValueError("predictions and targets must have shape [batch, augmentations]")

    target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)
    prediction_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    mask = target_diff > 0
    if not bool(mask.any()):
        return predictions.new_zeros(())

    return F.softplus(-prediction_diff[mask]).mean()


def selector_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rank_weight: float = 0.2,
) -> torch.Tensor:
    """SmoothL1 regression loss plus pairwise rank loss."""

    regression = F.smooth_l1_loss(predictions, targets)
    ranking = pairwise_rank_loss(predictions, targets)
    return regression + rank_weight * ranking


def spearman_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean per-sample Spearman rank correlation."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")

    pred_ranks = _ranks(predictions)
    target_ranks = _ranks(targets)
    pred_centered = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
    target_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (pred_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)).clamp_min(
            1e-12
        )
    )
    return float((numerator / denominator).mean().item())


def save_checkpoint_if_best(
    state: CheckpointState,
    val_nll: float,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    aug_ids: list[str] | None = None,
    target_stats: TargetStats | None = None,
) -> CheckpointState:
    """Save model checkpoint when validation NLL improves."""

    if val_nll >= state.best_val_nll:
        return state

    state.path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, object] = {
        "epoch": epoch,
        "val_nll": val_nll,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if aug_ids is not None:
        checkpoint["aug_ids"] = list(aug_ids)
    if target_stats is not None:
        checkpoint["target_mean"] = target_stats.mean
        checkpoint["target_std"] = target_stats.std
    torch.save(checkpoint, state.path)
    return CheckpointState(best_val_nll=val_nll, best_epoch=epoch, path=state.path)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    rank_weight: float = 0.2,
) -> dict[str, float]:
    """Train selector for one epoch over `(images, target_z)` batches."""

    model.train()
    losses: list[float] = []
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = selector_loss(predictions, targets, rank_weight=rank_weight)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return {"loss": _mean(losses)}


@torch.inference_mode()
def evaluate_regression(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    rank_weight: float = 0.2,
) -> dict[str, float]:
    """Evaluate selector regression loss and Spearman rank correlation."""

    model.eval()
    losses: list[float] = []
    correlations: list[float] = []
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images)
        loss = selector_loss(predictions, targets, rank_weight=rank_weight)
        losses.append(float(loss.cpu().item()))
        correlations.append(spearman_correlation(predictions.cpu(), targets.cpu()))

    return {"loss": _mean(losses), "spearman": _mean(correlations)}


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1)
    ranks = torch.zeros_like(values, dtype=torch.float32)
    rank_values = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)
    return ranks.scatter(dim=1, index=order, src=rank_values.expand_as(values))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
