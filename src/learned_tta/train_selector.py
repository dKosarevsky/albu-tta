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


@dataclass(frozen=True, slots=True)
class SelectorLossParts:
    """Named selector loss components."""

    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    usefulness_bce: torch.Tensor
    listwise_topk: torch.Tensor


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


def listwise_topk_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Cross-entropy against the target top-k augmentation membership per image."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if predictions.ndim != 2:
        raise ValueError("predictions and targets must have shape [batch, augmentations]")
    if top_k <= 0:
        return predictions.new_zeros(())

    capped_top_k = min(top_k, predictions.shape[1])
    target_topk = targets.topk(k=capped_top_k, dim=1).indices
    target_distribution = torch.zeros_like(targets).scatter(
        dim=1,
        index=target_topk,
        value=1.0 / capped_top_k,
    )
    log_prob = F.log_softmax(predictions, dim=1)
    return -(target_distribution * log_prob).sum(dim=1).mean()


def selector_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rank_weight: float = 0.2,
    useful_logits: torch.Tensor | None = None,
    gain: torch.Tensor | None = None,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    identity_index: int | None = None,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
) -> torch.Tensor:
    """SmoothL1 regression loss plus pairwise rank loss."""

    return selector_loss_components(
        predictions=predictions,
        targets=targets,
        rank_weight=rank_weight,
        useful_logits=useful_logits,
        gain=gain,
        usefulness_tau=usefulness_tau,
        usefulness_weight=usefulness_weight,
        identity_index=identity_index,
        listwise_weight=listwise_weight,
        listwise_top_k=listwise_top_k,
    ).total


def selector_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rank_weight: float = 0.2,
    useful_logits: torch.Tensor | None = None,
    gain: torch.Tensor | None = None,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    identity_index: int | None = None,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
) -> SelectorLossParts:
    """Return SmoothL1, pairwise rank, usefulness BCE, and weighted total."""

    regression = F.smooth_l1_loss(predictions, targets)
    ranking = pairwise_rank_loss(predictions, targets)
    listwise = listwise_topk_loss(predictions, targets, top_k=listwise_top_k)
    usefulness_bce = _usefulness_bce_loss(
        useful_logits=useful_logits,
        gain=gain,
        usefulness_tau=usefulness_tau,
        identity_index=identity_index,
    )
    total = (
        regression
        + rank_weight * ranking
        + usefulness_weight * usefulness_bce
        + listwise_weight * listwise
    )
    return SelectorLossParts(
        total=total,
        regression=regression,
        ranking=ranking,
        usefulness_bce=usefulness_bce,
        listwise_topk=listwise,
    )


def _usefulness_bce_loss(
    useful_logits: torch.Tensor | None,
    gain: torch.Tensor | None,
    usefulness_tau: float,
    identity_index: int | None,
) -> torch.Tensor:
    if useful_logits is None:
        reference = gain if gain is not None else torch.zeros(())
        return reference.new_zeros(())
    if gain is None:
        raise ValueError("gain is required when useful_logits are provided")
    if useful_logits.shape != gain.shape:
        raise ValueError("useful_logits and gain must have matching shapes")
    if useful_logits.ndim != 2:
        raise ValueError("useful_logits and gain must have shape [batch, augmentations]")

    labels = (gain > usefulness_tau).to(dtype=useful_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(useful_logits, labels, reduction="none")
    mask = torch.ones_like(loss, dtype=torch.bool)
    if identity_index is not None:
        if identity_index < 0 or identity_index >= loss.shape[1]:
            raise ValueError("identity_index is out of bounds")
        mask[:, identity_index] = False
    if not bool(mask.any()):
        return useful_logits.new_zeros(())
    return loss[mask].mean()


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
        (pred_centered.square().sum(dim=1) * target_centered.square().sum(dim=1)).clamp_min(1e-12)
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
    target_kind: str = "gain",
    higher_is_better: bool = True,
    usefulness_head: bool = False,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
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
    checkpoint["target_kind"] = target_kind
    checkpoint["higher_is_better"] = higher_is_better
    checkpoint["usefulness_head"] = usefulness_head
    checkpoint["usefulness_tau"] = usefulness_tau
    checkpoint["usefulness_weight"] = usefulness_weight
    checkpoint["listwise_weight"] = listwise_weight
    checkpoint["listwise_top_k"] = listwise_top_k
    torch.save(checkpoint, state.path)
    return CheckpointState(best_val_nll=val_nll, best_epoch=epoch, path=state.path)


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    device: torch.device,
    rank_weight: float = 0.2,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    identity_index: int | None = None,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
) -> dict[str, float]:
    """Train selector for one epoch over `(images, target_z)` batches."""

    model.train()
    losses: list[float] = []
    regression_losses: list[float] = []
    rank_losses: list[float] = []
    usefulness_bces: list[float] = []
    listwise_topk_losses: list[float] = []
    for batch in dataloader:
        images, targets, gain = _unpack_selector_batch(batch)
        images = images.to(device)
        targets = targets.to(device)
        gain = gain.to(device) if gain is not None else None
        optimizer.zero_grad(set_to_none=True)
        predictions, useful_logits = _forward_selector_heads(model, images)
        loss_parts = selector_loss_components(
            predictions=predictions,
            targets=targets,
            rank_weight=rank_weight,
            useful_logits=useful_logits,
            gain=gain,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            identity_index=identity_index,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
        )
        loss = loss_parts.total
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        regression_losses.append(float(loss_parts.regression.detach().cpu().item()))
        rank_losses.append(float(loss_parts.ranking.detach().cpu().item()))
        usefulness_bces.append(float(loss_parts.usefulness_bce.detach().cpu().item()))
        listwise_topk_losses.append(float(loss_parts.listwise_topk.detach().cpu().item()))

    return {
        "loss": _mean(losses),
        "regression_loss": _mean(regression_losses),
        "rank_loss": _mean(rank_losses),
        "usefulness_bce": _mean(usefulness_bces),
        "listwise_topk_loss": _mean(listwise_topk_losses),
    }


@torch.inference_mode()
def evaluate_regression(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    rank_weight: float = 0.2,
    usefulness_tau: float = 0.01,
    usefulness_weight: float = 0.0,
    identity_index: int | None = None,
    listwise_weight: float = 0.0,
    listwise_top_k: int = 1,
) -> dict[str, float]:
    """Evaluate selector regression loss and Spearman rank correlation."""

    model.eval()
    losses: list[float] = []
    regression_losses: list[float] = []
    rank_losses: list[float] = []
    usefulness_bces: list[float] = []
    listwise_topk_losses: list[float] = []
    correlations: list[float] = []
    for batch in dataloader:
        images, targets, gain = _unpack_selector_batch(batch)
        images = images.to(device)
        targets = targets.to(device)
        gain = gain.to(device) if gain is not None else None
        predictions, useful_logits = _forward_selector_heads(model, images)
        loss_parts = selector_loss_components(
            predictions=predictions,
            targets=targets,
            rank_weight=rank_weight,
            useful_logits=useful_logits,
            gain=gain,
            usefulness_tau=usefulness_tau,
            usefulness_weight=usefulness_weight,
            identity_index=identity_index,
            listwise_weight=listwise_weight,
            listwise_top_k=listwise_top_k,
        )
        losses.append(float(loss_parts.total.cpu().item()))
        regression_losses.append(float(loss_parts.regression.cpu().item()))
        rank_losses.append(float(loss_parts.ranking.cpu().item()))
        usefulness_bces.append(float(loss_parts.usefulness_bce.cpu().item()))
        listwise_topk_losses.append(float(loss_parts.listwise_topk.cpu().item()))
        correlations.append(spearman_correlation(predictions.cpu(), targets.cpu()))

    return {
        "loss": _mean(losses),
        "regression_loss": _mean(regression_losses),
        "rank_loss": _mean(rank_losses),
        "usefulness_bce": _mean(usefulness_bces),
        "listwise_topk_loss": _mean(listwise_topk_losses),
        "spearman": _mean(correlations),
    }


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1)
    ranks = torch.zeros_like(values, dtype=torch.float32)
    rank_values = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)
    return ranks.scatter(dim=1, index=order, src=rank_values.expand_as(values))


def _unpack_selector_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 2:
        images, targets = batch
        return images, targets, None
    if len(batch) == 3:
        images, targets, gain = batch
        return images, targets, gain
    raise ValueError("selector batch must contain images, targets, and optional gain")


def _forward_selector_heads(
    model: nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if hasattr(model, "forward_heads"):
        outputs = model.forward_heads(images)  # type: ignore[attr-defined]
        return outputs.gain, outputs.useful_logits
    return model(images), None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
