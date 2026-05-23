"""Split-role validation for leakage-safe experiment commands."""

from __future__ import annotations

PUBLIC_TRAIN_SPLIT = "public_train"
PUBLIC_VAL_SPLIT = "public_val"
PRIVATE_SPLIT = "private"


def validate_selector_target_splits(train_split: str, val_split: str) -> None:
    """Require selector targets to be built only from public train/val splits."""

    if train_split != PUBLIC_TRAIN_SPLIT:
        raise ValueError(
            f"train_split must be {PUBLIC_TRAIN_SPLIT}; got {train_split!r}. "
            "Private split is reserved for final evaluation."
        )
    if val_split != PUBLIC_VAL_SPLIT:
        raise ValueError(
            f"val_split must be {PUBLIC_VAL_SPLIT}; got {val_split!r}. "
            "Private split is reserved for final evaluation."
        )


def validate_public_tuning_split(split: str, command: str) -> None:
    """Require model selection, TTA tuning, and aggregation training on public-val."""

    if split != PUBLIC_VAL_SPLIT:
        raise ValueError(
            f"{command} split must be {PUBLIC_VAL_SPLIT}; got {split!r}. "
            "Tune hyperparameters and aggregation weights only on public validation."
        )


def validate_private_evaluation_split(split: str, command: str) -> None:
    """Require final evaluation commands to run only on the private split."""

    if split != PRIVATE_SPLIT:
        raise ValueError(
            f"{command} split must be {PRIVATE_SPLIT}; got {split!r}. "
            "Use public validation for tuning and private only for final metrics."
        )
