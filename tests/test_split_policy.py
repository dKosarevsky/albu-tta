from __future__ import annotations

import pytest

from learned_tta.split_policy import (
    validate_private_evaluation_split,
    validate_public_tuning_split,
    validate_selector_target_splits,
)


def test_selector_target_splits_accept_public_contract() -> None:
    validate_selector_target_splits(train_split="public_train", val_split="public_val")


@pytest.mark.parametrize(
    ("train_split", "val_split", "match"),
    [
        ("private", "public_val", "train_split must be public_train"),
        ("public_train", "private", "val_split must be public_val"),
    ],
)
def test_selector_target_splits_reject_private_leakage(
    train_split: str,
    val_split: str,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_selector_target_splits(train_split=train_split, val_split=val_split)


@pytest.mark.parametrize(
    "split",
    ["public_train", "private", "public"],
)
def test_public_tuning_split_rejects_non_public_val(split: str) -> None:
    with pytest.raises(ValueError, match="tune-tta split must be public_val"):
        validate_public_tuning_split(split, command="tune-tta")


def test_public_tuning_split_accepts_public_val() -> None:
    validate_public_tuning_split("public_val", command="tune-tta")


@pytest.mark.parametrize(
    "split",
    ["public_train", "public_val", "public"],
)
def test_private_evaluation_split_rejects_non_private(split: str) -> None:
    with pytest.raises(ValueError, match="evaluate-private split must be private"):
        validate_private_evaluation_split(split, command="evaluate-private")


def test_private_evaluation_split_accepts_private() -> None:
    validate_private_evaluation_split("private", command="evaluate-private")
